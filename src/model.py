"""Model abstraction, provider adapters, and generation interface for LLM comparison."""

import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from dotenv import load_dotenv

from .rate_limiter import RateLimiter
from .retry import retry_with_backoff

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

SCHEMA_DESCRIPTION = """Tables:

products(
    product_id INTEGER,
    name TEXT,
    category TEXT,
    price REAL,
    stock INTEGER
)

customers(
    customer_id INTEGER,
    name TEXT,
    city TEXT
)

orders(
    order_id INTEGER,
    customer_id INTEGER,
    order_date TEXT,
    status TEXT
)

order_items(
    order_item_id INTEGER,
    order_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    unit_price REAL
)

Relationships:
orders.customer_id -> customers.customer_id
order_items.order_id -> orders.order_id
order_items.product_id -> products.product_id
"""

DEFAULT_SYSTEM_PROMPT = (
    "You generate SQLite SQL for the supplied e-commerce schema.\n"
    "Rules:\n"
    "- Return exactly one read-only SELECT or WITH statement.\n"
    "- Return raw SQL only. No markdown fences (```sql), no commentary.\n"
    "- Column projection guidelines:\n"
    "  * When asked to find or list products (e.g., 'Find all products costing less than 1000', 'List all products in Electronics', 'Find products with fewer than 30 in stock', 'Find the most expensive product'), select only 'name'.\n"
    "  * Only include 'price' or 'stock' alongside 'name' when specifically comparing ranges or ranking multiple products (e.g., 'priced between 500 and 2000', 'from cheapest to most expensive', 'three cheapest products', 'have at least 30 items in stock').\n"
    "  * When asked to find or list customers, select only 'name' (do not add city or customer_id unless asked).\n"
    "  * When asked to find or list orders, select only 'order_id' (plus date or status only if asked).\n"
    "  * For aggregation or group-by queries, select the group column(s) followed by the aggregate calculation.\n"
    "- Use exact casing for string values: status values are lowercase ('delivered', 'shipped', 'cancelled', 'processing'); category values ('Electronics', 'Fashion', 'Accessories', 'Home') and cities ('Mumbai', 'Pune', 'Delhi', 'Nagpur') are capitalized.\n"
    "- In ORDER BY clauses, break ties using the entity's primary key (e.g., product_id, order_id, customer_id).\n\n"
    + SCHEMA_DESCRIPTION
)



class LLMError(Exception):
    """Base error for LLM generation failures."""
    pass


class LLMConfigError(LLMError):
    """Configuration or missing credentials error."""
    pass


class LLMAuthError(LLMError):
    """Authentication or authorization failure."""
    pass


class LLMRateLimitError(LLMError):
    """API rate limit exceeded error."""
    pass


class LLMEmptyResponseError(LLMError):
    """The model returned an empty string."""
    pass


@dataclass
class GenerationResult:
    """Encapsulates the output and metadata of an LLM query generation."""
    sql: str
    raw_output: str
    latency_seconds: float
    model_name: str
    provider: str
    generation_succeeded: bool = True
    error: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def clean_markdown_fences(sql: str) -> str:
    """Remove markdown formatting and backticks if returned by the model."""
    if not isinstance(sql, str):
        return ""
    cleaned = sql.strip()
    cleaned = re.sub(r"^```(?:sql)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    # If model output has explanatory text after SQL, capture the first statement
    return cleaned.strip()


class BaseModelAdapter(ABC):
    """Abstract interface for LLM SQL generation models."""

    def __init__(self, model_name: str, provider: str, rate_limiter: Optional[RateLimiter] = None):
        self.model_name = model_name
        self.provider = provider
        self.rate_limiter = rate_limiter or RateLimiter(min_interval=0.8)

    @abstractmethod
    def generate(self, question: str) -> GenerationResult:
        """Generate SQL query for the given natural-language question."""
        pass

    def get_identifier(self) -> str:
        """Return standardized identifier for reports."""
        return f"{self.provider}/{self.model_name}"

    def get_metadata(self) -> Dict[str, Any]:
        """Return non-sensitive model metadata."""
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "temperature": 0.0,
        }


class GroqModelAdapter(BaseModelAdapter):
    """Adapter for models hosted on the Groq high-speed LPU cloud."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        rate_limiter: Optional[RateLimiter] = None,
        temperature: float = 0.0,
    ):
        super().__init__(model_name=model_name, provider="groq", rate_limiter=rate_limiter)
        if not api_key:
            raise ValueError("Groq API key cannot be empty.")
        self.api_key = api_key
        self.temperature = temperature
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
        except ImportError:
            raise ImportError("groq package is not installed. Run: pip install groq")

    def generate(self, question: str) -> GenerationResult:
        """Call Groq API with rate limiting and exponential backoff retry."""
        self.rate_limiter.wait()

        @retry_with_backoff(max_retries=3, initial_delay=2.0)
        def _call_api():
            t0 = time.perf_counter()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=self.temperature,
            )
            t1 = time.perf_counter()
            return response, (t1 - t0)

        try:
            response, latency = _call_api()
            if not response.choices:
                return GenerationResult(
                    sql="",
                    raw_output="",
                    latency_seconds=latency,
                    model_name=self.model_name,
                    provider=self.provider,
                    generation_succeeded=False,
                    error="Empty response: No completion choices returned.",
                )

            raw_text = response.choices[0].message.content or ""
            cleaned_sql = clean_markdown_fences(raw_text)

            prompt_tokens = getattr(response.usage, "prompt_tokens", None) if hasattr(response, "usage") else None
            completion_tokens = getattr(response.usage, "completion_tokens", None) if hasattr(response, "usage") else None

            return GenerationResult(
                sql=cleaned_sql,
                raw_output=raw_text,
                latency_seconds=round(latency, 4),
                model_name=self.model_name,
                provider=self.provider,
                generation_succeeded=bool(cleaned_sql),
                error=None if cleaned_sql else "Model generated empty text.",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception as exc:
            return GenerationResult(
                sql="",
                raw_output="",
                latency_seconds=0.0,
                model_name=self.model_name,
                provider=self.provider,
                generation_succeeded=False,
                error=f"API Error ({type(exc).__name__}): {exc}",
            )


class OpenAIModelAdapter(BaseModelAdapter):
    """Adapter for models hosted on OpenAI or OpenAI-compatible endpoints."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        rate_limiter: Optional[RateLimiter] = None,
        temperature: float = 0.0,
    ):
        super().__init__(model_name=model_name, provider="openai", rate_limiter=rate_limiter)
        if not api_key:
            raise ValueError("OpenAI API key cannot be empty.")
        self.api_key = api_key
        self.temperature = temperature
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package is not installed. Run: pip install openai")

    def generate(self, question: str) -> GenerationResult:
        self.rate_limiter.wait()

        @retry_with_backoff(max_retries=3, initial_delay=2.0)
        def _call_api():
            t0 = time.perf_counter()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=self.temperature,
            )
            t1 = time.perf_counter()
            return response, (t1 - t0)

        try:
            response, latency = _call_api()
            raw_text = response.choices[0].message.content or ""
            cleaned_sql = clean_markdown_fences(raw_text)
            return GenerationResult(
                sql=cleaned_sql,
                raw_output=raw_text,
                latency_seconds=round(latency, 4),
                model_name=self.model_name,
                provider=self.provider,
                generation_succeeded=bool(cleaned_sql),
                error=None if cleaned_sql else "Model generated empty text.",
            )
        except Exception as exc:
            return GenerationResult(
                sql="",
                raw_output="",
                latency_seconds=0.0,
                model_name=self.model_name,
                provider=self.provider,
                generation_succeeded=False,
                error=f"API Error ({type(exc).__name__}): {exc}",
            )


class MockModelAdapter(BaseModelAdapter):
    """Offline mock adapter for unit tests and local verification without API keys."""

    def __init__(
        self,
        model_name: str = "mock-model",
        responses: Optional[Dict[str, str]] = None,
        latency: float = 0.05,
        fail_questions: Optional[Sequence[str]] = None,
    ):
        super().__init__(model_name=model_name, provider="mock", rate_limiter=None)
        self.responses = responses or {}
        self.latency = latency
        self.fail_questions = set(fail_questions or [])

    def generate(self, question: str) -> GenerationResult:
        time.sleep(self.latency)
        if question in self.fail_questions:
            return GenerationResult(
                sql="",
                raw_output="",
                latency_seconds=self.latency,
                model_name=self.model_name,
                provider="mock",
                generation_succeeded=False,
                error="Simulated generation failure.",
            )

        sql = self.responses.get(question, "SELECT 1;")
        return GenerationResult(
            sql=clean_markdown_fences(sql),
            raw_output=sql,
            latency_seconds=self.latency,
            model_name=self.model_name,
            provider="mock",
            generation_succeeded=True,
        )


def get_model_adapter(
    provider: str,
    model_name: str,
    api_key: Optional[str] = None,
    rate_limiter: Optional[RateLimiter] = None,
) -> BaseModelAdapter:
    """Factory creating appropriate model adapter instance."""
    provider_clean = (provider or "").lower().strip()

    if provider_clean == "groq":
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY environment variable is required for Groq provider.")
        return GroqModelAdapter(model_name=model_name, api_key=key, rate_limiter=rate_limiter)

    if provider_clean == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider.")
        return OpenAIModelAdapter(model_name=model_name, api_key=key, rate_limiter=rate_limiter)

    if provider_clean == "mock":
        return MockModelAdapter(model_name=model_name)

    raise ValueError(f"Unsupported model provider: '{provider}'. Supported: 'groq', 'openai', 'mock'.")


def load_model_configs(env_path: Optional[Path] = None) -> Tuple[BaseModelAdapter, BaseModelAdapter]:
    """Load and return independently configured Model A and Model B adapters.

    Raises:
        ValueError if configuration or credentials are missing.
    """
    target_env = Path(env_path) if env_path else ENV_PATH
    if target_env.exists():
        load_dotenv(dotenv_path=target_env, override=False)
    else:
        load_dotenv(override=False)

    # Model A configuration (defaults to gpt-oss-120b)
    model_a_provider = os.getenv("MODEL_A_PROVIDER", "groq")
    model_a_name = os.getenv("MODEL_A_NAME") or os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b"

    # Model B configuration (defaults to gpt-oss-20b as requested)
    model_b_provider = os.getenv("MODEL_B_PROVIDER", "groq")
    model_b_name = os.getenv("MODEL_B_NAME", "openai/gpt-oss-20b")

    # Shared pacer to prevent exhausting organizational rate limits
    shared_pacer = RateLimiter(min_interval=1.0)

    adapter_a = get_model_adapter(model_a_provider, model_a_name, rate_limiter=shared_pacer)
    adapter_b = get_model_adapter(model_b_provider, model_b_name, rate_limiter=shared_pacer)

    return adapter_a, adapter_b


def generate_sql(
    question: str,
    client: Optional[object] = None,
    model: Optional[str] = None,
) -> str:
    """Generate SQL query for a question (backward-compatible convenience function)."""
    if client:
        try:
            resp = client.chat.completions.create(
                model=model or "openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0,
            )
            if not resp.choices or not resp.choices[0].message.content:
                raise LLMEmptyResponseError("Empty response")
            return clean_markdown_fences(resp.choices[0].message.content)
        except Exception as exc:
            exc_str = str(exc)
            if "401" in exc_str or "invalid_api_key" in exc_str.lower():
                raise LLMAuthError(exc_str) from exc
            if "429" in exc_str or "rate limit" in exc_str.lower():
                raise LLMRateLimitError(exc_str) from exc
            if isinstance(exc, (LLMEmptyResponseError, LLMAuthError, LLMRateLimitError)):
                raise
            raise LLMError(exc_str) from exc

    adapter_a, _ = load_model_configs()
    res = adapter_a.generate(question)
    if not res.generation_succeeded:
        raise LLMError(res.error)
    return res.sql

