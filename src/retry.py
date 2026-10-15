"""Retry utilities with exponential backoff and jitter for transient API failures."""

import random
import time
from functools import wraps
from typing import Any, Callable, Optional, Sequence, Tuple, Type


class PermanentFailureError(Exception):
    """Raised when an error is permanent and should never be retried."""
    pass


def should_retry_exception(exc: Exception) -> bool:
    """Determine whether an exception is transient and safe to retry."""
    exc_str = str(exc).lower()

    # Never retry permanent authentication or bad request errors
    if any(auth_kw in exc_str for auth_kw in ("401", "unauthorized", "invalid_api_key", "invalid api key")):
        return False
    if "403" in exc_str or "forbidden" in exc_str:
        return False
    if "model_not_found" in exc_str or "404" in exc_str:
        return False

    # Retry transient rate limits and server errors
    if any(transient_kw in exc_str for transient_kw in ("429", "rate limit", "rate_limit", "tpm", "rpm")):
        return True
    if any(srv_kw in exc_str for srv_kw in ("500", "502", "503", "504", "server error", "overloaded")):
        return True
    if any(net_kw in exc_str for net_kw in ("timeout", "connection reset", "connection refused", "econnreset")):
        return True

    return False


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int, float], None]] = None,
) -> Callable:
    """Decorator for retrying a function with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exception = None

            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exception = exc
                    if not should_retry_exception(exc) or attempt == max_retries:
                        raise

                    actual_delay = delay
                    if jitter:
                        actual_delay += random.uniform(0.1, 0.5 * delay)

                    if on_retry:
                        on_retry(exc, attempt, actual_delay)

                    time.sleep(actual_delay)
                    delay *= backoff_factor

            if last_exception:
                raise last_exception

        return wrapper
    return decorator
