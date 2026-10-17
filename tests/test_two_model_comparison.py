"""Automated tests for two-model evaluation framework, statistics, and reporting."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.contamination import check_question_sql_leakage, compute_token_jaccard, run_contamination_audit
from src.graders import canonicalize_value, compare_results, grade_query
from src.metrics import compute_comparison_metrics, compute_model_metrics
from src.model import (
    BaseModelAdapter,
    GenerationResult,
    MockModelAdapter,
    clean_markdown_fences,
    get_model_adapter,
)
from src.rate_limiter import RateLimiter
from src.report import (
    format_comparison_text_summary,
    format_model_text_summary,
    recreate_comparison_from_files,
    save_evaluation_reports,
)
from src.retry import retry_with_backoff, should_retry_exception
from src.runner import evaluate_single_model_case
from src.statistics import (
    compute_cohens_kappa,
    compute_latency_summary,
    compute_mcnemar_test,
    wilson_score_interval,
)


class TestModelAbstraction(unittest.TestCase):
    """Test model interface, markdown cleaning, and mock adapters."""

    def test_clean_markdown_fences(self):
        self.assertEqual(clean_markdown_fences("```sql\nSELECT 1;\n```"), "SELECT 1;")
        self.assertEqual(clean_markdown_fences("```SQL\nSELECT 1;\n```"), "SELECT 1;")
        self.assertEqual(clean_markdown_fences("  SELECT 1;  "), "SELECT 1;")

    def test_mock_adapter_generation(self):
        adapter = MockModelAdapter(
            model_name="test-model",
            responses={"Question 1": "SELECT name FROM products;"},
            latency=0.01,
        )
        res = adapter.generate("Question 1")
        self.assertTrue(res.generation_succeeded)
        self.assertEqual(res.sql, "SELECT name FROM products;")
        self.assertGreater(res.latency_seconds, 0)
        self.assertEqual(res.model_name, "test-model")
        self.assertEqual(res.provider, "mock")

    def test_mock_adapter_failure(self):
        adapter = MockModelAdapter(
            model_name="failing-model",
            fail_questions=["Question Fail"],
        )
        res = adapter.generate("Question Fail")
        self.assertFalse(res.generation_succeeded)
        self.assertIn("Simulated", res.error)

    def test_factory_creates_mock(self):
        adapter = get_model_adapter(provider="mock", model_name="custom-mock")
        self.assertEqual(adapter.model_name, "custom-mock")
        self.assertEqual(adapter.provider, "mock")

    def test_factory_rejects_unknown_provider(self):
        with self.assertRaises(ValueError):
            get_model_adapter(provider="unknown_provider", model_name="xyz")


class TestRateLimiterAndRetry(unittest.TestCase):
    """Test API request pacing and retry mechanisms."""

    def test_rate_limiter_min_interval(self):
        limiter = RateLimiter(min_interval=0.05)
        w1 = limiter.wait()
        w2 = limiter.wait()
        self.assertGreaterEqual(w2, 0.0)

    def test_should_retry_exception_classification(self):
        self.assertTrue(should_retry_exception(Exception("429 rate limit exceeded")))
        self.assertTrue(should_retry_exception(Exception("503 Service Unavailable")))
        self.assertTrue(should_retry_exception(Exception("Connection reset by peer")))

        # Non-retryable permanent errors
        self.assertFalse(should_retry_exception(Exception("401 Unauthorized invalid_api_key")))
        self.assertFalse(should_retry_exception(Exception("403 Forbidden")))
        self.assertFalse(should_retry_exception(Exception("404 model_not_found")))

    def test_retry_with_backoff_succeeds_after_transient(self):
        attempts = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.01, jitter=False)
        def flaky_function():
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise Exception("429 transient rate limit")
            return "SUCCESS"

        result = flaky_function()
        self.assertEqual(result, "SUCCESS")
        self.assertEqual(attempts, 2)


class TestComparativeMetricsAndStatistics(unittest.TestCase):
    """Test paired metrics, McNemar significance, and Cohen's kappa."""

    def test_wilson_score_interval(self):
        low, high = wilson_score_interval(41, 50)
        self.assertGreater(low, 65.0)
        self.assertLess(high, 95.0)
        self.assertLess(low, high)

    def test_latency_summary(self):
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0]
        summary = compute_latency_summary(latencies)
        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["mean_seconds"], 3.0)
        self.assertEqual(summary["median_seconds"], 3.0)
        self.assertEqual(summary["min_seconds"], 1.0)
        self.assertEqual(summary["max_seconds"], 5.0)

    def test_cohens_kappa_perfect_agreement(self):
        kappa = compute_cohens_kappa(both_pass=40, both_fail=10, a_pass_b_fail=0, b_pass_a_fail=0)
        self.assertAlmostEqual(kappa, 1.0, places=2)

    def test_mcnemar_identical_models(self):
        mcnemar = compute_mcnemar_test(b=0, c=0)
        self.assertEqual(mcnemar["statistic"], 0.0)
        self.assertEqual(mcnemar["p_value_approx"], 1.0)
        self.assertFalse(mcnemar["is_statistically_significant"])

    def test_mcnemar_different_models(self):
        # 10 cases Model A passed where Model B failed, 1 case B passed where A failed
        mcnemar = compute_mcnemar_test(b=10, c=1)
        self.assertGreater(mcnemar["statistic"], 0.0)
        self.assertIn("interpretation", mcnemar)

    def test_compute_comparison_metrics(self):
        results_a = [
            {"id": 1, "passed": True, "category": "filtering", "difficulty": "easy"},
            {"id": 2, "passed": True, "category": "filtering", "difficulty": "easy"},
            {"id": 3, "passed": False, "category": "sorting", "difficulty": "medium"},
        ]
        results_b = [
            {"id": 1, "passed": True, "category": "filtering", "difficulty": "easy"},
            {"id": 2, "passed": False, "category": "filtering", "difficulty": "easy"},
            {"id": 3, "passed": False, "category": "sorting", "difficulty": "medium"},
        ]
        cm = compute_comparison_metrics(results_a, results_b, "model_a", "model_b")
        self.assertEqual(cm["paired_cases_count"], 3)
        self.assertEqual(cm["both_passed_count"], 1)  # Case 1
        self.assertEqual(cm["both_failed_count"], 1)  # Case 3
        self.assertEqual(cm["model_a_only_passed_count"], 1)  # Case 2
        self.assertEqual(cm["model_b_only_passed_count"], 0)
        self.assertEqual(cm["agreement_count"], 2)
        self.assertAlmostEqual(cm["agreement_percent"], 66.67, places=2)


class TestTwoModelExecutionIndependence(unittest.TestCase):
    """Test that two models are evaluated independently on the exact same dataset case."""

    def test_independent_evaluation_on_same_case(self):
        example = {
            "id": 1,
            "question": "Find all products costing less than 1000.",
            "expected_sql": "SELECT name FROM products WHERE price < 1000 ORDER BY product_id",
            "category": "filtering",
            "difficulty": "easy",
            "order_matters": False,
        }

        # Model A outputs correct SQL
        model_a = MockModelAdapter(
            model_name="model-a-120b",
            responses={example["question"]: "SELECT name FROM products WHERE price < 1000;"},
        )

        # Model B outputs incorrect SQL
        model_b = MockModelAdapter(
            model_name="model-b-20b",
            responses={example["question"]: "SELECT name FROM products WHERE price > 5000;"},
        )

        res_a = evaluate_single_model_case(example, model_a)
        res_b = evaluate_single_model_case(example, model_b)

        # Confirm results are recorded independently
        self.assertTrue(res_a["passed"])
        self.assertEqual(res_a["failure_type"], "PASS")

        self.assertFalse(res_b["passed"])
        self.assertEqual(res_b["failure_type"], "RESULT_MISMATCH")

        self.assertEqual(res_a["model_name"], "model-a-120b")
        self.assertEqual(res_b["model_name"], "model-b-20b")


class TestReportGenerationAndRecreation(unittest.TestCase):
    """Test generating all 6 report files and regenerating without API calls."""

    def test_save_all_reports_and_recreate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir)
            results_a = [
                {
                    "id": 1,
                    "question": "Q1",
                    "category": "filtering",
                    "difficulty": "easy",
                    "generated_sql": "SELECT 1;",
                    "expected_sql": "SELECT 1;",
                    "passed": True,
                    "generated_sql_executed": True,
                    "generation_succeeded": True,
                    "failure_type": "PASS",
                    "error": None,
                    "latency_seconds": 0.45,
                }
            ]
            results_b = [
                {
                    "id": 1,
                    "question": "Q1",
                    "category": "filtering",
                    "difficulty": "easy",
                    "generated_sql": "SELECT 2;",
                    "expected_sql": "SELECT 1;",
                    "passed": False,
                    "generated_sql_executed": True,
                    "generation_succeeded": True,
                    "failure_type": "RESULT_MISMATCH",
                    "error": "Mismatch",
                    "latency_seconds": 0.25,
                }
            ]

            comp = save_evaluation_reports(
                results_a=results_a,
                results_b=results_b,
                model_a_meta={"model_name": "model-120b", "provider": "groq"},
                model_b_meta={"model_name": "model-20b", "provider": "groq"},
                total_planned=1,
                output_dir=out_path,
            )

            # Assert all 6 files are created
            self.assertTrue((out_path / "model_a_results.json").exists())
            self.assertTrue((out_path / "model_a_summary.txt").exists())
            self.assertTrue((out_path / "model_b_results.json").exists())
            self.assertTrue((out_path / "model_b_summary.txt").exists())
            self.assertTrue((out_path / "comparison_results.json").exists())
            self.assertTrue((out_path / "comparison_summary.txt").exists())

            # Recreate from files without API calls
            recreated = recreate_comparison_from_files(out_path)
            self.assertEqual(
                recreated["comparison_metrics"]["both_passed_count"],
                comp["comparison_metrics"]["both_passed_count"],
            )


class TestContaminationAndLeakage(unittest.TestCase):
    """Test benchmark contamination detection."""

    def test_sql_keyword_leakage_detection(self):
        clean_q = "Find products cheaper than 500 dollars."
        self.assertEqual(check_question_sql_leakage(clean_q), [])

        leaked_q = "Select all products from products where price is 10."
        leaks = check_question_sql_leakage(leaked_q)
        self.assertIn("SELECT", leaks)
        self.assertIn("FROM", leaks)
        self.assertIn("WHERE", leaks)

    def test_token_jaccard(self):
        sql1 = "SELECT name, price FROM products"
        sql2 = "SELECT name, price FROM products"
        self.assertEqual(compute_token_jaccard(sql1, sql2), 1.0)


if __name__ == "__main__":
    unittest.main()
