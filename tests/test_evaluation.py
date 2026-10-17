"""Comprehensive test suite for the LLM evaluation harness.

Runs seamlessly via both:
    python -m unittest discover -s tests -v
and:
    python -m pytest
"""

import json
import sqlite3
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.database import create_database, DB_PATH
from src.dataset import load_dataset, validate_dataset
from src.model import (
    LLMAuthError,
    LLMEmptyResponseError,
    LLMError,
    LLMRateLimitError,
    clean_markdown_fences,
    generate_sql,
)
from src.graders import compare_results, grade_sql
from src.metrics import compute_metrics
from src.sql_executor import execute_query, execute_query_detailed


class TestDatabaseInitialization(unittest.TestCase):
    """Test database setup, schema integrity, and table constraints."""

    @classmethod
    def setUpClass(cls):
        create_database()

    def test_tables_exist(self):
        with sqlite3.connect(f"{DB_PATH.resolve().as_uri()}?mode=ro", uri=True) as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = {row[0] for row in cur.fetchall()}
            expected = {"products", "customers", "orders", "order_items"}
            self.assertTrue(expected.issubset(tables), f"Missing tables: {expected - tables}")

    def test_table_row_counts(self):
        counts = {
            "products": execute_query("SELECT COUNT(*) FROM products")[0][0],
            "customers": execute_query("SELECT COUNT(*) FROM customers")[0][0],
            "orders": execute_query("SELECT COUNT(*) FROM orders")[0][0],
            "order_items": execute_query("SELECT COUNT(*) FROM order_items")[0][0],
        }
        self.assertGreater(counts["products"], 0)
        self.assertGreater(counts["customers"], 0)
        self.assertGreater(counts["orders"], 0)
        self.assertGreater(counts["order_items"], 0)


class TestDatasetIntegrity(unittest.TestCase):
    """Test golden dataset schema and execute all 50 reference queries."""

    def test_dataset_size_and_fields(self):
        dataset = load_dataset()
        self.assertEqual(len(dataset), 50, "Golden dataset must contain exactly 50 examples.")
        seen_ids = set()
        for ex in dataset:
            self.assertIn("id", ex)
            self.assertIn("question", ex)
            self.assertIn("expected_sql", ex)
            self.assertIn("category", ex)
            self.assertIn("difficulty", ex)
            self.assertNotIn(ex["id"], seen_ids, f"Duplicate ID: {ex['id']}")
            seen_ids.add(ex["id"])

    def test_all_reference_queries_execute(self):
        valid, failures = validate_dataset()
        self.assertTrue(valid, f"Reference queries failed validation: {failures}")
        self.assertEqual(len(failures), 0)


class TestSqlExecutorSecurityAndExecution(unittest.TestCase):
    """Test read-only enforcement, multi-statement rejection, and error handling."""

    @classmethod
    def setUpClass(cls):
        create_database()

    def test_valid_select_query(self):
        rows, err = execute_query_detailed("SELECT name FROM products WHERE price < 1000;")
        self.assertIsNone(err)
        self.assertIsInstance(rows, list)
        self.assertGreater(len(rows), 0)

    def test_valid_with_cte_query(self):
        query = """
            WITH cheap_products AS (
                SELECT name, price FROM products WHERE price < 1000
            )
            SELECT name FROM cheap_products ORDER BY price;
        """
        rows, err = execute_query_detailed(query)
        self.assertIsNone(err)
        self.assertIsNotNone(rows)

    def test_multiline_select_execution(self):
        query = "SELECT\n    product_id,\n    name\nFROM products;"
        rows, err = execute_query_detailed(query)
        self.assertIsNone(err)
        self.assertIsNotNone(rows)

    def test_reject_write_queries(self):
        before = execute_query("SELECT COUNT(*) FROM products")[0][0]

        write_queries = [
            "DELETE FROM products;",
            "DROP TABLE products;",
            "INSERT INTO products VALUES (99, 'Test', 'Misc', 10, 10);",
            "UPDATE products SET price = 0;",
        ]
        for wq in write_queries:
            rows, err = execute_query_detailed(wq)
            self.assertIsNone(rows, f"Write query did not return None: {wq}")
            self.assertIsNotNone(err)

        after = execute_query("SELECT COUNT(*) FROM products")[0][0]
        self.assertEqual(before, after, "Database was modified despite read-only protections.")

    def test_reject_multiple_statements(self):
        multi = "SELECT 1; SELECT 2;"
        rows, err = execute_query_detailed(multi)
        self.assertIsNone(rows)
        self.assertIn("Multiple SQL statements", err)

    def test_reject_empty_and_invalid_queries(self):
        empty_res, empty_err = execute_query_detailed("   ")
        self.assertIsNone(empty_res)
        self.assertIsNotNone(empty_err)

        bad_syntax_res, bad_syntax_err = execute_query_detailed("SELECT * FROM non_existent_table")
        self.assertIsNone(bad_syntax_res)
        self.assertIn("OperationalError", bad_syntax_err)


class TestProgrammaticGrader(unittest.TestCase):
    """Test semantic comparison contract, order awareness, duplicates, and NULLs."""

    def test_order_ignored_when_order_matters_false(self):
        actual = [("A",), ("B",)]
        expected = [("B",), ("A",)]
        passed, err = compare_results(actual, expected, order_matters=False)
        self.assertTrue(passed)
        self.assertIsNone(err)

    def test_order_enforced_when_order_matters_true(self):
        actual = [("A",), ("B",)]
        expected = [("B",), ("A",)]
        passed, err = compare_results(actual, expected, order_matters=True)
        self.assertFalse(passed)
        self.assertIn("Row order/value mismatch", err)

    def test_duplicate_row_handling(self):
        actual = [("A",), ("A",)]
        expected = [("A",)]
        passed, err = compare_results(actual, expected, order_matters=False)
        self.assertFalse(passed)
        self.assertIn("Row count mismatch", err)

    def test_null_value_handling(self):
        actual = [(1, None)]
        expected = [(1, None)]
        passed, err = compare_results(actual, expected, order_matters=True)
        self.assertTrue(passed)

        mismatched_actual = [(1, "None")]
        passed_mismatch, _ = compare_results(mismatched_actual, expected, order_matters=True)
        self.assertFalse(passed_mismatch)

    def test_column_count_mismatch(self):
        actual = [(1, "Wireless Mouse")]
        expected = [("Wireless Mouse",)]
        passed, err = compare_results(actual, expected, order_matters=False)
        self.assertFalse(passed)
        self.assertIn("Column count mismatch", err)

    def test_grade_sql_failure_types(self):
        # Successful match
        res_pass = grade_sql(
            "SELECT name FROM products WHERE price < 1000",
            "SELECT name FROM products WHERE price < 1000",
        )
        self.assertTrue(res_pass["passed"])
        self.assertEqual(res_pass["failure_type"], "PASS")

        # Execution error in generated SQL
        res_exec_err = grade_sql(
            "SELECT * FROM non_existent_table",
            "SELECT name FROM products",
        )
        self.assertFalse(res_exec_err["passed"])
        self.assertEqual(res_exec_err["failure_type"], "SQL_EXECUTION_ERROR")

        # Result mismatch
        res_mismatch = grade_sql(
            "SELECT name FROM products WHERE price < 500",
            "SELECT name FROM products WHERE price < 1000",
        )
        self.assertFalse(res_mismatch["passed"])
        self.assertEqual(res_mismatch["failure_type"], "RESULT_MISMATCH")


class TestLLMGeneratorMocking(unittest.TestCase):
    """Test SQL generation error handling and formatting using mocked responses."""

    def test_clean_markdown_fences(self):
        fenced = "```sql\nSELECT * FROM products;\n```"
        self.assertEqual(clean_markdown_fences(fenced), "SELECT * FROM products;")

    def test_mock_successful_generation(self):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "SELECT name FROM products WHERE price < 1000;"
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        sql = generate_sql("Find products under 1000", client=mock_client, model="mock-model")
        self.assertEqual(sql, "SELECT name FROM products WHERE price < 1000;")

    def test_mock_empty_response_raises_error(self):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = ""
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        with self.assertRaises(LLMEmptyResponseError):
            generate_sql("Question", client=mock_client, model="mock-model")

    def test_mock_auth_failure_raises_typed_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("401 invalid_api_key")

        with self.assertRaises(LLMAuthError):
            generate_sql("Question", client=mock_client, model="mock-model")

    def test_mock_rate_limit_failure_raises_typed_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("429 rate limit exceeded")

        with self.assertRaises(LLMRateLimitError):
            generate_sql("Question", client=mock_client, model="mock-model")


class TestBatchMetricsAndReporting(unittest.TestCase):
    """Test batch metrics calculation and reporting logic."""

    def test_compute_metrics_calculation(self):
        sample_results = [
            {
                "id": 1,
                "category": "filtering",
                "difficulty": "easy",
                "generated_sql_executed": True,
                "passed": True,
                "failure_type": "PASS",
            },
            {
                "id": 2,
                "category": "filtering",
                "difficulty": "easy",
                "generated_sql_executed": True,
                "passed": False,
                "failure_type": "RESULT_MISMATCH",
            },
            {
                "id": 3,
                "category": "joins",
                "difficulty": "medium",
                "generated_sql_executed": False,
                "passed": False,
                "failure_type": "SQL_EXECUTION_ERROR",
            },
        ]

        metrics = compute_metrics(sample_results, total_planned=3)
        self.assertEqual(metrics["total_planned"], 3)
        self.assertEqual(metrics["completed_cases"], 3)
        self.assertEqual(metrics["passed"], 1)
        self.assertEqual(metrics["failed"], 2)
        self.assertAlmostEqual(metrics["accuracy_percent"], 33.33, places=2)
        self.assertAlmostEqual(metrics["sql_execution_success_rate"], 66.67, places=2)
        self.assertEqual(metrics["failures_by_type"]["RESULT_MISMATCH"], 1)
        self.assertEqual(metrics["failures_by_type"]["SQL_EXECUTION_ERROR"], 1)
        self.assertEqual(metrics["accuracy_by_category"]["filtering"]["passed"], 1)
        self.assertEqual(metrics["accuracy_by_category"]["joins"]["passed"], 0)


class TestNineCasesRegression(unittest.TestCase):
    """Regression tests verifying diagnostic grading and contracts for the 9 investigated cases."""

    @classmethod
    def setUpClass(cls):
        create_database()
        cls.dataset = {ex["id"]: ex for ex in load_dataset()}

    def test_case_6_products_fewer_than_30_stock(self):
        ex = self.dataset[6]
        # Intended answer: 1 column (name)
        correct_sql = "SELECT name FROM products WHERE stock < 30 ORDER BY product_id;"
        res = grade_sql(correct_sql, ex["expected_sql"], order_matters=ex["order_matters"])
        self.assertTrue(res["passed"])

        # Over-projected variant (2 columns) must fail with column count mismatch
        over_projected = "SELECT name, stock FROM products WHERE stock < 30;"
        res_fail = grade_sql(over_projected, ex["expected_sql"], order_matters=ex["order_matters"])
        self.assertFalse(res_fail["passed"])
        self.assertIn("Column count mismatch", res_fail["error"])

    def test_case_17_max_price_per_category(self):
        ex = self.dataset[17]
        correct_sql = "SELECT category, MAX(price) FROM products GROUP BY category ORDER BY category;"
        res = grade_sql(correct_sql, ex["expected_sql"], order_matters=ex["order_matters"])
        self.assertTrue(res["passed"])

    def test_case_18_categories_more_than_one_product(self):
        ex = self.dataset[18]
        correct_sql = "SELECT category FROM products GROUP BY category HAVING COUNT(*) > 1 ORDER BY category;"
        res = grade_sql(correct_sql, ex["expected_sql"], order_matters=ex["order_matters"])
        self.assertTrue(res["passed"])

        # Unrequested count in SELECT must fail
        with_count = "SELECT category, COUNT(*) FROM products GROUP BY category HAVING COUNT(*) > 1;"
        res_fail = grade_sql(with_count, ex["expected_sql"], order_matters=ex["order_matters"])
        self.assertFalse(res_fail["passed"])
        self.assertIn("Column count mismatch", res_fail["error"])

    def test_case_20_non_delivered_orders_with_status(self):
        ex = self.dataset[20]
        correct_sql = "SELECT order_id, status FROM orders WHERE status <> 'delivered' ORDER BY order_id;"
        res = grade_sql(correct_sql, ex["expected_sql"], order_matters=ex["order_matters"])
        self.assertTrue(res["passed"])

    def test_case_21_orders_with_order_date(self):
        ex = self.dataset[21]
        correct_sql = "SELECT order_id, order_date FROM orders WHERE order_date >= '2026-09-05' ORDER BY order_date, order_id;"
        res = grade_sql(correct_sql, ex["expected_sql"], order_matters=ex["order_matters"])
        self.assertTrue(res["passed"])

    def test_case_38_products_and_category_in_electronics_or_accessories(self):
        ex = self.dataset[38]
        correct_sql = "SELECT name, category FROM products WHERE category IN ('Electronics', 'Accessories') ORDER BY product_id;"
        res = grade_sql(correct_sql, ex["expected_sql"], order_matters=ex["order_matters"])
        self.assertTrue(res["passed"])

    def test_case_40_second_most_expensive_product_and_price(self):
        ex = self.dataset[40]
        correct_sql = "SELECT name, price FROM products ORDER BY price DESC, product_id LIMIT 1 OFFSET 1;"
        res = grade_sql(correct_sql, ex["expected_sql"], order_matters=ex["order_matters"])
        self.assertTrue(res["passed"])

    def test_case_43_cities_more_than_one_customer(self):
        ex = self.dataset[43]
        correct_sql = "SELECT city FROM customers GROUP BY city HAVING COUNT(*) > 1 ORDER BY city;"
        res = grade_sql(correct_sql, ex["expected_sql"], order_matters=ex["order_matters"])
        self.assertTrue(res["passed"])

    def test_case_46_most_recently_placed_order_and_date(self):
        ex = self.dataset[46]
        correct_sql = "SELECT order_id, order_date FROM orders ORDER BY order_date DESC, order_id DESC LIMIT 1;"
        res = grade_sql(correct_sql, ex["expected_sql"], order_matters=ex["order_matters"])
        self.assertTrue(res["passed"])

    def test_case_11_products_priced_between_500_and_2000(self):
        ex = self.dataset[11]
        correct_sql = "SELECT name, price FROM products WHERE price BETWEEN 500 AND 2000 ORDER BY price, product_id;"
        res = grade_sql(correct_sql, ex["expected_sql"], order_matters=ex["order_matters"])
        self.assertTrue(res["passed"])

        single_col = "SELECT name FROM products WHERE price BETWEEN 500 AND 2000;"
        res_fail = grade_sql(single_col, ex["expected_sql"], order_matters=ex["order_matters"])
        self.assertFalse(res_fail["passed"])
        self.assertIn("Column count mismatch", res_fail["error"])

    def test_case_12_products_at_least_30_stock(self):
        ex = self.dataset[12]
        correct_sql = "SELECT name, stock FROM products WHERE stock >= 30 ORDER BY product_id;"
        res = grade_sql(correct_sql, ex["expected_sql"], order_matters=ex["order_matters"])
        self.assertTrue(res["passed"])

        single_col = "SELECT name FROM products WHERE stock >= 30;"
        res_fail = grade_sql(single_col, ex["expected_sql"], order_matters=ex["order_matters"])
        self.assertFalse(res_fail["passed"])
        self.assertIn("Column count mismatch", res_fail["error"])

    def test_case_13_products_cheapest_to_most_expensive(self):
        ex = self.dataset[13]
        correct_sql = "SELECT name, price FROM products ORDER BY price, product_id;"
        res = grade_sql(correct_sql, ex["expected_sql"], order_matters=ex["order_matters"])
        self.assertTrue(res["passed"])

        single_col = "SELECT name FROM products ORDER BY price;"
        res_fail = grade_sql(single_col, ex["expected_sql"], order_matters=ex["order_matters"])
        self.assertFalse(res_fail["passed"])
        self.assertIn("Column count mismatch", res_fail["error"])

    def test_case_14_three_cheapest_products(self):
        ex = self.dataset[14]
        correct_sql = "SELECT name, price FROM products ORDER BY price, product_id LIMIT 3;"
        res = grade_sql(correct_sql, ex["expected_sql"], order_matters=ex["order_matters"])
        self.assertTrue(res["passed"])

        single_col = "SELECT name FROM products ORDER BY price LIMIT 3;"
        res_fail = grade_sql(single_col, ex["expected_sql"], order_matters=ex["order_matters"])
        self.assertFalse(res_fail["passed"])
        self.assertIn("Column count mismatch", res_fail["error"])

    def test_case_39_subquery_products_priced_above_average(self):
        ex = self.dataset[39]
        correct_sql = "SELECT name, price FROM products WHERE price > (SELECT AVG(price) FROM products) ORDER BY price DESC, product_id;"
        res = grade_sql(correct_sql, ex["expected_sql"], order_matters=ex["order_matters"])
        self.assertTrue(res["passed"])

        single_col = "SELECT name FROM products WHERE price > (SELECT AVG(price) FROM products);"
        res_fail = grade_sql(single_col, ex["expected_sql"], order_matters=ex["order_matters"])
        self.assertFalse(res_fail["passed"])
        self.assertIn("Column count mismatch", res_fail["error"])


if __name__ == "__main__":
    unittest.main()