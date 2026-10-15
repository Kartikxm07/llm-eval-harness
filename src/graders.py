"""Evaluation grader and semantic result comparator for SQL benchmarks."""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .sql_executor import execute_query_detailed


@dataclass
class GradeResult:
    """Diagnostic grading outcome for a query evaluation."""
    passed: bool
    failure_type: str
    error: Optional[str]
    actual_rows: Optional[List[Tuple[Any, ...]]]
    expected_rows: Optional[List[Tuple[Any, ...]]]
    order_matters: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "failure_type": self.failure_type,
            "error": self.error,
            "actual_result": self.actual_rows,
            "expected_result": self.expected_rows,
            "order_matters": self.order_matters,
        }


def canonicalize_value(val: Any) -> Any:
    """Canonicalize values for robust comparison while preserving strict types.

    Floats are rounded to 6 decimal places to prevent floating-point noise.
    Types like int, str, bool, and None remain strictly distinct.
    """
    if val is None:
        return None
    if isinstance(val, float):
        if math.isnan(val):
            return "NaN"
        return ("float", round(val, 6))
    if isinstance(val, bool):
        return ("bool", val)
    if isinstance(val, int):
        return ("int", val)
    if isinstance(val, str):
        return ("str", val)
    return (type(val).__name__, val)


def canonicalize_row(row: Tuple[Any, ...]) -> Tuple[Any, ...]:
    return tuple(canonicalize_value(v) for v in row)


def compare_results(
    actual: Optional[List[Tuple[Any, ...]]],
    expected: Optional[List[Tuple[Any, ...]]],
    order_matters: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Compare database query results according to benchmark grading contract.

    Returns:
        (passed, error_reason)
    """
    if actual is None and expected is None:
        return True, None
    if actual is None:
        return False, "Actual query result is None."
    if expected is None:
        return False, "Expected reference result is None."

    # Compare row counts
    if len(actual) != len(expected):
        return False, f"Row count mismatch: got {len(actual)} rows, expected {len(expected)} rows."

    # Both empty
    if len(actual) == 0 and len(expected) == 0:
        return True, None

    # Compare column counts on first row
    actual_cols = len(actual[0])
    expected_cols = len(expected[0])
    if actual_cols != expected_cols:
        return (
            False,
            f"Column count mismatch: got {actual_cols} columns, expected {expected_cols} columns.",
        )

    # Order-sensitive comparison
    if order_matters:
        for idx, (act_row, exp_row) in enumerate(zip(actual, expected)):
            if len(act_row) != len(exp_row):
                return False, f"Column count mismatch at row {idx}."
            if canonicalize_row(act_row) != canonicalize_row(exp_row):
                return (
                    False,
                    f"Row order/value mismatch at row {idx}: got {act_row}, expected {exp_row}.",
                )
        return True, None

    # Order-insensitive comparison (multiset equality)
    actual_canon = [canonicalize_row(r) for r in actual]
    expected_canon = [canonicalize_row(r) for r in expected]

    actual_counts = Counter(actual_canon)
    expected_counts = Counter(expected_canon)

    if actual_counts == expected_counts:
        return True, None

    missing = expected_counts - actual_counts
    unexpected = actual_counts - expected_counts
    return (
        False,
        f"Result content mismatch (missing rows: {len(missing)}, unexpected rows: {len(unexpected)}).",
    )


def grade_query(
    generated_sql: str,
    expected_sql: str,
    order_matters: bool = False,
    db_path: Optional[Any] = None,
) -> GradeResult:
    """Execute both generated and expected SQL queries and grade the results."""
    gen_rows, gen_error = execute_query_detailed(generated_sql, db_path=db_path)
    exp_rows, exp_error = execute_query_detailed(expected_sql, db_path=db_path)

    if exp_error:
        return GradeResult(
            passed=False,
            failure_type="INVALID_EXPECTED_SQL",
            error=f"Reference SQL failed: {exp_error}",
            actual_rows=gen_rows,
            expected_rows=None,
            order_matters=order_matters,
        )

    if gen_error:
        return GradeResult(
            passed=False,
            failure_type="SQL_EXECUTION_ERROR",
            error=gen_error,
            actual_rows=None,
            expected_rows=exp_rows,
            order_matters=order_matters,
        )

    passed, diff_reason = compare_results(gen_rows, exp_rows, order_matters=order_matters)

    return GradeResult(
        passed=passed,
        failure_type="PASS" if passed else "RESULT_MISMATCH",
        error=None if passed else diff_reason,
        actual_rows=gen_rows,
        expected_rows=exp_rows,
        order_matters=order_matters,
    )


# Backward-compatible function alias
def grade_sql(
    generated_sql: str,
    expected_sql: str,
    order_matters: bool = False,
    db_path: Optional[Any] = None,
) -> Dict[str, Any]:
    """Legacy dict-returning wrapper for grade_query."""
    res = grade_query(generated_sql, expected_sql, order_matters=order_matters, db_path=db_path)
    return {
        "passed": res.passed,
        "failure_type": res.failure_type,
        "error": res.error,
        "generated_result": res.actual_rows,
        "expected_result": res.expected_rows,
        "order_matters": res.order_matters,
    }


def normalize_result(result: Optional[List[Tuple[Any, ...]]]) -> Optional[List[Tuple[Any, ...]]]:
    """Sort tuples for legacy comparisons."""
    if result is None:
        return None
    normalized = [tuple(row) for row in result]
    try:
        normalized.sort()
    except TypeError:
        normalized = sorted(normalized, key=lambda r: str(r))
    return normalized
