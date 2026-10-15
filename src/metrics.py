"""Benchmark evaluation metrics for single models and two-model comparisons."""

from collections import Counter
from typing import Any, Dict, List, Optional


def compute_model_metrics(results: List[Dict[str, Any]], total_planned: int) -> Dict[str, Any]:
    """Compute benchmark summary metrics for a single model run."""
    completed = len(results)
    passed_count = sum(1 for r in results if r.get("passed"))
    failed_count = completed - passed_count

    accuracy = (passed_count / completed * 100) if completed else 0.0

    exec_successes = sum(1 for r in results if r.get("generated_sql_executed"))
    exec_rate = (exec_successes / completed * 100) if completed else 0.0

    gen_successes = sum(1 for r in results if r.get("generation_succeeded"))
    gen_rate = (gen_successes / completed * 100) if completed else 0.0

    failure_counts = Counter(r.get("failure_type") for r in results if not r.get("passed") and r.get("failure_type"))

    # Category breakdown
    cat_total = Counter(r.get("category") for r in results)
    cat_passed = Counter(r.get("category") for r in results if r.get("passed"))
    category_breakdown = {}
    for cat, total in cat_total.items():
        pass_n = cat_passed.get(cat, 0)
        pct = (pass_n / total * 100) if total else 0.0
        category_breakdown[cat] = {
            "total": total,
            "passed": pass_n,
            "failed": total - pass_n,
            "accuracy_percent": round(pct, 2),
        }

    # Difficulty breakdown
    diff_total = Counter(r.get("difficulty") for r in results)
    diff_passed = Counter(r.get("difficulty") for r in results if r.get("passed"))
    difficulty_breakdown = {}
    for diff, total in diff_total.items():
        pass_n = diff_passed.get(diff, 0)
        pct = (pass_n / total * 100) if total else 0.0
        difficulty_breakdown[diff] = {
            "total": total,
            "passed": pass_n,
            "failed": total - pass_n,
            "accuracy_percent": round(pct, 2),
        }

    return {
        "total_planned": total_planned,
        "completed_cases": completed,
        "passed": passed_count,
        "failed": failed_count,
        "accuracy_percent": round(accuracy, 2),
        "sql_execution_success_rate": round(exec_rate, 2),
        "generation_success_rate": round(gen_rate, 2),
        "failures_by_type": dict(failure_counts),
        "accuracy_by_category": category_breakdown,
        "accuracy_by_difficulty": difficulty_breakdown,
    }


def compute_comparison_metrics(
    results_a: List[Dict[str, Any]],
    results_b: List[Dict[str, Any]],
    model_a_id: str,
    model_b_id: str,
) -> Dict[str, Any]:
    """Compute paired comparative metrics between Model A and Model B."""
    map_a = {r["id"]: r for r in results_a}
    map_b = {r["id"]: r for r in results_b}

    common_ids = sorted(set(map_a.keys()) & set(map_b.keys()))
    total_evaluated = len(common_ids)

    both_passed = []
    both_failed = []
    model_a_only = []
    model_b_only = []

    for cid in common_ids:
        pass_a = bool(map_a[cid].get("passed"))
        pass_b = bool(map_b[cid].get("passed"))

        diff_record = {
            "id": cid,
            "question": map_a[cid].get("question"),
            "category": map_a[cid].get("category"),
            "difficulty": map_a[cid].get("difficulty"),
            "model_a_passed": pass_a,
            "model_b_passed": pass_b,
            "model_a_sql": map_a[cid].get("generated_sql"),
            "model_b_sql": map_b[cid].get("generated_sql"),
            "model_a_error": map_a[cid].get("error"),
            "model_b_error": map_b[cid].get("error"),
        }

        if pass_a and pass_b:
            both_passed.append(diff_record)
        elif not pass_a and not pass_b:
            both_failed.append(diff_record)
        elif pass_a and not pass_b:
            model_a_only.append(diff_record)
        elif not pass_a and pass_b:
            model_b_only.append(diff_record)

    agreed_count = len(both_passed) + len(both_failed)
    agreement_percent = (agreed_count / total_evaluated * 100) if total_evaluated else 0.0

    return {
        "model_a_identifier": model_a_id,
        "model_b_identifier": model_b_id,
        "paired_cases_count": total_evaluated,
        "both_passed_count": len(both_passed),
        "both_failed_count": len(both_failed),
        "model_a_only_passed_count": len(model_a_only),
        "model_b_only_passed_count": len(model_b_only),
        "agreement_count": agreed_count,
        "agreement_percent": round(agreement_percent, 2),
        "disagreement_count": total_evaluated - agreed_count,
        "model_a_wins": [r["id"] for r in model_a_only],
        "model_b_wins": [r["id"] for r in model_b_only],
        "disagreement_details": model_a_only + model_b_only,
    }


# Backward-compatible alias
compute_metrics = compute_model_metrics

