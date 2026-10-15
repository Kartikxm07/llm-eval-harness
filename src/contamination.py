"""Benchmark contamination and prompt leakage audit utilities."""

import re
from typing import Any, Dict, List, Optional


def check_question_sql_leakage(question: str) -> List[str]:
    """Check if natural language question accidentally leaks raw SQL keywords."""
    sql_keywords = ["SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "HAVING", "ORDER BY", "LIMIT"]
    leaks = []
    q_upper = question.upper()
    for kw in sql_keywords:
        # Check whole word match
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, q_upper):
            leaks.append(kw)
    return leaks


def compute_token_jaccard(sql_a: str, sql_b: str) -> float:
    """Compute token Jaccard similarity between two SQL queries."""
    tokens_a = set(re.findall(r"\w+", sql_a.upper()))
    tokens_b = set(re.findall(r"\w+", sql_b.upper()))
    if not tokens_a and not tokens_b:
        return 1.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def run_contamination_audit(
    dataset: List[Dict[str, Any]],
    results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Audit the benchmark dataset and optional model outputs for potential leakage."""
    question_leaks = []
    for item in dataset:
        leaks = check_question_sql_leakage(item.get("question", ""))
        if leaks:
            question_leaks.append({
                "id": item.get("id"),
                "question": item.get("question"),
                "leaked_keywords": leaks,
            })

    exact_memorizations = 0
    high_similarity_count = 0
    total_comparisons = 0

    if results:
        for r in results:
            gen = (r.get("generated_sql") or "").strip()
            exp = (r.get("expected_sql") or "").strip()
            if gen and exp:
                total_comparisons += 1
                if gen.upper() == exp.upper():
                    exact_memorizations += 1
                elif compute_token_jaccard(gen, exp) > 0.85:
                    high_similarity_count += 1

    return {
        "dataset_total_cases": len(dataset),
        "questions_with_sql_keywords_leaked": len(question_leaks),
        "leaked_question_details": question_leaks,
        "is_dataset_clean": len(question_leaks) == 0,
        "model_exact_sql_match_rate": round(exact_memorizations / total_comparisons * 100, 2) if total_comparisons else 0.0,
        "model_high_similarity_rate": round(high_similarity_count / total_comparisons * 100, 2) if total_comparisons else 0.0,
    }
