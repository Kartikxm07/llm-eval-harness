"""Optional qualitative LLM-as-a-judge for secondary SQL quality evaluations."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .model import BaseModelAdapter


@dataclass
class JudgeEvaluation:
    """Qualitative review of SQL queries."""
    model_a_score: int  # 1 to 5
    model_b_score: int  # 1 to 5
    winner: str         # 'model_a', 'model_b', or 'tie'
    reasoning: str


def heuristic_judge_query(
    question: str,
    sql_a: str,
    sql_b: str,
    expected_sql: str,
) -> JudgeEvaluation:
    """Deterministic offline heuristic judge based on SQL conventions."""
    score_a = 3
    score_b = 3

    # Capitalization of SQL keywords
    if any(kw in sql_a for kw in ["SELECT", "FROM", "WHERE"]):
        score_a += 1
    if any(kw in sql_b for kw in ["SELECT", "FROM", "WHERE"]):
        score_b += 1

    # Indentation / formatting bonus
    if "\n" in sql_a.strip():
        score_a += 1
    if "\n" in sql_b.strip():
        score_b += 1

    if score_a > score_b:
        winner = "model_a"
        reason = "Model A used cleaner SQL formatting and keyword capitalization."
    elif score_b > score_a:
        winner = "model_b"
        reason = "Model B used cleaner SQL formatting and keyword capitalization."
    else:
        winner = "tie"
        reason = "Both models exhibited similar SQL structural style."

    return JudgeEvaluation(
        model_a_score=min(5, score_a),
        model_b_score=min(5, score_b),
        winner=winner,
        reasoning=reason,
    )


def judge_two_queries(
    question: str,
    sql_a: str,
    sql_b: str,
    expected_sql: str,
    judge_model: Optional[BaseModelAdapter] = None,
) -> JudgeEvaluation:
    """Evaluate query quality using LLM judge or fallback heuristic."""
    if not judge_model:
        return heuristic_judge_query(question, sql_a, sql_b, expected_sql)

    judge_prompt = f"""You are an expert database administrator. Compare two generated SQL queries for the question:
Question: {question}
Expected SQL: {expected_sql}

Model A SQL:
{sql_a}

Model B SQL:
{sql_b}

Compare their readability, indexing friendliness, and schema adherence.
Provide your verdict in the format:
WINNER: [model_a|model_b|tie]
SCORE_A: [1-5]
SCORE_B: [1-5]
REASON: [Brief explanation]
"""
    try:
        res = judge_model.generate(judge_prompt)
        text = res.sql or res.raw_output
        winner = "tie"
        if "WINNER: model_a" in text:
            winner = "model_a"
        elif "WINNER: model_b" in text:
            winner = "model_b"
        return JudgeEvaluation(model_a_score=4, model_b_score=4, winner=winner, reasoning=text.strip())
    except Exception as exc:
        return heuristic_judge_query(question, sql_a, sql_b, expected_sql)
