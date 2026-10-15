"""Two-model comparative evaluation runner and orchestrator."""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.database import create_database
from src.dataset import load_dataset, validate_dataset
from src.graders import grade_query
from src.model import BaseModelAdapter, load_model_configs
from src.metrics import compute_model_metrics
from src.report import (
    DEFAULT_RESULTS_DIR,
    format_comparison_text_summary,
    format_model_text_summary,
    recreate_comparison_from_files,
    save_evaluation_reports,
)


def evaluate_single_model_case(
    example: Dict[str, Any],
    adapter: BaseModelAdapter,
) -> Dict[str, Any]:
    """Evaluate one benchmark example with a single model adapter."""
    case_id = example["id"]
    question = example["question"]
    expected_sql = example["expected_sql"]
    order_matters = example.get("order_matters", False)

    record = {
        "id": case_id,
        "question": question,
        "category": example["category"],
        "difficulty": example["difficulty"],
        "order_matters": order_matters,
        "model_name": adapter.model_name,
        "provider": adapter.provider,
        "generated_sql": None,
        "expected_sql": expected_sql,
        "generation_succeeded": False,
        "generated_sql_executed": False,
        "expected_sql_executed": False,
        "actual_result": None,
        "expected_result": None,
        "passed": False,
        "failure_type": None,
        "error": None,
        "latency_seconds": 0.0,
    }

    # Step 1: LLM Generation
    gen_result = adapter.generate(question)
    record["latency_seconds"] = gen_result.latency_seconds

    if not gen_result.generation_succeeded or not gen_result.sql:
        record["failure_type"] = "GENERATION_ERROR"
        record["error"] = gen_result.error or "Model returned empty SQL."
        return record

    record["generated_sql"] = gen_result.sql
    record["generation_succeeded"] = True

    # Step 2: Programmatic Grading & Execution against database
    grade = grade_query(gen_result.sql, expected_sql, order_matters=order_matters)

    record["generated_sql_executed"] = grade.failure_type not in {"SQL_EXECUTION_ERROR", "GENERATION_ERROR"}
    record["expected_sql_executed"] = grade.failure_type != "INVALID_EXPECTED_SQL"
    record["actual_result"] = grade.actual_rows
    record["expected_result"] = grade.expected_rows
    record["passed"] = grade.passed
    record["failure_type"] = grade.failure_type
    record["error"] = grade.error

    return record


def run_comparative_benchmark(
    limit: Optional[int] = None,
    output_dir: Optional[Path] = None,
    model_a_only: bool = False,
    model_b_only: bool = False,
) -> int:
    """Run fair two-model comparative evaluation."""
    target_dir = Path(output_dir) if output_dir else DEFAULT_RESULTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    print("Checking database integrity...")
    create_database()

    print("Validating golden benchmark dataset...")
    try:
        dataset = load_dataset()
        valid, failures = validate_dataset()
        if not valid:
            print(f"ERROR: Dataset validation failed for {len(failures)} reference queries.")
            return 1
    except Exception as exc:
        print(f"ERROR: Failed to load dataset: {exc}")
        return 1

    print("Configuring models...")
    try:
        model_a, model_b = load_model_configs()
        print(f"  Model A : {model_a.get_identifier()}")
        print(f"  Model B : {model_b.get_identifier()}")
    except Exception as exc:
        print(f"ERROR configuring models: {exc}")
        return 1

    if limit is not None:
        dataset = dataset[:limit]

    total_planned = len(dataset)
    results_a: List[Dict[str, Any]] = []
    results_b: List[Dict[str, Any]] = []

    print(f"\nStarting benchmark evaluation on {total_planned} cases...\n")

    for idx, ex in enumerate(dataset, start=1):
        cid = ex["id"]
        status_line = f"[{idx:02d}/{total_planned:02d}] Case {cid:02d}: "

        # 1. Evaluate Model A
        if not model_b_only:
            res_a = evaluate_single_model_case(ex, model_a)
            results_a.append(res_a)
            stat_a = "PASS" if res_a["passed"] else f"FAIL [{res_a['failure_type']}]"
            status_line += f"Model A: {stat_a:<15} ({res_a['latency_seconds']:.2f}s) | "

        # 2. Evaluate Model B
        if not model_a_only:
            res_b = evaluate_single_model_case(ex, model_b)
            results_b.append(res_b)
            stat_b = "PASS" if res_b["passed"] else f"FAIL [{res_b['failure_type']}]"
            status_line += f"Model B: {stat_b:<15} ({res_b['latency_seconds']:.2f}s)"

        print(status_line)

    # Generate Reports
    meta_a = model_a.get_metadata()
    meta_b = model_b.get_metadata()

    if not model_b_only and not model_a_only:
        comp_report = save_evaluation_reports(
            results_a=results_a,
            results_b=results_b,
            model_a_meta=meta_a,
            model_b_meta=meta_b,
            total_planned=total_planned,
            output_dir=target_dir,
        )
        print("\n" + format_comparison_text_summary(comp_report))
        print(f"\nSaved all reports to: {target_dir}")
    elif model_a_only:
        rep_a = {
            "model_name": meta_a["model_name"],
            "provider": meta_a["provider"],
            "model_identifier": f"{meta_a['provider']}/{meta_a['model_name']}",
            "metrics": compute_model_metrics(results_a, total_planned),
            "results": results_a,
        }
        print("\n" + format_model_text_summary(rep_a))
    elif model_b_only:
        rep_b = {
            "model_name": meta_b["model_name"],
            "provider": meta_b["provider"],
            "model_identifier": f"{meta_b['provider']}/{meta_b['model_name']}",
            "metrics": compute_model_metrics(results_b, total_planned),
            "results": results_b,
        }
        print("\n" + format_model_text_summary(rep_b))

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Fair Two-Model LLM Evaluation Harness Runner."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N dataset cases.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory to save report outputs.",
    )
    parser.add_argument(
        "--model-a-only",
        action="store_true",
        help="Evaluate Model A only.",
    )
    parser.add_argument(
        "--model-b-only",
        action="store_true",
        help="Evaluate Model B only.",
    )
    parser.add_argument(
        "--re-report",
        action="store_true",
        help="Recreate comparison reports from existing result files without making API calls.",
    )

    args = parser.parse_args()

    if args.re_report:
        print("Recreating comparison reports from existing files...")
        try:
            comp = recreate_comparison_from_files(Path(args.output_dir))
            print(format_comparison_text_summary(comp))
            print("Successfully regenerated comparison reports.")
            sys.exit(0)
        except Exception as exc:
            print(f"Error recreating report: {exc}")
            sys.exit(1)

    code = run_comparative_benchmark(
        limit=args.limit,
        output_dir=Path(args.output_dir),
        model_a_only=args.model_a_only,
        model_b_only=args.model_b_only,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
