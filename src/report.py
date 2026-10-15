"""Report generation for individual model benchmarks and two-model comparisons."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics import compute_comparison_metrics, compute_model_metrics
from .statistics import (
    compute_cohens_kappa,
    compute_latency_summary,
    compute_mcnemar_test,
    wilson_score_interval,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = BASE_DIR / "results"


def format_model_text_summary(report: Dict[str, Any]) -> str:
    """Format single model evaluation report into an easy-to-read text dashboard."""
    m = report["metrics"]
    lat = report.get("latency_summary", {})
    model_id = report.get("model_identifier", report.get("model_name", "Unknown"))
    provider = report.get("provider", "Unknown")

    lines = [
        "=" * 60,
        f"       EVALUATION REPORT: {model_id}",
        "=" * 60,
        f"Provider                : {provider}",
        f"Timestamp (UTC)         : {report.get('run_timestamp_utc')}",
        f"Total Planned Cases     : {m['total_planned']}",
        f"Completed Cases         : {m['completed_cases']}",
        f"Passed                  : {m['passed']}",
        f"Failed                  : {m['failed']}",
        f"Result Accuracy         : {m['accuracy_percent']:.2f}%",
        f"95% Confidence Interval : {report.get('confidence_interval_95', 'N/A')}",
        f"SQL Execution Rate      : {m['sql_execution_success_rate']:.2f}%",
        f"Generation Success Rate : {m['generation_success_rate']:.2f}%",
    ]

    if lat and lat.get("count", 0) > 0:
        lines.extend([
            "-" * 60,
            "Latency Profile (Wall Clock):",
            f"  Mean Latency          : {lat.get('mean_seconds', 0.0):.3f}s",
            f"  Median Latency        : {lat.get('median_seconds', 0.0):.3f}s",
            f"  P95 Latency           : {lat.get('p95_seconds', 0.0):.3f}s",
            f"  Min / Max Latency     : {lat.get('min_seconds', 0.0):.3f}s / {lat.get('max_seconds', 0.0):.3f}s",
        ])

    if m.get("failures_by_type"):
        lines.extend([
            "-" * 60,
            "Failures by Type:",
        ])
        for ftype, count in m["failures_by_type"].items():
            lines.append(f"  - {ftype:<22} : {count}")

    if m.get("accuracy_by_category"):
        lines.extend([
            "-" * 60,
            "Accuracy by Category:",
        ])
        for cat, data in m["accuracy_by_category"].items():
            lines.append(f"  - {cat:<15} : {data['passed']}/{data['total']} ({data['accuracy_percent']}%)")

    if m.get("accuracy_by_difficulty"):
        lines.extend([
            "-" * 60,
            "Accuracy by Difficulty:",
        ])
        for diff, data in m["accuracy_by_difficulty"].items():
            lines.append(f"  - {diff:<15} : {data['passed']}/{data['total']} ({data['accuracy_percent']}%)")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_comparison_text_summary(comp: Dict[str, Any]) -> str:
    """Format comparative side-by-side analysis into an easy-to-read text dashboard."""
    meta_a = comp["model_a"]
    meta_b = comp["model_b"]
    cm = comp["comparison_metrics"]
    mcn = comp.get("mcnemar_test", {})
    lat_a = meta_a.get("latency_summary", {})
    lat_b = meta_b.get("latency_summary", {})

    lines = [
        "=" * 70,
        "         FAIR TWO-MODEL COMPARATIVE EVALUATION BENCHMARK",
        "=" * 70,
        f"Benchmark Date (UTC)   : {comp.get('run_timestamp_utc')}",
        f"Total Paired Cases     : {cm['paired_cases_count']}",
        "",
        f"Model A (High Param)   : {meta_a['model_name']} ({meta_a['provider']})",
        f"Model B (Small Param)  : {meta_b['model_name']} ({meta_b['provider']})",
        "-" * 70,
        f"{'Metric':<28} | {'Model A':<18} | {'Model B':<18}",
        "-" * 70,
        f"{'Passed / Total':<28} | {meta_a['metrics']['passed']}/{meta_a['metrics']['completed_cases']:<16} | {meta_b['metrics']['passed']}/{meta_b['metrics']['completed_cases']:<16}",
        f"{'Result Accuracy':<28} | {meta_a['metrics']['accuracy_percent']:<17.2f}% | {meta_b['metrics']['accuracy_percent']:<17.2f}%",
        f"{'95% Confidence Interval':<28} | {str(meta_a.get('confidence_interval_95')):<18} | {str(meta_b.get('confidence_interval_95')):<18}",
        f"{'SQL Execution Rate':<28} | {meta_a['metrics']['sql_execution_success_rate']:<17.2f}% | {meta_b['metrics']['sql_execution_success_rate']:<17.2f}%",
        f"{'Mean Latency':<28} | {lat_a.get('mean_seconds', 0.0):<17.3f}s | {lat_b.get('mean_seconds', 0.0):<17.3f}s",
        f"{'Median Latency':<28} | {lat_a.get('median_seconds', 0.0):<17.3f}s | {lat_b.get('median_seconds', 0.0):<17.3f}s",
        "-" * 70,
        "Paired Agreement Breakdown:",
        f"  Both Models Passed (Agreement)   : {cm['both_passed_count']} cases",
        f"  Both Models Failed (Agreement)   : {cm['both_failed_count']} cases",
        f"  Model A Passed, Model B Failed   : {cm['model_a_only_passed_count']} cases (Model A Wins)",
        f"  Model B Passed, Model A Failed   : {cm['model_b_only_passed_count']} cases (Model B Wins)",
        f"  Inter-Model Agreement Rate       : {cm['agreement_percent']:.2f}%",
        f"  Cohen's Kappa (Agreement Chance) : {comp.get('cohens_kappa', 0.0):.4f}",
        "-" * 70,
        "Statistical Significance (McNemar's Test):",
        f"  Discordant Pairs (b+c)           : {mcn.get('discordant_pairs', 0)}",
        f"  Chi-squared Statistic            : {mcn.get('statistic', 0.0)}",
        f"  p-value (approximate)            : {mcn.get('p_value_approx', 1.0):.4f}",
        f"  Verdict                          : {mcn.get('interpretation', 'N/A')}",
        "-" * 70,
    ]

    # Category comparisons
    lines.append("Category-by-Category Accuracy Comparison:")
    all_cats = sorted(set(meta_a['metrics']['accuracy_by_category'].keys()) | set(meta_b['metrics']['accuracy_by_category'].keys()))
    lines.append(f"  {'Category':<16} | {'Model A Acc':<14} | {'Model B Acc':<14}")
    lines.append("  " + "-" * 48)
    for cat in all_cats:
        acc_a = meta_a['metrics']['accuracy_by_category'].get(cat, {}).get('accuracy_percent', 0.0)
        acc_b = meta_b['metrics']['accuracy_by_category'].get(cat, {}).get('accuracy_percent', 0.0)
        lines.append(f"  {cat:<16} | {acc_a:<13.1f}% | {acc_b:<13.1f}%")

    if cm["disagreement_details"]:
        lines.append("-" * 70)
        lines.append("Discordant Cases Summary (Where Models Disagreed):")
        for d in cm["disagreement_details"]:
            winner = "Model A" if d["model_a_passed"] else "Model B"
            lines.append(f"  [Case {d['id']:02d}] Winner: {winner:<7} | Category: {d['category']} | Diff: {d['difficulty']}")
            lines.append(f"    Q: {d['question']}")
            if d["model_a_error"]:
                lines.append(f"    Model A Error: {d['model_a_error']}")
            if d["model_b_error"]:
                lines.append(f"    Model B Error: {d['model_b_error']}")

    lines.append("=" * 70)
    return "\n".join(lines)


def save_evaluation_reports(
    results_a: List[Dict[str, Any]],
    results_b: List[Dict[str, Any]],
    model_a_meta: Dict[str, Any],
    model_b_meta: Dict[str, Any],
    total_planned: int,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate and persist all 6 individual and comparative reports."""
    target_dir = Path(output_dir) if output_dir else DEFAULT_RESULTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    now_utc = datetime.now(timezone.utc).isoformat()

    # 1. Process Model A
    metrics_a = compute_model_metrics(results_a, total_planned)
    latencies_a = [r.get("latency_seconds", 0.0) for r in results_a]
    lat_summary_a = compute_latency_summary(latencies_a)
    ci_a = wilson_score_interval(metrics_a["passed"], metrics_a["completed_cases"])

    report_a = {
        "run_timestamp_utc": now_utc,
        "model_name": model_a_meta["model_name"],
        "provider": model_a_meta["provider"],
        "model_identifier": f"{model_a_meta['provider']}/{model_a_meta['model_name']}",
        "confidence_interval_95": f"[{ci_a[0]}%, {ci_a[1]}%]",
        "metrics": metrics_a,
        "latency_summary": lat_summary_a,
        "results": results_a,
    }

    # 2. Process Model B
    metrics_b = compute_model_metrics(results_b, total_planned)
    latencies_b = [r.get("latency_seconds", 0.0) for r in results_b]
    lat_summary_b = compute_latency_summary(latencies_b)
    ci_b = wilson_score_interval(metrics_b["passed"], metrics_b["completed_cases"])

    report_b = {
        "run_timestamp_utc": now_utc,
        "model_name": model_b_meta["model_name"],
        "provider": model_b_meta["provider"],
        "model_identifier": f"{model_b_meta['provider']}/{model_b_meta['model_name']}",
        "confidence_interval_95": f"[{ci_b[0]}%, {ci_b[1]}%]",
        "metrics": metrics_b,
        "latency_summary": lat_summary_b,
        "results": results_b,
    }

    # 3. Process Comparative Metrics
    comp_metrics = compute_comparison_metrics(
        results_a,
        results_b,
        report_a["model_identifier"],
        report_b["model_identifier"],
    )

    kappa = compute_cohens_kappa(
        comp_metrics["both_passed_count"],
        comp_metrics["both_failed_count"],
        comp_metrics["model_a_only_passed_count"],
        comp_metrics["model_b_only_passed_count"],
    )

    mcnemar = compute_mcnemar_test(
        comp_metrics["model_a_only_passed_count"],
        comp_metrics["model_b_only_passed_count"],
    )

    comparison_report = {
        "run_timestamp_utc": now_utc,
        "model_a": {
            "model_name": model_a_meta["model_name"],
            "provider": model_a_meta["provider"],
            "metrics": metrics_a,
            "latency_summary": lat_summary_a,
            "confidence_interval_95": f"[{ci_a[0]}%, {ci_a[1]}%]",
        },
        "model_b": {
            "model_name": model_b_meta["model_name"],
            "provider": model_b_meta["provider"],
            "metrics": metrics_b,
            "latency_summary": lat_summary_b,
            "confidence_interval_95": f"[{ci_b[0]}%, {ci_b[1]}%]",
        },
        "comparison_metrics": comp_metrics,
        "cohens_kappa": kappa,
        "mcnemar_test": mcnemar,
    }

    # Write files
    (target_dir / "model_a_results.json").write_text(json.dumps(report_a, indent=2, default=str), encoding="utf-8")
    (target_dir / "model_a_summary.txt").write_text(format_model_text_summary(report_a), encoding="utf-8")

    (target_dir / "model_b_results.json").write_text(json.dumps(report_b, indent=2, default=str), encoding="utf-8")
    (target_dir / "model_b_summary.txt").write_text(format_model_text_summary(report_b), encoding="utf-8")

    (target_dir / "comparison_results.json").write_text(json.dumps(comparison_report, indent=2, default=str), encoding="utf-8")
    (target_dir / "comparison_summary.txt").write_text(format_comparison_text_summary(comparison_report), encoding="utf-8")
    return comparison_report


def recreate_comparison_from_files(results_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Recreate comparative reports from existing model_a_results.json and model_b_results.json."""
    target_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    path_a = target_dir / "model_a_results.json"
    path_b = target_dir / "model_b_results.json"

    if not path_a.exists() or not path_b.exists():
        raise FileNotFoundError(f"Missing required result files: {path_a} or {path_b}")

    with open(path_a, "r", encoding="utf-8") as f:
        rep_a = json.load(f)
    with open(path_b, "r", encoding="utf-8") as f:
        rep_b = json.load(f)

    return save_evaluation_reports(
        results_a=rep_a["results"],
        results_b=rep_b["results"],
        model_a_meta={"model_name": rep_a["model_name"], "provider": rep_a["provider"]},
        model_b_meta={"model_name": rep_b["model_name"], "provider": rep_b["provider"]},
        total_planned=rep_a["metrics"]["total_planned"],
        output_dir=target_dir,
    )
