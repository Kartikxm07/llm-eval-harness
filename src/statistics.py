"""Statistical analysis, latency profiling, and paired hypothesis testing."""

import math
from typing import Any, Dict, List, Optional, Tuple


def compute_latency_summary(latencies: List[float]) -> Dict[str, Any]:
    """Compute comprehensive latency distribution metrics in seconds."""
    valid = [float(x) for x in latencies if x is not None and x > 0]
    if not valid:
        return {
            "count": 0,
            "mean_seconds": 0.0,
            "median_seconds": 0.0,
            "p90_seconds": 0.0,
            "p95_seconds": 0.0,
            "min_seconds": 0.0,
            "max_seconds": 0.0,
        }

    valid.sort()
    n = len(valid)

    mean_val = sum(valid) / n

    def percentile(p: float) -> float:
        idx = int(math.ceil(p * n)) - 1
        return valid[max(0, min(idx, n - 1))]

    return {
        "count": n,
        "mean_seconds": round(mean_val, 4),
        "median_seconds": round(percentile(0.50), 4),
        "p90_seconds": round(percentile(0.90), 4),
        "p95_seconds": round(percentile(0.95), 4),
        "min_seconds": round(valid[0], 4),
        "max_seconds": round(valid[-1], 4),
    }


def wilson_score_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Calculate Wilson score 95% confidence interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + (z ** 2) / total
    center_adjusted = (p + (z ** 2) / (2 * total)) / denominator
    spread = (z / denominator) * math.sqrt((p * (1 - p) / total) + (z ** 2) / (4 * (total ** 2)))
    low = max(0.0, center_adjusted - spread)
    high = min(1.0, center_adjusted + spread)
    return round(low * 100, 2), round(high * 100, 2)


def compute_cohens_kappa(
    both_pass: int,
    both_fail: int,
    a_pass_b_fail: int,
    b_pass_a_fail: int,
) -> float:
    """Compute Cohen's Kappa coefficient of inter-model agreement."""
    total = both_pass + both_fail + a_pass_b_fail + b_pass_a_fail
    if total == 0:
        return 0.0

    po = (both_pass + both_fail) / total

    p_a_yes = (both_pass + a_pass_b_fail) / total
    p_b_yes = (both_pass + b_pass_a_fail) / total
    p_a_no = 1.0 - p_a_yes
    p_b_no = 1.0 - p_b_yes

    pe = (p_a_yes * p_b_yes) + (p_a_no * p_b_no)

    if math.isclose(pe, 1.0):
        return 1.0

    kappa = (po - pe) / (1.0 - pe)
    return round(kappa, 4)


def compute_mcnemar_test(
    b: int,
    c: int,
) -> Dict[str, Any]:
    """Perform McNemar's test with continuity correction on discordant pairs.

    b: Model A passed, Model B failed
    c: Model B passed, Model A failed
    """
    discordant_total = b + c
    if discordant_total == 0:
        return {
            "statistic": 0.0,
            "p_value_approx": 1.0,
            "is_statistically_significant": False,
            "interpretation": "Both models had identical accuracy across all cases.",
        }

    # McNemar's statistic with Edward's continuity correction
    stat = ((abs(b - c) - 1.0) ** 2) / discordant_total

    # Upper-tail p-value for chi-squared with df=1 approximation using standard normal
    # p ≈ 2 * (1 - Φ(sqrt(stat)))
    z = math.sqrt(stat)
    p_approx = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2))))
    p_approx = max(0.0, min(1.0, p_approx))

    is_significant = p_approx < 0.05

    if is_significant:
        interp = (
            f"Statistically significant difference detected (p ~= {p_approx:.4f} < 0.05). "
            f"The observed performance gap is unlikely due to random variation."
        )
    else:
        interp = (
            f"Difference is NOT statistically significant (p ~= {p_approx:.4f} >= 0.05). "
            f"Due to sample size (N=50), performance difference could be random variation."
        )

    return {
        "statistic": round(stat, 4),
        "discordant_pairs": discordant_total,
        "p_value_approx": round(p_approx, 4),
        "is_statistically_significant": is_significant,
        "interpretation": interp,
    }
