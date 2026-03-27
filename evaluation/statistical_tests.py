"""
Statistical testing for NeurIPS rigor.
Bootstrap confidence intervals and paired significance tests.
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 1000,
                  ci: float = 0.95) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval.
    Returns: (mean, lower, upper)
    """
    n = len(values)
    bootstraps = np.array([
        np.random.choice(values, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    lower = np.percentile(bootstraps, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstraps, (1 + ci) / 2 * 100)
    return float(values.mean()), float(lower), float(upper)


def paired_bootstrap_test(scores_a: np.ndarray, scores_b: np.ndarray,
                           n_bootstrap: int = 10000) -> float:
    """
    Paired bootstrap significance test.
    Returns p-value for the null hypothesis that A and B perform equally.
    """
    n = len(scores_a)
    assert len(scores_b) == n

    observed_diff = scores_a.mean() - scores_b.mean()

    count = 0
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        diff = scores_a[idx].mean() - scores_b[idx].mean()
        if diff <= 0:  # One-sided: is A not better than B?
            count += 1

    return count / n_bootstrap


def compute_per_query_scores(rankings: np.ndarray, gt_indices: np.ndarray,
                              k: int = 10) -> np.ndarray:
    """Compute per-query binary hit scores for statistical testing."""
    N = len(gt_indices)
    scores = np.zeros(N)
    for i in range(N):
        if gt_indices[i] in rankings[i, :k]:
            scores[i] = 1.0
    return scores


def full_significance_report(
    model_a_scores: Dict[str, np.ndarray],  # {metric_name: per_query_scores}
    model_b_scores: Dict[str, np.ndarray],
    model_a_name: str = "QA-MRL",
    model_b_name: str = "MRL Baseline",
    n_bootstrap: int = 5000,
) -> str:
    """Generate a full significance report."""
    lines = []
    lines.append(f"\nStatistical Significance Report")
    lines.append(f"{'='*70}")
    lines.append(f"{'Metric':25s} {model_a_name:>12s} {model_b_name:>12s} {'Diff':>8s} {'p-value':>10s} {'Sig?':>6s}")
    lines.append("-" * 70)

    for metric in model_a_scores:
        a = model_a_scores[metric]
        b = model_b_scores.get(metric, np.zeros_like(a))

        mean_a, lo_a, hi_a = bootstrap_ci(a, n_bootstrap)
        mean_b, lo_b, hi_b = bootstrap_ci(b, n_bootstrap)
        diff = mean_a - mean_b
        p_val = paired_bootstrap_test(a, b, n_bootstrap)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

        a_str = f"{mean_a:.4f}"
        b_str = f"{mean_b:.4f}"
        d_str = f"{diff:+.4f}"
        p_str = f"{p_val:.4f}"

        lines.append(f"{metric:25s} {a_str:>12s} {b_str:>12s} {d_str:>8s} {p_str:>10s} {sig:>6s}")

    lines.append("-" * 70)
    lines.append("Significance: * p<0.05, ** p<0.01, *** p<0.001")
    return "\n".join(lines)