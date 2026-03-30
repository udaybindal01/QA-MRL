"""
Bloom's taxonomy stratified evaluation - the key novel metric.

v3: Bloom level is a property of the QUERY only — documents have no Bloom
label. Metrics are stratified by query Bloom level, with bootstrap CIs
for levels with small sample sizes.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply",
               4: "Analyze", 5: "Evaluate", 6: "Create"}


def bootstrap_ci(hits: np.ndarray, n_bootstrap: int = 1000,
                 ci: float = 0.95, seed: int = 42):
    """Compute bootstrap confidence interval for a binary metric."""
    rng = np.random.RandomState(seed)
    n = len(hits)
    if n == 0:
        return 0.0, 0.0, 0.0
    means = np.array([rng.choice(hits, size=n, replace=True).mean()
                      for _ in range(n_bootstrap)])
    means.sort()
    lo = float(means[int((1 - ci) / 2 * n_bootstrap)])
    hi = float(means[int((1 + ci) / 2 * n_bootstrap)])
    return float(hits.mean()), lo, hi


def bloom_stratified_evaluation(
    similarity_matrix: np.ndarray,   # [N_q, N_d]
    query_blooms: np.ndarray,        # [N_q] values 1-6
    gt_indices: np.ndarray,          # [N_q] ground-truth doc index per query
    ks: List[int] = [1, 5, 10],
    compute_bootstrap: bool = True,
) -> Dict[str, float]:
    """
    Compute retrieval metrics stratified by query Bloom level.

    v3: No doc_blooms parameter — Bloom is a query-only property.
    Ground truth is specified by gt_indices (index of the correct doc
    for each query), not by diagonal assumption.

    Args:
        similarity_matrix: [N_q, N_d] query-document similarity scores
        query_blooms: [N_q] Bloom level per query (1-indexed, 1-6)
        gt_indices: [N_q] index of the ground-truth document per query
        ks: list of K values for Recall@K
        compute_bootstrap: whether to compute 95% bootstrap CIs
    """
    n_q = similarity_matrix.shape[0]
    rankings = np.argsort(-similarity_matrix, axis=1)

    metrics = {}

    # Per-level recall with bootstrap CIs
    for level in range(1, 7):
        mask = query_blooms == level
        n_level = int(mask.sum())
        if n_level == 0:
            continue
        name = BLOOM_NAMES[level]
        level_rankings = rankings[mask]
        level_gt = gt_indices[mask]

        for k in ks:
            topk = level_rankings[:, :k]
            hits = np.array([level_gt[i] in topk[i] for i in range(n_level)])
            metrics[f"bloom_{name}_recall@{k}"] = float(hits.mean())

            if k == 10 and compute_bootstrap:
                mean, lo, hi = bootstrap_ci(hits.astype(float))
                metrics[f"bloom_{name}_recall@10_ci_lo"] = lo
                metrics[f"bloom_{name}_recall@10_ci_hi"] = hi

        metrics[f"bloom_{name}_n"] = n_level

    return metrics


def format_bloom_results(metrics: Dict[str, float]) -> str:
    """Pretty-format Bloom-stratified results."""
    lines = ["\nBloom-Stratified Results (query Bloom only):", "-" * 60]
    for level in range(1, 7):
        name = BLOOM_NAMES[level]
        n = metrics.get(f"bloom_{name}_n", 0)
        r1 = metrics.get(f"bloom_{name}_recall@1", 0)
        r10 = metrics.get(f"bloom_{name}_recall@10", 0)
        ci_lo = metrics.get(f"bloom_{name}_recall@10_ci_lo", 0)
        ci_hi = metrics.get(f"bloom_{name}_recall@10_ci_hi", 0)
        if n > 0:
            ci_str = f"  95% CI=[{ci_lo:.3f}, {ci_hi:.3f}]" if ci_lo or ci_hi else ""
            lines.append(f"  {name:12s} (n={n:4d})  R@1={r1:.4f}  R@10={r10:.4f}{ci_str}")
    return "\n".join(lines)
