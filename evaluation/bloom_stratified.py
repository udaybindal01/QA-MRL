"""Bloom's taxonomy stratified evaluation - the key novel metric."""

import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply",
               4: "Analyze", 5: "Evaluate", 6: "Create"}


def bloom_stratified_evaluation(
    similarity_matrix: np.ndarray,   # [N, N]
    query_blooms: np.ndarray,        # [N] values 1-6
    doc_blooms: np.ndarray,          # [N] values 1-6
    ks: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Compute retrieval metrics stratified by query Bloom level.

    Also computes Bloom-Aligned Recall: fraction of top-K results
    that match the query's Bloom level.
    """
    n = similarity_matrix.shape[0]
    rankings = np.argsort(-similarity_matrix, axis=1)
    relevant = np.arange(n)

    metrics = {}

    # Per-level recall
    for level in range(1, 7):
        mask = query_blooms == level
        if mask.sum() == 0:
            continue
        name = BLOOM_NAMES[level]
        level_rankings = rankings[mask]
        level_relevant = relevant[mask]

        for k in ks:
            topk = level_rankings[:, :k]
            hits = np.any(topk == level_relevant[:, None], axis=1)
            metrics[f"bloom_{name}_recall@{k}"] = float(hits.mean())

    # Bloom-Aligned Recall@K: do retrieved docs match query's Bloom level?
    for k in ks:
        alignments = []
        for i in range(n):
            top_ids = rankings[i, :k]
            top_blooms = doc_blooms[top_ids]
            aligned = (top_blooms == query_blooms[i]).sum()
            alignments.append(aligned / k)
        metrics[f"bloom_aligned_recall@{k}"] = float(np.mean(alignments))

    return metrics


def format_bloom_results(metrics: Dict[str, float]) -> str:
    """Pretty-format Bloom-stratified results."""
    lines = ["\nBloom-Stratified Results:", "-" * 50]
    for level in range(1, 7):
        name = BLOOM_NAMES[level]
        r1 = metrics.get(f"bloom_{name}_recall@1", 0)
        r10 = metrics.get(f"bloom_{name}_recall@10", 0)
        lines.append(f"  {name:12s}  R@1={r1:.4f}  R@10={r10:.4f}")

    ba = metrics.get("bloom_aligned_recall@10", 0)
    lines.append(f"\n  Bloom-Aligned R@10: {ba:.4f}")
    return "\n".join(lines)
