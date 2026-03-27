"""Standard retrieval metrics: Recall@K, NDCG@K, MRR, MAP."""

import numpy as np
from typing import Dict, List


def recall_at_k(rankings: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """Fraction of queries where the relevant doc appears in top-k."""
    topk = rankings[:, :k]
    hits = np.any(topk == relevant[:, None], axis=1)
    return float(hits.mean())


def mrr(rankings: np.ndarray, relevant: np.ndarray) -> float:
    """Mean Reciprocal Rank."""
    rrs = []
    for i in range(len(relevant)):
        matches = np.where(rankings[i] == relevant[i])[0]
        rrs.append(1.0 / (matches[0] + 1) if len(matches) > 0 else 0.0)
    return float(np.mean(rrs))


def ndcg_at_k(rankings: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """NDCG@K with binary relevance."""
    ndcgs = []
    for i in range(len(relevant)):
        dcg = 0.0
        for j, doc in enumerate(rankings[i, :k]):
            if doc == relevant[i]:
                dcg += 1.0 / np.log2(j + 2)
        ndcgs.append(dcg)  # IDCG = 1.0 for binary relevance
    return float(np.mean(ndcgs))


def mean_average_precision(rankings: np.ndarray, relevant: np.ndarray) -> float:
    """MAP (single relevant doc per query)."""
    aps = []
    for i in range(len(relevant)):
        matches = np.where(rankings[i] == relevant[i])[0]
        if len(matches) > 0:
            aps.append(1.0 / (matches[0] + 1))
        else:
            aps.append(0.0)
    return float(np.mean(aps))


def compute_all_metrics(
    similarity_matrix: np.ndarray,
    ks: List[int] = [1, 5, 10, 20, 50],
) -> Dict[str, float]:
    """
    Compute all standard metrics from a [N_q x N_d] similarity matrix.
    Assumes the diagonal is the correct match (query i -> doc i).
    """
    n = similarity_matrix.shape[0]
    rankings = np.argsort(-similarity_matrix, axis=1)  # descending
    relevant = np.arange(n)

    metrics = {}
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(rankings, relevant, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(rankings, relevant, k)
    metrics["mrr"] = mrr(rankings, relevant)
    metrics["map"] = mean_average_precision(rankings, relevant)
    return metrics
