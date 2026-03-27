"""Tail-topic evaluation: performance on rare/specialized topics."""

import numpy as np
from typing import Dict, List
from collections import Counter


def tail_topic_evaluation(
    similarity_matrix: np.ndarray,
    query_topics: List[str],
    corpus_topics: List[str],
    ks: List[int] = [1, 5, 10],
    tail_threshold: int = 50,
) -> Dict[str, float]:
    """
    Evaluate retrieval on tail (rare) vs head (common) topics.

    Topics with fewer than `tail_threshold` passages in the corpus
    are considered "tail" topics. MRL's coarse subspace collapse
    hurts tail topics disproportionately.
    """
    n = similarity_matrix.shape[0]
    rankings = np.argsort(-similarity_matrix, axis=1)
    relevant = np.arange(n)

    topic_counts = Counter(corpus_topics)
    tail_topics = {t for t, c in topic_counts.items() if c < tail_threshold}

    tail_mask = np.array([t in tail_topics for t in query_topics])
    head_mask = ~tail_mask

    metrics = {}
    for name, mask in [("tail", tail_mask), ("head", head_mask)]:
        if mask.sum() == 0:
            continue
        for k in ks:
            topk = rankings[mask][:, :k]
            hits = np.any(topk == relevant[mask][:, None], axis=1)
            metrics[f"{name}_recall@{k}"] = float(hits.mean())

    if "tail_recall@10" in metrics and "head_recall@10" in metrics:
        metrics["tail_head_gap@10"] = metrics["head_recall@10"] - metrics["tail_recall@10"]

    metrics["num_tail_queries"] = int(tail_mask.sum())
    metrics["num_head_queries"] = int(head_mask.sum())

    return metrics
