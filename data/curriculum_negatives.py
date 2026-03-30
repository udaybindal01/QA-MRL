"""
Curriculum-Guided Hard Negative Mining v4.

v4 changes: No subject/topic labels used. Negatives are mined purely by
BM25-style keyword overlap between query and corpus passages:

  Tier 1 (hardest): High keyword overlap with query (most lexically confusing)
  Tier 2 (medium):  Low/nonzero keyword overlap
  Tier 3 (easy):    Zero overlap — unrelated passages

Curriculum progression controls the mix of easy vs hard negatives.
No gold-labeled subject or topic fields required.
"""

import json
import random
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm


STOPWORDS = {"the", "and", "for", "are", "was", "were", "that", "this",
             "with", "from", "have", "has", "had", "not", "but", "what",
             "which", "when", "where", "how", "who", "why", "can", "will",
             "would", "could", "should", "does", "did", "been", "being",
             "than", "then", "them", "they", "their", "there", "these",
             "those", "into", "about", "between", "through", "during",
             "before", "after", "above", "below", "each", "every",
             "some", "such", "only", "other", "also", "most", "more"}


def _get_keywords(text: str) -> set:
    words = text.lower().split()
    return {w.strip(".,!?;:\"'()[]{}") for w in words
            if len(w) >= 3 and w.strip(".,!?;:\"'()[]{}").isalpha()} - STOPWORDS


class CurriculumNegativeMiner:
    """
    Mines hard negatives by BM25-style keyword overlap.

    v4: No subject/topic labels required. Hardness is determined entirely
    by keyword overlap between the query and each corpus passage — the
    higher the overlap, the more lexically confusing the negative.
    """

    def __init__(self, corpus_path: str, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

        self.corpus = []
        with open(corpus_path) as f:
            for line in f:
                self.corpus.append(json.loads(line.strip()))

        # Precompute keywords for every passage (BM25-style overlap scoring)
        self.corpus_kw = [_get_keywords(p["text"]) for p in self.corpus]

        print(f"  CurriculumNegativeMiner v4: {len(self.corpus)} passages, "
              f"BM25 keyword overlap mining (no subject/topic labels needed)")

    def mine_tiered_negatives(
        self,
        query_text: str,
        positive_idx: int,
        num_per_tier: int = 1,
    ) -> Dict[str, List[int]]:
        """
        Mine negatives tiered by BM25-style keyword overlap with query.

        Tier 1 (hardest): High overlap — top half of overlapping passages
        Tier 2 (medium):  Low overlap — bottom half of overlapping passages
        Tier 3 (easy):    Zero overlap — unrelated passages
        """
        query_kw = _get_keywords(query_text)

        # Score every passage by keyword overlap with query
        scored = []
        for j, ckw in enumerate(self.corpus_kw):
            if j == positive_idx:
                continue
            overlap = len(query_kw & ckw) if query_kw else 0
            scored.append((j, overlap))

        # Sort descending by overlap
        scored.sort(key=lambda x: -x[1])

        # Split into tiers by overlap score
        overlapping = [(j, s) for j, s in scored if s > 0]
        zero_overlap = [j for j, s in scored if s == 0]

        mid = max(1, len(overlapping) // 2)
        tier1 = [j for j, _ in overlapping[:mid]]   # high overlap (hardest)
        tier2 = [j for j, _ in overlapping[mid:]]   # low overlap (medium)
        tier3 = zero_overlap                         # no overlap (easy)

        return {
            "bm25_hard": random.sample(tier1, min(num_per_tier, len(tier1))) if tier1 else [],
            "bm25_medium": random.sample(tier2, min(num_per_tier, len(tier2))) if tier2 else [],
            "random": random.sample(tier3, min(num_per_tier * 2, len(tier3))) if tier3 else [],
        }

    def mine_curriculum_batch(
        self,
        query_text: str,
        positive_idx: int,
        num_negatives: int = 3,
        curriculum_stage: float = 0.0,
    ) -> List[int]:
        """
        Mine negatives with curriculum-aware tier mixing.

        Early training (stage~0): mostly zero-overlap passages (easy)
        Late training (stage~1): mostly high-overlap passages (hard)

        Returns:
            negative_indices: list of corpus indices
        """
        tiered = self.mine_tiered_negatives(
            query_text, positive_idx,
            num_per_tier=max(1, num_negatives),
        )

        tier_names = ["bm25_hard", "bm25_medium", "random"]
        easy_probs = [0.05, 0.20, 0.75]
        hard_probs = [0.60, 0.30, 0.10]
        probs = [
            easy_probs[i] * (1 - curriculum_stage) + hard_probs[i] * curriculum_stage
            for i in range(3)
        ]

        all_negs = []
        for tier_name in tier_names:
            for idx in tiered[tier_name]:
                all_negs.append((idx, tier_name))

        if not all_negs:
            return random.sample(range(len(self.corpus)), min(num_negatives, len(self.corpus)))

        tier_weights = {t: p for t, p in zip(tier_names, probs)}
        selected = []

        for _ in range(num_negatives):
            if not all_negs:
                break
            weights = [tier_weights[t] for _, t in all_negs]
            total = sum(weights)
            weights = [w / total for w in weights]
            idx = np.random.choice(len(all_negs), p=weights)
            selected.append(all_negs[idx][0])
            all_negs.pop(idx)

        while len(selected) < num_negatives:
            selected.append(random.randrange(len(self.corpus)))

        return selected


def rebuild_pairs_with_curriculum_negatives(
    pairs_path: str,
    corpus_path: str,
    output_path: str,
    num_negatives: int = 3,
    curriculum_stage: float = 0.5,
    seed: int = 42,
):
    """
    Rebuild training pairs with curriculum-guided BM25 negatives.

    v4: No subject/topic labels used. Hardness determined by keyword overlap.
    """
    miner = CurriculumNegativeMiner(corpus_path, seed)
    corpus_id_to_idx = {p["id"]: i for i, p in enumerate(miner.corpus)}

    pairs = []
    with open(pairs_path) as f:
        for line in f:
            pairs.append(json.loads(line.strip()))

    print(f"  Rebuilding {len(pairs)} pairs with BM25 curriculum negatives v4 (stage={curriculum_stage})...")

    new_pairs = []
    for pair in tqdm(pairs, desc="  mining"):
        pos_id = pair.get("positive_id", "")
        pos_idx = corpus_id_to_idx.get(pos_id, -1)
        if pos_idx < 0:
            new_pairs.append(pair)
            continue

        query_text = pair.get("query", "")

        neg_indices = miner.mine_curriculum_batch(
            query_text, pos_idx,
            num_negatives=num_negatives,
            curriculum_stage=curriculum_stage,
        )

        pair["negative_ids"] = [miner.corpus[j]["id"] for j in neg_indices]
        pair["negative_texts"] = [miner.corpus[j]["text"] for j in neg_indices]
        pair.pop("negative_blooms", None)
        pair["neg_tier_info"] = "curriculum_v4_bm25"
        new_pairs.append(pair)

    with open(output_path, "w") as f:
        for p in new_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"  Saved to {output_path}")
    return new_pairs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, help="Input pairs JSONL")
    parser.add_argument("--corpus", required=True, help="Corpus JSONL")
    parser.add_argument("--output", required=True, help="Output pairs JSONL")
    parser.add_argument("--stage", type=float, default=0.7)
    parser.add_argument("--num_neg", type=int, default=3)
    args = parser.parse_args()

    rebuild_pairs_with_curriculum_negatives(
        args.pairs, args.corpus, args.output,
        num_negatives=args.num_neg, curriculum_stage=args.stage,
    )
