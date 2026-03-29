"""
Curriculum-Guided Hard Negative Mining v3.

v3 changes: Documents have no Bloom labels. Negatives are mined by
topical confusion only (standard IR hard negative approach):

  Tier 1 (hardest): Same topic, high keyword overlap with query
  Tier 2 (hard):    Same topic, different passages
  Tier 3 (medium):  Same subject, different topic
  Tier 4 (easy):    Random from corpus

Curriculum progression controls the mix of easy vs hard negatives.
"""

import json
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import re


STOPWORDS = {"the", "and", "for", "are", "was", "were", "that", "this",
             "with", "from", "have", "has", "had", "not", "but", "what",
             "which", "when", "where", "how", "who", "why", "can", "will",
             "would", "could", "should", "does", "did", "been", "being",
             "than", "then", "them", "they", "their", "there", "these",
             "those", "into", "about", "between", "through", "during",
             "before", "after", "above", "below", "each", "every",
             "some", "such", "only", "other", "also", "most", "more"}


def _get_keywords(text: str) -> set:
    return set(re.findall(r'\b[a-z]{3,}\b', text.lower())) - STOPWORDS


class CurriculumNegativeMiner:
    """
    Mines hard negatives based on topical confusion.

    v3: No document Bloom labels. Hardness comes from topical similarity
    (same topic = harder negative), not cognitive distance.
    """

    def __init__(self, corpus_path: str, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

        # Load and index corpus
        self.corpus = []
        with open(corpus_path) as f:
            for line in f:
                self.corpus.append(json.loads(line.strip()))

        # Build indices
        self.by_topic = defaultdict(list)
        self.by_subject = defaultdict(list)

        # Precompute keywords for keyword-overlap based hardness
        self.corpus_kw = []
        for i, p in enumerate(self.corpus):
            topic = p.get("topic", "general")
            subject = p.get("subject", "general")
            self.by_topic[topic].append(i)
            self.by_subject[subject].append(i)
            self.corpus_kw.append(_get_keywords(p["text"]))

        print(f"  CurriculumNegativeMiner v3: {len(self.corpus)} passages, "
              f"{len(self.by_topic)} topics (topic-based mining, no doc Bloom)")

    def mine_tiered_negatives(
        self,
        query_text: str,
        query_topic: str,
        query_subject: str,
        positive_idx: int,
        num_per_tier: int = 1,
    ) -> Dict[str, List[int]]:
        """
        Mine negatives in tiers of difficulty.

        Tier 1 (hardest): Same topic, high keyword overlap with query
        Tier 2 (hard):    Same topic, any passage
        Tier 3 (medium):  Same subject, different topic
        Tier 4 (easy):    Random
        """
        result = {
            "topic_overlap": [],    # Tier 1: same topic + keyword overlap
            "topic_any": [],        # Tier 2: same topic, any
            "cross_topic": [],      # Tier 3: same subject, diff topic
            "random": [],           # Tier 4: random
        }
        used = {positive_idx}
        query_kw = _get_keywords(query_text)

        # Tier 1: Same topic, ranked by keyword overlap with query
        topic_cands = [j for j in self.by_topic.get(query_topic, []) if j not in used]
        if topic_cands and query_kw:
            # Sort by keyword overlap (descending) — most confusing first
            scored = [(j, len(query_kw & self.corpus_kw[j])) for j in topic_cands]
            scored.sort(key=lambda x: -x[1])
            # Top overlap = tier 1
            for j, score in scored[:num_per_tier]:
                if score > 0:
                    result["topic_overlap"].append(j)
                    used.add(j)

        # Tier 2: Same topic, remaining passages
        topic_remaining = [j for j in self.by_topic.get(query_topic, []) if j not in used]
        if topic_remaining:
            selected = random.sample(topic_remaining, min(num_per_tier, len(topic_remaining)))
            result["topic_any"].extend(selected)
            used.update(selected)

        # Tier 3: Same subject, different topic
        candidates = [
            j for j in self.by_subject.get(query_subject, [])
            if j not in used and self.corpus[j].get("topic") != query_topic
        ]
        if candidates:
            selected = random.sample(candidates, min(num_per_tier * 2, len(candidates)))
            result["cross_topic"].extend(selected)
            used.update(selected)

        # Tier 4: Random
        candidates = [j for j in range(len(self.corpus)) if j not in used]
        if candidates:
            selected = random.sample(candidates, min(num_per_tier, len(candidates)))
            result["random"].extend(selected)

        return result

    def mine_curriculum_batch(
        self,
        query_text: str,
        query_topic: str,
        query_subject: str,
        positive_idx: int,
        num_negatives: int = 3,
        curriculum_stage: float = 0.0,
    ) -> List[int]:
        """
        Mine negatives with curriculum-aware tier mixing.

        Early training (stage~0): mostly random/cross-topic (easy)
        Late training (stage~1): mostly same-topic overlap (hard)

        Returns:
            negative_indices: list of corpus indices
        """
        tiered = self.mine_tiered_negatives(
            query_text, query_topic, query_subject, positive_idx,
            num_per_tier=max(1, num_negatives),
        )

        # Tier selection probabilities based on curriculum stage
        tier_names = ["topic_overlap", "topic_any", "cross_topic", "random"]
        easy_probs = [0.05, 0.15, 0.30, 0.50]
        hard_probs = [0.40, 0.30, 0.20, 0.10]
        probs = [
            easy_probs[i] * (1 - curriculum_stage) + hard_probs[i] * curriculum_stage
            for i in range(4)
        ]

        all_negs = []
        for tier_name in tier_names:
            for idx in tiered[tier_name]:
                all_negs.append((idx, tier_name))

        if not all_negs:
            fallback = random.sample(range(len(self.corpus)), min(num_negatives, len(self.corpus)))
            return fallback

        # Sample with tier-weighted probabilities
        selected = []
        tier_weights = {t: p for t, p in zip(tier_names, probs)}

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
            selected.append(random.choice(range(len(self.corpus))))

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
    Rebuild training pairs with curriculum-guided negatives.

    v3: No document Bloom labels used in mining.
    """
    miner = CurriculumNegativeMiner(corpus_path, seed)
    corpus_id_to_idx = {p["id"]: i for i, p in enumerate(miner.corpus)}

    pairs = []
    with open(pairs_path) as f:
        for line in f:
            pairs.append(json.loads(line.strip()))

    print(f"  Rebuilding {len(pairs)} pairs with curriculum negatives v3 (stage={curriculum_stage})...")

    new_pairs = []
    for pair in tqdm(pairs, desc="  mining"):
        pos_id = pair.get("positive_id", "")
        pos_idx = corpus_id_to_idx.get(pos_id, -1)
        if pos_idx < 0:
            new_pairs.append(pair)
            continue

        topic = pair.get("topic", "general")
        subject = pair.get("subject", "general")
        query_text = pair.get("query", "")

        neg_indices = miner.mine_curriculum_batch(
            query_text, topic, subject, pos_idx,
            num_negatives=num_negatives,
            curriculum_stage=curriculum_stage,
        )

        pair["negative_ids"] = [miner.corpus[j]["id"] for j in neg_indices]
        pair["negative_texts"] = [miner.corpus[j]["text"] for j in neg_indices]
        # v3: no negative_blooms field — docs don't have Bloom labels
        pair.pop("negative_blooms", None)
        pair["neg_tier_info"] = "curriculum_v3"
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
