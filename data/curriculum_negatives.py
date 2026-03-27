"""
Curriculum-Guided Hard Negative Mining.

Novel contribution: mines negatives by Bloom-level confusion,
not just semantic similarity.

Standard hard negatives: BM25/dense retrieval neighbors (topically similar)
Curriculum negatives: same-topic, WRONG Bloom level (cognitively confusing)

The mining produces three tiers of negatives per query:
  Tier 1 (hardest): Same topic, adjacent Bloom level (e.g., Remember vs Understand)
  Tier 2 (hard):    Same topic, distant Bloom level (e.g., Remember vs Evaluate)
  Tier 3 (medium):  Same subject, different topic
  Tier 4 (easy):    Random from corpus

Each training batch mixes tiers based on curriculum stage.
"""

import json
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm


class CurriculumNegativeMiner:
    """
    Mines hard negatives based on Bloom-level cognitive distance.

    This is novel because it introduces curriculum structure into
    the negative sampling process for contrastive retrieval training.
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
        self.by_topic = defaultdict(list)       # topic -> [indices]
        self.by_subject = defaultdict(list)     # subject -> [indices]
        self.by_bloom = defaultdict(list)       # bloom_level -> [indices]
        self.by_topic_bloom = defaultdict(list) # (topic, bloom) -> [indices]

        for i, p in enumerate(self.corpus):
            topic = p.get("topic", "general")
            subject = p.get("subject", "general")
            bloom = p.get("bloom_level", 2)
            self.by_topic[topic].append(i)
            self.by_subject[subject].append(i)
            self.by_bloom[bloom].append(i)
            self.by_topic_bloom[(topic, bloom)].append(i)

        print(f"  CurriculumNegativeMiner: {len(self.corpus)} passages, "
              f"{len(self.by_topic)} topics, {len(self.by_bloom)} Bloom levels")

    def mine_tiered_negatives(
        self,
        query_topic: str,
        query_subject: str,
        query_bloom: int,
        positive_idx: int,
        num_per_tier: int = 1,
    ) -> Dict[str, List[int]]:
        """
        Mine negatives in tiers of difficulty.

        Returns dict mapping tier name to list of corpus indices.
        """
        result = {
            "bloom_adjacent": [],   # Tier 1: same topic, Bloom ±1
            "bloom_distant": [],    # Tier 2: same topic, Bloom ±2+
            "cross_topic": [],      # Tier 3: same subject, different topic
            "random": [],           # Tier 4: random
        }
        used = {positive_idx}

        # Tier 1: Same topic, adjacent Bloom (hardest)
        for delta in [1, -1]:
            adj_bloom = query_bloom + delta
            candidates = [
                j for j in self.by_topic_bloom.get((query_topic, adj_bloom), [])
                if j not in used
            ]
            if candidates:
                selected = random.sample(candidates, min(num_per_tier, len(candidates)))
                result["bloom_adjacent"].extend(selected)
                used.update(selected)

        # Tier 2: Same topic, distant Bloom
        for delta in [2, -2, 3, -3, 4, -4]:
            dist_bloom = query_bloom + delta
            if dist_bloom < 1 or dist_bloom > 6:
                continue
            candidates = [
                j for j in self.by_topic_bloom.get((query_topic, dist_bloom), [])
                if j not in used
            ]
            if candidates:
                selected = random.sample(candidates, min(num_per_tier, len(candidates)))
                result["bloom_distant"].extend(selected)
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
        query_topic: str,
        query_subject: str,
        query_bloom: int,
        positive_idx: int,
        num_negatives: int = 3,
        curriculum_stage: float = 0.0,  # 0.0 = early (easy negs), 1.0 = late (hard negs)
    ) -> Tuple[List[int], List[int]]:
        """
        Mine negatives with curriculum-aware tier mixing.

        Early training (stage~0): mostly random/cross-topic negatives (easy)
        Late training (stage~1): mostly Bloom-adjacent negatives (hard)

        Returns:
            negative_indices: list of corpus indices
            negative_bloom_levels: list of Bloom levels for each negative
        """
        tiered = self.mine_tiered_negatives(
            query_topic, query_subject, query_bloom, positive_idx,
            num_per_tier=max(1, num_negatives),
        )

        # Tier selection probabilities based on curriculum stage
        # Early: [0.1, 0.1, 0.3, 0.5] (easy)
        # Late:  [0.5, 0.3, 0.1, 0.1] (hard)
        tier_names = ["bloom_adjacent", "bloom_distant", "cross_topic", "random"]
        easy_probs = [0.1, 0.1, 0.3, 0.5]
        hard_probs = [0.5, 0.3, 0.1, 0.1]
        probs = [
            easy_probs[i] * (1 - curriculum_stage) + hard_probs[i] * curriculum_stage
            for i in range(4)
        ]

        # Collect all available negatives with their tier
        all_negs = []
        for tier_name in tier_names:
            for idx in tiered[tier_name]:
                all_negs.append((idx, tier_name))

        if not all_negs:
            # Fallback: random
            fallback = random.sample(range(len(self.corpus)), min(num_negatives, len(self.corpus)))
            return fallback, [self.corpus[j].get("bloom_level", 2) for j in fallback]

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

        # Pad if needed
        while len(selected) < num_negatives:
            selected.append(random.choice(range(len(self.corpus))))

        bloom_levels = [self.corpus[j].get("bloom_level", 2) for j in selected]
        return selected, bloom_levels


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

    Usage:
        rebuild_pairs_with_curriculum_negatives(
            "data/real/train.jsonl", "data/real/corpus.jsonl",
            "data/real/train_curriculum.jsonl",
            curriculum_stage=0.7,  # Mostly hard negatives
        )
    """
    miner = CurriculumNegativeMiner(corpus_path, seed)
    corpus_id_to_idx = {p["id"]: i for i, p in enumerate(miner.corpus)}

    pairs = []
    with open(pairs_path) as f:
        for line in f:
            pairs.append(json.loads(line.strip()))

    print(f"  Rebuilding {len(pairs)} pairs with curriculum negatives (stage={curriculum_stage})...")

    new_pairs = []
    for pair in tqdm(pairs, desc="  mining"):
        pos_id = pair.get("positive_id", "")
        pos_idx = corpus_id_to_idx.get(pos_id, -1)
        if pos_idx < 0:
            new_pairs.append(pair)
            continue

        topic = pair.get("topic", "general")
        subject = pair.get("subject", "general")
        bloom = pair.get("bloom_level", 2)

        neg_indices, neg_blooms = miner.mine_curriculum_batch(
            topic, subject, bloom, pos_idx,
            num_negatives=num_negatives,
            curriculum_stage=curriculum_stage,
        )

        pair["negative_ids"] = [miner.corpus[j]["id"] for j in neg_indices]
        pair["negative_texts"] = [miner.corpus[j]["text"] for j in neg_indices]
        pair["negative_blooms"] = neg_blooms
        pair["neg_tier_info"] = "curriculum"
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
