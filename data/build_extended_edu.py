"""
Additional Educational Datasets for broader evaluation.

Adds:
  - ScienceQA: Multi-modal science QA with grade levels and topics
  - RACE: Reading comprehension at middle/high school levels
  - ARC with challenge reasoning

These provide more Bloom level diversity and larger evaluation sets.
Bloom classification uses cip29/bert-blooms-taxonomy-classifier (HuggingFace),
not regex patterns.

Usage:
    python data/build_extended_edu.py --output_dir /tmp/data/edu_extended
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.bloom_classifier import classify_bloom_batch


def load_scienceqa(max_samples=5000):
    """
    ScienceQA: science questions with grade levels, topics, hints, and explanations.
    The hints and explanations serve as retrieval passages.
    """
    from datasets import load_dataset
    print("  Loading ScienceQA...")

    try:
        ds = load_dataset("derek-thomas/ScienceQA", split="train", trust_remote_code=True)
    except Exception:
        try:
            ds = load_dataset("tasksource/ScienceQA", split="train", trust_remote_code=True)
        except Exception as e:
            print(f"    Failed: {e}")
            return [], []

    raw = []
    for row in ds:
        if len(raw) >= max_samples:
            break
        question = row.get("question", "")
        hint = row.get("hint", "")
        solution = row.get("solution", "")
        passage = f"{hint} {solution}".strip()
        if len(passage) < 30:
            continue
        topic = row.get("topic", "general")
        grade = row.get("grade", None)
        subject = row.get("subject", "science")
        choices = row.get("choices", [])
        answer_idx = row.get("answer", 0)
        raw.append((question, passage, topic, grade, subject, choices, answer_idx))

    print(f"    Classifying {len(raw)} ScienceQA queries...")
    bloom_levels = classify_bloom_batch([r[0] for r in raw])

    corpus = []
    pairs = []
    for pid, (question, passage, topic, grade, subject, choices, answer_idx), bloom in \
            zip(range(len(raw)), raw, bloom_levels):
        answer = choices[answer_idx] if answer_idx < len(choices) else ""
        p_id = f"sciqa_{pid}"
        corpus.append({
            "id": p_id, "text": passage,
            "subject": subject.lower() if subject else "science",
            "topic": topic.lower() if topic else "general",
            "bloom_level": bloom, "source": "scienceqa",
            "difficulty": "easy" if (grade and grade <= 5) else "medium",
        })
        pairs.append({
            "query": question,
            "positive_text": passage,
            "positive_id": p_id,
            "negative_texts": [],
            "negative_ids": [],
            "bloom_level": bloom,
            "subject": subject.lower() if subject else "science",
            "topic": topic.lower() if topic else "general",
            "query_type": "factual" if bloom <= 2 else "conceptual",
        })

    print(f"    Loaded {len(corpus)} passages, {len(pairs)} pairs from ScienceQA")
    return corpus, pairs


def load_race(max_samples=3000):
    """
    RACE: Reading comprehension with middle school and high school levels.
    Passages are reading texts, queries are comprehension questions.
    """
    from datasets import load_dataset
    print("  Loading RACE...")

    raw = []  # (question, article, level, bloom_base)
    for level, bloom_base in [("middle", 2), ("high", 3)]:
        try:
            ds = load_dataset("ehovy/race", level, split="train", trust_remote_code=True)
        except Exception as e:
            print(f"    Failed to load RACE-{level}: {e}")
            continue

        for row in ds:
            if len(raw) >= max_samples:
                break
            article = row.get("article", "")
            question = row.get("question", "")
            if len(article) < 50 or not question:
                continue
            raw.append((question, article[:500], level, bloom_base))

    print(f"    Classifying {len(raw)} RACE queries...")
    bloom_levels = classify_bloom_batch([r[0] for r in raw])

    corpus = []
    pairs = []
    for pid, (question, article, level, bloom_base), bloom in \
            zip(range(len(raw)), raw, bloom_levels):
        bloom = max(bloom, bloom_base)
        p_id = f"race_{pid}"
        corpus.append({
            "id": p_id, "text": article,
            "subject": "reading_comprehension",
            "topic": "general",
            "bloom_level": bloom, "source": f"race_{level}",
            "difficulty": level,
        })
        pairs.append({
            "query": question,
            "positive_text": article,
            "positive_id": p_id,
            "negative_texts": [],
            "negative_ids": [],
            "bloom_level": bloom,
            "subject": "reading_comprehension",
            "topic": "general",
            "query_type": "conceptual" if bloom >= 3 else "factual",
        })

    print(f"    Loaded {len(corpus)} passages, {len(pairs)} pairs from RACE")
    return corpus, pairs


def build_extended_dataset(output_dir, seed=42):
    """Build extended educational dataset from multiple sources."""
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    print("=" * 60)
    print("Building Extended Educational Dataset")
    print("=" * 60)

    all_corpus = []
    all_pairs = []

    # Load ScienceQA
    c, p = load_scienceqa(max_samples=5000)
    all_corpus.extend(c)
    all_pairs.extend(p)

    # Load RACE
    c, p = load_race(max_samples=3000)
    all_corpus.extend(c)
    all_pairs.extend(p)

    # Also load the original educational data if available
    original_corpus = "/tmp/data/real/corpus.jsonl"
    original_train = "/tmp/data/real/train.jsonl"
    if os.path.exists(original_corpus) and os.path.exists(original_train):
        print("  Merging with original educational data...")
        with open(original_corpus) as f:
            for line in f:
                all_corpus.append(json.loads(line))
        with open(original_train) as f:
            for line in f:
                all_pairs.append(json.loads(line))

    print(f"\n  Total corpus: {len(all_corpus)}")
    print(f"  Total pairs: {len(all_pairs)}")

    # Bloom distribution
    bloom_dist = defaultdict(int)
    for p in all_pairs:
        bloom_dist[p["bloom_level"]] += 1
    print(f"  Bloom distribution: {dict(sorted(bloom_dist.items()))}")

    # Save corpus
    with open(os.path.join(output_dir, "corpus.jsonl"), "w") as f:
        for c in all_corpus:
            f.write(json.dumps(c) + "\n")

    # Split pairs
    random.shuffle(all_pairs)
    n = len(all_pairs)
    splits = {
        "train": all_pairs[:int(0.8 * n)],
        "val": all_pairs[int(0.8 * n):int(0.9 * n)],
        "test": all_pairs[int(0.9 * n):],
    }
    for name, sp in splits.items():
        with open(os.path.join(output_dir, f"{name}.jsonl"), "w") as f:
            for p in sp:
                f.write(json.dumps(p) + "\n")
        print(f"  {name}: {len(sp)} pairs")

    print(f"\nDone! Data at {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/tmp/data/edu_extended")
    args = parser.parse_args()
    build_extended_dataset(args.output_dir)
