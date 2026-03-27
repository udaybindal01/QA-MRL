"""
MS MARCO data builder - matches actual HuggingFace schema.

MS MARCO v2.1 format:
  - query: str
  - passages: {is_selected: [0,1,...], passage_text: ["...", ...]}
  - is_selected=1 marks the positive passage

Usage:
    export HF_HOME=/tmp/hf_cache
    python data/build_msmarco.py --output_dir /tmp/data/msmarco --max_train 100000
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_msmarco_dataset(output_dir, max_train=100000, num_hard_negatives=1, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    print("=" * 60)
    print("Building MS MARCO Training Data")
    print("=" * 60)

    from datasets import load_dataset

    print("\nLoading MS MARCO v2.1 (streaming)...")
    ds = load_dataset("ms_marco", "v2.1", split="train", streaming=True, trust_remote_code=True)

    corpus = {}  # passage_id -> text
    pairs = []
    pid_counter = 0
    skipped = 0

    print(f"Processing up to {max_train} queries...")
    for i, row in enumerate(tqdm(ds, total=max_train, desc="  processing")):
        if len(pairs) >= max_train:
            break

        query = row["query"]
        passages = row["passages"]

        if not passages or not passages.get("passage_text"):
            skipped += 1
            continue

        texts = passages["passage_text"]
        selected = passages["is_selected"]

        # Find positive passage (is_selected=1)
        positive_text = None
        positive_id = None
        negative_texts = []
        negative_ids = []

        for j, (text, sel) in enumerate(zip(texts, selected)):
            text = text.strip()
            if len(text) < 20:
                continue

            # Add to corpus
            p_id = f"p_{pid_counter}"
            corpus[p_id] = text
            pid_counter += 1

            if sel == 1 and positive_text is None:
                positive_text = text
                positive_id = p_id
            else:
                negative_texts.append(text)
                negative_ids.append(p_id)

        if positive_text is None:
            skipped += 1
            continue

        # Take up to num_hard_negatives from same passage set (BM25 hard negatives)
        neg_t = negative_texts[:num_hard_negatives]
        neg_i = negative_ids[:num_hard_negatives]

        # Pad with random if needed
        while len(neg_t) < num_hard_negatives:
            neg_t.append(negative_texts[0] if negative_texts else positive_text)
            neg_i.append(negative_ids[0] if negative_ids else positive_id)

        pairs.append({
            "query": query,
            "positive_text": positive_text,
            "positive_id": positive_id,
            "negative_texts": neg_t,
            "negative_ids": neg_i,
            "bloom_level": 2,
            "subject": "general",
            "topic": "general",
            "query_type": row.get("query_type", "general"),
        })

    print(f"\n  Processed: {len(pairs)} pairs, {len(corpus)} passages, skipped {skipped}")

    # Save corpus
    print("\nSaving...")
    corpus_path = os.path.join(output_dir, "corpus.jsonl")
    with open(corpus_path, "w") as f:
        for cid, text in corpus.items():
            f.write(json.dumps({
                "id": cid, "text": text, "subject": "general",
                "topic": "general", "bloom_level": 2,
                "source": "msmarco", "difficulty": "medium",
            }) + "\n")
    print(f"  Corpus: {len(corpus)} passages -> {corpus_path}")

    # Split
    random.shuffle(pairs)
    n = len(pairs)
    splits = {
        "train": pairs[:int(0.95 * n)],
        "val": pairs[int(0.95 * n):int(0.975 * n)],
        "test": pairs[int(0.975 * n):],
    }

    for name, sp in splits.items():
        path = os.path.join(output_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for p in sp:
                f.write(json.dumps(p) + "\n")
        print(f"  {name}: {len(sp)} pairs -> {path}")

    meta = {"num_corpus": len(corpus), "num_train": len(splits["train"]),
            "num_val": len(splits["val"]), "num_test": len(splits["test"])}
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Data at {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/tmp/data/msmarco")
    parser.add_argument("--max_train", type=int, default=100000)
    parser.add_argument("--num_hard_negatives", type=int, default=1)
    args = parser.parse_args()

    build_msmarco_dataset(args.output_dir, args.max_train, args.num_hard_negatives)