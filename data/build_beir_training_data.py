"""
Build training data from BEIR dataset train splits.

Downloads BEIR train splits from HuggingFace, annotates queries with Bloom
levels, mines hard negatives from the corpus, and writes JSONL files in the
same format as our educational data — ready to be combined with it.

Supported datasets: scifact, nfcorpus, fiqa
(These are the same datasets used for BEIR evaluation, test split held out.)

Usage:
    python data/build_beir_training_data.py \
        --datasets scifact nfcorpus \
        --output_dir /tmp/data/beir_train \
        --num_neg 7

    # Then combine with educational data:
    cat /tmp/data/real/train_curriculum.jsonl \
        /tmp/data/beir_train/combined_train.jsonl \
        > /tmp/data/mixed/train_curriculum.jsonl
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_beir_train_split(dataset_name: str):
    """Load corpus, train queries, and train qrels from HuggingFace BEIR."""
    from datasets import load_dataset

    print(f"  Loading {dataset_name} corpus...")
    ds_corpus = load_dataset(f"BeIR/{dataset_name}", "corpus", split="corpus")
    corpus = {}
    for row in tqdm(ds_corpus, desc="  corpus", leave=False):
        corpus[str(row["_id"])] = {
            "id": str(row["_id"]),
            "text": (row.get("title", "") + " " + row.get("text", "")).strip(),
        }

    print(f"  Loading {dataset_name} queries (train split)...")
    try:
        ds_q = load_dataset(f"BeIR/{dataset_name}", "queries", split="queries")
        queries = {str(row["_id"]): row.get("text", "") for row in ds_q}
    except Exception as e:
        print(f"  WARNING: Could not load queries: {e}")
        return None, None, None

    print(f"  Loading {dataset_name} qrels (train split)...")
    try:
        ds_qrels = load_dataset(f"BeIR/{dataset_name}-qrels", split="train")
        qrels = defaultdict(dict)
        for row in ds_qrels:
            qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])
        qrels = dict(qrels)
    except Exception as e:
        print(f"  WARNING: No train qrels for {dataset_name}: {e}")
        return None, None, None

    # Only keep queries that have at least one relevant passage
    valid_qids = {qid for qid, rels in qrels.items() if any(v > 0 for v in rels.values())}
    queries = {qid: text for qid, text in queries.items() if qid in valid_qids}

    print(f"  {dataset_name}: {len(corpus)} passages, {len(queries)} train queries with relevance")
    return corpus, queries, qrels


def mine_negatives(query_text: str, positive_ids: set, corpus: dict,
                   num_neg: int = 7, seed: int = 42) -> list:
    """
    Simple BM25-style hard negative mining via random sampling from non-relevant passages.
    For a proper paper run, replace with BM25 retrieval negatives.
    """
    rng = random.Random(hash(query_text) + seed)
    candidate_ids = [pid for pid in corpus if pid not in positive_ids]
    if len(candidate_ids) < num_neg:
        return candidate_ids
    return rng.sample(candidate_ids, num_neg)


def annotate_bloom(query_texts: list, device: str) -> list:
    """Annotate queries with Bloom levels (0-indexed)."""
    from data.annotate_bloom_pretrained import load_pretrained_classifier, predict_bloom
    model, tok, id2label = load_pretrained_classifier(device=device)
    labels_1idx = predict_bloom(query_texts, model, tok, device=device, id2label=id2label)
    return [l - 1 for l in labels_1idx]  # convert to 0-indexed


def build_dataset(dataset_name: str, output_dir: str, num_neg: int, device: str) -> str:
    """Build train JSONL for a single BEIR dataset. Returns output path."""
    corpus, queries, qrels = load_beir_train_split(dataset_name)
    if corpus is None:
        print(f"  Skipping {dataset_name} — could not load train split.")
        return None

    print(f"  Annotating {len(queries)} queries with Bloom levels...")
    query_list = list(queries.items())  # [(qid, text), ...]
    texts = [t for _, t in query_list]
    bloom_labels = annotate_bloom(texts, device)

    # Print Bloom distribution
    from collections import Counter
    names = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
    dist = Counter(bloom_labels)
    print(f"  Bloom distribution:")
    for i, name in enumerate(names):
        print(f"    {name}: {dist[i]} ({dist[i]/len(bloom_labels)*100:.1f}%)")

    print(f"  Building training pairs...")
    records = []
    for (qid, query_text), bloom_label in zip(query_list, bloom_labels):
        if qid not in qrels:
            continue
        positive_ids = {pid for pid, score in qrels[qid].items() if score > 0}
        if not positive_ids:
            continue
        positive_id = list(positive_ids)[0]
        if positive_id not in corpus:
            continue

        negative_ids = mine_negatives(query_text, positive_ids, corpus, num_neg)
        negative_texts = [corpus[nid]["text"] for nid in negative_ids if nid in corpus]
        negative_ids_list = [nid for nid in negative_ids if nid in corpus]

        records.append({
            "query": query_text,
            "positive_text": corpus[positive_id]["text"],
            "positive_id": positive_id,
            "negative_texts": negative_texts,
            "negative_ids": negative_ids_list,
            "bloom_level": bloom_label + 1,   # store 1-indexed to match educational data
            "subject": dataset_name,
            "source": f"beir_{dataset_name}",
        })

    out_path = os.path.join(output_dir, f"{dataset_name}_train.jsonl")
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  Wrote {len(records)} records → {out_path}")

    # Also write corpus file for this dataset
    corpus_path = os.path.join(output_dir, f"{dataset_name}_corpus.jsonl")
    with open(corpus_path, "w") as f:
        for pid, pdata in corpus.items():
            f.write(json.dumps({"id": pid, "text": pdata["text"],
                                "source": f"beir_{dataset_name}"}) + "\n")
    print(f"  Wrote {len(corpus)} corpus passages → {corpus_path}")

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"],
                        help="BEIR datasets to build training data from")
    parser.add_argument("--output_dir", default="/tmp/data/beir_train")
    parser.add_argument("--num_neg", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    all_train_paths = []
    for ds_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"  Processing {ds_name}")
        print(f"{'='*60}")
        path = build_dataset(ds_name, args.output_dir, args.num_neg, args.device)
        if path:
            all_train_paths.append(path)

    if not all_train_paths:
        print("ERROR: No datasets built successfully.")
        return

    # Combine all BEIR train splits into one file
    combined_path = os.path.join(args.output_dir, "combined_train.jsonl")
    total = 0
    with open(combined_path, "w") as fout:
        for path in all_train_paths:
            with open(path) as fin:
                for line in fin:
                    fout.write(line)
                    total += 1
    print(f"\nCombined {total} records → {combined_path}")
    print(f"\nNext step: mix with educational data:")
    print(f"  cat /tmp/data/real/train_curriculum.jsonl \\")
    print(f"      {combined_path} \\")
    print(f"      > /tmp/data/mixed/train_curriculum.jsonl")


if __name__ == "__main__":
    main()
