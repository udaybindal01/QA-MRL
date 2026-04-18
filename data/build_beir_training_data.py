"""
Build per-dataset train/val/test splits from BEIR datasets.

For each dataset, outputs:
  {output_dir}/{dataset}/train.jsonl      — 90% of train qrels
  {output_dir}/{dataset}/val.jsonl        — 10% of train qrels (for epoch selection)
  {output_dir}/{dataset}/test.jsonl       — test qrels (never seen during training)
  {output_dir}/{dataset}/corpus.jsonl     — full corpus

All files use the same JSONL format as educational data so existing
training/eval scripts work without modification.

Supported datasets: scifact, nfcorpus, fiqa (any BEIR dataset with train qrels)

Usage:
    python data/build_beir_training_data.py \
        --datasets scifact nfcorpus fiqa \
        --output_dir /tmp/data/beir \
        --num_neg 7
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_beir_corpus(dataset_name: str) -> dict:
    """Download and return corpus dict {id: {id, text}}."""
    from datasets import load_dataset
    print(f"  Loading {dataset_name} corpus...")
    ds = load_dataset(f"BeIR/{dataset_name}", "corpus", split="corpus")
    corpus = {}
    for row in tqdm(ds, desc="  corpus", leave=False):
        text = (row.get("title", "") + " " + row.get("text", "")).strip()
        corpus[str(row["_id"])] = {"id": str(row["_id"]), "text": text}
    return corpus


def load_beir_qrels(dataset_name: str, split: str) -> tuple:
    """Load queries and qrels for a given split. Returns (queries, qrels)."""
    from datasets import load_dataset
    try:
        ds_q = load_dataset(f"BeIR/{dataset_name}", "queries", split="queries")
        queries = {str(row["_id"]): row.get("text", "") for row in ds_q}
    except Exception as e:
        print(f"  WARNING: Could not load queries: {e}")
        return None, None

    try:
        ds_qrels = load_dataset(f"BeIR/{dataset_name}-qrels", split=split)
        qrels = defaultdict(dict)
        for row in ds_qrels:
            qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])
        qrels = dict(qrels)
    except Exception as e:
        print(f"  WARNING: No {split} qrels for {dataset_name}: {e}")
        return None, None

    # Keep only queries with at least one relevant passage
    valid_qids = {qid for qid, rels in qrels.items() if any(v > 0 for v in rels.values())}
    queries = {qid: text for qid, text in queries.items() if qid in valid_qids}
    return queries, qrels


def mine_negatives(query_text: str, positive_ids: set, corpus: dict, num_neg: int = 7) -> list:
    """Sample hard negatives randomly from non-relevant passages."""
    rng = random.Random(hash(query_text))
    candidates = [pid for pid in corpus if pid not in positive_ids]
    return rng.sample(candidates, min(num_neg, len(candidates)))


def annotate_bloom(query_texts: list, device: str) -> list:
    """Annotate queries with Bloom levels (returns 0-indexed)."""
    from data.annotate_bloom_pretrained import load_pretrained_classifier, predict_bloom
    model, tok, id2label = load_pretrained_classifier(device=device)
    labels_1idx = predict_bloom(query_texts, model, tok, device=device, id2label=id2label)
    return [l - 1 for l in labels_1idx]


def build_pairs(queries: dict, qrels: dict, corpus: dict,
                bloom_map: dict, num_neg: int, dataset_name: str) -> list:
    """Build training/eval pair records from queries + qrels."""
    records = []
    for qid, query_text in queries.items():
        if qid not in qrels:
            continue
        positive_ids = {pid for pid, score in qrels[qid].items() if score > 0}
        if not positive_ids:
            continue
        positive_id = list(positive_ids)[0]
        if positive_id not in corpus:
            continue
        negative_ids = mine_negatives(query_text, positive_ids, corpus, num_neg)
        records.append({
            "query":          query_text,
            "positive_text":  corpus[positive_id]["text"],
            "positive_id":    positive_id,
            "negative_texts": [corpus[nid]["text"] for nid in negative_ids if nid in corpus],
            "negative_ids":   [nid for nid in negative_ids if nid in corpus],
            "bloom_level":    bloom_map[qid] + 1,   # store 1-indexed
            "subject":        dataset_name,
            "source":         f"beir_{dataset_name}",
        })
    return records


def write_jsonl(records: list, path: str):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  Wrote {len(records)} records → {path}")


def build_dataset(dataset_name: str, output_dir: str, num_neg: int,
                  device: str, val_ratio: float = 0.1):
    """
    Build train/val/test splits for one BEIR dataset.
    val is a held-out 10% of the train qrels.
    test uses the official BEIR test qrels.
    """
    ds_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(ds_dir, exist_ok=True)

    # Skip if already built
    if all(os.path.exists(os.path.join(ds_dir, f)) for f in
           ["train.jsonl", "val.jsonl", "test.jsonl", "corpus.jsonl"]):
        print(f"  {dataset_name}: already built at {ds_dir} — skipping.")
        return ds_dir

    # ── Corpus ───────────────────────────────────────────────────────────────
    corpus = load_beir_corpus(dataset_name)
    corpus_path = os.path.join(ds_dir, "corpus.jsonl")
    with open(corpus_path, "w") as f:
        for pdata in corpus.values():
            f.write(json.dumps(pdata) + "\n")
    print(f"  Corpus: {len(corpus)} passages → {corpus_path}")

    # ── Train + Val ──────────────────────────────────────────────────────────
    train_queries, train_qrels = load_beir_qrels(dataset_name, "train")
    if train_queries is None:
        print(f"  ERROR: no train qrels for {dataset_name}, skipping.")
        return None
    print(f"  Train queries: {len(train_queries)}")

    # Annotate all train queries with Bloom levels
    print(f"  Annotating {len(train_queries)} train queries with Bloom levels...")
    qids = list(train_queries.keys())
    texts = [train_queries[qid] for qid in qids]
    bloom_labels = annotate_bloom(texts, device)
    bloom_map = {qid: label for qid, label in zip(qids, bloom_labels)}

    names = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
    dist = Counter(bloom_labels)
    print(f"  Bloom distribution (train):")
    for i, name in enumerate(names):
        print(f"    {name}: {dist[i]} ({dist[i]/len(bloom_labels)*100:.1f}%)")

    # 90/10 train/val split (deterministic)
    rng = random.Random(42)
    shuffled_qids = qids[:]
    rng.shuffle(shuffled_qids)
    n_val = max(1, int(len(shuffled_qids) * val_ratio))
    val_qids = set(shuffled_qids[:n_val])
    train_qids = set(shuffled_qids[n_val:])

    train_q = {qid: train_queries[qid] for qid in train_qids}
    val_q   = {qid: train_queries[qid] for qid in val_qids}

    train_records = build_pairs(train_q, train_qrels, corpus, bloom_map, num_neg, dataset_name)
    val_records   = build_pairs(val_q,   train_qrels, corpus, bloom_map, num_neg, dataset_name)

    write_jsonl(train_records, os.path.join(ds_dir, "train.jsonl"))
    write_jsonl(val_records,   os.path.join(ds_dir, "val.jsonl"))

    # ── Test ─────────────────────────────────────────────────────────────────
    test_queries, test_qrels = load_beir_qrels(dataset_name, "test")
    if test_queries is None:
        print(f"  WARNING: no test qrels for {dataset_name}.")
        write_jsonl(val_records, os.path.join(ds_dir, "test.jsonl"))  # fallback
    else:
        print(f"  Annotating {len(test_queries)} test queries with Bloom levels...")
        test_qids = list(test_queries.keys())
        test_texts = [test_queries[qid] for qid in test_qids]
        test_bloom = annotate_bloom(test_texts, device)
        test_bloom_map = {qid: label for qid, label in zip(test_qids, test_bloom)}
        test_records = build_pairs(test_queries, test_qrels, corpus,
                                   test_bloom_map, num_neg, dataset_name)
        write_jsonl(test_records, os.path.join(ds_dir, "test.jsonl"))

    print(f"  Done: {ds_dir}/")
    return ds_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"],
                        help="BEIR datasets to build")
    parser.add_argument("--output_dir", default="/tmp/data/beir",
                        help="Root output dir; each dataset gets a subdirectory")
    parser.add_argument("--num_neg", type=int, default=7)
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of train queries held out for val")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    for ds_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"  Building: {ds_name}")
        print(f"{'='*60}")
        build_dataset(ds_name, args.output_dir, args.num_neg, args.device, args.val_ratio)

    print(f"\nAll datasets built under {args.output_dir}/")
    for ds_name in args.datasets:
        print(f"  {ds_name:12s}: {args.output_dir}/{ds_name}/{{train,val,test,corpus}}.jsonl")


if __name__ == "__main__":
    main()
