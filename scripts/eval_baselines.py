"""
Evaluate additional baselines for NeurIPS submission.

Baselines:
  - BM25 (sparse lexical)
  - Contriever (unsupervised dense)
  - E5-base (supervised dense)
  - BGE-base (our backbone, no MRL/routing)
  - MRL (our backbone + MRL training)
  - QA-MRL (ours)

Usage:
    python scripts/eval_baselines.py --config configs/neurips.yaml \
        --datasets scifact nfcorpus fiqa \
        --output_dir results/baselines/
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from typing import Dict, List
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from transformers import AutoModel, AutoTokenizer


# ─────────────────────── Pretrained Model Encoders ───────────────────────

BASELINE_MODELS = {
    "contriever": "facebook/contriever",
    "e5-base": "intfloat/e5-base-v2",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "gte-base": "thenlper/gte-base",
}


@torch.no_grad()
def encode_with_pretrained(model_name_or_path: str, texts: List[str],
                            device: torch.device, batch_size: int = 128,
                            is_query: bool = False) -> np.ndarray:
    """Encode texts using a pretrained HF model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path).to(device).eval()

    all_embs = []
    # E5 needs "query: " or "passage: " prefix
    is_e5 = "e5" in model_name_or_path.lower()

    for i in tqdm(range(0, len(texts), batch_size), desc=f"  encoding", leave=False):
        batch = texts[i:i+batch_size]
        if is_e5:
            prefix = "query: " if is_query else "passage: "
            batch = [prefix + t for t in batch]

        enc = tokenizer(batch, padding=True, truncation=True,
                       max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)

        # CLS pooling
        emb = out.last_hidden_state[:, 0]
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        all_embs.append(emb.cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return np.concatenate(all_embs)


def bm25_search(corpus_texts: List[str], query_texts: List[str],
                 corpus_ids: List[str], k: int = 100) -> Dict[str, Dict[str, float]]:
    """BM25 baseline using rank_bm25."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("  rank_bm25 not installed. pip install rank_bm25")
        return {}

    print("  Building BM25 index...")
    tokenized = [doc.lower().split() for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized)

    results = {}
    for i, query in enumerate(tqdm(query_texts, desc="  BM25 search", leave=False)):
        scores = bm25.get_scores(query.lower().split())
        top_k = np.argsort(scores)[::-1][:k]
        results[str(i)] = {corpus_ids[j]: float(scores[j]) for j in top_k}

    return results


# ─────────────────────── Main ───────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/neurips.yaml")
    parser.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus", "fiqa"])
    parser.add_argument("--baselines", nargs="+",
                        default=["bm25", "contriever", "e5-base", "bge-base"])
    parser.add_argument("--output_dir", default="results/baselines/")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Import BEIR evaluation functions
    from scripts.eval_beir import (
        load_beir_dataset, retrieve_faiss, compute_beir_metrics
    )

    all_results = {}

    for ds_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        corpus, queries, qrels = load_beir_dataset(ds_name)
        if corpus is None:
            continue

        corpus_ids = list(corpus.keys())
        corpus_texts = [
            (corpus[cid].get("title", "") + " " + corpus[cid].get("text", "")).strip()
            for cid in corpus_ids
        ]
        query_ids = [qid for qid in queries if qid in qrels]
        query_texts = [queries[qid] for qid in query_ids]

        print(f"  Corpus: {len(corpus_ids)}, Queries: {len(query_ids)}")

        for baseline_name in args.baselines:
            print(f"\n  --- {baseline_name} ---")

            if baseline_name == "bm25":
                # BM25 returns results directly
                bm25_results = bm25_search(corpus_texts, query_texts, corpus_ids)
                # Remap to query_ids
                results = {query_ids[i]: bm25_results.get(str(i), {})
                          for i in range(len(query_ids))}

            elif baseline_name in BASELINE_MODELS:
                model_path = BASELINE_MODELS[baseline_name]
                print(f"    Model: {model_path}")

                corpus_embs = encode_with_pretrained(
                    model_path, corpus_texts, device, is_query=False
                )
                query_embs = encode_with_pretrained(
                    model_path, query_texts, device, is_query=True
                )

                scores, indices = retrieve_faiss(query_embs, corpus_embs, k=100)

                results = {}
                for i, qid in enumerate(query_ids):
                    results[qid] = {}
                    for j in range(indices.shape[1]):
                        cidx = int(indices[i, j])
                        if cidx < len(corpus_ids):
                            results[qid][corpus_ids[cidx]] = float(scores[i, j])
            else:
                print(f"    Unknown baseline: {baseline_name}")
                continue

            metrics = compute_beir_metrics(qrels, results)

            key = f"{baseline_name}"
            if key not in all_results:
                all_results[key] = {}
            all_results[key][ds_name] = metrics

            print(f"    NDCG@10: {metrics.get('ndcg@10', 0):.4f}")
            print(f"    R@10:    {metrics.get('recall@10', 0):.4f}")

    # Save
    with open(os.path.join(args.output_dir, "baseline_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    # Print table
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON (NDCG@10)")
    print("=" * 80)

    models = list(all_results.keys())
    header = f"{'Dataset':15s}" + "".join(f"{m:>15s}" for m in models)
    print(header)
    print("-" * len(header))
    for ds in args.datasets:
        row = f"{ds:15s}"
        for m in models:
            v = all_results.get(m, {}).get(ds, {}).get("ndcg@10", 0)
            row += f"{v:>15.4f}"
        print(row)

    print(f"\nSaved to {args.output_dir}/baseline_results.json")


if __name__ == "__main__":
    main()