"""
Cluster-routing retrieval baseline: replaces BAM-PQ's supervised Bloom-level
routing with unsupervised k-means clusters over the training query
embeddings. Each cluster learns its own top-D dimension mask; test queries
are assigned to their nearest cluster centroid and use that cluster's mask.

Directly parallels BAM-PQ's mask head:
    BAM-PQ:      query -> predict Bloom b -> apply mask E[b]  -> masked query
    This script: query -> predict cluster c -> apply mask M[c] -> masked query

Corpus is encoded once at full dimensionality (identical to BAM-PQ's
deployment), so retrieval scores are (masked_q) . (full_d) — same
asymmetric inner-product BAM-PQ uses.

Comparison target: BAM-PQ. If cluster-routing achieves lower R@10 at
matched active dimensions, unsupervised clusters cannot substitute for
supervised Bloom labels — the paper's core claim.

Outputs (schema matches scripts/eval_csr_baseline.py):
    {output_dir}/cluster_routing_results.json  -- aggregate + per-Bloom
    {output_dir}/cluster_routing_per_query.csv -- per-query metrics
    {output_dir}/cluster_routing_sweep.csv     -- (active_dims, R@10, ...) curve

Usage:
    python scripts/eval_cluster_routing.py \\
        --config     configs/mrl_bge_base.yaml \\
        --checkpoint /scratch/ishaan.karan/bampq-checkpoints/educational/mrl_bge_base/best/ \\
        --train_path data/real/train_curriculum.jsonl \\
        --output_dir /scratch/ishaan.karan/cluster_routing/bge-base/ \\
        --k 6 \\
        --active_dims_sweep 128 256 384 512
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from evaluation.retrieval_metrics import per_query_recall, per_query_ndcg
from scripts.eval_bam import load_mrl


BLOOM_NAMES = {1: "L1 Remember", 2: "L2 Understand", 3: "L3 Apply",
               4: "L4 Analyze",  5: "L5 Evaluate",   6: "L6 Create"}


# ---------------------------------------------------------------------------
# Encoder (identical to eval_csr_baseline.py — keep scripts independent)
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_texts(model, tokenizer, texts, device,
                 is_query: bool, batch_size: int = 128):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=256 if not is_query else 128,
                        return_tensors="pt").to(device)
        if hasattr(model, "encode_queries") and is_query:
            out = model.encode_queries(enc["input_ids"], enc["attention_mask"])
            emb = out.get("masked_embedding", out.get("full")) \
                if isinstance(out, dict) else out
        elif hasattr(model, "encode_passages") and not is_query:
            emb = model.encode_passages(enc["input_ids"], enc["attention_mask"])
        else:
            out = model(enc["input_ids"], enc["attention_mask"])
            emb = out["full"] if isinstance(out, dict) else out
        all_embs.append(emb.float().cpu().numpy())
    return normalize(np.concatenate(all_embs, axis=0), norm="l2", axis=1)


# ---------------------------------------------------------------------------
# Cluster-based mask learning
# ---------------------------------------------------------------------------
def fit_kmeans(train_embs: np.ndarray, k: int, seed: int) -> KMeans:
    print(f"  Fitting k-means (k={k}, n_train={len(train_embs)})...")
    t0 = time.time()
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    km.fit(train_embs)
    print(f"  k-means fit in {time.time() - t0:.1f} s")
    # Report cluster sizes
    _, counts = np.unique(km.labels_, return_counts=True)
    for c, n in enumerate(counts):
        print(f"    cluster {c}: N={n:>5} ({n / len(train_embs):.1%})")
    return km


def learn_cluster_masks(train_embs: np.ndarray, cluster_ids: np.ndarray,
                        k: int, active_dims: int, D: int) -> np.ndarray:
    """
    For each cluster, learn a binary mask over the D encoder dimensions
    that keeps the `active_dims` dimensions with highest mean absolute
    activation on cluster members.

    Returns: (k, D) binary array.
    """
    masks = np.zeros((k, D), dtype=np.float32)
    for c in range(k):
        members = train_embs[cluster_ids == c]
        if len(members) == 0:
            masks[c] = 1.0  # degenerate: no members -> keep all dims
            continue
        importance = np.abs(members).mean(axis=0)   # (D,)
        top_dims = np.argsort(-importance)[:active_dims]
        masks[c, top_dims] = 1.0
    return masks


# ---------------------------------------------------------------------------
# Retrieval + metrics
# ---------------------------------------------------------------------------
def retrieve_and_score(query_embs: np.ndarray, corpus_embs: np.ndarray,
                       relevant_idx: np.ndarray, ks=(1, 10, 50)):
    """Full-dim query x full-dim corpus dot product; top-k rankings."""
    q = normalize(query_embs, norm="l2", axis=1)
    d = normalize(corpus_embs, norm="l2", axis=1)
    max_k = max(ks)
    N = len(q)
    rankings = np.zeros((N, max_k), dtype=np.int64)
    chunk = 256
    for i in range(0, N, chunk):
        sim = q[i:i + chunk] @ d.T
        topk = np.argpartition(-sim, kth=max_k - 1, axis=1)[:, :max_k]
        row_scores = np.take_along_axis(sim, topk, axis=1)
        order = np.argsort(-row_scores, axis=1)
        rankings[i:i + chunk] = np.take_along_axis(topk, order, axis=1)

    per_q = {f"recall@{k}": per_query_recall(rankings, relevant_idx, k)
             for k in ks}
    per_q["ndcg@10"] = per_query_ndcg(rankings, relevant_idx, 10)
    return rankings, per_q


def aggregate(per_q: dict, blooms_1idx: np.ndarray) -> dict:
    agg = {k: float(v.mean()) for k, v in per_q.items()}
    by_bloom = {}
    for b in range(1, 7):
        mask = blooms_1idx == b
        if mask.sum() == 0:
            continue
        by_bloom[BLOOM_NAMES[b]] = {k: float(v[mask].mean())
                                    for k, v in per_q.items()}
    return {"overall": agg, "by_bloom": by_bloom,
            "n_queries": int(len(blooms_1idx))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",       required=True)
    ap.add_argument("--checkpoint",   required=True,
                    help="MRL baseline checkpoint dir")
    ap.add_argument("--train_path",   default="data/real/train_curriculum.jsonl",
                    help="Training queries used to fit clusters + masks")
    ap.add_argument("--test_path",    default="data/real/test.jsonl")
    ap.add_argument("--corpus_path",  default="data/real/corpus.jsonl")
    ap.add_argument("--output_dir",   required=True)
    ap.add_argument("--k", type=int, default=6,
                    help="Number of clusters (default: 6 to match Bloom levels)")
    ap.add_argument("--active_dims_sweep", type=int, nargs="+",
                    default=[128, 256, 384, 512],
                    help="Active dimensions per cluster mask to sweep")
    ap.add_argument("--max_train_queries", type=int, default=20000,
                    help="Cap on training queries used for clustering")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load MRL encoder ===
    print(f"Loading config: {args.config}")
    cfg = load_config(args.config)
    set_seed(cfg["training"].get("seed", args.seed))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])
    model = load_mrl(cfg, args.checkpoint, device)
    model.eval()

    # === Load data ===
    with open(args.corpus_path) as f:
        corpus = [json.loads(l) for l in f]
    with open(args.test_path) as f:
        test = [json.loads(l) for l in f]
    with open(args.train_path) as f:
        train = [json.loads(l) for l in f]
    if len(train) > args.max_train_queries:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(train), args.max_train_queries, replace=False)
        train = [train[i] for i in idx]

    print(f"Corpus: {len(corpus)}  Test: {len(test)}  "
          f"Train queries (for clustering): {len(train)}")

    corpus_id_to_idx = {p["id"]: i for i, p in enumerate(corpus)}
    relevant_idx = np.array([corpus_id_to_idx[s["positive_id"]] for s in test])
    blooms_1idx  = np.array([s["bloom_level"] for s in test])

    # === Encode everything with MRL ===
    print("\nEncoding corpus with MRL...")
    corpus_embs = encode_texts(model, tokenizer, [p["text"] for p in corpus],
                               device, is_query=False)
    print(f"  Corpus shape: {corpus_embs.shape}")
    D = corpus_embs.shape[1]

    print("Encoding train queries (for clustering)...")
    train_query_embs = encode_texts(model, tokenizer,
                                    [s["query"] for s in train],
                                    device, is_query=True)

    print("Encoding test queries...")
    test_query_embs = encode_texts(model, tokenizer,
                                   [s["query"] for s in test],
                                   device, is_query=True)
    print(f"  Test query shape: {test_query_embs.shape}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Reference: MRL full-dim retrieval
    print("\n=== Reference: MRL full-dim retrieval ===")
    _, per_q_mrl = retrieve_and_score(test_query_embs, corpus_embs, relevant_idx)
    agg_mrl = aggregate(per_q_mrl, blooms_1idx)
    print(f"  R@1={agg_mrl['overall']['recall@1']:.4f}  "
          f"R@10={agg_mrl['overall']['recall@10']:.4f}  "
          f"R@50={agg_mrl['overall']['recall@50']:.4f}  "
          f"NDCG@10={agg_mrl['overall']['ndcg@10']:.4f}")

    # === Fit k-means on training queries ===
    print(f"\nFitting k-means on {len(train_query_embs)} training queries...")
    km = fit_kmeans(train_query_embs, k=args.k, seed=args.seed)

    # === Assign test queries to nearest cluster ===
    print("Assigning test queries to clusters...")
    test_cluster_ids = km.predict(test_query_embs)
    _, test_counts = np.unique(test_cluster_ids, return_counts=True)
    for c, n in enumerate(test_counts):
        print(f"  test cluster {c}: N={n:>5} ({n / len(test):.1%})")

    # === Sweep active_dims (matched-budget rows for the paper table) ===
    sweep_rows = []
    best_row_per_query = None
    for active in args.active_dims_sweep:
        if active > D:
            print(f"Skipping active_dims={active} > D={D}")
            continue
        print(f"\n=== Cluster routing: active_dims={active} / {D} ===")
        masks = learn_cluster_masks(train_query_embs, km.labels_,
                                    args.k, active, D)   # (k, D)
        # Broadcast: each test query gets its cluster's mask
        query_masks   = masks[test_cluster_ids]           # (N_test, D)
        masked_queries = test_query_embs * query_masks

        _, per_q = retrieve_and_score(masked_queries, corpus_embs, relevant_idx)
        agg = aggregate(per_q, blooms_1idx)
        print(f"  R@1={agg['overall']['recall@1']:.4f}  "
              f"R@10={agg['overall']['recall@10']:.4f}  "
              f"R@50={agg['overall']['recall@50']:.4f}  "
              f"NDCG@10={agg['overall']['ndcg@10']:.4f}")

        sweep_rows.append({
            "k":               args.k,
            "active_dims":     active,
            "recall@1":        agg["overall"]["recall@1"],
            "recall@10":       agg["overall"]["recall@10"],
            "recall@50":       agg["overall"]["recall@50"],
            "ndcg@10":         agg["overall"]["ndcg@10"],
        })
        # Save per-query CSV for the setting closest to BAM-PQ's typical
        # active-dims (~320 on bge-base, ~418 on e5-large, etc.)
        if (best_row_per_query is None
                or abs(active - 320) < abs(best_row_per_query - 320)):
            best_row_per_query = active
            pd.DataFrame({
                "bloom_1idx":  blooms_1idx,
                "cluster_id":  test_cluster_ids,
                "recall@1":    per_q["recall@1"],
                "recall@10":   per_q["recall@10"],
                "recall@50":   per_q["recall@50"],
                "ndcg@10":     per_q["ndcg@10"],
            }).to_csv(
                os.path.join(args.output_dir, "cluster_routing_per_query.csv"),
                index=False,
            )

    # === Save ===
    pd.DataFrame(sweep_rows).to_csv(
        os.path.join(args.output_dir, "cluster_routing_sweep.csv"), index=False
    )
    with open(os.path.join(args.output_dir, "cluster_routing_results.json"),
              "w") as f:
        json.dump({
            "config":               args.config,
            "checkpoint":           args.checkpoint,
            "k":                    args.k,
            "active_dims_sweep":    args.active_dims_sweep,
            "reference_mrl_full":   agg_mrl,
            "cluster_sweep":        sweep_rows,
            "test_cluster_counts":  test_counts.tolist(),
        }, f, indent=2)

    # === Pretty print ===
    print("\n" + "=" * 78)
    print(f"CLUSTER-ROUTING SWEEP — {args.config}  (k={args.k})")
    print("=" * 78)
    print(f"  {'MRL@full':<26}  R@1={agg_mrl['overall']['recall@1']:.4f}  "
          f"R@10={agg_mrl['overall']['recall@10']:.4f}  "
          f"R@50={agg_mrl['overall']['recall@50']:.4f}  "
          f"NDCG@10={agg_mrl['overall']['ndcg@10']:.4f}")
    for r in sweep_rows:
        label = f"Cluster@d={r['active_dims']}"
        print(f"  {label:<26}  R@1={r['recall@1']:.4f}  "
              f"R@10={r['recall@10']:.4f}  R@50={r['recall@50']:.4f}  "
              f"NDCG@10={r['ndcg@10']:.4f}")
    print("=" * 78)
    print(f"Saved: {args.output_dir}/cluster_routing_results.json")
    print(f"Saved: {args.output_dir}/cluster_routing_sweep.csv")
    print(f"Saved: {args.output_dir}/cluster_routing_per_query.csv "
          f"(active_dims={best_row_per_query})")


if __name__ == "__main__":
    main()
