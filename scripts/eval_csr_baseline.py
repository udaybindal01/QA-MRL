"""
CSR-style sparse-coding retrieval baseline.

Reproduces Wen et al. 2025 (Contrastive Sparse Representation) in a
post-hoc form: an overcomplete sparse dictionary is fit on MRL corpus
embeddings, then queries and passages are sparse-coded with a hard
sparsity budget (Orthogonal Matching Pursuit, `--n_nonzero` active atoms
per vector). Retrieval scores are sparse dot-products.

Comparison target: BAM-PQ. Both methods reduce effective query-side
dimensionality; CSR does so via sparse atom selection, BAM-PQ via
Bloom-conditioned mask routing. At matched or comparable sparsity,
CSR should underperform BAM-PQ on R@10 and NDCG@10 (per Wen et al.'s
own framing — CSR beats MRL, but not supervised query-adaptive routing).

Outputs (identical schema to scripts/eval_bam.py / paper Table 3):
    {output_dir}/csr_results.json               -- aggregate + per-Bloom
    {output_dir}/csr_per_query.csv              -- per-query metrics
    {output_dir}/csr_sweep.csv                  -- (n_nonzero, R@10, ...) curve

Usage:
    python scripts/eval_csr_baseline.py \\
        --config     configs/mrl_bge_base.yaml \\
        --checkpoint /scratch/ishaan.karan/bampq-checkpoints/educational/mrl_bge_base/best/ \\
        --output_dir /scratch/ishaan.karan/csr_baseline/bge-base/ \\
        --n_atoms    512 \\
        --n_nonzero_sweep 20 40 80 160 320
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from evaluation.retrieval_metrics import per_query_recall, per_query_ndcg
from scripts.eval_bam import load_mrl


BLOOM_NAMES = {1: "L1 Remember", 2: "L2 Understand", 3: "L3 Apply",
               4: "L4 Analyze",  5: "L5 Evaluate",   6: "L6 Create"}


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_texts(model, tokenizer, texts, device,
                 is_query: bool, batch_size: int = 128):
    """Encode texts and return L2-normalised numpy array (N, D)."""
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
# CSR-style sparse coding
# ---------------------------------------------------------------------------
def fit_dictionary(sample_embeddings: np.ndarray, n_atoms: int,
                   max_iter: int, batch_size: int, seed: int
                   ) -> MiniBatchDictionaryLearning:
    """Fit an overcomplete dictionary on a sample of corpus embeddings."""
    print(f"  Fitting dictionary: n_atoms={n_atoms}, "
          f"train_size={len(sample_embeddings)}, max_iter={max_iter}")
    t0 = time.time()
    dl = MiniBatchDictionaryLearning(
        n_components=n_atoms,
        max_iter=max_iter,
        batch_size=batch_size,
        transform_algorithm="omp",
        transform_n_nonzero_coefs=1,       # placeholder; reset per sweep
        n_jobs=-1,
        random_state=seed,
    )
    dl.fit(sample_embeddings)
    print(f"  Dictionary fit in {time.time() - t0:.1f} s")
    return dl


def sparse_transform(dl: MiniBatchDictionaryLearning, embeddings: np.ndarray,
                     n_nonzero: int, chunk_size: int = 2048) -> np.ndarray:
    """OMP-encode embeddings with a hard sparsity budget of n_nonzero atoms."""
    dl.transform_algorithm = "omp"
    dl.transform_n_nonzero_coefs = n_nonzero
    codes = np.empty((len(embeddings), dl.n_components), dtype=np.float32)
    for i in range(0, len(embeddings), chunk_size):
        codes[i:i + chunk_size] = dl.transform(embeddings[i:i + chunk_size])
    return codes


# ---------------------------------------------------------------------------
# Retrieval + metrics
# ---------------------------------------------------------------------------
def retrieve_and_score(query_codes: np.ndarray, corpus_codes: np.ndarray,
                       relevant_idx: np.ndarray, ks=(1, 10, 50)):
    """
    Compute sparse dot-product similarity and return rankings + per-query metrics.
    """
    # Scale L2 for cosine-like scoring on the sparse space
    q = normalize(query_codes, norm="l2", axis=1)
    d = normalize(corpus_codes, norm="l2", axis=1)
    max_k = max(ks) if isinstance(ks, (list, tuple)) else max(ks + (10,))

    # Chunked matmul → top-k
    N = len(q)
    rankings = np.zeros((N, max_k), dtype=np.int64)
    chunk = 256
    for i in range(0, N, chunk):
        sim = q[i:i + chunk] @ d.T  # (chunk, N_corpus)
        topk = np.argpartition(-sim, kth=max_k - 1, axis=1)[:, :max_k]
        # Sort within top-k by score
        row_scores = np.take_along_axis(sim, topk, axis=1)
        order = np.argsort(-row_scores, axis=1)
        rankings[i:i + chunk] = np.take_along_axis(topk, order, axis=1)

    per_q = {
        f"recall@{k}": per_query_recall(rankings, relevant_idx, k)
        for k in ks
    }
    per_q["ndcg@10"] = per_query_ndcg(rankings, relevant_idx, 10)
    return rankings, per_q


def aggregate(per_q: dict, blooms_1idx: np.ndarray) -> dict:
    """Aggregate per-query arrays → overall + per-Bloom means."""
    agg = {k: float(v.mean()) for k, v in per_q.items()}
    by_bloom = {}
    for b in range(1, 7):
        mask = blooms_1idx == b
        if mask.sum() == 0:
            continue
        by_bloom[BLOOM_NAMES[b]] = {
            k: float(v[mask].mean()) for k, v in per_q.items()
        }
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
    ap.add_argument("--output_dir",   required=True)
    ap.add_argument("--test_path",    default="data/real/test.jsonl")
    ap.add_argument("--corpus_path",  default="data/real/corpus.jsonl")
    # CSR hyperparameters
    ap.add_argument("--n_atoms",      type=int, default=512,
                    help="Overcomplete dictionary size (default: 512)")
    ap.add_argument("--n_nonzero_sweep", type=int, nargs="+",
                    default=[20, 40, 80, 160, 320],
                    help="OMP sparsity budgets to sweep (each = 1 row)")
    ap.add_argument("--dict_train_size", type=int, default=10000,
                    help="Corpus sample size for dictionary fitting")
    ap.add_argument("--csr_max_iter",   type=int, default=50)
    ap.add_argument("--csr_batch_size", type=int, default=256)
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
    print(f"Corpus: {len(corpus)}  Test queries: {len(test)}")

    corpus_id_to_idx = {p["id"]: i for i, p in enumerate(corpus)}
    relevant_idx = np.array([corpus_id_to_idx[s["positive_id"]] for s in test])
    blooms_1idx  = np.array([s["bloom_level"] for s in test])

    # === Encode corpus + queries ===
    print("Encoding corpus with MRL...")
    corpus_embs = encode_texts(model, tokenizer, [p["text"] for p in corpus],
                               device, is_query=False)
    print(f"  Corpus shape: {corpus_embs.shape}")

    print("Encoding queries with MRL...")
    query_embs = encode_texts(model, tokenizer, [s["query"] for s in test],
                              device, is_query=True)
    print(f"  Query shape:  {query_embs.shape}")

    # Baseline (before CSR) — MRL full-dim retrieval for reference
    print("\n=== Reference: MRL full-dim retrieval ===")
    _, per_q_mrl = retrieve_and_score(query_embs, corpus_embs, relevant_idx)
    agg_mrl = aggregate(per_q_mrl, blooms_1idx)
    print(f"  R@1={agg_mrl['overall']['recall@1']:.4f}  "
          f"R@10={agg_mrl['overall']['recall@10']:.4f}  "
          f"R@50={agg_mrl['overall']['recall@50']:.4f}  "
          f"NDCG@10={agg_mrl['overall']['ndcg@10']:.4f}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # === Fit CSR dictionary on a corpus sample ===
    rng = np.random.RandomState(args.seed)
    train_sample_size = min(args.dict_train_size, len(corpus_embs))
    train_idx = rng.choice(len(corpus_embs), size=train_sample_size,
                           replace=False)
    print(f"\nFitting CSR dictionary on {train_sample_size} corpus samples...")
    dl = fit_dictionary(
        corpus_embs[train_idx],
        n_atoms=args.n_atoms,
        max_iter=args.csr_max_iter,
        batch_size=args.csr_batch_size,
        seed=args.seed,
    )

    # === Sweep sparsity budgets ===
    sweep_rows = []
    best_row_per_query = None
    for n_nz in args.n_nonzero_sweep:
        if n_nz > args.n_atoms:
            print(f"Skipping n_nonzero={n_nz} > n_atoms={args.n_atoms}")
            continue
        print(f"\n=== CSR retrieval: n_nonzero={n_nz} "
              f"(effective active dims per query) ===")
        t0 = time.time()
        query_sparse  = sparse_transform(dl, query_embs,  n_nz)
        corpus_sparse = sparse_transform(dl, corpus_embs, n_nz)
        t_enc = time.time() - t0
        _, per_q = retrieve_and_score(query_sparse, corpus_sparse, relevant_idx)
        agg = aggregate(per_q, blooms_1idx)
        print(f"  R@1={agg['overall']['recall@1']:.4f}  "
              f"R@10={agg['overall']['recall@10']:.4f}  "
              f"R@50={agg['overall']['recall@50']:.4f}  "
              f"NDCG@10={agg['overall']['ndcg@10']:.4f}  "
              f"(encode {t_enc:.1f}s)")

        sweep_rows.append({
            "n_atoms":   args.n_atoms,
            "n_nonzero": n_nz,
            "recall@1":  agg["overall"]["recall@1"],
            "recall@10": agg["overall"]["recall@10"],
            "recall@50": agg["overall"]["recall@50"],
            "ndcg@10":   agg["overall"]["ndcg@10"],
        })
        # Save per-query CSV for the middle-of-sweep budget (or the
        # one closest to BAM-PQ's typical avg active dims ~320 on bge-base)
        if (best_row_per_query is None
                or abs(n_nz - 320) < abs(best_row_per_query - 320)):
            best_row_per_query = n_nz
            pd.DataFrame({
                "bloom_1idx": blooms_1idx,
                "recall@1":   per_q["recall@1"],
                "recall@10":  per_q["recall@10"],
                "recall@50":  per_q["recall@50"],
                "ndcg@10":    per_q["ndcg@10"],
            }).to_csv(os.path.join(args.output_dir, "csr_per_query.csv"),
                      index=False)

    # === Save ===
    pd.DataFrame(sweep_rows).to_csv(
        os.path.join(args.output_dir, "csr_sweep.csv"), index=False
    )
    with open(os.path.join(args.output_dir, "csr_results.json"), "w") as f:
        json.dump({
            "config":      args.config,
            "checkpoint":  args.checkpoint,
            "n_atoms":     args.n_atoms,
            "n_nonzero_sweep": args.n_nonzero_sweep,
            "reference_mrl_full": agg_mrl,
            "csr_sweep":   sweep_rows,
        }, f, indent=2)

    # === Pretty print ===
    print("\n" + "=" * 78)
    print(f"CSR SWEEP — {args.config}  (dictionary K={args.n_atoms})")
    print("=" * 78)
    print(f"  {'MRL@full':<24}  R@1={agg_mrl['overall']['recall@1']:.4f}  "
          f"R@10={agg_mrl['overall']['recall@10']:.4f}  "
          f"R@50={agg_mrl['overall']['recall@50']:.4f}  "
          f"NDCG@10={agg_mrl['overall']['ndcg@10']:.4f}")
    for r in sweep_rows:
        label = f"CSR@nnz={r['n_nonzero']}"
        print(f"  {label:<24}  R@1={r['recall@1']:.4f}  "
              f"R@10={r['recall@10']:.4f}  R@50={r['recall@50']:.4f}  "
              f"NDCG@10={r['ndcg@10']:.4f}")
    print("=" * 78)
    print(f"Saved: {args.output_dir}/csr_results.json")
    print(f"Saved: {args.output_dir}/csr_sweep.csv")
    print(f"Saved: {args.output_dir}/csr_per_query.csv "
          f"(n_nonzero={best_row_per_query})")


if __name__ == "__main__":
    main()
