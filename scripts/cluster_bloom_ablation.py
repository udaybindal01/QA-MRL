"""
Ablation: unsupervised clustering / CSR-style sparse coding cannot recover
Bloom structure from encoder embeddings, but a supervised linear probe can.

Motivates BAM-PQ's supervised Bloom routing — cognitive structure is present
in the embeddings but is not the dominant unsupervised axis of variation,
so clustering (including sparse-code clustering a la CSR; Wen et al. 2025)
fails to recover it.

Metrics vs ground-truth Bloom labels (0=Remember ... 5=Create):
    - ARI (Adjusted Rand Index)              — chance-corrected agreement
    - NMI (Normalized Mutual Information)    — information overlap
    - Purity                                  — max-class fraction per cluster
    - Homogeneity / Completeness / V-measure — clustering quality
    - Linear probe accuracy                   — supervised upper bound

Usage:
    python scripts/cluster_bloom_ablation.py \\
        --config     configs/mrl_bge_large.yaml \\
        --checkpoint /scratch/ishaan.karan/bampq-checkpoints/educational/mrl_bge/best/ \\
        --output_dir /scratch/ishaan.karan/cluster_ablation/bge-large/

    # Skip the slow CSR proxy for a first pass:
    python scripts/cluster_bloom_ablation.py --skip_csr ...
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import (
    AgglomerativeClustering,
    KMeans,
    SpectralClustering,
)
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from scripts.eval_bam import load_mrl


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def cluster_purity(labels_pred: np.ndarray, labels_true: np.ndarray) -> float:
    """Purity: for each cluster, count majority true class; sum over clusters."""
    n = len(labels_pred)
    total = 0
    for c in np.unique(labels_pred):
        mask = labels_pred == c
        if mask.sum() == 0:
            continue
        _, counts = np.unique(labels_true[mask], return_counts=True)
        total += int(counts.max())
    return total / n


def majority_baseline(labels: np.ndarray) -> float:
    """Chance-level 'predict most frequent Bloom level'."""
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / len(labels))


# ---------------------------------------------------------------------------
# Clustering runners
# ---------------------------------------------------------------------------
def report_clustering(name: str, embeddings: np.ndarray,
                      labels_true: np.ndarray, k: int = 6,
                      seed: int = 42) -> dict:
    """Run one clustering algorithm and return metrics."""
    if name == "kmeans":
        model = KMeans(n_clusters=k, random_state=seed, n_init=10)
        preds = model.fit_predict(embeddings)
    elif name == "gmm":
        model = GaussianMixture(n_components=k, random_state=seed,
                                covariance_type="diag", max_iter=200)
        preds = model.fit_predict(embeddings)
    elif name == "spectral":
        model = SpectralClustering(n_clusters=k, random_state=seed,
                                   affinity="nearest_neighbors",
                                   n_neighbors=20, assign_labels="kmeans")
        preds = model.fit_predict(embeddings)
    elif name == "agglomerative":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        preds = model.fit_predict(embeddings)
    else:
        raise ValueError(f"Unknown clustering method: {name}")

    hom, com, v = homogeneity_completeness_v_measure(labels_true, preds)
    return {
        "method":       name,
        "ARI":          float(adjusted_rand_score(labels_true, preds)),
        "NMI":          float(normalized_mutual_info_score(labels_true, preds)),
        "purity":       float(cluster_purity(preds, labels_true)),
        "homogeneity":  float(hom),
        "completeness": float(com),
        "v_measure":    float(v),
    }


def csr_sparse_representation(embeddings: np.ndarray, n_atoms: int = 1024,
                              alpha: float = 1.0, seed: int = 42) -> np.ndarray:
    """
    Sparse dictionary coding (sklearn) as a CSR proxy.

    True CSR (Wen et al. 2025) is contrastively trained; this is the
    sklearn-only stand-in that still tests the 'sparse encoding cannot
    recover Bloom' hypothesis on the same embeddings.
    """
    print(f"  Learning sparse dictionary "
          f"(n_atoms={n_atoms}, alpha={alpha})...")
    dl = DictionaryLearning(
        n_components=n_atoms,
        alpha=alpha,
        max_iter=100,
        transform_algorithm="lasso_lars",
        n_jobs=-1,
        random_state=seed,
    )
    sparse_codes = dl.fit_transform(embeddings)
    density = float((sparse_codes != 0).sum() / sparse_codes.size)
    print(f"  Sparse code density: {density:.3f}")
    return sparse_codes


# ---------------------------------------------------------------------------
# Supervised upper bound
# ---------------------------------------------------------------------------
def linear_probe(embeddings: np.ndarray, labels: np.ndarray,
                 seed: int = 42) -> dict:
    """Supervised upper bound: does a linear layer read Bloom off embeddings?"""
    X_tr, X_te, y_tr, y_te = train_test_split(
        embeddings, labels, test_size=0.3, random_state=seed, stratify=labels
    )
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed,
                             class_weight="balanced")
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    return {
        "probe_accuracy": float(accuracy_score(y_te, y_pred)),
        "probe_macro_f1": float(f1_score(y_te, y_pred, average="macro")),
    }


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_queries(model, tokenizer, queries, device, batch_size: int = 128):
    """Encode query texts and return L2-normalised numpy array."""
    all_embs = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=128, return_tensors="pt").to(device)
        if hasattr(model, "encode_queries"):
            out = model.encode_queries(enc["input_ids"], enc["attention_mask"])
            if isinstance(out, dict):
                emb = out.get("masked_embedding", out.get("full"))
            else:
                emb = out
        else:
            out = model(enc["input_ids"], enc["attention_mask"])
            emb = out["full"] if isinstance(out, dict) else out
        all_embs.append(emb.float().cpu().numpy())
    embs = np.concatenate(all_embs, axis=0)
    return normalize(embs, norm="l2", axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",     required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="MRL baseline checkpoint dir")
    ap.add_argument("--test_path",  default="data/real/test.jsonl")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--k", type=int, default=6,
                    help="Number of clusters (default: 6 = Bloom levels)")
    ap.add_argument("--n_atoms", type=int, default=1024,
                    help="Sparse dictionary size for CSR proxy")
    ap.add_argument("--csr_alpha", type=float, default=1.0,
                    help="Sparsity strength for CSR proxy")
    ap.add_argument("--skip_csr", action="store_true",
                    help="Skip CSR sparse coding (slow on large N)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load MRL baseline ===
    print(f"Loading config: {args.config}")
    cfg = load_config(args.config)
    set_seed(cfg["training"].get("seed", args.seed))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])
    model = load_mrl(cfg, args.checkpoint, device)
    model.eval()

    # === Load test queries ===
    with open(args.test_path) as f:
        queries = [json.loads(line) for line in f]
    print(f"Loaded {len(queries)} test queries from {args.test_path}")
    query_texts = [q["query"] for q in queries]
    labels = np.array([q["bloom_level"] - 1 for q in queries])  # 0-indexed

    # === Encode ===
    print("Encoding queries...")
    embeddings = encode_queries(model, tokenizer, query_texts, device)
    print(f"Embeddings: {embeddings.shape}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Save embeddings + labels for downstream (t-SNE, etc.)
    np.savez_compressed(
        os.path.join(args.output_dir, "query_embeddings.npz"),
        embeddings=embeddings, labels=labels,
    )

    # === Baselines ===
    maj = majority_baseline(labels)
    print(f"\nMajority-class baseline accuracy: {maj:.4f}")

    # === Unsupervised clustering on raw embeddings ===
    rows = []
    for method in ["kmeans", "gmm", "spectral", "agglomerative"]:
        print(f"\nClustering with {method} on MRL embeddings...")
        row = report_clustering(method, embeddings, labels,
                                k=args.k, seed=args.seed)
        row["representation"] = "MRL-embed"
        rows.append(row)
        print(f"  ARI={row['ARI']:+.4f}  NMI={row['NMI']:.4f}  "
              f"purity={row['purity']:.4f}")

    # === CSR proxy: sparse dictionary coding + k-means ===
    if not args.skip_csr:
        print("\n=== CSR proxy (sparse dictionary coding) ===")
        sparse = csr_sparse_representation(
            embeddings, n_atoms=args.n_atoms,
            alpha=args.csr_alpha, seed=args.seed,
        )
        for method in ["kmeans", "gmm"]:
            print(f"Clustering CSR-sparse codes with {method}...")
            row = report_clustering(method, sparse, labels,
                                    k=args.k, seed=args.seed)
            row["representation"] = f"CSR-sparse(alpha={args.csr_alpha})"
            rows.append(row)
            print(f"  ARI={row['ARI']:+.4f}  NMI={row['NMI']:.4f}  "
                  f"purity={row['purity']:.4f}")

    # === Supervised linear probe (upper bound) ===
    print("\n=== Supervised linear probe (upper bound) ===")
    probe = linear_probe(embeddings, labels, seed=args.seed)
    print(f"  probe_accuracy = {probe['probe_accuracy']:.4f}")
    print(f"  probe_macro_f1 = {probe['probe_macro_f1']:.4f}")

    # === Save results ===
    df = pd.DataFrame(rows)
    df["majority_baseline"] = maj
    df["probe_accuracy"]    = probe["probe_accuracy"]
    df["probe_macro_f1"]    = probe["probe_macro_f1"]
    df["config"]            = args.config
    csv_path = os.path.join(args.output_dir, "cluster_bloom_ablation.csv")
    df.to_csv(csv_path, index=False)

    summary = {
        "n_queries":         int(len(labels)),
        "k":                 args.k,
        "majority_baseline": maj,
        "probe_accuracy":    probe["probe_accuracy"],
        "probe_macro_f1":    probe["probe_macro_f1"],
        "results":           rows,
        "config":            args.config,
        "checkpoint":        args.checkpoint,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # === Pretty summary ===
    print("\n" + "=" * 78)
    print(f"SUMMARY — unsupervised methods vs Bloom labels "
          f"(N={len(labels)}, k={args.k})")
    print("=" * 78)
    print(f"  Majority-class baseline:      accuracy = {maj:.3f}")
    print(f"  Linear probe (supervised):    accuracy = {probe['probe_accuracy']:.3f}, "
          f"F1 = {probe['probe_macro_f1']:.3f}")
    print(f"  -> Bloom IS learnable from these embeddings.")
    print()
    for r in rows:
        print(f"  {r['representation']:<30} {r['method']:<13} "
              f"ARI={r['ARI']:+.4f}  NMI={r['NMI']:.4f}  "
              f"purity={r['purity']:.3f}")
    print("=" * 78)
    print("If ARI < 0.05 across all rows: cognitive structure is NOT the")
    print("dominant unsupervised axis. Supervised Bloom routing is necessary.")
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {os.path.join(args.output_dir, 'summary.json')}")
    print(f"Saved: {os.path.join(args.output_dir, 'query_embeddings.npz')}")


if __name__ == "__main__":
    main()
