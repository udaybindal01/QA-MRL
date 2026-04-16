"""
Query-Adaptive Baseline (No Bloom Taxonomy).

Isolates the contribution of Bloom's taxonomy structure by comparing:
  1. BAM with correct Bloom labels (full model)
  2. BAM with K-Means cluster labels (query-adaptive, no Bloom)
  3. BAM with random labels (no structure at all)
  4. MRL at matched dims (no routing at all)

The K-Means baseline clusters query embeddings into 6 groups (same as Bloom)
and uses cluster assignment as the "routing label". This tests whether
Bloom-specific routing provides benefit over generic query-adaptive routing.

If BAM-Bloom >> BAM-KMeans >> BAM-Random, then Bloom structure matters.
If BAM-Bloom ≈ BAM-KMeans >> BAM-Random, then any query clustering works equally well.

Usage:
    python scripts/eval_query_adaptive_baseline.py \
        --config configs/bam_optionb_e5large.yaml \
        --checkpoint /tmp/bam-b-e5large-ckpts5/best_bsr/ \
        --baseline /tmp/mrl-e5large-ckpts/best/ \
        --output_dir results/query_adaptive_baseline/
"""

import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from models.bam import BloomAlignedMRL
from models.encoder import MRLEncoder
from transformers import AutoTokenizer

BLOOM_NAMES = {0: "Remember", 1: "Understand", 2: "Apply",
               3: "Analyze", 4: "Evaluate", 5: "Create"}


@torch.no_grad()
def encode_corpus(model, corpus, tokenizer, device):
    """Encode corpus with model (documents always at full dims)."""
    model.eval()
    embs = []
    for i in tqdm(range(0, len(corpus), 128), desc="  corpus", leave=False):
        batch = [c["text"] for c in corpus[i:i + 128]]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        if hasattr(model, "encode_documents"):
            out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
            embs.append(out["full_embedding"].cpu())
        else:
            out = model(enc["input_ids"], enc["attention_mask"])
            embs.append(out["full"].cpu())
    return torch.cat(embs)


@torch.no_grad()
def get_query_embeddings(model, valid, tokenizer, device):
    """Get full (unmasked) query embeddings for clustering."""
    model.eval()
    embs = []
    for i in range(0, len(valid), 64):
        batch = [s["query"] for s in valid[i:i + 64]]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.encoder(enc["input_ids"], enc["attention_mask"])
        embs.append(out["full"].cpu())
    return torch.cat(embs)


@torch.no_grad()
def evaluate_with_labels(model, valid, corpus_embs, corpus_id_to_idx,
                          bloom_labels, tokenizer, device):
    """Evaluate BAM model using provided Bloom labels (any assignment)."""
    model.eval()
    query_masked_list = []

    for i in range(0, len(valid), 64):
        batch = valid[i:i + 64]
        enc = tokenizer([s["query"] for s in batch], padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        batch_labels = torch.tensor(bloom_labels[i:i + len(batch)],
                                     dtype=torch.long, device=device)
        out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                   bloom_labels=batch_labels)
        query_masked_list.append(out["masked_embedding"].cpu())

    query_masked = torch.cat(query_masked_list)
    N = len(valid)
    gt_indices = np.array([corpus_id_to_idx[s["positive_id"]] for s in valid])
    query_blooms = np.array([s["bloom_level"] for s in valid])

    rankings = []
    for i in range(0, N, 256):
        q = query_masked[i:i + 256].to(device)
        sim = torch.mm(q, corpus_embs.to(device).t())
        rankings.append(sim.topk(100, dim=-1).indices.cpu().numpy())
    rankings = np.concatenate(rankings)

    metrics = {}
    for k in [1, 5, 10, 50]:
        hits = np.array([gt_indices[i] in rankings[i, :k] for i in range(N)])
        metrics[f"recall@{k}"] = float(hits.mean())

    mrrs = []
    for i in range(N):
        pos = np.where(rankings[i] == gt_indices[i])[0]
        mrrs.append(1.0 / (pos[0] + 1) if len(pos) > 0 else 0.0)
    metrics["mrr"] = float(np.mean(mrrs))

    ndcgs = []
    for i in range(N):
        for j, idx in enumerate(rankings[i, :10]):
            if idx == gt_indices[i]:
                ndcgs.append(1.0 / np.log2(j + 2))
                break
        else:
            ndcgs.append(0.0)
    metrics["ndcg@10"] = float(np.mean(ndcgs))

    # Per-Bloom breakdown
    for level in range(1, 7):
        mask = query_blooms == level
        if mask.sum() == 0:
            continue
        lr = rankings[mask]
        lg = gt_indices[mask]
        nl = int(mask.sum())
        hits = np.array([lg[i] in lr[i, :10] for i in range(nl)])
        metrics[f"bloom_{BLOOM_NAMES[level-1]}_recall@10"] = float(hits.mean())

    return metrics


@torch.no_grad()
def evaluate_mrl_at_dims(model, valid, corpus_embs, corpus_id_to_idx,
                          tokenizer, device, target_dims):
    """Evaluate MRL baseline truncated to target_dims."""
    model.eval()
    query_embs = []
    for i in range(0, len(valid), 64):
        batch = [s["query"] for s in valid[i:i + 64]]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(enc["input_ids"], enc["attention_mask"])
        query_embs.append(out["full"].cpu())
    query_embs = torch.cat(query_embs)

    # Truncate to target dims
    d = int(target_dims)
    q_trunc = F.normalize(query_embs[:, :d], p=2, dim=-1)
    c_trunc = F.normalize(corpus_embs[:, :d], p=2, dim=-1)

    N = len(valid)
    gt_indices = np.array([corpus_id_to_idx[s["positive_id"]] for s in valid])

    rankings = []
    for i in range(0, N, 256):
        q = q_trunc[i:i + 256].to(device)
        sim = torch.mm(q, c_trunc.to(device).t())
        rankings.append(sim.topk(100, dim=-1).indices.cpu().numpy())
    rankings = np.concatenate(rankings)

    metrics = {"avg_dims": d}
    for k in [1, 5, 10, 50]:
        hits = np.array([gt_indices[i] in rankings[i, :k] for i in range(N)])
        metrics[f"recall@{k}"] = float(hits.mean())

    mrrs = []
    for i in range(N):
        pos = np.where(rankings[i] == gt_indices[i])[0]
        mrrs.append(1.0 / (pos[0] + 1) if len(pos) > 0 else 0.0)
    metrics["mrr"] = float(np.mean(mrrs))

    ndcgs = []
    for i in range(N):
        for j, idx in enumerate(rankings[i, :10]):
            if idx == gt_indices[i]:
                ndcgs.append(1.0 / np.log2(j + 2))
                break
        else:
            ndcgs.append(0.0)
    metrics["ndcg@10"] = float(np.mean(ndcgs))

    return metrics


def kmeans_cluster(embeddings: np.ndarray, n_clusters: int = 6,
                   n_iter: int = 100, seed: int = 42) -> np.ndarray:
    """Simple K-Means clustering (no sklearn dependency)."""
    np.random.seed(seed)
    N, D = embeddings.shape

    # K-Means++ initialization
    centroids = np.zeros((n_clusters, D))
    centroids[0] = embeddings[np.random.randint(N)]
    for k in range(1, n_clusters):
        dists = np.min([np.sum((embeddings - centroids[j])**2, axis=1)
                        for j in range(k)], axis=0)
        probs = dists / dists.sum()
        centroids[k] = embeddings[np.random.choice(N, p=probs)]

    for _ in range(n_iter):
        # Assign
        dists = np.stack([np.sum((embeddings - c)**2, axis=1) for c in centroids], axis=1)
        labels = np.argmin(dists, axis=1)
        # Update
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                centroids[k] = embeddings[mask].mean(axis=0)

    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Query-Adaptive Baseline: Bloom routing vs K-Means routing"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="BAM checkpoint dir")
    parser.add_argument("--baseline", default=None, help="MRL baseline checkpoint dir")
    parser.add_argument("--output_dir", default="results/query_adaptive_baseline/")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])

    test_path = config["data"]["test_path"]
    corpus_path = config["data"]["corpus_path"]

    # Load data
    corpus = []
    with open(corpus_path) as f:
        for line in f:
            corpus.append(json.loads(line.strip()))
    corpus_id_to_idx = {p["id"]: i for i, p in enumerate(corpus)}

    samples = []
    with open(test_path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    valid = [s for s in samples if s.get("positive_id", "") in corpus_id_to_idx]
    print(f"Test queries: {len(valid)}, Corpus: {len(corpus)}")

    # Load BAM model
    print("\nLoading BAM model...")
    config["training"]["loss"].setdefault("bloom_frequencies", [1/6] * 6)
    bam_model = BloomAlignedMRL(config)
    ckpt = os.path.join(args.checkpoint, "checkpoint.pt")
    if os.path.exists(ckpt):
        bam_model.load_state_dict(
            torch.load(ckpt, map_location=device)["model_state_dict"], strict=False
        )
    bam_model.to(device).eval()

    # Encode corpus
    print("Encoding corpus...")
    corpus_embs = encode_corpus(bam_model, corpus, tokenizer, device)

    # Get true Bloom labels (0-indexed)
    bloom_cache_path = test_path + ".bloom_cache.json"
    if os.path.exists(bloom_cache_path):
        with open(bloom_cache_path) as f:
            true_bloom = json.load(f)
        true_bloom = true_bloom[:len(samples)]
        true_bloom_valid = [true_bloom[samples.index(s)] if s in samples else s["bloom_level"] - 1
                           for s in valid]
    else:
        true_bloom_valid = [s["bloom_level"] - 1 for s in valid]

    # Get query embeddings for K-Means clustering
    print("Computing query embeddings for clustering...")
    query_embs = get_query_embeddings(bam_model, valid, tokenizer, device)
    query_embs_np = query_embs.numpy()

    # K-Means cluster into 6 groups
    print("Running K-Means clustering (6 clusters)...")
    kmeans_labels = kmeans_cluster(query_embs_np, n_clusters=6)

    # Random labels
    np.random.seed(42)
    random_labels = np.random.randint(0, 6, size=len(valid))

    all_results = {}

    # 1. BAM with correct Bloom labels
    print("\n--- BAM with Bloom labels (full model) ---")
    all_results["BAM (Bloom routing)"] = evaluate_with_labels(
        bam_model, valid, corpus_embs, corpus_id_to_idx,
        true_bloom_valid, tokenizer, device
    )

    # 2. BAM with K-Means labels
    print("--- BAM with K-Means labels (query-adaptive, no Bloom) ---")
    all_results["BAM (K-Means routing)"] = evaluate_with_labels(
        bam_model, valid, corpus_embs, corpus_id_to_idx,
        kmeans_labels.tolist(), tokenizer, device
    )

    # 3. BAM with random labels
    print("--- BAM with random labels (no structure) ---")
    all_results["BAM (Random routing)"] = evaluate_with_labels(
        bam_model, valid, corpus_embs, corpus_id_to_idx,
        random_labels.tolist(), tokenizer, device
    )

    # 4. BAM with all-Remember (majority class)
    print("--- BAM with all-Remember (majority class) ---")
    all_results["BAM (All-Remember)"] = evaluate_with_labels(
        bam_model, valid, corpus_embs, corpus_id_to_idx,
        [0] * len(valid), tokenizer, device
    )

    # 5. MRL at matched dims (no routing)
    if args.baseline:
        print("--- MRL Baseline ---")
        mc = config["model"]
        mrl_model = MRLEncoder(
            model_name=mc["backbone"], embedding_dim=mc["embedding_dim"],
            mrl_dims=mc["mrl_dims"]
        )
        bl_ckpt = os.path.join(args.baseline, "checkpoint.pt")
        if os.path.exists(bl_ckpt):
            mrl_model.load_state_dict(
                torch.load(bl_ckpt, map_location=device)["model_state_dict"],
                strict=False
            )
        mrl_model.to(device).eval()
        mrl_corpus_embs = encode_corpus(mrl_model, corpus, tokenizer, device)

        # MRL at full dims
        all_results["MRL (full dims)"] = evaluate_mrl_at_dims(
            mrl_model, valid, mrl_corpus_embs, corpus_id_to_idx,
            tokenizer, device, mc["embedding_dim"]
        )

        # MRL at matched avg dims (approximate BAM's avg)
        bam_r10 = all_results["BAM (Bloom routing)"]
        # Use 456 or compute from BAM if available
        matched_dim = 456
        all_results[f"MRL (truncated @{matched_dim})"] = evaluate_mrl_at_dims(
            mrl_model, valid, mrl_corpus_embs, corpus_id_to_idx,
            tokenizer, device, matched_dim
        )

    # Print Bloom vs K-Means label alignment
    print("\n=== Bloom vs K-Means Label Alignment ===")
    bloom_arr = np.array(true_bloom_valid)
    # For each K-Means cluster, show dominant Bloom level
    for k in range(6):
        mask = kmeans_labels == k
        if mask.sum() == 0:
            continue
        bloom_dist = Counter(bloom_arr[mask].tolist())
        dominant = bloom_dist.most_common(1)[0]
        total = mask.sum()
        print(f"  Cluster {k} (n={total}): dominant Bloom = {BLOOM_NAMES[dominant[0]]} "
              f"({dominant[1]}/{total} = {dominant[1]/total:.0%})")

    # Print comparison table
    print("\n" + "=" * 90)
    print("QUERY-ADAPTIVE BASELINE COMPARISON")
    print("=" * 90)
    hdr = f"{'Method':35s}{'R@1':>8s}{'R@10':>8s}{'R@50':>8s}{'NDCG@10':>9s}{'MRR':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for name, res in all_results.items():
        print(f"{name:35s}"
              f"{res.get('recall@1', 0):>8.4f}"
              f"{res.get('recall@10', 0):>8.4f}"
              f"{res.get('recall@50', 0):>8.4f}"
              f"{res.get('ndcg@10', 0):>9.4f}"
              f"{res.get('mrr', 0):>8.4f}")

    # Save
    out_path = os.path.join(args.output_dir, "query_adaptive_baseline.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
