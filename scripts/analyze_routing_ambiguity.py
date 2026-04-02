"""
Analyze routing ambiguity vs retrieval quality (Challenge 2 + 7).

Computes per-query "routing ambiguity" as the entropy of the k-nearest-neighbor
Bloom label distribution in embedding space. Queries surrounded by neighbors with
diverse Bloom labels are "ambiguous" — the routing signal is noisy.

Bins queries by ambiguity (low/medium/high), computes R@10 per bin, and plots
entropy vs R@10 scatter. This is the evidence for:
  - Challenge 2: routing ambiguity degrades performance
  - Challenge 7: uncertainty in routing hurts retrieval

Output:
  routing_ambiguity_analysis.pdf   — scatter plot + binned bar chart
  ambiguity_table.json             — quantitative table for paper

Usage:
    python scripts/analyze_routing_ambiguity.py \
        --config configs/bam.yaml \
        --checkpoint /tmp/bam-ckpts/best/ \
        --output_dir results/analysis/
"""

import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from models.bam import BloomAlignedMRL
from transformers import AutoTokenizer

BLOOM_NAMES = {0: "Remember", 1: "Understand", 2: "Apply",
               3: "Analyze", 4: "Evaluate", 5: "Create"}


def compute_knn_bloom_entropy(query_embs, bloom_labels, k=10):
    """
    For each query, find k nearest neighbors (by cosine similarity) and
    compute Shannon entropy of their Bloom label distribution.

    High entropy → neighbors span many Bloom levels → ambiguous routing signal.
    """
    normed = F.normalize(query_embs, p=2, dim=-1)
    sim = torch.mm(normed, normed.t())  # [N, N]
    # Mask self-similarity
    sim.fill_diagonal_(-1.0)

    entropies = []
    for i in range(len(query_embs)):
        topk_idx = sim[i].topk(k).indices
        neighbor_blooms = bloom_labels[topk_idx]
        # Count per Bloom level
        counts = torch.bincount(neighbor_blooms, minlength=6).float()
        probs = counts / counts.sum().clamp(min=1)
        # Shannon entropy
        probs = probs.clamp(min=1e-9)
        entropy = -(probs * probs.log()).sum().item()
        entropies.append(entropy)
    return np.array(entropies)


@torch.no_grad()
def run_analysis(args):
    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])
    os.makedirs(args.output_dir, exist_ok=True)

    config["training"]["loss"]["bloom_frequencies"] = [1/6] * 6
    model = BloomAlignedMRL(config)
    ckpt = os.path.join(args.checkpoint, "checkpoint.pt")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"], strict=False)
        print(f"Loaded from {ckpt}")
    model.to(device).eval()

    # Load test data
    samples = []
    with open(config["data"]["test_path"]) as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    corpus = []
    with open(config["data"]["corpus_path"]) as f:
        for line in f:
            corpus.append(json.loads(line.strip()))
    corpus_id_to_idx = {p["id"]: i for i, p in enumerate(corpus)}
    valid = [s for s in samples if s.get("positive_id", "") in corpus_id_to_idx]
    print(f"Valid queries: {len(valid)}")

    # Encode queries
    query_embs, bloom_labels_list = [], []
    for i in tqdm(range(0, len(valid), 64), desc="Encoding queries"):
        batch = valid[i:i + 64]
        enc = tokenizer([s["query"] for s in batch], padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        bloom_lbl = torch.tensor([s["bloom_level"] - 1 for s in batch],
                                 dtype=torch.long, device=device)
        out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                   bloom_labels=bloom_lbl)
        query_embs.append(out["full_embedding"].cpu())
        bloom_labels_list.append(bloom_lbl.cpu())

    query_embs = torch.cat(query_embs)
    bloom_labels = torch.cat(bloom_labels_list)

    # Encode corpus
    print("Encoding corpus...")
    corpus_embs = []
    for i in tqdm(range(0, len(corpus), 128), desc="Corpus"):
        batch = [c["text"] for c in corpus[i:i + 128]]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
        corpus_embs.append(out["full_embedding"].cpu())
    corpus_embs = torch.cat(corpus_embs)

    # Compute R@10 per query
    gt_indices = np.array([corpus_id_to_idx[s["positive_id"]] for s in valid])
    N = len(valid)

    # Re-encode with routing for retrieval
    query_masked, query_dims_list = [], []
    for i in tqdm(range(0, len(valid), 64), desc="Encoding (masked)"):
        batch = valid[i:i + 64]
        enc = tokenizer([s["query"] for s in batch], padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        bloom_lbl = torch.tensor([s["bloom_level"] - 1 for s in batch],
                                 dtype=torch.long, device=device)
        out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                   bloom_labels=bloom_lbl)
        query_masked.append(out["masked_embedding"].cpu())
        if "discrete_dim" in out:
            query_dims_list.append(out["discrete_dim"].detach().cpu())

    query_masked = torch.cat(query_masked)
    hits_at_10 = np.zeros(N, dtype=bool)
    chunk = 256
    for i in range(0, N, chunk):
        q = query_masked[i:i + chunk].to(device)
        sim = torch.mm(q, corpus_embs.to(device).t())
        topk = sim.topk(10, dim=-1).indices.cpu().numpy()
        for j, row in enumerate(topk):
            hits_at_10[i + j] = (gt_indices[i + j] in row)

    # Compute kNN entropy
    print("Computing kNN Bloom entropy...")
    entropies = compute_knn_bloom_entropy(query_embs, bloom_labels, k=min(20, N - 1))

    # Bin by entropy
    n_bins = 3
    percentiles = np.percentile(entropies, [0, 33.3, 66.7, 100])
    bins = ["low", "medium", "high"]
    bin_results = {}
    for b_idx in range(n_bins):
        lo, hi = percentiles[b_idx], percentiles[b_idx + 1]
        mask = (entropies >= lo) & (entropies <= hi if b_idx == n_bins - 1 else entropies < hi)
        if mask.sum() == 0:
            continue
        r10 = hits_at_10[mask].mean()
        bloom_dist = {}
        for level in range(6):
            bloom_mask = (bloom_labels.numpy() == level) & mask
            if bloom_mask.sum() > 0:
                bloom_dist[BLOOM_NAMES[level]] = int(bloom_mask.sum())
        bin_results[bins[b_idx]] = {
            "n": int(mask.sum()),
            "recall@10": float(r10),
            "entropy_range": [float(lo), float(hi)],
            "mean_entropy": float(entropies[mask].mean()),
            "bloom_distribution": bloom_dist,
        }

    # Table output
    table = {
        "description": "Per-query kNN Bloom entropy vs R@10. High entropy = ambiguous routing context.",
        "bins": bin_results,
        "correlation": {
            "pearson_r": float(np.corrcoef(entropies, hits_at_10.astype(float))[0, 1]),
        },
        "overall": {
            "n": N,
            "recall@10": float(hits_at_10.mean()),
            "mean_entropy": float(entropies.mean()),
            "std_entropy": float(entropies.std()),
        },
    }

    out_path = os.path.join(args.output_dir, "ambiguity_table.json")
    with open(out_path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\nSaved table to {out_path}")

    # Print summary
    print("\n=== Routing Ambiguity Analysis ===")
    print(f"{'Bin':10s}  {'N':>6s}  {'R@10':>8s}  {'Mean H':>8s}  {'Entropy range'}")
    print("-" * 55)
    for b_name, res in bin_results.items():
        print(f"{b_name:10s}  {res['n']:>6d}  {res['recall@10']:>8.4f}  "
              f"{res['mean_entropy']:>8.3f}  [{res['entropy_range'][0]:.3f}, {res['entropy_range'][1]:.3f}]")
    print(f"\nPearson r(entropy, R@10): {table['correlation']['pearson_r']:.4f}")
    print("(negative r = higher entropy hurts retrieval)")

    # Plot (optional — skip gracefully if matplotlib not available)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Scatter: entropy vs R@10 per query (binned for visibility)
        ax = axes[0]
        entropy_bins = np.linspace(entropies.min(), entropies.max(), 30)
        bin_centers, bin_r10 = [], []
        for lo, hi in zip(entropy_bins[:-1], entropy_bins[1:]):
            mask = (entropies >= lo) & (entropies < hi)
            if mask.sum() > 5:
                bin_centers.append((lo + hi) / 2)
                bin_r10.append(hits_at_10[mask].mean())
        ax.plot(bin_centers, bin_r10, "o-", color="steelblue")
        ax.set_xlabel("Routing Ambiguity (kNN Bloom entropy)")
        ax.set_ylabel("R@10")
        ax.set_title("Ambiguity vs Retrieval Quality")
        ax.grid(alpha=0.3)

        # Bar chart: R@10 per ambiguity bin
        ax2 = axes[1]
        bin_names = list(bin_results.keys())
        bin_r10s = [bin_results[b]["recall@10"] for b in bin_names]
        colors = ["#4CAF50", "#FFC107", "#F44336"]
        ax2.bar(bin_names, bin_r10s, color=colors[:len(bin_names)])
        ax2.set_ylabel("R@10")
        ax2.set_title("R@10 by Routing Ambiguity Bin")
        ax2.set_ylim(0, 1)
        for i, v in enumerate(bin_r10s):
            ax2.text(i, v + 0.01, f"{v:.3f}", ha="center")

        plt.tight_layout()
        pdf_path = os.path.join(args.output_dir, "routing_ambiguity_analysis.pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved plot to {pdf_path}")
        plt.close()
    except ImportError:
        print("matplotlib not available — skipping plot generation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bam.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="results/analysis/")
    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
