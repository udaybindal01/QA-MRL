"""
t-SNE / UMAP visualisation of MRL encoder embeddings coloured by Bloom level.

Visual proof that unsupervised structure in the embedding space does not
align with Bloom taxonomy — pairs with scripts/cluster_bloom_ablation.py
(low ARI/NMI). Produces a figure suitable for the Appendix H "cognitive
structure is not unsupervised" discussion.

Two input modes:
    (1) Precomputed embeddings from cluster_bloom_ablation.py:
        --embeddings /scratch/.../cluster_ablation/bge-large/query_embeddings.npz

    (2) Encode from scratch (requires config + checkpoint):
        --config     configs/mrl_bge_large.yaml
        --checkpoint /scratch/.../mrl_bge/best/
        --test_path  data/real/test.jsonl

Usage examples:
    # Fast path — reuse embeddings saved by cluster_bloom_ablation.py
    python scripts/plot_bloom_tsne.py \\
        --embeddings /scratch/ishaan.karan/cluster_ablation/bge-large/query_embeddings.npz \\
        --output_dir /scratch/ishaan.karan/cluster_ablation/bge-large/ \\
        --method tsne

    # Slow path — encode from scratch and also plot UMAP
    python scripts/plot_bloom_tsne.py \\
        --config     configs/mrl_bge_large.yaml \\
        --checkpoint /scratch/ishaan.karan/bampq-checkpoints/educational/mrl_bge/best/ \\
        --output_dir /scratch/ishaan.karan/cluster_ablation/bge-large/ \\
        --method both
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from scripts.eval_bam import load_mrl


BLOOM_NAMES = {
    0: "L1 Remember",
    1: "L2 Understand",
    2: "L3 Apply",
    3: "L4 Analyze",
    4: "L5 Evaluate",
    5: "L6 Create",
}
# Colour-blind-friendly palette (Wong 2011) — 6 distinct hues
BLOOM_COLOURS = {
    0: "#0072B2",  # blue
    1: "#009E73",  # green
    2: "#E69F00",  # orange
    3: "#CC79A7",  # pink
    4: "#F0E442",  # yellow
    5: "#D55E00",  # red
}


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


def load_embeddings(args):
    """Either load a precomputed .npz or encode from scratch."""
    if args.embeddings:
        data = np.load(args.embeddings)
        return data["embeddings"], data["labels"]

    if not (args.config and args.checkpoint):
        raise ValueError(
            "Provide either --embeddings <path.npz> or "
            "--config <yaml> + --checkpoint <ckpt_dir>."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading config: {args.config}")
    cfg = load_config(args.config)
    set_seed(cfg["training"].get("seed", 42))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])
    model = load_mrl(cfg, args.checkpoint, device)
    model.eval()

    with open(args.test_path) as f:
        queries = [json.loads(line) for line in f]
    print(f"Loaded {len(queries)} queries from {args.test_path}")
    texts  = [q["query"] for q in queries]
    labels = np.array([q["bloom_level"] - 1 for q in queries])

    print("Encoding queries...")
    embs = encode_queries(model, tokenizer, texts, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return embs, labels


def reduce_tsne(embeddings, perplexity: int = 30, seed: int = 42):
    print(f"Running t-SNE (perplexity={perplexity})...")
    reducer = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                   init="pca", learning_rate="auto", n_iter=1000)
    return reducer.fit_transform(embeddings)


def reduce_umap(embeddings, n_neighbors: int = 15, min_dist: float = 0.1,
                seed: int = 42):
    try:
        import umap
    except ImportError:
        print("  UMAP not installed — skipping. Install: pip install umap-learn")
        return None
    print(f"Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, random_state=seed, metric="cosine")
    return reducer.fit_transform(embeddings)


def scatter_by_bloom(coords, labels, title, out_path,
                     point_size: float = 6.0):
    """Save a 2-D scatter plot coloured by Bloom level."""
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=150)
    for b in range(6):
        mask = labels == b
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=point_size, c=BLOOM_COLOURS[b],
                   label=BLOOM_NAMES[b], alpha=0.7,
                   edgecolors="none")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="best", fontsize=8, framealpha=0.9,
              markerscale=1.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"),
                bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default=None,
                    help="Precomputed .npz from cluster_bloom_ablation.py")
    ap.add_argument("--config",     default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--test_path",  default="data/real/test.jsonl")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--method", choices=["tsne", "umap", "both"],
                    default="tsne")
    ap.add_argument("--perplexity",  type=int,   default=30)
    ap.add_argument("--n_neighbors", type=int,   default=15)
    ap.add_argument("--min_dist",    type=float, default=0.1)
    ap.add_argument("--seed",        type=int,   default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    embeddings, labels = load_embeddings(args)
    print(f"Embeddings: {embeddings.shape}   labels: {labels.shape}")

    _, counts = np.unique(labels, return_counts=True)
    print("Bloom-level distribution (test set):")
    for b, n in enumerate(counts):
        print(f"  {BLOOM_NAMES[b]:<14}  N={n:>5}  ({n / len(labels):.1%})")

    if args.method in ("tsne", "both"):
        coords = reduce_tsne(embeddings, perplexity=args.perplexity,
                             seed=args.seed)
        scatter_by_bloom(
            coords, labels,
            title=(f"t-SNE of MRL query embeddings by Bloom level "
                   f"(perplexity={args.perplexity})"),
            out_path=os.path.join(args.output_dir, "tsne_bloom.pdf"),
        )
        np.save(os.path.join(args.output_dir, "tsne_coords.npy"), coords)

    if args.method in ("umap", "both"):
        coords = reduce_umap(embeddings, n_neighbors=args.n_neighbors,
                             min_dist=args.min_dist, seed=args.seed)
        if coords is not None:
            scatter_by_bloom(
                coords, labels,
                title=(f"UMAP of MRL query embeddings by Bloom level "
                       f"(k={args.n_neighbors}, min_dist={args.min_dist})"),
                out_path=os.path.join(args.output_dir, "umap_bloom.pdf"),
            )
            np.save(os.path.join(args.output_dir, "umap_coords.npy"), coords)

    print("\nDone. If Bloom-level colours are visually intermixed with no")
    print("class-coherent clusters, this is the visual counterpart to the")
    print("low ARI/NMI in scripts/cluster_bloom_ablation.py.")


if __name__ == "__main__":
    main()
