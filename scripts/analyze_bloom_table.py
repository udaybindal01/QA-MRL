"""
Analyze the learned Bloom → Dimension mapping in a trained BAM model.

This is the paper's key qualitative result: does the model learn that
cognitively simpler queries (Bloom 1-2) require fewer dimensions than
complex queries (Bloom 5-6)?

Outputs:
  1. Printed Bloom → dim table with monotonicity check
  2. bloom_logit_heatmap.pdf  — heatmap of logit distributions per Bloom level
  3. bloom_dim_table.pdf      — bar chart of the learned dim per Bloom level
  4. bloom_table.json         — serialized table for paper reporting

Usage:
    python scripts/analyze_bloom_table.py \
        --config configs/bam.yaml \
        --checkpoint /tmp/bam-ckpts/best/ \
        --output_dir results/bloom_analysis/
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config
from models.bam import BloomAlignedMRL


BLOOM_NAMES = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
BLOOM_COLORS = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759", "#b07aa1", "#76b7b2"]

# Research-ready matplotlib style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def load_bam(config, checkpoint_path, device):
    model = BloomAlignedMRL(config)
    ckpt = os.path.join(checkpoint_path, "checkpoint.pt")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"], strict=False)
        print(f"  Loaded from {ckpt}")
    else:
        print(f"  Warning: checkpoint not found at {ckpt}, using random weights")
    model.to(device).eval()
    return model


def get_bloom_dim_info(model):
    """Extract full dim mapping info from trained BloomDimMapping."""
    logits = model.bloom_dim_map.bloom_dim_logits.detach().cpu()  # [6, K]
    dims = model.bloom_dim_map.dims                               # [64,128,256,384,512,768]
    probs = F.softmax(logits, dim=-1).numpy()                     # [6, K]

    # Argmax dim per Bloom level
    selected_idx = probs.argmax(axis=-1)                          # [6]
    selected_dims = [dims[i] for i in selected_idx]              # [6]

    # Soft expected dim (weighted average)
    dim_arr = np.array(dims, dtype=float)
    expected_dims = (probs * dim_arr[None, :]).sum(axis=-1)       # [6]

    return {
        "logits": logits.numpy(),
        "probs": probs,
        "dims": dims,
        "selected_dims": selected_dims,
        "expected_dims": expected_dims.tolist(),
        "selected_idx": selected_idx.tolist(),
    }


def check_monotonicity(selected_dims):
    """Check if learned dims are monotonically non-decreasing across Bloom levels."""
    is_mono = all(selected_dims[i] <= selected_dims[i+1] for i in range(len(selected_dims)-1))
    inversions = [(i+1, i+2, selected_dims[i], selected_dims[i+1])
                  for i in range(len(selected_dims)-1)
                  if selected_dims[i] > selected_dims[i+1]]
    return is_mono, inversions


def plot_logit_heatmap(info, output_path):
    """
    Heatmap of softmax probabilities: Bloom level (y) × truncation dim (x).
    Bright cells = model strongly prefers that dim for that Bloom level.
    """
    probs = info["probs"]   # [6, K]
    dims = info["dims"]

    fig, ax = plt.subplots(figsize=(7, 3.5))

    cmap = LinearSegmentedColormap.from_list(
        "bloom_heat", ["#f7fbff", "#2171b5"], N=256
    )
    im = ax.imshow(probs, aspect="auto", cmap=cmap, vmin=0, vmax=probs.max())

    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([str(d) for d in dims], fontsize=10)
    ax.set_yticks(range(6))
    ax.set_yticklabels(BLOOM_NAMES, fontsize=10)
    ax.set_xlabel("Truncation Dimension", fontsize=11)
    ax.set_ylabel("Bloom Level", fontsize=11)
    ax.set_title("Learned Bloom → Dimension Mapping (Softmax Probability)", fontsize=12, pad=8)

    # Annotate each cell with probability
    for i in range(6):
        for j in range(len(dims)):
            val = probs[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold" if val == probs[i].max() else "normal")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Probability", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_dim_bar(info, output_path):
    """
    Bar chart: Bloom level (x) → selected truncation dimension (y).
    The key figure for the paper: shows monotonic increase if model learned correctly.
    """
    selected = info["selected_dims"]
    expected = info["expected_dims"]
    dims = info["dims"]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(6)
    width = 0.38

    bars1 = ax.bar(x - width/2, selected, width, color=BLOOM_COLORS,
                   edgecolor="white", linewidth=0.5, label="Argmax dim", zorder=3)
    bars2 = ax.bar(x + width/2, expected, width, color=BLOOM_COLORS,
                   edgecolor="white", linewidth=0.5, alpha=0.5, label="Expected dim", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(BLOOM_NAMES, rotation=20, ha="right", fontsize=10)
    ax.set_yticks(dims)
    ax.set_yticklabels([str(d) for d in dims], fontsize=9)
    ax.set_ylabel("Truncation Dimension", fontsize=11)
    ax.set_title("Learned Bloom → Truncation Dimension", fontsize=12, pad=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, framealpha=0.7)

    # Annotate argmax bars
    for bar, val in zip(bars1, selected):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                str(int(val)), ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bam.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="results/bloom_analysis/")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_bam(config, args.checkpoint, device)
    info = get_bloom_dim_info(model)

    # ── Print table ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LEARNED BLOOM → DIMENSION TABLE")
    print("=" * 60)
    print(f"{'Bloom Level':20s}{'Argmax Dim':>12s}{'Expected Dim':>14s}{'Probability':>12s}")
    print("-" * 60)
    for i, name in enumerate(BLOOM_NAMES):
        d = info["selected_dims"][i]
        ed = info["expected_dims"][i]
        p = info["probs"][i, info["selected_idx"][i]]
        print(f"  {name:18s}{d:>12d}{ed:>14.1f}{p:>12.3f}")

    # ── Monotonicity check ────────────────────────────────────────────
    is_mono, inversions = check_monotonicity(info["selected_dims"])
    print(f"\n  Monotonically ordered: {'YES ✓' if is_mono else 'NO ✗'}")
    if inversions:
        for b1, b2, d1, d2 in inversions:
            print(f"    Inversion: Bloom {b1} ({BLOOM_NAMES[b1-1]}) → {d1}d "
                  f"> Bloom {b2} ({BLOOM_NAMES[b2-1]}) → {d2}d")
    else:
        print("  The model learned that cognitive complexity requires more dimensions.")

    # Efficiency estimate
    avg_dim = np.mean(info["selected_dims"])
    saving = (768 - avg_dim) / 768 * 100
    print(f"\n  Average dims across Bloom levels: {avg_dim:.0f} / 768")
    print(f"  Theoretical FAISS speedup (vs full 768): {768/avg_dim:.1f}x")
    print(f"  Dimension savings: {saving:.0f}%")

    # ── Save JSON ─────────────────────────────────────────────────────
    result = {
        "bloom_dim_table": {BLOOM_NAMES[i]: int(info["selected_dims"][i]) for i in range(6)},
        "bloom_expected_dim": {BLOOM_NAMES[i]: float(info["expected_dims"][i]) for i in range(6)},
        "bloom_dim_probs": {BLOOM_NAMES[i]: info["probs"][i].tolist() for i in range(6)},
        "dims": info["dims"],
        "is_monotonic": is_mono,
        "avg_dim": float(avg_dim),
        "theoretical_speedup": float(768 / avg_dim),
    }
    json_path = os.path.join(args.output_dir, "bloom_table.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved table: {json_path}")

    # ── Figures ───────────────────────────────────────────────────────
    plot_logit_heatmap(info, os.path.join(args.output_dir, "bloom_logit_heatmap.pdf"))
    plot_dim_bar(info, os.path.join(args.output_dir, "bloom_dim_table.pdf"))

    print(f"\nDone. All outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
