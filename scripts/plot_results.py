"""
BAM-PQ Results Plots

Generates:
  1. Efficiency curves — R@10 vs dim budget (MRL curve + BAM-PQ operating points)
  2. Storage efficiency — index size (MB) and compression ratio per backbone
  3. Inference time efficiency — relative latency vs R@10 achieved
  4. Mask cosine similarity heatmaps — per-Bloom specialization
  5. Kendall tau & dim spread — cognitive ordering across backbones
  6. Per-Bloom R@10 improvement bars

Usage:
    python scripts/plot_results.py [--output_dir results/plots]
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    "e5large":  "#2563EB",
    "bge":      "#16A34A",
    "arctic":   "#D97706",
    "roberta":  "#DC2626",
    "qwen06b":  "#7C3AED",
    "bam_b":    "#0891B2",
    "mrl":      "#6B7280",
}
BLOOM_COLORS = ["#6366F1","#06B6D4","#10B981","#F59E0B","#EF4444","#8B5CF6"]
BLOOM_NAMES  = ["Remember","Understand","Apply","Analyze","Evaluate","Create"]

# ── Known results ─────────────────────────────────────────────────────────────

# MRL truncation curve (e5-large)
MRL_CURVE_DIMS  = [64,128,192,256,320,384,448,512,576,640,704,768]
MRL_CURVE_R10   = [0.3844,0.4278,0.4381,0.4463,0.4490,0.4536,0.4536,
                   0.4557,0.4609,0.4606,0.4663,0.4639]

# BAM-PQ operating points per backbone: (avg_dims, overall_R@10, MRL_R@10_full)
BACKBONES = {
    "e5-large":  {"dims": 473, "pq_r10": 0.5297, "mrl_r10": 0.4669, "ndcg": 0.4070,
                  "bsr": 0.6177, "color": C["e5large"]},
    "bge-large": {"dims": 384, "pq_r10": 0.5358, "mrl_r10": 0.4779, "ndcg": 0.4063,
                  "bsr": 0.6522, "color": C["bge"]},
    "arctic":    {"dims": 302, "pq_r10": 0.4548, "mrl_r10": 0.4508, "ndcg": 0.3586,
                  "bsr": 0.5429, "color": C["arctic"]},
    "roberta":   {"dims": 272, "pq_r10": 0.3908, "mrl_r10": 0.1374, "ndcg": 0.2982,
                  "bsr": 0.4726, "color": C["roberta"]},
    "qwen06b":   {"dims": 366, "pq_r10": 0.4915, "mrl_r10": 0.0270, "ndcg": 0.3564,
                  "bsr": 0.5876, "color": C["qwen06b"]},
}

# Per-Bloom dims for each backbone [Rem, Und, App, Ana, Eva, Cre]
BLOOM_DIMS = {
    "e5-large":  [468, 447, 474, 486, 495, 510],
    "bge-large": [362, 364, 394, 395, 441, 417],
    "arctic":    [298, 299, 302, 308, 307, 311],
    "roberta":   [272, 272, 271, 275, 271, 272],
    "qwen06b":   [354, 356, 371, 372, 392, 388],
}

# Per-Bloom R@10 for BAM-PQ and MRL (at full 768 dims)
BLOOM_R10_BAMPQ = {
    "e5-large":  [0.6352, 0.5466, 0.4031, 0.5365, 0.3846, 0.4936],
    "bge-large": [0.6324, 0.5675, 0.3984, 0.5365, 0.3846, 0.5353],
    "arctic":    [0.6105, 0.5161, 0.3500, 0.4922, 0.3279, 0.4551],
    "roberta":   [0.4432, 0.3633, 0.3063, 0.3906, 0.3117, 0.3750],
    "qwen06b":   [0.6050, 0.5338, 0.3984, 0.5365, 0.3846, 0.4904],
}
BLOOM_R10_MRL = {
    "e5-large":  [0.5665, 0.5048, 0.3187, 0.4870, 0.3239, 0.4359],
    "bge-large": [0.5930, 0.5193, 0.3328, 0.4818, 0.2955, 0.4295],
    "arctic":    [0.5600, 0.5016, 0.3094, 0.4661, 0.3117, 0.4103],
    "roberta":   [0.1697, 0.1254, 0.0859, 0.1302, 0.0850, 0.1218],
    "qwen06b":   [0.0362, 0.0257, 0.0203, 0.0234, 0.0202, 0.0192],
}

# Mask cosine similarity matrices (off-diagonal mean)
MASK_SIM = {
    "e5-large": np.array([
        [1.000, 0.447, 0.466, 0.472, 0.486, 0.485],
        [0.447, 1.000, 0.481, 0.451, 0.507, 0.480],
        [0.466, 0.481, 1.000, 0.496, 0.455, 0.469],
        [0.472, 0.451, 0.496, 1.000, 0.485, 0.494],
        [0.486, 0.507, 0.455, 0.485, 1.000, 0.504],
        [0.485, 0.480, 0.469, 0.494, 0.504, 1.000],
    ]),
    "qwen06b": np.array([
        [1.000, 0.647, 0.637, 0.606, 0.653, 0.637],
        [0.647, 1.000, 0.646, 0.629, 0.654, 0.638],
        [0.637, 0.646, 1.000, 0.651, 0.648, 0.651],
        [0.606, 0.629, 0.651, 1.000, 0.627, 0.630],
        [0.653, 0.654, 0.648, 0.627, 1.000, 0.674],
        [0.637, 0.638, 0.651, 0.630, 0.674, 1.000],
    ]),
    "roberta": np.array([
        [1.000, 0.597, 0.585, 0.585, 0.601, 0.553],
        [0.597, 1.000, 0.592, 0.599, 0.602, 0.566],
        [0.585, 0.592, 1.000, 0.620, 0.606, 0.626],
        [0.585, 0.599, 0.620, 1.000, 0.594, 0.580],
        [0.601, 0.602, 0.606, 0.594, 1.000, 0.626],
        [0.553, 0.566, 0.626, 0.580, 0.626, 1.000],
    ]),
}

# Dynamically load bge and arctic mask sim if available
import json as _json
_MASK_SIM_PATHS = {
    "bge":    "results/bloom_analysis/bge_large/mask_specialization_table.json",
    "arctic": "results/bloom_analysis/arctic/mask_specialization_table.json",
}
_MASK_SIM_DISPLAY = {"bge": "bge-large", "arctic": "arctic"}
for _bk, _path in _MASK_SIM_PATHS.items():
    if os.path.exists(_path):
        with open(_path) as _f:
            _d = _json.load(_f)
        MASK_SIM[_MASK_SIM_DISPLAY[_bk]] = np.array(_d["cosine_similarity"])

# Kendall tau
KENDALL = {
    "e5-large":  {"tau": 0.867, "p": 0.011},
    "bge-large": {"tau": 0.733, "p": 0.041},
    "arctic":    {"tau": 0.600, "p": 0.136},
    "qwen06b":   {"tau": 0.467, "p": 0.272},
    "roberta":   {"tau":-0.200, "p": 0.719},
}

CORPUS_SIZE = 40_640
FLOAT32_BYTES = 4
FULL_DIMS = 768


# ── Helper: save figure ───────────────────────────────────────────────────────
def savefig(fig, path, dpi=150):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  Saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Efficiency curves — R@10 vs dim budget + BAM-PQ operating points
# ─────────────────────────────────────────────────────────────────────────────
def plot_efficiency_curves(out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # MRL curve
    ax.plot(MRL_CURVE_DIMS, MRL_CURVE_R10, "o-", color=C["mrl"],
            linewidth=2, markersize=5, label="MRL (e5-large, prefix trunc)", zorder=3)

    # BAM-B reference point
    ax.scatter([463], [0.5337], marker="D", s=120, color=C["bam_b"],
               zorder=5, label="BAM-B (e5-large)")

    # BAM-PQ operating points
    for name, v in BACKBONES.items():
        ax.scatter([v["dims"]], [v["pq_r10"]],
                   marker="*", s=220, color=v["color"], zorder=5,
                   label=f"BAM-PQ ({name})")
        ax.annotate(f"  {name}\n  {v['dims']}d / {v['pq_r10']:.3f}",
                    (v["dims"], v["pq_r10"]),
                    fontsize=7.5, va="center", color=v["color"])

    # Horizontal ref: MRL@1024 (≈best MRL)
    ax.axhline(0.4669, ls="--", lw=1, color=C["mrl"], alpha=0.5)
    ax.text(70, 0.470, "MRL@1024 best", fontsize=8, color=C["mrl"], alpha=0.7)

    ax.set_xlabel("Active Embedding Dimensions", fontsize=12)
    ax.set_ylabel("R@10", fontsize=12)
    ax.set_title("R@10 vs Dimension Budget\n(Educational dataset, N=3296 queries)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(50, 800)
    ax.set_ylim(0.33, 0.58)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "01_efficiency_curves.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2a: Storage efficiency — index size in MB
# ─────────────────────────────────────────────────────────────────────────────
def plot_storage_efficiency(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    names  = list(BACKBONES.keys())
    dims   = [BACKBONES[n]["dims"] for n in names]
    colors = [BACKBONES[n]["color"] for n in names]

    # Full 768-dim index size (MRL baseline — all models share same corpus size)
    full_mb  = CORPUS_SIZE * FULL_DIMS * FLOAT32_BYTES / 1e6   # ~125 MB
    bampq_mb = [CORPUS_SIZE * d * FLOAT32_BYTES / 1e6 for d in dims]
    savings  = [100 * (1 - d / FULL_DIMS) for d in dims]

    # Left: absolute sizes
    ax = axes[0]
    x = np.arange(len(names))
    bars = ax.bar(x, bampq_mb, color=colors, alpha=0.85, width=0.5, label="BAM-PQ index")
    ax.axhline(full_mb, ls="--", lw=1.8, color=C["mrl"], label=f"MRL@768 ({full_mb:.0f} MB)")

    for bar, mb, dim in zip(bars, bampq_mb, dims):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{mb:.0f} MB\n({dim}d)", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Embedding Index Size (MB)", fontsize=11)
    ax.set_title("Storage: Embedding Index Size\n(Corpus = 40,640 docs, float32)",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, full_mb * 1.3)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Right: compression %  + R@10 lift scatter
    ax2 = axes[1]
    r10_lifts = [BACKBONES[n]["pq_r10"] - BACKBONES[n]["mrl_r10"] for n in names]

    sc = ax2.scatter(savings, r10_lifts, c=colors, s=180, zorder=5, edgecolors="white", lw=1.5)
    for i, n in enumerate(names):
        ax2.annotate(n, (savings[i], r10_lifts[i]),
                     textcoords="offset points", xytext=(6, 4),
                     fontsize=8.5, color=colors[i])

    ax2.axhline(0, ls="-", lw=0.8, color="black", alpha=0.4)
    ax2.set_xlabel("Storage Saved vs MRL@768 (%)", fontsize=11)
    ax2.set_ylabel("ΔR@10 (BAM-PQ − MRL@768)", fontsize=11)
    ax2.set_title("Storage Saving vs Retrieval Gain\n(top-right = best trade-off)",
                  fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "02_storage_efficiency.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2b: Inference time efficiency
# ─────────────────────────────────────────────────────────────────────────────
def plot_time_efficiency(out_dir):
    """
    Dot-product retrieval time scales as O(N * D).
    Relative latency = avg_dims / 768.
    We also show the quality-per-latency (R@10 / relative_latency).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    names  = list(BACKBONES.keys())
    dims   = np.array([BACKBONES[n]["dims"] for n in names])
    r10    = np.array([BACKBONES[n]["pq_r10"] for n in names])
    mrl_r10= np.array([BACKBONES[n]["mrl_r10"] for n in names])
    colors = [BACKBONES[n]["color"] for n in names]

    rel_lat    = dims / FULL_DIMS          # relative dot-product cost
    mrl_lat    = np.ones(len(names))       # MRL@768 = 1.0 (baseline)
    qpl_bampq  = r10 / rel_lat             # quality per unit latency
    qpl_mrl    = mrl_r10 / mrl_lat

    # Left: relative latency bar + R@10 overlay
    ax = axes[0]
    x = np.arange(len(names))
    w = 0.35
    b1 = ax.bar(x - w/2, mrl_lat, w, color=C["mrl"], alpha=0.7, label="MRL@768 latency")
    b2 = ax.bar(x + w/2, rel_lat, w, color=colors, alpha=0.85, label="BAM-PQ latency")

    ax2 = ax.twinx()
    ax2.plot(x - w/2, mrl_r10, "s--", color=C["mrl"], markersize=7, lw=1.5, label="MRL R@10")
    ax2.plot(x + w/2, r10,     "o-",  color="black",  markersize=7, lw=1.5, label="BAM-PQ R@10")
    ax2.set_ylabel("R@10", fontsize=10, color="black")
    ax2.set_ylim(0, 0.75)

    for i in range(len(names)):
        ax.text(x[i] + w/2, rel_lat[i] + 0.01, f"{rel_lat[i]:.2f}×",
                ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Relative Retrieval Latency (1.0 = MRL@768)", fontsize=10)
    ax.set_ylim(0, 1.35)
    ax.set_title("Inference Latency vs Quality\n(latency ∝ active dims / 768)",
                 fontsize=11, fontweight="bold")
    lines1, lbs1 = ax.get_legend_handles_labels()
    lines2, lbs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbs1 + lbs2, fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Right: quality-per-latency improvement
    ax3 = axes[1]
    qpl_ratio = qpl_bampq / np.where(qpl_mrl > 0, qpl_mrl, 1e-6)

    bars = ax3.bar(x, qpl_ratio, color=colors, alpha=0.85, width=0.5)
    ax3.axhline(1.0, ls="--", lw=1.5, color=C["mrl"], label="MRL@768 baseline (1.0×)")
    for bar, v in zip(bars, qpl_ratio):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{v:.2f}×", ha="center", fontsize=9, fontweight="bold")

    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax3.set_ylabel("Quality-per-Latency vs MRL@768 (higher=better)", fontsize=10)
    ax3.set_title("Quality / Latency Ratio\n(R@10 per unit dot-product cost)",
                  fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(axis="y", alpha=0.3)
    ax3.set_ylim(0, max(qpl_ratio) * 1.25)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "03_time_efficiency.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Mask cosine similarity heatmaps
# ─────────────────────────────────────────────────────────────────────────────
def plot_mask_similarity(out_dir):
    available = [k for k in ["e5-large", "bge-large", "arctic", "roberta", "qwen06b"]
                 if k in MASK_SIM]
    n = len(available)
    if n == 0:
        print("  No mask similarity data — skipping heatmaps.")
        return

    cmap = LinearSegmentedColormap.from_list(
        "spec", ["#1e40af", "#3b82f6", "#bfdbfe", "#fef3c7", "#fca5a5", "#dc2626"])

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
    if n == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]

    for idx, name in enumerate(available):
        ax = axes[idx // ncols][idx % ncols]
        mat = MASK_SIM[name]
        off_diag = mat[np.triu_indices(6, k=1)]
        mean_sim = off_diag.mean()

        im = ax.imshow(mat, vmin=0.4, vmax=1.0, cmap=cmap, aspect="equal")
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(BLOOM_NAMES, rotation=40, ha="right", fontsize=8)
        ax.set_yticklabels(BLOOM_NAMES, fontsize=8)
        ax.set_title(f"BAM-PQ ({name})\nmean off-diag = {mean_sim:.3f}",
                     fontsize=10, fontweight="bold")

        for i in range(6):
            for j in range(6):
                val = mat[i, j]
                color = "white" if val < 0.65 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7.5, color=color)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Pairwise Mask Cosine Similarity per Bloom Level\n"
                 "(lower off-diagonal = more dimension specialization per level)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "04_mask_similarity_heatmaps.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Kendall tau + dim spread — cognitive ordering
# ─────────────────────────────────────────────────────────────────────────────
def plot_cognitive_ordering(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: per-Bloom dim allocation lines
    ax = axes[0]
    for name, bk in BACKBONES.items():
        dims = BLOOM_DIMS[name]
        ax.plot(range(6), dims, "o-", color=bk["color"], lw=2,
                markersize=7, label=name)
        ax.annotate(f"{name}", (5, dims[5]),
                    textcoords="offset points", xytext=(4, 0),
                    fontsize=8, color=bk["color"], va="center")

    ax.set_xticks(range(6))
    ax.set_xticklabels(BLOOM_NAMES, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Active Embedding Dimensions", fontsize=11)
    ax.set_title("Per-Bloom Dimension Allocation\n(cognitive ordering hypothesis: monotone ↑)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=8.5, loc="upper left")

    # Right: Kendall tau bar chart colored by significance
    ax2 = axes[1]
    k_names = list(KENDALL.keys())
    k_taus  = [KENDALL[n]["tau"] for n in k_names]
    k_ps    = [KENDALL[n]["p"]   for n in k_names]
    k_cols  = [BACKBONES[n]["color"] for n in k_names]
    k_alpha = [1.0 if p < 0.05 else 0.45 for p in k_ps]

    x = np.arange(len(k_names))
    bars = ax2.bar(x, k_taus, color=k_cols, width=0.5)
    for bar, alp in zip(bars, k_alpha):
        bar.set_alpha(float(alp))
    ax2.axhline(0, ls="-", lw=1, color="black", alpha=0.5)
    ax2.axhline(0.5, ls="--", lw=1, color="gray", alpha=0.5, label="τ=0.5 reference")

    for bar, tau, p, name in zip(bars, k_taus, k_ps, k_names):
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        yoff = 0.03 if tau >= 0 else -0.06
        ax2.text(bar.get_x() + bar.get_width()/2, tau + yoff,
                 f"{tau:+.3f}\n({sig})", ha="center", fontsize=8.5, fontweight="bold",
                 color=BACKBONES[name]["color"])

    ax2.set_xticks(x)
    ax2.set_xticklabels(k_names, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Kendall's τ  (Bloom level vs active dims)", fontsize=11)
    ax2.set_title("Cognitive Ordering: Kendall's τ\n(positive + significant → dims increase with Bloom level)",
                  fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.5, 1.15)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    # Add annotation for sig threshold
    ax2.text(len(k_names) - 0.5, 0.05, "solid = p<0.05\nfaded = not sig",
             fontsize=8, ha="right", style="italic", color="gray")

    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "05_cognitive_ordering.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Per-Bloom R@10 improvements
# ─────────────────────────────────────────────────────────────────────────────
def plot_bloom_improvements(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: absolute R@10 per Bloom for strong models
    ax = axes[0]
    strong = ["e5-large", "bge-large", "arctic"]
    x = np.arange(6)
    w = 0.13
    offsets = np.linspace(-w * len(strong)/2, w * len(strong)/2, len(strong))

    for i, name in enumerate(strong):
        bk_color = BACKBONES[name]["color"]
        pq  = BLOOM_R10_BAMPQ[name]
        mrl = BLOOM_R10_MRL[name]
        ax.bar(x + offsets[i], pq,  w, color=bk_color, alpha=0.85, label=f"BAM-PQ {name}")
        ax.bar(x + offsets[i], mrl, w, color=bk_color, alpha=0.3,
               hatch="///", label=f"MRL {name}" if i == 0 else "")

    ax.set_xticks(x)
    ax.set_xticklabels(BLOOM_NAMES, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("R@10", fontsize=11)
    ax.set_title("Per-Bloom R@10: BAM-PQ vs MRL\n(strong backbones; hatched = MRL@768)",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.75)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2)

    # Right: delta heatmap (BAM-PQ - MRL) across all backbones
    ax2 = axes[1]
    names  = list(BACKBONES.keys())
    deltas = np.array([[BLOOM_R10_BAMPQ[n][b] - BLOOM_R10_MRL[n][b]
                        for b in range(6)] for n in names])

    vmax = np.abs(deltas).max()
    im = ax2.imshow(deltas, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax2.set_xticks(range(6)); ax2.set_yticks(range(len(names)))
    ax2.set_xticklabels(BLOOM_NAMES, rotation=20, ha="right", fontsize=9)
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_title("ΔR@10 Heatmap (BAM-PQ − MRL@768)\nper Bloom level × backbone",
                  fontsize=11, fontweight="bold")

    for i in range(len(names)):
        for j in range(6):
            v = deltas[i, j]
            color = "white" if abs(v) > vmax * 0.6 else "black"
            ax2.text(j, i, f"{v:+.3f}", ha="center", va="center",
                     fontsize=8, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="ΔR@10")
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "06_bloom_improvements.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6: Summary overview — 2×3 composite
# ─────────────────────────────────────────────────────────────────────────────
def plot_summary_overview(out_dir):
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    names  = list(BACKBONES.keys())
    colors = [BACKBONES[n]["color"] for n in names]
    dims   = np.array([BACKBONES[n]["dims"] for n in names])
    pq_r10 = np.array([BACKBONES[n]["pq_r10"] for n in names])
    mrl_r10= np.array([BACKBONES[n]["mrl_r10"] for n in names])
    comp   = dims / FULL_DIMS

    # (0,0) R@10 comparison bars
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(names))
    ax.bar(x, mrl_r10, 0.4, color=C["mrl"],  alpha=0.6, label="MRL@768")
    ax.bar(x, pq_r10,  0.4, color=colors,    alpha=0.85, label="BAM-PQ")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("R@10"); ax.set_title("Overall R@10", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # (0,1) compression ratio
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(x, (1 - comp) * 100, color=colors, alpha=0.85)
    for i, v in enumerate((1 - comp) * 100):
        ax.text(i, v + 0.5, f"{v:.0f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Dims Saved (%)"); ax.set_title("Storage Compression vs MRL@768", fontweight="bold")
    ax.set_ylim(0, 75); ax.grid(axis="y", alpha=0.3)

    # (0,2) quality-per-latency
    ax = fig.add_subplot(gs[0, 2])
    qpl_bampq = pq_r10 / comp
    qpl_mrl   = np.where(mrl_r10 > 0, mrl_r10, 1e-6)
    ratio = qpl_bampq / qpl_mrl
    bars = ax.bar(x, ratio, color=colors, alpha=0.85)
    ax.axhline(1.0, ls="--", lw=1.5, color=C["mrl"])
    for bar, v in zip(bars, ratio):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.05, f"{v:.2f}×",
                ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Quality/Latency ratio"); ax.set_title("Quality-per-Latency vs MRL", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # (1,0) cognitive ordering (Kendall tau)
    ax = fig.add_subplot(gs[1, 0])
    k_names = list(KENDALL.keys())
    k_taus  = [KENDALL[n]["tau"] for n in k_names]
    k_ps    = [KENDALL[n]["p"]   for n in k_names]
    k_cols  = [BACKBONES[n]["color"] for n in k_names]
    k_alpha = [1.0 if p < 0.05 else 0.4 for p in k_ps]
    xi = np.arange(len(k_names))
    for i, (tau, col, alp) in enumerate(zip(k_taus, k_cols, k_alpha)):
        ax.bar(i, tau, color=col, alpha=float(alp), width=0.5)
    ax.axhline(0, lw=1, color="black", alpha=0.4)
    ax.set_xticks(xi); ax.set_xticklabels(k_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Kendall's τ"); ax.set_title("Cognitive Ordering (τ)", fontweight="bold")
    ax.set_ylim(-0.5, 1.1); ax.grid(axis="y", alpha=0.3)

    # (1,1) mask similarity summary bar
    ax = fig.add_subplot(gs[1, 1])
    sim_names = [n for n in names if n in MASK_SIM]
    sim_means = []
    sim_colors= []
    for n in sim_names:
        mat = MASK_SIM[n]
        off = mat[np.triu_indices(6, k=1)]
        sim_means.append(off.mean())
        sim_colors.append(BACKBONES[n]["color"])
    xi2 = np.arange(len(sim_names))
    bars2 = ax.bar(xi2, sim_means, color=sim_colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars2, sim_means):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.003, f"{v:.3f}",
                ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(xi2); ax.set_xticklabels(sim_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean Mask Cosine Sim.\n(lower = more specialized)")
    ax.set_title("Per-Bloom Mask Specialization", fontweight="bold")
    ax.set_ylim(0.3, 0.85); ax.grid(axis="y", alpha=0.3)

    # (1,2) efficiency curve (compact)
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(MRL_CURVE_DIMS, MRL_CURVE_R10, "o-", color=C["mrl"], lw=1.5,
            markersize=4, label="MRL (e5)")
    for name, v in BACKBONES.items():
        ax.scatter([v["dims"]], [v["pq_r10"]], marker="*", s=150,
                   color=v["color"], zorder=5)
        ax.annotate(name[:3], (v["dims"], v["pq_r10"]),
                    textcoords="offset points", xytext=(3, 2),
                    fontsize=7, color=v["color"])
    ax.set_xlabel("Dims"); ax.set_ylabel("R@10")
    ax.set_title("Efficiency Curve", fontweight="bold")
    ax.set_xlim(50, 800); ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle("BAM-PQ Results Overview — Educational Dataset",
                 fontsize=15, fontweight="bold", y=1.01)
    savefig(fig, os.path.join(out_dir, "00_summary_overview.png"))


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/plots")
    args = parser.parse_args()

    out = args.output_dir
    print(f"\nGenerating plots → {out}/")

    plot_summary_overview(out)
    plot_efficiency_curves(out)
    plot_storage_efficiency(out)
    plot_time_efficiency(out)
    plot_mask_similarity(out)
    plot_cognitive_ordering(out)
    plot_bloom_improvements(out)

    print(f"\nDone. {len(os.listdir(out))} files in {out}/")


if __name__ == "__main__":
    main()
