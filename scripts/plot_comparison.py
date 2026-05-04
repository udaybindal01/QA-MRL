"""
Comparison plots: Standard FT vs MRL vs BAM-PQ
Reads from results/standard_ft/educational/summary.json
         and hardcoded BAM-PQ / MRL results.

Usage:
    python scripts/plot_comparison.py
    python scripts/plot_comparison.py --std_ft_dir results/standard_ft/educational
    python scripts/plot_comparison.py --output_dir results/plots/comparison
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ── Palette ───────────────────────────────────────────────────────────────────
CLR = {
    "std_ft":  "#F97316",   # orange
    "mrl":     "#6B7280",   # gray
    "bam_pq":  "#2563EB",   # blue
    "bam_b":   "#0891B2",   # teal
}
BLOOM_COLORS = ["#6366F1","#06B6D4","#10B981","#F59E0B","#EF4444","#8B5CF6"]
BLOOM_NAMES  = ["Remember","Understand","Apply","Analyze","Evaluate","Create"]

BACKBONES = ["e5large","bge","arctic","roberta","qwen06b"]
BK_LABELS = {"e5large":"e5-large","bge":"bge-large",
             "arctic":"arctic","roberta":"roberta","qwen06b":"qwen06b"}

# ── Known BAM-PQ + MRL results ────────────────────────────────────────────────
BAMPQ_R10  = {"e5large":0.5297,"bge":0.5358,"arctic":0.4548,"roberta":0.3908,"qwen06b":0.4915}
BAMPQ_NDCG = {"e5large":0.4070,"bge":0.4063,"arctic":0.3586,"roberta":0.2982,"qwen06b":0.3564}
BAMPQ_DIMS = {"e5large":473,   "bge":384,   "arctic":302,   "roberta":272,   "qwen06b":366}

MRL_R10    = {"e5large":0.4669,"bge":0.4779,"arctic":0.4508,"roberta":0.1374,"qwen06b":0.0270}
MRL_NDCG   = {"e5large":0.3603,"bge":0.3710,"arctic":0.3545,"roberta":0.0945,"qwen06b":0.0183}

BAMPQ_BLOOM_R10 = {
    "e5large": [0.6352,0.5466,0.4031,0.5365,0.3846,0.4936],
    "bge":     [0.6324,0.5675,0.3984,0.5365,0.3846,0.5353],
    "arctic":  [0.6105,0.5161,0.3500,0.4922,0.3279,0.4551],
    "roberta": [0.4432,0.3633,0.3063,0.3906,0.3117,0.3750],
    "qwen06b": [0.6050,0.5338,0.3984,0.5365,0.3846,0.4904],
}
MRL_BLOOM_R10 = {
    "e5large": [0.5665,0.5048,0.3187,0.4870,0.3239,0.4359],
    "bge":     [0.5930,0.5193,0.3328,0.4818,0.2955,0.4295],
    "arctic":  [0.5600,0.5016,0.3094,0.4661,0.3117,0.4103],
    "roberta": [0.1697,0.1254,0.0859,0.1302,0.0850,0.1218],
    "qwen06b": [0.0362,0.0257,0.0203,0.0234,0.0202,0.0192],
}


def savefig(fig, path, dpi=150):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  Saved → {path}")
    plt.close(fig)


def load_std_ft(std_ft_dir):
    """Load standard FT summary. Returns dict or empty dict if missing."""
    p = os.path.join(std_ft_dir, "summary.json")
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    print(f"  WARNING: {p} not found — plotting without standard FT results")
    return {}


# ── Plot 1: Overall R@10 — 3-way comparison bars ─────────────────────────────
def plot_overall_comparison(std_ft, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    bks   = BACKBONES
    xlabs = [BK_LABELS[b] for b in bks]
    x     = np.arange(len(bks))
    w     = 0.25

    for ax_idx, (metric, title, bampq_d, mrl_d) in enumerate([
        ("r10",    "Recall@10",  BAMPQ_R10,  MRL_R10),
        ("ndcg10", "NDCG@10",   BAMPQ_NDCG, MRL_NDCG),
    ]):
        ax = axes[ax_idx]

        mrl_vals   = [mrl_d[b] for b in bks]
        bampq_vals = [bampq_d[b] for b in bks]
        std_vals   = [std_ft.get(b, {}).get(metric) for b in bks]
        has_std    = any(v is not None for v in std_vals)

        if has_std:
            offset = [-w, 0, w]
            ax.bar(x + offset[0], mrl_vals,  w, color=CLR["mrl"],   alpha=0.85, label="MRL@768")
            ax.bar(x + offset[1], std_vals,  w, color=CLR["std_ft"],alpha=0.85, label="Standard FT",
                   hatch="")
            ax.bar(x + offset[2], bampq_vals,w, color=CLR["bam_pq"],alpha=0.85, label="BAM-PQ")
        else:
            ax.bar(x - w/2, mrl_vals,   w, color=CLR["mrl"],   alpha=0.85, label="MRL@768")
            ax.bar(x + w/2, bampq_vals, w, color=CLR["bam_pq"],alpha=0.85, label="BAM-PQ")

        # Delta annotations on BAM-PQ bars
        for i, (bk, bpq, mrl) in enumerate(zip(bks, bampq_vals, mrl_vals)):
            delta = bpq - mrl
            off   = offset[2] if has_std else w/2
            ax.text(x[i] + off, bpq + 0.005, f"+{delta:.3f}",
                    ha="center", fontsize=7, color=CLR["bam_pq"], fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(xlabs, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f"{title} — Standard FT vs MRL vs BAM-PQ", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(bampq_vals) * 1.2)

    fig.suptitle("Overall Retrieval Performance — Educational Benchmark",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "C1_overall_comparison.png"))


# ── Plot 2: Per-Bloom R@10 — 3-way for strong backbones ──────────────────────
def plot_per_bloom(std_ft, out_dir):
    strong = ["e5large", "bge"]
    fig, axes = plt.subplots(1, len(strong), figsize=(7 * len(strong), 5.5))
    if len(strong) == 1:
        axes = [axes]

    for ax, bk in zip(axes, strong):
        x   = np.arange(6)
        w   = 0.25
        pq  = BAMPQ_BLOOM_R10[bk]
        mrl = MRL_BLOOM_R10[bk]
        std = [std_ft.get(bk, {}).get("per_bloom_r10", {}).get(bl) for bl in BLOOM_NAMES]
        has_std = any(v is not None for v in std)

        if has_std:
            ax.bar(x - w, mrl, w, color=CLR["mrl"],   alpha=0.8,  label="MRL@768")
            ax.bar(x,     std, w, color=CLR["std_ft"], alpha=0.85, label="Standard FT")
            ax.bar(x + w, pq,  w, color=CLR["bam_pq"],alpha=0.85, label="BAM-PQ")
        else:
            ax.bar(x - w/2, mrl, w, color=CLR["mrl"],   alpha=0.8,  label="MRL@768")
            ax.bar(x + w/2, pq,  w, color=CLR["bam_pq"],alpha=0.85, label="BAM-PQ")

        ax.set_xticks(x)
        ax.set_xticklabels(BLOOM_NAMES, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Recall@10", fontsize=11)
        ax.set_title(f"Per-Bloom R@10 — {BK_LABELS[bk]}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 0.80)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-Bloom Level Recall@10 — Standard FT vs MRL vs BAM-PQ",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "C2_per_bloom_comparison.png"))


# ── Plot 3: Gain over Standard FT heatmap ────────────────────────────────────
def plot_gain_heatmap(std_ft, out_dir):
    if not std_ft:
        print("  Skipping gain heatmap — no standard FT data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (label, bampq_d, mrl_d) in enumerate([
        ("BAM-PQ − Standard FT  (R@10)", BAMPQ_R10, None),
        ("BAM-PQ − MRL  (R@10)",         BAMPQ_R10, MRL_R10),
    ]):
        ax = axes[ax_idx]

        if mrl_d is None:
            # BAM-PQ vs std FT
            deltas = np.array([
                [BAMPQ_BLOOM_R10[bk][b] -
                 (std_ft.get(bk, {}).get("per_bloom_r10", {}).get(BLOOM_NAMES[b]) or 0)
                 for b in range(6)] for bk in BACKBONES
            ])
        else:
            deltas = np.array([
                [BAMPQ_BLOOM_R10[bk][b] - MRL_BLOOM_R10[bk][b]
                 for b in range(6)] for bk in BACKBONES
            ])

        vmax = max(abs(deltas).max(), 0.05)
        im = ax.imshow(deltas, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

        ax.set_xticks(range(6)); ax.set_yticks(range(len(BACKBONES)))
        ax.set_xticklabels(BLOOM_NAMES, rotation=20, ha="right", fontsize=9)
        ax.set_yticklabels([BK_LABELS[b] for b in BACKBONES], fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")

        for i in range(len(BACKBONES)):
            for j in range(6):
                v = deltas[i, j]
                color = "white" if abs(v) > vmax * 0.6 else "black"
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ΔR@10")

    fig.suptitle("Per-Bloom ΔR@10 Heatmaps — Educational Benchmark",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "C3_gain_heatmap.png"))


# ── Plot 4: Dims vs R@10 — efficiency frontier ───────────────────────────────
def plot_efficiency_frontier(std_ft, out_dir):
    fig, ax = plt.subplots(figsize=(9, 6))

    # MRL truncation curve (e5-large)
    mrl_dims = [64,128,192,256,320,384,448,512,576,640,704,768]
    mrl_r10  = [0.3844,0.4278,0.4381,0.4463,0.4490,0.4536,0.4536,
                0.4557,0.4609,0.4606,0.4663,0.4639]
    ax.plot(mrl_dims, mrl_r10, "o-", color=CLR["mrl"], lw=1.8,
            markersize=4, label="MRL (e5-large, prefix trunc)", zorder=3)

    bk_colors = {"e5large":"#2563EB","bge":"#16A34A","arctic":"#D97706",
                 "roberta":"#DC2626","qwen06b":"#7C3AED"}

    for bk in BACKBONES:
        col = bk_colors[bk]
        # BAM-PQ point
        ax.scatter(BAMPQ_DIMS[bk], BAMPQ_R10[bk], marker="*",
                   s=220, color=col, zorder=5, label=f"BAM-PQ ({BK_LABELS[bk]})")
        ax.annotate(f"BAM-PQ\n{BK_LABELS[bk]}",
                    (BAMPQ_DIMS[bk], BAMPQ_R10[bk]),
                    textcoords="offset points", xytext=(6, 3),
                    fontsize=7, color=col)

        # Standard FT point (at full 768)
        sft_r10 = std_ft.get(bk, {}).get("r10")
        if sft_r10:
            ax.scatter(768, sft_r10, marker="s", s=80, color=col,
                       alpha=0.6, zorder=4)
            ax.annotate(f"StdFT\n{BK_LABELS[bk]}",
                        (768, sft_r10),
                        textcoords="offset points", xytext=(6, -10),
                        fontsize=6.5, color=col, alpha=0.8)

        # MRL full point
        ax.scatter(768, MRL_R10[bk], marker="D", s=60, color=col,
                   alpha=0.5, zorder=4)

    ax.axhline(0.4669, ls="--", lw=1, color=CLR["mrl"], alpha=0.4)
    ax.text(70, 0.471, "MRL@768 (e5)", fontsize=7.5, color=CLR["mrl"], alpha=0.6)
    ax.set_xlabel("Active Embedding Dimensions", fontsize=12)
    ax.set_ylabel("Recall@10", fontsize=12)
    ax.set_title("Efficiency Frontier — R@10 vs Dimension Budget\n"
                 "(★ = BAM-PQ, ■ = Standard FT@768, ◆ = MRL@768)",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(50, 850)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=7.5, loc="lower right", ncol=2)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "C4_efficiency_frontier.png"))


# ── Plot 5: Standard FT vs MRL vs BAM-PQ — dim-normalized quality ────────────
def plot_quality_per_dim(std_ft, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    x  = np.arange(len(BACKBONES))
    w  = 0.25
    xlabs = [BK_LABELS[b] for b in BACKBONES]

    mrl_qpd   = [MRL_R10[b] / 1.0          for b in BACKBONES]  # MRL at full 768
    bampq_qpd = [BAMPQ_R10[b] / (BAMPQ_DIMS[b]/768) for b in BACKBONES]

    std_qpd = []
    for b in BACKBONES:
        sft = std_ft.get(b, {}).get("r10")
        std_qpd.append(sft / 1.0 if sft else None)

    has_std = any(v is not None for v in std_qpd)

    if has_std:
        ax.bar(x - w, mrl_qpd,  w, color=CLR["mrl"],   alpha=0.85, label="MRL@768")
        ax.bar(x,     [v or 0 for v in std_qpd], w,
               color=CLR["std_ft"], alpha=0.85, label="Standard FT@768")
        ax.bar(x + w, bampq_qpd, w, color=CLR["bam_pq"],alpha=0.85,
               label="BAM-PQ (R@10 / rel_latency)")
    else:
        ax.bar(x - w/2, mrl_qpd,   w, color=CLR["mrl"],   alpha=0.85, label="MRL@768")
        ax.bar(x + w/2, bampq_qpd, w, color=CLR["bam_pq"],alpha=0.85,
               label="BAM-PQ (R@10 / rel_latency)")

    for i, v in enumerate(bampq_qpd):
        off = w if has_std else w/2
        ax.text(x[i] + off, v + 0.005, f"{v:.3f}", ha="center",
                fontsize=8, color=CLR["bam_pq"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabs, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("R@10 / Relative Latency", fontsize=11)
    ax.set_title("Quality-per-Latency  (BAM-PQ normalized by active dims / 768)\n"
                 "Standard FT and MRL shown at full 768 dims",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "C5_quality_per_dim.png"))


# ── Plot 6: Radar chart — per-Bloom profile for best backbone ─────────────────
def plot_radar(std_ft, out_dir, backbone="bge"):
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.patches as mpatches

    cats   = BLOOM_NAMES
    N      = len(cats)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    def add_line(vals, color, label, ls="-", lw=2, alpha=0.7):
        v = vals + vals[:1]
        ax.plot(angles, v, ls=ls, lw=lw, color=color, label=label)
        ax.fill(angles, v, color=color, alpha=0.08)

    add_line(MRL_BLOOM_R10[backbone],   CLR["mrl"],   "MRL@768",     ls="--")
    add_line(BAMPQ_BLOOM_R10[backbone], CLR["bam_pq"],"BAM-PQ",      ls="-")

    std_bloom = [std_ft.get(backbone, {}).get("per_bloom_r10", {}).get(bl, 0)
                 for bl in BLOOM_NAMES]
    if any(v > 0 for v in std_bloom):
        add_line(std_bloom, CLR["std_ft"], "Standard FT", ls="-.")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=10)
    ax.set_ylim(0, 0.75)
    ax.set_title(f"Per-Bloom R@10 Radar — {BK_LABELS[backbone]}",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, f"C6_radar_{backbone}.png"))


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--std_ft_dir",  default="results/standard_ft/educational")
    parser.add_argument("--output_dir",  default="results/plots/comparison")
    args = parser.parse_args()

    std_ft = load_std_ft(args.std_ft_dir)
    out    = args.output_dir
    print(f"\nGenerating comparison plots → {out}/")

    plot_overall_comparison(std_ft, out)
    plot_per_bloom(std_ft, out)
    plot_gain_heatmap(std_ft, out)
    plot_efficiency_frontier(std_ft, out)
    plot_quality_per_dim(std_ft, out)
    plot_radar(std_ft, out, backbone="bge")
    plot_radar(std_ft, out, backbone="e5large")

    print(f"\nDone. {len(os.listdir(out))} files in {out}/")


if __name__ == "__main__":
    main()
