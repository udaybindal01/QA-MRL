"""
BAM-PQ Architecture Diagram — landscape overview
Output: results/plots/bampq_architecture.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

C = dict(
    enc="#EBEBEB",   enc_e="#888888",
    bloom="#D6CCF0", bloom_e="#6B5BBE",
    res="#A8D8C5",   res_e="#2A9070",
    comb="#F5E0B0",  comb_e="#C8922A",
    mask="#B8D4EA",  mask_e="#3A7FB5",
    score="#EBEBEB", score_e="#888888",
    loss="#F0F0F0",  loss_e="#BBBBBB",
)

W, H = 20, 10.5
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.axis("off"); fig.patch.set_facecolor("white")


def box(cx, cy, w, h, fc, ec, lw=1.5, zorder=3):
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle="round,pad=0.1",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(p)

def txt(x, y, s, fs=9, bold=False, color="#111", ha="center", va="center",
        style="normal", zorder=5):
    ax.text(x, y, s, fontsize=fs, fontweight="bold" if bold else "normal",
            fontstyle=style, color=color, ha=ha, va=va, zorder=zorder)

def arr(x1, y1, x2, y2, color="#333", lw=1.5, rad=0.0, zorder=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}",
                                shrinkA=6, shrinkB=6), zorder=zorder)

def darr(x1, y1, x2, y2, color="#AAAAAA", lw=1.2, rad=0.0, zorder=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle="dashed",
                                connectionstyle=f"arc3,rad={rad}",
                                shrinkA=6, shrinkB=6), zorder=zorder)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 1  — query input + encoder
# ─────────────────────────────────────────────────────────────────────────────
txt(2.5,  9.9, "Q UERY   S IDE",   fs=8, color="#666")
txt(17.5, 9.9, "C ORPUS   ( OFFLINE )", fs=8, color="#666")

# Query text
box(1.5, 9.1, 2.4, 0.65, C["enc"], C["enc_e"])
txt(1.5, 9.1, "Query text", fs=10)

arr(2.7, 9.1, 3.6, 9.1, color=C["enc_e"])

# Transformer encoder
box(4.85, 9.1, 2.3, 0.8, C["enc"], C["enc_e"])
txt(4.85, 9.28, "Transformer", fs=9.5, bold=True)
txt(4.85, 9.02, "backbone encoder", fs=9.5, bold=True)
txt(4.85, 8.78, "frozen 5 ep → fine-tuned 15 ep", fs=7, color="#666")

# z label
arr(6.0, 9.1, 6.9, 9.1, color=C["enc_e"])
txt(6.7, 9.35, r"$\mathbf{z}\in\mathbb{R}^{768}$", fs=8.5, color="#333", ha="center")

# Corpus passage
box(18.5, 9.1, 2.4, 0.65, C["enc"], C["enc_e"])
txt(18.5, 9.1, "Passage text", fs=10)

arr(17.3, 9.1, 16.5, 9.1, color=C["enc_e"])

# Corpus encoder
box(15.3, 9.1, 2.2, 0.8, C["enc"], C["enc_e"])
txt(15.3, 9.28, "Transformer", fs=9.5, bold=True)
txt(15.3, 9.02, "backbone", fs=9.5, bold=True)
txt(15.3, 8.78, "no mask applied", fs=7, color="#666")

# d label + dashed arrow going down
darr(14.2, 9.1, 13.5, 9.1, color="#AAAAAA")
txt(13.7, 9.35, r"$\mathbf{d}\in\mathbb{R}^{768}$", fs=8.5, color="#888", ha="center")

# ─────────────────────────────────────────────────────────────────────────────
# DASHED MASK HEAD BORDER
# ─────────────────────────────────────────────────────────────────────────────
border = FancyBboxPatch((0.3, 1.5), 12.9, 7.0,
                        boxstyle="round,pad=0.1", facecolor="none",
                        edgecolor="#AAAAAA", linewidth=1.2, linestyle="dashed", zorder=1)
ax.add_patch(border)
txt(0.85, 8.55, "BAM-PQ mask head", fs=8, color="#888", ha="left")

# ─────────────────────────────────────────────────────────────────────────────
# ROW 2  — Bloom council / prior  +  Query residual  (side by side)
# ─────────────────────────────────────────────────────────────────────────────

# z splits: down-left to Bloom council, down-right to residual MLP
arr(7.0, 8.7, 2.8, 7.95, color=C["bloom_e"], rad=0.15)
arr(7.0, 8.7, 7.0, 7.95, color=C["res_e"],   rad=0.0)

# Bloom council
box(2.0, 7.5, 3.2, 0.80, C["bloom"], C["bloom_e"])
txt(2.0, 7.72, "Bloom council", fs=9.5, bold=True, color="#3D2D8A")
txt(2.0, 7.48, "DeBERTa + RoBERTa + BERT + SVM", fs=7.2, color="#444")
txt(2.0, 7.28, "85–88% label accuracy", fs=7.2, color="#444")

arr(2.0, 7.10, 2.0, 6.45, color=C["bloom_e"])
txt(2.45, 6.75, r"$b\in\{0,\ldots,5\}$", fs=8, color=C["bloom_e"], ha="left")

# Bloom prior
box(2.0, 6.1, 3.2, 0.62, C["bloom"], C["bloom_e"])
txt(2.0, 6.26, r"Bloom prior  $\mathbf{E}[b]$", fs=9.5, bold=True, color="#3D2D8A")
txt(2.0, 6.03, "6 learned level embeddings  ·  Kaiming init", fs=7.2, color="#444")

# Query residual
box(7.0, 7.5, 3.2, 0.80, C["res"], C["res_e"])
txt(7.0, 7.72, r"Query residual  $\Delta(\mathbf{z})$", fs=9.5, bold=True, color="#1A6E50")
txt(7.0, 7.48, r"2-layer MLP on norm$(\mathbf{z})$", fs=7.5, color="#444")
txt(7.0, 7.28, "*zero-init output layer", fs=7.2, color="#666", style="italic")

# ─────────────────────────────────────────────────────────────────────────────
# ROW 3  — Combine
# ─────────────────────────────────────────────────────────────────────────────

# Arrows from prior + residual → combine
arr(2.0, 5.79, 4.5, 5.22, color=C["bloom_e"], rad=-0.1)
arr(7.0, 7.10, 5.8, 5.22, color=C["res_e"],   rad=0.1)

box(4.55, 4.85, 6.2, 0.70, C["comb"], C["comb_e"])
txt(4.55, 5.04,
    r"$\boldsymbol{\ell} = \mathbf{E}[b] + \alpha\cdot\Delta(\mathbf{z})$",
    fs=13, bold=True, color="#6B4500")
txt(4.55, 4.73,
    r"$\alpha=\sigma(\alpha_{\mathrm{raw}})$,   $\alpha_{\mathrm{raw}}\ \mathrm{init}=-3.0$",
    fs=8, color="#888")

# ─────────────────────────────────────────────────────────────────────────────
# ROW 4  — Mask generation  +  masked query  (side by side)
# ─────────────────────────────────────────────────────────────────────────────
arr(4.55, 4.50, 4.55, 3.82, color=C["mask_e"])

# Mask generation
box(3.4, 3.48, 3.8, 0.60, C["mask"], C["mask_e"])
txt(3.4, 3.66, "Mask generation", fs=9.5, bold=True, color="#1C4E7A")
txt(3.4, 3.42,
    r"Train: STE  →  $\mathbf{m}\in\{0,1\}^{768}$  (scattered)",
    fs=7.3, color="#444")

arr(5.3, 3.48, 6.1, 3.48, color=C["mask_e"])

# Masked query
box(7.35, 3.48, 2.5, 0.60, C["mask"], C["mask_e"])
txt(7.35, 3.66,
    r"$\tilde{\mathbf{q}}=\mathrm{norm}(\mathbf{z}\odot\mathbf{m})$",
    fs=11, bold=True, color="#1C4E7A")
txt(7.35, 3.40, "avg active dims: 302–473 / 768", fs=7.3, color="#555")

# ─────────────────────────────────────────────────────────────────────────────
# ROW 5  — Score
# ─────────────────────────────────────────────────────────────────────────────
arr(8.6, 3.18, 10.5, 2.42, color=C["mask_e"], rad=-0.1)

# Dashed d arrow comes down from corpus side to score
darr(13.5, 8.8, 11.8, 2.42, color="#AAAAAA", rad=0.15)

box(10.85, 2.08, 5.2, 0.60, C["score"], C["score_e"])
txt(10.85, 2.22,
    r"score $= \tilde{\mathbf{q}}^{\top}\mathbf{d}$     FAISS inner product over full-dim corpus",
    fs=9.5, color="#222")

# BSR selection
arr(10.85, 1.78, 10.85, 1.35, color="#AAAAAA", lw=1.0)
box(10.85, 1.05, 5.2, 0.52, "#F7F7F7", "#BBBBBB", lw=1.0)
txt(10.85, 1.08,
    "Post-hoc BSR:  Quality × (1 + 0.5 × Efficiency)  ·  best epoch from full corpus eval",
    fs=7.5, color="#777")

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOSSES  — right panel
# ─────────────────────────────────────────────────────────────────────────────
box(17.0, 5.5, 5.6, 5.8, "#F9F9F9", "#CCCCCC", lw=1.0, zorder=1)
txt(17.0, 8.22, "Training losses", fs=9, bold=True, color="#333")

losses = [
    ("1.  InfoNCE",    r"Masked contrastive (class-weighted $1/\sqrt{\text{freq}}$)"),
    ("2.  Efficiency", r"Per-Bloom avg penalty  ·  cog. weights $1-b/6$"),
    ("3.  Diversity",  r"Mean pairwise $|\dim_i - \dim_j|$  across Bloom levels"),
    ("4.  MRL anchor", r"InfoNCE at fixed dims  ·  $D/\sqrt{d}$ weighting"),
]
ly = 7.78
for name, desc in losses:
    txt(14.35, ly, name, fs=8.2, bold=True, color="#333", ha="left")
    txt(14.35, ly - 0.28, desc, fs=7.5, color="#666", ha="left")
    ly -= 0.95

# ─────────────────────────────────────────────────────────────────────────────
# LEGEND
# ─────────────────────────────────────────────────────────────────────────────
items = [
    (C["bloom"], C["bloom_e"], "Bloom prior"),
    (C["res"],   C["res_e"],   "Query residual"),
    (C["comb"],  C["comb_e"],  "Combine + α"),
    (C["mask"],  C["mask_e"],  "Mask + masked query"),
    (C["score"], C["score_e"], "Encoder / score"),
]
lx = 0.6
for fc, ec, label in items:
    p = FancyBboxPatch((lx, 0.18), 0.44, 0.30,
                       boxstyle="round,pad=0.03",
                       facecolor=fc, edgecolor=ec, linewidth=1.3, zorder=5)
    ax.add_patch(p)
    txt(lx + 0.62, 0.33, label, fs=8.5, ha="left", color="#222")
    lx += 2.5

plt.tight_layout(pad=0.2)
out = "results/plots/bampq_architecture.png"
os.makedirs("results/plots", exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
plt.close(fig)
