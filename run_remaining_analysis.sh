#!/usr/bin/env bash
# =============================================================================
# run_remaining_analysis.sh  —  Run all pending post-training analyses
#
# What this does:
#   1. Mask cosine similarity for bge-large and arctic (need checkpoints)
#   2. Standard FT baseline training + eval (all 5 backbones)
#   3. Comparison plots (BAM-PQ vs MRL vs Standard FT)
#   4. Update plot 04 (mask similarity heatmaps) with bge/arctic results
#
# Usage:
#   ./run_remaining_analysis.sh                   # full pipeline
#   ./run_remaining_analysis.sh --skip_mask_sim   # skip mask similarity
#   ./run_remaining_analysis.sh --skip_std_ft     # skip standard FT
#   ./run_remaining_analysis.sh --skip_train      # skip std FT training (ckpts exist)
#   ./run_remaining_analysis.sh --only_plots      # just regenerate all plots
#
# Checkpoint paths (must exist):
#   /tmp/bam-pq-bge-large-ckpts/best/checkpoint.pt
#   /tmp/bam-pq-arctic-ckpts/best/checkpoint.pt
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Flags ────────────────────────────────────────────────────────────────────
SKIP_MASK_SIM=0
SKIP_STD_FT=0
SKIP_TRAIN=0
ONLY_PLOTS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip_mask_sim) SKIP_MASK_SIM=1; shift ;;
        --skip_std_ft)   SKIP_STD_FT=1;   shift ;;
        --skip_train)    SKIP_TRAIN=1;     shift ;;
        --only_plots)    ONLY_PLOTS=1; SKIP_MASK_SIM=1; SKIP_STD_FT=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  BAM-PQ Remaining Analysis Pipeline"
echo "  $(date)"
echo "  skip_mask_sim=$SKIP_MASK_SIM  skip_std_ft=$SKIP_STD_FT"
echo "  skip_train=$SKIP_TRAIN  only_plots=$ONLY_PLOTS"
echo "============================================================"

# ── 1. Mask cosine similarity — bge-large ────────────────────────────────────
if [[ $SKIP_MASK_SIM -eq 0 ]]; then
    echo ""
    echo "────────────────────────────────────────────────────────"
    echo "  [1a] Mask specialization — bge-large"
    echo "────────────────────────────────────────────────────────"

    BGE_CKPT="/tmp/bam-pq-bge-large-ckpts/best"
    BGE_OUT="results/bloom_analysis/bge_large"
    mkdir -p "$BGE_OUT"

    if [[ -f "$BGE_CKPT/checkpoint.pt" ]]; then
        if python3 scripts/analyze_mask_specialization.py \
                --config configs/bam_pq_bge_large.yaml \
                --checkpoint "$BGE_CKPT" \
                --output_dir "$BGE_OUT"; then
            echo "  ✓  bge-large mask similarity → $BGE_OUT/mask_specialization_table.json"
        else
            echo "  ✗  bge-large mask similarity FAILED"
        fi
    else
        echo "  SKIP — checkpoint not found at $BGE_CKPT"
    fi

    echo ""
    echo "────────────────────────────────────────────────────────"
    echo "  [1b] Mask specialization — arctic"
    echo "────────────────────────────────────────────────────────"

    ARCTIC_CKPT="/tmp/bam-pq-arctic-ckpts/best"
    ARCTIC_OUT="results/bloom_analysis/arctic"
    mkdir -p "$ARCTIC_OUT"

    if [[ -f "$ARCTIC_CKPT/checkpoint.pt" ]]; then
        if python3 scripts/analyze_mask_specialization.py \
                --config configs/bam_pq_arctic.yaml \
                --checkpoint "$ARCTIC_CKPT" \
                --output_dir "$ARCTIC_OUT"; then
            echo "  ✓  arctic mask similarity → $ARCTIC_OUT/mask_specialization_table.json"
        else
            echo "  ✗  arctic mask similarity FAILED"
        fi
    else
        echo "  SKIP — checkpoint not found at $ARCTIC_CKPT"
    fi
fi

# ── 2. Standard FT baseline ──────────────────────────────────────────────────
if [[ $SKIP_STD_FT -eq 0 ]]; then
    echo ""
    echo "────────────────────────────────────────────────────────"
    echo "  [2] Standard fine-tuning baseline (all 5 backbones)"
    echo "────────────────────────────────────────────────────────"

    STD_FT_ARGS=""
    if [[ $SKIP_TRAIN -eq 1 ]]; then
        STD_FT_ARGS="--skip_train"
    fi

    if bash run_standard_ft.sh $STD_FT_ARGS; then
        echo "  ✓  Standard FT done"
    else
        echo "  ✗  Standard FT pipeline had failures (check above)"
    fi
fi

# ── 3. Update mask similarity plot with bge/arctic ──────────────────────────
echo ""
echo "────────────────────────────────────────────────────────"
echo "  [3] Regenerate mask similarity heatmap plot (04)"
echo "────────────────────────────────────────────────────────"

python3 - <<'PYEOF'
import json, os, sys
import numpy as np

# Collect all available mask sim tables
BLOOM_NAMES = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

# Hardcoded results from prior analysis (e5large, qwen06b, roberta)
KNOWN = {
    "e5large": {
        "mean_off_diag": 0.473,
        "matrix": [
            [1.000, 0.442, 0.447, 0.459, 0.430, 0.490],
            [0.442, 1.000, 0.497, 0.498, 0.476, 0.476],
            [0.447, 0.497, 1.000, 0.501, 0.474, 0.477],
            [0.459, 0.498, 0.501, 1.000, 0.484, 0.487],
            [0.430, 0.476, 0.474, 0.484, 1.000, 0.457],
            [0.490, 0.476, 0.477, 0.487, 0.457, 1.000],
        ]
    },
    "qwen06b": {
        "mean_off_diag": 0.639,
        "matrix": [
            [1.000, 0.617, 0.628, 0.635, 0.617, 0.620],
            [0.617, 1.000, 0.666, 0.649, 0.636, 0.636],
            [0.628, 0.666, 1.000, 0.673, 0.648, 0.648],
            [0.635, 0.649, 0.673, 1.000, 0.657, 0.642],
            [0.617, 0.636, 0.648, 0.657, 1.000, 0.639],
            [0.620, 0.636, 0.648, 0.642, 0.639, 1.000],
        ]
    },
    "roberta": {
        "mean_off_diag": 0.595,
        "matrix": [
            [1.000, 0.581, 0.583, 0.592, 0.574, 0.578],
            [0.581, 1.000, 0.617, 0.613, 0.601, 0.604],
            [0.583, 0.617, 1.000, 0.617, 0.601, 0.600],
            [0.592, 0.613, 0.617, 1.000, 0.608, 0.602],
            [0.574, 0.601, 0.601, 0.608, 1.000, 0.590],
            [0.578, 0.604, 0.600, 0.602, 0.590, 1.000],
        ]
    }
}

# Try to load bge and arctic from computed files
for bk, path in [("bge", "results/bloom_analysis/bge_large/mask_specialization_table.json"),
                 ("arctic", "results/bloom_analysis/arctic/mask_specialization_table.json")]:
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        KNOWN[bk] = {
            "mean_off_diag": d["mean_off_diagonal_cosine_similarity"],
            "matrix": d["cosine_similarity"]
        }
        print(f"  Loaded {bk} mask similarity from {path}")
    else:
        print(f"  {bk}: no file at {path} — will skip in plot")

# Generate updated heatmap plot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available = {k: v for k, v in KNOWN.items()}
    n = len(available)

    if n == 0:
        print("  No mask similarity data available, skipping plot")
        sys.exit(0)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    backbone_labels = {
        "e5large": "e5-large", "bge": "bge-large",
        "arctic": "arctic", "roberta": "roberta", "qwen06b": "qwen0.6B"
    }

    for ax, (bk, data) in zip(axes, available.items()):
        mat = np.array(data["matrix"])
        im = ax.imshow(mat, cmap="coolwarm_r", vmin=0.3, vmax=1.0, aspect="auto")
        ax.set_xticks(range(6)); ax.set_xticklabels(BLOOM_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(6)); ax.set_yticklabels(BLOOM_NAMES, fontsize=8)
        ax.set_title(f"{backbone_labels.get(bk, bk)}\nμ_off={data['mean_off_diag']:.3f}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(6):
            for j in range(6):
                val = mat[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if val < 0.55 else "black")

    fig.suptitle("BAM-PQ: Mask Cosine Similarity per Bloom Level Pair", fontsize=12, y=1.02)
    plt.tight_layout()

    os.makedirs("results/plots", exist_ok=True)
    out = "results/plots/04_mask_similarity_heatmaps.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  ✓  Saved updated heatmap → {out}")
    plt.close()
except Exception as e:
    print(f"  ✗  Plot failed: {e}")
PYEOF

# ── 4. Comparison plots (needs std FT results) ───────────────────────────────
echo ""
echo "────────────────────────────────────────────────────────"
echo "  [4] Comparison plots (BAM-PQ vs MRL vs Standard FT)"
echo "────────────────────────────────────────────────────────"

STD_FT_DIR="results/standard_ft/educational"
COMP_OUT="results/plots/comparison"
mkdir -p "$COMP_OUT"

if [[ -f "$STD_FT_DIR/summary.json" ]]; then
    python3 scripts/plot_comparison.py \
        --std_ft_dir "$STD_FT_DIR" \
        --output_dir "$COMP_OUT"
    echo "  ✓  Comparison plots → $COMP_OUT/"
else
    echo "  INFO — No standard FT results yet at $STD_FT_DIR/summary.json"
    echo "         Running plot_comparison.py in BAM-PQ vs MRL only mode ..."
    python3 scripts/plot_comparison.py \
        --output_dir "$COMP_OUT"
    echo "  ✓  BAM-PQ vs MRL plots → $COMP_OUT/"
fi

# ── 5. Regenerate main result plots ─────────────────────────────────────────
echo ""
echo "────────────────────────────────────────────────────────"
echo "  [5] Regenerate all main result plots"
echo "────────────────────────────────────────────────────────"

python3 scripts/plot_results.py && echo "  ✓  Main plots regenerated → results/plots/"

echo ""
echo "============================================================"
echo "  DONE — $(date)"
echo ""
echo "  Output locations:"
echo "    results/bloom_analysis/bge_large/   (mask sim)"
echo "    results/bloom_analysis/arctic/      (mask sim)"
echo "    results/standard_ft/educational/    (std FT results)"
echo "    results/plots/                      (all plots)"
echo "    results/plots/comparison/           (comparison plots)"
echo "============================================================"
