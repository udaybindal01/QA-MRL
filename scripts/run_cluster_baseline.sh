#!/usr/bin/env bash
# =============================================================================
# run_cluster_baseline.sh — one-shot job for the cluster-routing baseline
#
# Trains a fresh MRL baseline for the requested backbone and then runs
# scripts/eval_cluster_routing.py on top of it. Sequential in one job so
# you don't have to submit two separate scheduler entries.
#
# Usage:
#     ./scripts/run_cluster_baseline.sh <backbone>
#
# where <backbone> ∈ {bge_base, bge_large, e5large, arctic}
# (matches configs/mrl_<backbone>.yaml).
#
# Env overrides:
#     CKPT_ROOT   base dir for MRL checkpoints (default: /scratch/$USER/mrl-ckpts)
#     OUT_ROOT    base dir for cluster-routing results (default: /scratch/$USER/cluster_routing)
#     K           number of clusters (default: 6)
#     DIMS        space-separated active-dims sweep (default: "128 256 384 512")
#     TRAIN_PATH  training queries jsonl (default: data/real/train_curriculum.jsonl)
#     SKIP_TRAIN  set to 1 to reuse an existing checkpoint at $CKPT_ROOT/mrl_$BB/best/
# =============================================================================
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <backbone>   (bge_base | bge_large | e5large | arctic)"
    exit 1
fi

BB="$1"
CFG="configs/mrl_${BB}.yaml"
[[ -f "$CFG" ]] || { echo "ERROR: config not found: $CFG"; exit 1; }

CKPT_ROOT="${CKPT_ROOT:-/scratch/${USER}/mrl-ckpts}"
OUT_ROOT="${OUT_ROOT:-/scratch/${USER}/cluster_routing}"
K="${K:-6}"
DIMS="${DIMS:-128 256 384 512}"
TRAIN_PATH="${TRAIN_PATH:-data/real/train_curriculum.jsonl}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

CKPT_DIR="${CKPT_ROOT}/mrl_${BB}"
BEST_CKPT="${CKPT_DIR}/best"
OUT_DIR="${OUT_ROOT}/${BB}"

mkdir -p "$CKPT_DIR" "$OUT_DIR"

echo "======================================================================"
echo "  backbone      : $BB"
echo "  config        : $CFG"
echo "  ckpt dir      : $CKPT_DIR"
echo "  output dir    : $OUT_DIR"
echo "  k / dims sweep: $K / $DIMS"
echo "  train_path    : $TRAIN_PATH"
echo "  skip training : $SKIP_TRAIN"
echo "======================================================================"

# ── Step 1: MRL training ────────────────────────────────────────────────────
if [[ "$SKIP_TRAIN" == "1" ]] && [[ -f "$BEST_CKPT/checkpoint.pt" ]]; then
    echo "[step 1] SKIP_TRAIN=1 and $BEST_CKPT/checkpoint.pt exists — reusing."
else
    echo "[step 1] Training MRL baseline for $BB ..."
    python3 scripts/train_baseline_mrl.py \
        --config         "$CFG" \
        --checkpoint_dir "$CKPT_DIR"
    if [[ ! -f "$BEST_CKPT/checkpoint.pt" ]]; then
        # Some train scripts save 'final' but not 'best'; fall back to final.
        if [[ -f "$CKPT_DIR/final/checkpoint.pt" ]]; then
            ln -sfn "$CKPT_DIR/final" "$BEST_CKPT"
            echo "[step 1] Linked best -> final"
        else
            echo "ERROR: no MRL checkpoint found at $BEST_CKPT or $CKPT_DIR/final"
            exit 1
        fi
    fi
fi

# ── Step 2: cluster-routing baseline eval ──────────────────────────────────
echo ""
echo "[step 2] Running cluster-routing baseline on top of MRL checkpoint ..."
python3 scripts/eval_cluster_routing.py \
    --config     "$CFG" \
    --checkpoint "$BEST_CKPT" \
    --train_path "$TRAIN_PATH" \
    --output_dir "$OUT_DIR" \
    --k          "$K" \
    --active_dims_sweep $DIMS

echo ""
echo "======================================================================"
echo "  DONE.  MRL ckpt : $BEST_CKPT/checkpoint.pt"
echo "         Results  : $OUT_DIR/cluster_routing_results.json"
echo "                    $OUT_DIR/cluster_routing_sweep.csv"
echo "                    $OUT_DIR/cluster_routing_per_query.csv"
echo "======================================================================"
