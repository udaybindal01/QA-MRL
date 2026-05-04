#!/usr/bin/env bash
# ============================================================
#  BAM-PQ — arctic
#  Run this on a GPU node to train + evaluate
#
#  To change backbone: edit the 4 lines under "CONFIG" below
# ============================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# ── CONFIG ──────────────────────────────────────────────────
BACKBONE_NAME="arctic"
CONFIG="configs/bam_pq_arctic.yaml"
CKPT_DIR="/tmp/bam-pq-arctic-ckpts"
STD_FT_CFG="configs/standard_ft_arctic.yaml"
STD_FT_CKPT="/tmp/uday/standard-ft/arctic"
# ────────────────────────────────────────────────────────────

OUT_DIR="results/per_backbone/${BACKBONE_NAME}"
mkdir -p "$OUT_DIR" "$CKPT_DIR" "$STD_FT_CKPT"

echo "======================================================"
echo "  BAM-PQ  |  $BACKBONE_NAME"
echo "  Config  : $CONFIG"
echo "  Ckpt    : $CKPT_DIR"
echo "  $(date)"
echo "======================================================"

# 1. Train BAM-PQ
echo "[1/5] Training BAM-PQ ..."
python3 scripts/train_bam.py --config "$CONFIG" --checkpoint_dir "$CKPT_DIR"
echo "  Done training"

# 2. Find best epoch (corpus-level, not in-batch)
echo "[2/5] Selecting best epoch ..."
python3 scripts/find_best_epoch.py \
    --config "$CONFIG" \
    --checkpoint_dir "$CKPT_DIR" \
    --model_type bam_pq \
    --metric "recall@10"

# 3. Evaluate BAM-PQ
echo "[3/5] Evaluating BAM-PQ ..."
python3 scripts/eval_edu_baselines.py \
    --config      "$CONFIG" \
    --checkpoint  "$CKPT_DIR/best" \
    --model_type  bam_pq \
    --output_dir  "$OUT_DIR/bam_pq" \
    --bloom_stratified

# 4. Standard FT baseline
echo "[4/5] Standard FT — train ..."
python3 scripts/train_baseline_mrl.py \
    --config "$STD_FT_CFG" \
    --checkpoint_dir "$STD_FT_CKPT"

python3 scripts/find_best_epoch.py \
    --config "$STD_FT_CFG" \
    --checkpoint_dir "$STD_FT_CKPT" \
    --model_type mrl \
    --metric "recall@10"

echo "[5/5] Standard FT — evaluate ..."
python3 scripts/eval_edu_baselines.py \
    --config      "$STD_FT_CFG" \
    --checkpoint  "$STD_FT_CKPT/best" \
    --model_type  mrl \
    --output_dir  "$OUT_DIR/standard_ft" \
    --bloom_stratified

# Alpha
echo ""
echo "── Alpha (per-query residual weight) ──────────────────"
python3 scripts/extract_alpha.py
echo ""
echo "Results → $OUT_DIR/"
echo "Done — $(date)"
