#!/bin/bash
# BAM v3 Training Pipeline
# Query-Only Bloom — documents have no Bloom labels
#
# Run on gnode077 or wherever your GPU is.
# All data/checkpoints go to /tmp to avoid disk quota issues.

set -e

# ─────────────────────── Setup ───────────────────────
export HF_HOME=/tmp/$USER/hf_cache
export TRANSFORMERS_CACHE=/tmp/$USER/hf_cache
export HF_DATASETS_CACHE=/tmp/$USER/hf_cache/datasets
mkdir -p /tmp/$USER/hf_cache

echo "============================================================"
echo "BAM v3 Pipeline: Query-Only Bloom"
echo "============================================================"

# ─────────────────────── Step 1: Rebuild Data ───────────────────────
echo ""
echo "[Step 1/5] Rebuilding dataset with v3 pipeline (no doc Bloom)..."
python data/build_real_data.py \
    --config configs/bam.yaml \
    --output_dir /tmp/data/real

# ─────────────────────── Step 2: Mine Curriculum Negatives ──────────
echo ""
echo "[Step 2/5] Mining curriculum negatives (topic-based, no doc Bloom)..."
python data/curriculum_negatives.py \
    --pairs /tmp/data/real/train.jsonl \
    --corpus /tmp/data/real/corpus.jsonl \
    --output /tmp/data/real/train_curriculum.jsonl \
    --stage 0.7 \
    --num_neg 3

# Update config to use curriculum data
sed -i 's|/tmp/data/real/train.jsonl|/tmp/data/real/train_curriculum.jsonl|' configs/bam.yaml
echo "  Updated config to use train_curriculum.jsonl"

# ─────────────────────── Step 3: Train MRL Baseline ─────────────────
echo ""
echo "[Step 3/5] Training MRL baseline..."
python scripts/train_baseline_mrl.py --config configs/bam.yaml

# ─────────────────────── Step 4: Train BAM v3 ──────────────────────
echo ""
echo "[Step 4/5] Training BAM v3 (query-only Bloom, no early stopping)..."
python scripts/train_bam.py \
    --config configs/bam.yaml \
    --init_encoder /tmp/bam-ckpts/mrl_baseline_best/

# ─────────────────────── Step 5: Evaluate All Epochs ────────────────
echo ""
echo "[Step 5/5] Evaluating all epoch checkpoints..."

# Revert config to use original train.jsonl for eval consistency
sed -i 's|/tmp/data/real/train_curriculum.jsonl|/tmp/data/real/train.jsonl|' configs/bam.yaml

# Evaluate each epoch checkpoint to find true best
for epoch in $(seq 0 14); do
    ckpt="/tmp/bam-ckpts/epoch_${epoch}"
    if [ -d "$ckpt" ]; then
        echo "  Evaluating epoch $epoch..."
        python scripts/eval_bam.py \
            --config configs/bam.yaml \
            --checkpoint "$ckpt" \
            --baseline /tmp/bam-ckpts/mrl_baseline_best/ \
            --output_dir "results/bam_v3_eval/epoch_${epoch}/" \
            2>&1 | tail -20
        echo ""
    fi
done

echo "============================================================"
echo "Done! Check results/bam_v3_eval/ for per-epoch results."
echo "Pick the epoch with best corpus-level R@10 (not val NDCG)."
echo "============================================================"
