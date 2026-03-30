#!/bin/bash
# Full QA-MRL training pipeline
# Usage: nohup bash run_pipeline.sh > logs/pipeline.log 2>&1 &

set -e  # exit on any error

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="/tmp/data/real"
BAM_CKPT="/tmp/bam-ckpts"
MRL_CKPT="/tmp/mrl-ckpts"
LOG_DIR="$REPO_DIR/logs"
RESULTS_DIR="$REPO_DIR/results"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"
cd "$REPO_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Step 1: Build corpus and initial pairs ────────────────────────────────────
log "=== STEP 1/5: Building real educational dataset ==="
python data/build_real_data.py \
    --config configs/real_data.yaml \
    --output_dir "$DATA_DIR" \
    2>&1 | tee "$LOG_DIR/build_real.log"
log "Step 1 done."

# ── Step 2: Rebuild training pairs with curriculum BM25 negatives ─────────────
log "=== STEP 2/5: Curriculum negative mining (stage=0.7) ==="
python data/curriculum_negatives.py \
    --pairs  "$DATA_DIR/train.jsonl" \
    --corpus "$DATA_DIR/corpus.jsonl" \
    --output "$DATA_DIR/train.jsonl" \
    --stage 0.7 \
    --num_neg 3 \
    2>&1 | tee "$LOG_DIR/curriculum.log"
log "Step 2 done."

# ── Step 3: Train MRL Baseline ────────────────────────────────────────────────
log "=== STEP 3/5: Training MRL Baseline ==="
python scripts/train_baseline_mrl.py \
    --config configs/neurips.yaml \
    2>&1 | tee "$LOG_DIR/train_mrl_baseline.log"
log "Step 3 done. Checkpoint: $MRL_CKPT"

# ── Step 4: Train BAM ─────────────────────────────────────────────────────────
log "=== STEP 4/5: Training BAM ==="
python scripts/train_bam.py \
    --config configs/bam.yaml \
    2>&1 | tee "$LOG_DIR/train_bam.log"
log "Step 4 done. Checkpoint: $BAM_CKPT"

# ── Step 5: Evaluate both models ──────────────────────────────────────────────
log "=== STEP 5/5: Evaluation ==="
python scripts/eval_bam.py \
    --config    configs/bam.yaml \
    --checkpoint "$BAM_CKPT/best/" \
    --baseline   "$MRL_CKPT/best/" \
    --output_dir "$RESULTS_DIR/bam_eval/" \
    2>&1 | tee "$LOG_DIR/eval.log"
log "Step 5 done. Results: $RESULTS_DIR/bam_eval/results.json"

log "=== PIPELINE COMPLETE ==="
