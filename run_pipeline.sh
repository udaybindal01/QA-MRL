#!/bin/bash
# Full QA-MRL training + evaluation pipeline
#
# Usage:
#   bash run_pipeline.sh                        # full pipeline
#   bash run_pipeline.sh --skip-data            # skip data prep
#   bash run_pipeline.sh --skip-data --skip-mrl # skip data + MRL training
#
# Background:
#   nohup bash run_pipeline.sh > logs/pipeline.log 2>&1 &

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="/tmp/data/real"
BAM_CKPT="/tmp/bam-ckpts"
MRL_CKPT="/tmp/mrl-ckpts"
LOG_DIR="$REPO_DIR/logs"
RESULTS_DIR="$REPO_DIR/results"

# ── Flags ─────────────────────────────────────────────────────────────────────
SKIP_DATA=false
SKIP_MRL=false
SKIP_BAM=false
SKIP_EVAL=false

for arg in "$@"; do
  case $arg in
    --skip-data) SKIP_DATA=true ;;
    --skip-mrl)  SKIP_MRL=true  ;;
    --skip-bam)  SKIP_BAM=true  ;;
    --skip-eval) SKIP_EVAL=true ;;
  esac
done

mkdir -p "$LOG_DIR" "$RESULTS_DIR"
cd "$REPO_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Step 1: Data preparation ──────────────────────────────────────────────────
if [ "$SKIP_DATA" = false ]; then
  log "=== STEP 1/6: Building real educational dataset ==="
  python data/build_real_data.py \
      --config configs/real_data.yaml \
      --output_dir "$DATA_DIR" \
      2>&1 | tee "$LOG_DIR/build_real.log"

  log "=== STEP 1b/6: Bloom annotation ==="
  python data/annotate_bloom_pretrained.py \
      2>&1 | tee "$LOG_DIR/annotate_bloom.log"

  log "=== STEP 1c/6: Curriculum negative mining ==="
  python data/curriculum_negatives.py \
      --pairs  "$DATA_DIR/train.jsonl" \
      --corpus "$DATA_DIR/corpus.jsonl" \
      --output "$DATA_DIR/train.jsonl" \
      --stage 0.7 \
      --num_neg 3 \
      2>&1 | tee "$LOG_DIR/curriculum.log"

  log "Step 1 done."
else
  log "Skipping data preparation."
fi

# ── Step 2: Train MRL Baseline ────────────────────────────────────────────────
if [ "$SKIP_MRL" = false ]; then
  log "=== STEP 2/6: Training MRL Baseline (saves epoch_N/ every epoch) ==="
  python scripts/train_baseline_mrl.py \
      --config configs/neurips.yaml \
      2>&1 | tee "$LOG_DIR/train_mrl.log"
  log "Step 2 done. Checkpoints at: $MRL_CKPT"
else
  log "Skipping MRL baseline training."
fi

# ── Step 3: Find best MRL epoch (full-corpus eval) ───────────────────────────
log "=== STEP 3/6: Selecting best MRL checkpoint via full-corpus eval ==="
python scripts/find_best_epoch.py \
    --config configs/neurips.yaml \
    --checkpoint_dir "$MRL_CKPT" \
    --model_type mrl \
    --metric recall@10 \
    2>&1 | tee "$LOG_DIR/find_best_mrl.log"
log "Best MRL checkpoint copied to: $MRL_CKPT/best/"

# ── Step 4: Train BAM (warm-start from best MRL checkpoint) ───────────────────
if [ "$SKIP_BAM" = false ]; then
  log "=== STEP 4/6: Training BAM (init from $MRL_CKPT/best/) ==="
  python scripts/train_bam.py \
      --config configs/bam.yaml \
      --init_encoder "$MRL_CKPT/best/" \
      2>&1 | tee "$LOG_DIR/train_bam.log"
  log "Step 4 done. Checkpoints at: $BAM_CKPT"
else
  log "Skipping BAM training."
fi

# ── Step 5: Find best BAM epoch (full-corpus eval) ────────────────────────────
log "=== STEP 5/6: Selecting best BAM checkpoint via full-corpus eval ==="
python scripts/find_best_epoch.py \
    --config configs/bam.yaml \
    --checkpoint_dir "$BAM_CKPT" \
    --model_type bam \
    --metric recall@10 \
    2>&1 | tee "$LOG_DIR/find_best_bam.log"
log "Best BAM checkpoint copied to: $BAM_CKPT/best/"

# ── Step 6: Evaluation ────────────────────────────────────────────────────────
if [ "$SKIP_EVAL" = false ]; then
  log "=== STEP 6/6: Evaluation ==="

  log "  6a: BAM vs MRL comparison..."
  python scripts/eval_bam.py \
      --config     configs/bam.yaml \
      --checkpoint "$BAM_CKPT/best/" \
      --baseline   "$MRL_CKPT/best/" \
      --output_dir "$RESULTS_DIR/bam_eval/" \
      2>&1 | tee "$LOG_DIR/eval_bam.log"

  log "  6b: Full evaluation suite..."
  python scripts/run_evaluation.py \
      --config configs/neurips.yaml \
      --checkpoint "$BAM_CKPT/best/" \
      --baseline   "$MRL_CKPT/best/" \
      --output_dir "$RESULTS_DIR/full_eval/" \
      2>&1 | tee "$LOG_DIR/run_evaluation.log"

  log "  6c: Ablations..."
  python scripts/run_ablations.py \
      --config     configs/bam.yaml \
      --checkpoint "$BAM_CKPT/best/" \
      --output_dir "$RESULTS_DIR/ablations/" \
      2>&1 | tee "$LOG_DIR/ablations.log"

  log "Step 6 done. Results at: $RESULTS_DIR/"
else
  log "Skipping evaluation."
fi

log "=== PIPELINE COMPLETE ==="
log "Checkpoint summary:"
log "  MRL best : $MRL_CKPT/best/"
log "  BAM best : $BAM_CKPT/best/"
log "  Results  : $RESULTS_DIR/"
