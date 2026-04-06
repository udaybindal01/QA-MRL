#!/usr/bin/env bash
# =============================================================================
# BAM Option A — Working Pipeline (matches v8 / 3ecb0f7 conditions)
#
# Steps:
#   1. Mine curriculum negatives (BM25, stage=0.7, 3 negatives per query)
#   2. Patch bam.yaml to use train_curriculum.jsonl
#   3. Train MRL baseline encoder (15 epochs, saves to MRL_CKPT_DIR)
#   4. Find best MRL epoch (corpus-level recall@10)
#   5. Train BAM Option A (warm-starts from best MRL encoder)
#   6. Find best BAM-A epoch
#
# Usage:
#   chmod +x optionA-working_pipeline.sh
#   ./optionA-working_pipeline.sh
#   ./optionA-working_pipeline.sh --from train_mrl   # skip data/curriculum steps
#
# Prerequisites:
#   pip install -r requirements.txt
#   data/real/train.jsonl and corpus.jsonl must exist (run data/build_real_data.py first)
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR="./data/real"
MRL_CKPT_DIR="/tmp/mrl-ckpts"
BAM_CKPT_DIR="/tmp/bam-ckpts"
RESULTS_DIR="./results"

MRL_CONFIG="configs/neurips.yaml"
BAM_CONFIG="configs/bam.yaml"

CURRICULUM_STAGE="0.7"
NUM_NEG=3
# ─────────────────────────────────────────────────────────────────────────────

# ── Argument parsing ──────────────────────────────────────────────────────────
FROM_STEP=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM_STEP="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ALL_STEPS=(curriculum patch_config train_mrl find_mrl train_bam find_best)

found=0
SKIP_STEPS=()
if [[ -n "$FROM_STEP" ]]; then
    for s in "${ALL_STEPS[@]}"; do
        if [[ "$s" == "$FROM_STEP" ]]; then found=1; fi
        if [[ $found -eq 0 ]]; then SKIP_STEPS+=("$s"); fi
    done
    if [[ $found -eq 0 ]]; then
        echo "Unknown step: $FROM_STEP"
        echo "Valid steps: ${ALL_STEPS[*]}"
        exit 1
    fi
fi

should_run() {
    local s="$1"
    for skip in "${SKIP_STEPS[@]:-}"; do
        [[ "$skip" == "$s" ]] && return 1
    done
    return 0
}

log() {
    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  [$(date '+%H:%M:%S')]  $1"
    echo "══════════════════════════════════════════════════════"
}
die() { echo "ERROR: $1" >&2; exit 1; }

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — CURRICULUM NEGATIVES
# ─────────────────────────────────────────────────────────────────────────────
if should_run curriculum; then
    log "STEP 1/6 — CURRICULUM NEGATIVES (stage=$CURRICULUM_STAGE, neg=$NUM_NEG)"

    [[ -f "$DATA_DIR/train.jsonl" ]]  || die "train.jsonl not found at $DATA_DIR"
    [[ -f "$DATA_DIR/corpus.jsonl" ]] || die "corpus.jsonl not found at $DATA_DIR"

    python3 data/curriculum_negatives.py \
        --pairs   "$DATA_DIR/train.jsonl" \
        --corpus  "$DATA_DIR/corpus.jsonl" \
        --output  "$DATA_DIR/train_curriculum.jsonl" \
        --stage   "$CURRICULUM_STAGE" \
        --num_neg "$NUM_NEG" \
        || die "curriculum_negatives.py failed"

    echo "  Curriculum negatives → $DATA_DIR/train_curriculum.jsonl"
    echo "  $(wc -l < "$DATA_DIR/train_curriculum.jsonl") pairs"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — PATCH bam.yaml: train.jsonl → train_curriculum.jsonl
# ─────────────────────────────────────────────────────────────────────────────
if should_run patch_config; then
    log "STEP 2/6 — PATCH bam.yaml (train_path → train_curriculum.jsonl)"

    [[ -f "$DATA_DIR/train_curriculum.jsonl" ]] \
        || die "train_curriculum.jsonl not found. Run curriculum step first."

    # Only patch if not already pointing at curriculum file
    if grep -q "train_curriculum.jsonl" "$BAM_CONFIG"; then
        echo "  bam.yaml already uses train_curriculum.jsonl — skipping sed."
    else
        sed -i 's|train\.jsonl|train_curriculum.jsonl|' "$BAM_CONFIG"
        echo "  Patched $BAM_CONFIG: train.jsonl → train_curriculum.jsonl"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TRAIN MRL BASELINE
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_mrl; then
    log "STEP 3/6 — TRAIN MRL BASELINE (15 epochs)"

    python3 scripts/train_baseline_mrl.py \
        --config "$MRL_CONFIG" \
        || die "train_baseline_mrl.py failed"

    echo "  MRL checkpoints → $MRL_CKPT_DIR/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — FIND BEST MRL EPOCH
# ─────────────────────────────────────────────────────────────────────────────
if should_run find_mrl; then
    log "STEP 4/6 — FIND BEST MRL EPOCH (corpus recall@10)"

    [[ -d "$MRL_CKPT_DIR/epoch_0" ]] \
        || die "No MRL epoch checkpoints at $MRL_CKPT_DIR. Run train_mrl first."

    mkdir -p "$RESULTS_DIR/best_epochs/mrl"

    python3 scripts/find_best_epoch.py \
        --checkpoint_dir "$MRL_CKPT_DIR" \
        --config         "$MRL_CONFIG" \
        --model_type     mrl \
        --metric         recall@10 \
        --output_dir     "$RESULTS_DIR/best_epochs/mrl/" \
        || die "find_best_epoch (MRL) failed"

    echo "  MRL best → $MRL_CKPT_DIR/best/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — TRAIN BAM OPTION A
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_bam; then
    log "STEP 5/6 — TRAIN BAM OPTION A (prefix routing, warm-start from MRL)"

    # Resolve best MRL checkpoint
    BEST_FILE="$RESULTS_DIR/best_epochs/mrl/best_checkpoint_path.txt"
    if [[ -f "$BEST_FILE" ]]; then
        MRL_BEST=$(cat "$BEST_FILE")
    else
        MRL_BEST="$MRL_CKPT_DIR/best"
    fi
    [[ -f "$MRL_BEST/checkpoint.pt" ]] \
        || die "MRL best checkpoint not found at $MRL_BEST. Run find_mrl first."

    [[ -f "$DATA_DIR/train_curriculum.jsonl" ]] \
        || die "train_curriculum.jsonl not found. Run curriculum step first."

    python3 scripts/train_bam.py \
        --config       "$BAM_CONFIG" \
        --init_encoder "$MRL_BEST" \
        || die "train_bam.py (Option A) failed"

    echo "  BAM Option A checkpoints → $BAM_CKPT_DIR/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — FIND BEST BAM-A EPOCH
# ─────────────────────────────────────────────────────────────────────────────
if should_run find_best; then
    log "STEP 6/6 — FIND BEST BAM-A EPOCH (corpus recall@10)"

    [[ -d "$BAM_CKPT_DIR/epoch_0" ]] \
        || die "No BAM epoch checkpoints at $BAM_CKPT_DIR. Run train_bam first."

    mkdir -p "$RESULTS_DIR/best_epochs/bam"

    python3 scripts/find_best_epoch.py \
        --checkpoint_dir "$BAM_CKPT_DIR" \
        --config         "$BAM_CONFIG" \
        --model_type     bam \
        --metric         recall@10 \
        --output_dir     "$RESULTS_DIR/best_epochs/bam/" \
        || die "find_best_epoch (BAM-A) failed"

    echo "  BAM-A best → $BAM_CKPT_DIR/best/"
fi

log "OPTION A PIPELINE COMPLETE"
echo ""
echo "  Best MRL checkpoint:   $MRL_CKPT_DIR/best/"
echo "  Best BAM-A checkpoint: $BAM_CKPT_DIR/best/"
echo ""
echo "  Run evaluation:"
echo "    python scripts/eval_bam.py \\"
echo "        --config $BAM_CONFIG \\"
echo "        --checkpoint $BAM_CKPT_DIR/best \\"
echo "        --baseline $MRL_CKPT_DIR/best"
