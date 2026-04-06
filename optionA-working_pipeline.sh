#!/usr/bin/env bash
# =============================================================================
# BAM Option A — Working Pipeline
#
# Steps:
#   1. build_real_data       — build train/val/test/corpus JSONL files
#   2. annotate              — label queries with Bloom taxonomy
#   3. curriculum_negatives  — mine BM25 hard negatives (stage=0.7)
#   4. patch_config          — sed train.jsonl → train_curriculum.jsonl in bam.yaml
#   5. train_mrl             — train MRL baseline encoder (configs/bam.yaml)
#   6. find_mrl              — find best MRL epoch (corpus recall@10)
#   7. train_bam             — train BAM Option A (init from best MRL epoch)
#
# Usage:
#   chmod +x optionA-working_pipeline.sh
#   ./optionA-working_pipeline.sh                  # run all steps
#   ./optionA-working_pipeline.sh --from annotate  # resume from a step
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR="./data/real"
MRL_CKPT_DIR="/tmp/mrl-ckpts"
BAM_CKPT_DIR="/tmp/bam-ckpts"
RESULTS_DIR="./results"
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

ALL_STEPS=(build_real_data annotate curriculum_negatives patch_config train_mrl find_mrl train_bam)

SKIP_STEPS=()
if [[ -n "$FROM_STEP" ]]; then
    found=0
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
    for skip in "${SKIP_STEPS[@]:-}"; do
        [[ "$skip" == "$1" ]] && return 1
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
# STEP 1 — BUILD REAL DATA
# ─────────────────────────────────────────────────────────────────────────────
if should_run build_real_data; then
    log "STEP 1/7 — BUILD REAL DATA"
    python3 data/build_real_data.py || die "build_real_data.py failed"
    echo "  train: $(wc -l < "$DATA_DIR/train.jsonl") pairs"
    echo "  val:   $(wc -l < "$DATA_DIR/val.jsonl") pairs"
    echo "  test:  $(wc -l < "$DATA_DIR/test.jsonl") pairs"
    echo "  corpus: $(wc -l < "$DATA_DIR/corpus.jsonl") passages"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — ANNOTATE WITH BLOOM TAXONOMY
# ─────────────────────────────────────────────────────────────────────────────
if should_run annotate; then
    log "STEP 2/7 — BLOOM ANNOTATION"
    python3 data/annotate_bloom_pretrained.py \
        --data_dir "$DATA_DIR" \
        --method   pretrained \
        || die "annotate_bloom_pretrained.py failed"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — CURRICULUM NEGATIVES
# ─────────────────────────────────────────────────────────────────────────────
if should_run curriculum_negatives; then
    log "STEP 3/7 — CURRICULUM NEGATIVES (stage=$CURRICULUM_STAGE, neg=$NUM_NEG)"
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
# STEP 4 — PATCH bam.yaml: train.jsonl → train_curriculum.jsonl
# ─────────────────────────────────────────────────────────────────────────────
if should_run patch_config; then
    log "STEP 4/7 — PATCH bam.yaml (train_path → train_curriculum.jsonl)"
    [[ -f "$DATA_DIR/train_curriculum.jsonl" ]] \
        || die "train_curriculum.jsonl not found — run curriculum_negatives step first"
    if grep -q "train_curriculum.jsonl" "$BAM_CONFIG"; then
        echo "  Already patched — skipping."
    else
        sed -i 's|train\.jsonl|train_curriculum.jsonl|' "$BAM_CONFIG"
        echo "  Patched $BAM_CONFIG"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — TRAIN MRL BASELINE
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_mrl; then
    log "STEP 5/7 — TRAIN MRL BASELINE (config: $BAM_CONFIG)"
    python3 scripts/train_baseline_mrl.py \
        --config         "$BAM_CONFIG" \
        --checkpoint_dir "$MRL_CKPT_DIR" \
        || die "train_baseline_mrl.py failed"
    echo "  MRL checkpoints → $MRL_CKPT_DIR/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — FIND BEST MRL EPOCH
# ─────────────────────────────────────────────────────────────────────────────
if should_run find_mrl; then
    log "STEP 6/7 — FIND BEST MRL EPOCH (corpus recall@10)"
    [[ -d "$MRL_CKPT_DIR/epoch_0" ]] \
        || die "No MRL epoch checkpoints at $MRL_CKPT_DIR — run train_mrl first"
    mkdir -p "$RESULTS_DIR/best_epochs/mrl"
    python3 scripts/find_best_epoch.py \
        --checkpoint_dir "$MRL_CKPT_DIR" \
        --config         "$BAM_CONFIG" \
        --model_type     mrl \
        --metric         recall@10 \
        --output_dir     "$RESULTS_DIR/best_epochs/mrl/" \
        || die "find_best_epoch (MRL) failed"
    echo "  MRL best → $MRL_CKPT_DIR/best/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — TRAIN BAM OPTION A
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_bam; then
    log "STEP 7/7 — TRAIN BAM OPTION A (config: $BAM_CONFIG)"

    BEST_FILE="$RESULTS_DIR/best_epochs/mrl/best_checkpoint_path.txt"
    if [[ -f "$BEST_FILE" ]]; then
        MRL_BEST=$(cat "$BEST_FILE")
    else
        MRL_BEST="$MRL_CKPT_DIR/best"
    fi
    [[ -f "$MRL_BEST/checkpoint.pt" ]] \
        || die "MRL best checkpoint not found at $MRL_BEST — run find_mrl first"

    python3 scripts/train_bam.py \
        --config       "$BAM_CONFIG" \
        --init_encoder "$MRL_BEST" \
        || die "train_bam.py (Option A) failed"

    echo "  BAM Option A checkpoints → $BAM_CKPT_DIR/"
fi

# ─────────────────────────────────────────────────────────────────────────────
log "PIPELINE COMPLETE"
echo ""
echo "  MRL best:   $MRL_CKPT_DIR/best/"
echo "  BAM-A ckpts: $BAM_CKPT_DIR/"
echo ""
echo "  Next — find best BAM-A epoch:"
echo "    python scripts/find_best_epoch.py \\"
echo "        --checkpoint_dir $BAM_CKPT_DIR \\"
echo "        --config $BAM_CONFIG \\"
echo "        --model_type bam --metric recall@10 \\"
echo "        --output_dir $RESULTS_DIR/best_epochs/bam/"
