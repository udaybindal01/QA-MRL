#!/usr/bin/env bash
# =============================================================================
# BAM BGE-large Pipeline — Option A + Option B, cold start (no MRL pre-training)
#
# Backbone: BAAI/bge-large-en-v1.5 (335M params, 1024-dim, encoder-only)
# Both Option A and Option B initialize directly from raw BGE-large pretrained weights.
# No intermediate MRL training step.
#
# Prerequisites:
#   ./data/real/ must exist (run optionA-working_pipeline.sh or build_real_data.py)
#   train_curriculum.jsonl must exist (run curriculum_negatives.py)
#
# Steps:
#   1. patch_configs    — point all configs at existing train_curriculum.jsonl
#   2. remine_negatives — (optional, REMINE=1) re-mine hard negatives
#   3. train_bam_a      — BAM Option A (prefix router, cold start from BGE-large)
#   4. train_bam_b      — BAM Option B (scattered mask, cold start from BGE-large)
#   5. find_bam_a       — BSR epoch selection for Option A
#   6. find_bam_b       — BSR epoch selection for Option B
#   7. eval_compare     — full eval: Option A vs Option B
#
# Usage:
#   chmod +x bgelarge-pipeline.sh
#   ./bgelarge-pipeline.sh                        # run all steps
#   ./bgelarge-pipeline.sh --from train_bam_b     # resume from a step
#   REMINE=1 ./bgelarge-pipeline.sh               # re-mine negatives before training
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BAM_A_CKPT_DIR="/tmp/bam-a-bgelarge-ckpts1"
BAM_B_CKPT_DIR="/tmp/bam-b-bgelarge-ckpts1"
RESULTS_DIR="./results/bam_bgelarge1"

BAM_A_CONFIG="configs/bam_optionA_bgelarge.yaml"
BAM_B_CONFIG="configs/bam_optionb_bgelarge.yaml"

CURRICULUM="./data/real/train_curriculum.jsonl"
BSR_ALPHA="0.5"
REMINE="${REMINE:-0}"
# ─────────────────────────────────────────────────────────────────────────────

# ── Argument parsing ──────────────────────────────────────────────────────────
FROM_STEP=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM_STEP="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ALL_STEPS=(patch_configs remine_negatives train_bam_a train_bam_b find_bam_a find_bam_b eval_compare)

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

# ── Prereq check ─────────────────────────────────────────────────────────────
log "PREREQ CHECK"
[[ -f "$CURRICULUM" ]] \
    || die "train_curriculum.jsonl not found at $CURRICULUM. Run optionA-working_pipeline.sh or curriculum_negatives.py first."
[[ -f "./data/real/corpus.jsonl" ]] \
    || die "corpus.jsonl not found. Run build_real_data.py first."

echo "  Curriculum : $CURRICULUM"
echo "  Backbone   : BAAI/bge-large-en-v1.5 (1024-dim, cold start — no MRL pre-training)"

mkdir -p "$BAM_A_CKPT_DIR" "$BAM_B_CKPT_DIR" "$RESULTS_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — PATCH CONFIGS: point train_path at existing curriculum
# ─────────────────────────────────────────────────────────────────────────────
if should_run patch_configs; then
    log "STEP 1/7 — PATCH CONFIGS (train_path → $CURRICULUM)"

    CURRICULUM_ESC=$(echo "$CURRICULUM" | sed 's|/|\\/|g')
    for cfg in "$BAM_A_CONFIG" "$BAM_B_CONFIG"; do
        cp "$cfg" "${cfg}.bak"
        sed -i.bak "s|train_path:.*|train_path: \"$CURRICULUM_ESC\"|" "$cfg"
        echo "  Patched: $cfg"
    done
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — RE-MINE HARD NEGATIVES (optional, set REMINE=1 to enable)
# ─────────────────────────────────────────────────────────────────────────────
if should_run remine_negatives; then
    if [[ "$REMINE" == "1" ]]; then
        log "STEP 2/7 — RE-MINE HARD NEGATIVES"
        NUM_NEG=$(python3 -c "
import yaml
with open('$BAM_A_CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg['data']['num_hard_negatives'])
")
        echo "  Re-mining $NUM_NEG hard negatives..."
        python3 data/curriculum_negatives.py \
            --pairs  "$CURRICULUM" \
            --corpus "./data/real/corpus.jsonl" \
            --output "$CURRICULUM" \
            --num_neg "$NUM_NEG" \
            --stage  0.7 \
            || die "curriculum_negatives.py failed"
        echo "  Done → $CURRICULUM"
    else
        log "STEP 2/7 — REMINE_NEGATIVES (skipped — set REMINE=1 to enable)"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TRAIN BAM OPTION A (cold start)
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_bam_a; then
    log "STEP 3/7 — TRAIN BAM OPTION A (prefix router, cold start from BGE-large)"
    echo "  Config  : $BAM_A_CONFIG"
    echo "  Init    : raw BAAI/bge-large-en-v1.5 pretrained weights (no MRL warm-start)"
    echo "  Output  : $BAM_A_CKPT_DIR/"

    # No --init_encoder: BloomAlignedMRL loads raw BGE-large HuggingFace weights
    python3 scripts/train_bam.py \
        --config "$BAM_A_CONFIG" \
        || die "train_bam.py (Option A) failed"

    echo "  BAM Option A checkpoints → $BAM_A_CKPT_DIR/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — TRAIN BAM OPTION B (cold start)
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_bam_b; then
    log "STEP 4/7 — TRAIN BAM OPTION B (scattered mask, cold start from BGE-large)"
    echo "  Config  : $BAM_B_CONFIG"
    echo "  Init    : raw BAAI/bge-large-en-v1.5 pretrained weights (no MRL warm-start)"
    echo "  Output  : $BAM_B_CKPT_DIR/"

    python3 scripts/train_bam.py \
        --config "$BAM_B_CONFIG" \
        || die "train_bam.py (Option B) failed"

    echo "  BAM Option B checkpoints → $BAM_B_CKPT_DIR/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — BSR EPOCH SELECTION — OPTION A
# ─────────────────────────────────────────────────────────────────────────────
if should_run find_bam_a; then
    log "STEP 5/7 — FIND BEST BAM OPTION A EPOCH (BSR)"
    [[ -d "$BAM_A_CKPT_DIR/epoch_0" ]] \
        || die "No BAM-A epoch checkpoints at $BAM_A_CKPT_DIR — run train_bam_a first"

    mkdir -p "$RESULTS_DIR/optionA_bsr"
    python3 scripts/find_best_epoch_bsr.py \
        --config         "$BAM_A_CONFIG" \
        --checkpoint_dir "$BAM_A_CKPT_DIR" \
        --output_dir     "$RESULTS_DIR/optionA_bsr/" \
        --alpha          "$BSR_ALPHA" \
        || die "find_best_epoch_bsr (Option A) failed"

    echo "  Option A best → $BAM_A_CKPT_DIR/best_bsr/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — BSR EPOCH SELECTION — OPTION B
# ─────────────────────────────────────────────────────────────────────────────
if should_run find_bam_b; then
    log "STEP 6/7 — FIND BEST BAM OPTION B EPOCH (BSR)"
    [[ -d "$BAM_B_CKPT_DIR/epoch_0" ]] \
        || die "No BAM-B epoch checkpoints at $BAM_B_CKPT_DIR — run train_bam_b first"

    mkdir -p "$RESULTS_DIR/optionB_bsr"
    python3 scripts/find_best_epoch_bsr.py \
        --config         "$BAM_B_CONFIG" \
        --checkpoint_dir "$BAM_B_CKPT_DIR" \
        --output_dir     "$RESULTS_DIR/optionB_bsr/" \
        --alpha          "$BSR_ALPHA" \
        || die "find_best_epoch_bsr (Option B) failed"

    echo "  Option B best → $BAM_B_CKPT_DIR/best_bsr/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — FULL EVAL: Option A vs Option B
# ─────────────────────────────────────────────────────────────────────────────
if should_run eval_compare; then
    log "STEP 7/7 — FULL EVALUATION (Option A vs Option B)"

    BAM_A_BEST="$BAM_A_CKPT_DIR/best_bsr"
    BAM_B_BEST="$BAM_B_CKPT_DIR/best_bsr"

    [[ -f "$BAM_A_BEST/checkpoint.pt" ]] || die "Option A best_bsr not found at $BAM_A_BEST — run find_bam_a first"
    [[ -f "$BAM_B_BEST/checkpoint.pt" ]] || die "Option B best_bsr not found at $BAM_B_BEST — run find_bam_b first"

    python3 scripts/eval_bam.py \
        --config          "$BAM_A_CONFIG" \
        --checkpoint      "$BAM_A_BEST" \
        --checkpoint_v4   "$BAM_B_BEST" \
        --config_v4       "$BAM_B_CONFIG" \
        --output_dir      "$RESULTS_DIR/" \
        || die "eval_bam.py failed"

    echo "  Full results → $RESULTS_DIR/results.json"
fi

# ─────────────────────────────────────────────────────────────────────────────
log "PIPELINE COMPLETE"
echo ""
echo "  Option A best (BSR): $BAM_A_CKPT_DIR/best_bsr/"
echo "  Option B best (BSR): $BAM_B_CKPT_DIR/best_bsr/"
echo "  BSR tables         : $RESULTS_DIR/optionA_bsr/  $RESULTS_DIR/optionB_bsr/"
echo "  Full eval results  : $RESULTS_DIR/results.json"
echo ""
echo "  Key metrics to compare:"
echo "    - recall@10, NDCG@10  (overall retrieval quality)"
echo "    - bloom_*_recall@10   (per-Bloom-level performance)"
echo "    - avg_active_dims     (efficiency — lower is better)"
echo "    NOTE: Option B dims are scattered — cannot use FAISS sub-index."
