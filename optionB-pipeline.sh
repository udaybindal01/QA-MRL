#!/usr/bin/env bash
# =============================================================================
# BAM Option B — Pipeline
#
# Prerequisite: optionA-working_pipeline.sh must have completed first.
#   Reuses:  ./data/real/  (train_curriculum, val, test, corpus)
#            /tmp/mrl-ckpts/best/  (MRL baseline encoder warm-start)
#
# Steps:
#   1. remine_negatives — re-mine hard negatives (num_neg from bam_optionb.yaml, default 7)
#   2. train_bam_b      — train BAM Option B (BloomMaskHead, scattered mask)
#   3. find_bam_a       — BSR selection on Option A checkpoints (for comparison)
#   4. find_bam_b       — BSR selection on Option B checkpoints
#   5. eval_compare     — full eval: Option B vs Option A vs MRL baseline
#
# Usage:
#   chmod +x optionB-pipeline.sh
#   ./optionB-pipeline.sh                    # run all steps
#   ./optionB-pipeline.sh --from find_bam_b  # resume from a step
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MRL_CKPT_DIR="/tmp/mrl-ckpts"
BAM_A_CKPT_DIR="/tmp/bam-ckpts"
BAM_B_CKPT_DIR="/tmp/bam-b-ckpts9"
RESULTS_DIR="./results/bam_optionb9"
BAM_A_CONFIG="configs/bam.yaml"
BAM_B_CONFIG="configs/bam_optionb.yaml"
BSR_ALPHA="0.5"
# ─────────────────────────────────────────────────────────────────────────────

# ── Argument parsing ──────────────────────────────────────────────────────────
FROM_STEP=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM_STEP="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ALL_STEPS=(remine_negatives train_bam_b find_bam_a find_bam_b eval_compare)

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

MRL_BEST="$MRL_CKPT_DIR/best"
BEST_FILE="$RESULTS_DIR/../best_epochs/mrl/best_checkpoint_path.txt"
if [[ -f "$BEST_FILE" ]]; then
    MRL_BEST=$(cat "$BEST_FILE")
fi

[[ -f "$MRL_BEST/checkpoint.pt" ]] \
    || die "MRL best checkpoint not found at $MRL_BEST. Run optionA-working_pipeline.sh first."
[[ -f "./data/real/train_curriculum.jsonl" ]] \
    || die "train_curriculum.jsonl not found. Run optionA-working_pipeline.sh first."

echo "  MRL warm-start : $MRL_BEST"
echo "  Data           : ./data/real/train_curriculum.jsonl"

mkdir -p "$BAM_B_CKPT_DIR" "$RESULTS_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — RE-MINE HARD NEGATIVES
# ─────────────────────────────────────────────────────────────────────────────
if should_run remine_negatives; then
    log "STEP 1/5 — RE-MINE HARD NEGATIVES (num_neg from $BAM_B_CONFIG)"

    NUM_NEG=$(python3 -c "
import yaml
with open('$BAM_B_CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg['data']['num_hard_negatives'])
")
    echo "  num_hard_negatives : $NUM_NEG"
    echo "  Input/Output       : ./data/real/train_curriculum.jsonl (overwrite in-place)"

    python3 data/curriculum_negatives.py \
        --pairs  "./data/real/train_curriculum.jsonl" \
        --corpus "./data/real/corpus.jsonl" \
        --output "./data/real/train_curriculum.jsonl" \
        --num_neg "$NUM_NEG" \
        --stage  0.7 \
        || die "curriculum_negatives.py failed"

    echo "  Re-mined $NUM_NEG negatives → ./data/real/train_curriculum.jsonl"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — TRAIN BAM OPTION B
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_bam_b; then
    log "STEP 2/5 — TRAIN BAM OPTION B (BloomMaskHead, scattered mask)"
    echo "  Config      : $BAM_B_CONFIG"
    echo "  Init encoder: $MRL_BEST"
    echo "  Output      : $BAM_B_CKPT_DIR/"

    # MRL init: start from Option A's best MRL checkpoint.
    # Base init was tried and failed — domain adaptation benefit outweighs prefix bias cost.
    # mrl_anchor_weight=0.0 in config means the encoder won't be pulled back toward prefix
    # structure during Option B training. MRL init only provides the domain-adapted starting
    # point; scatter mask training then reorganizes dims freely.
    python3 scripts/train_bam.py \
        --config       "$BAM_B_CONFIG" \
        --init_encoder "$MRL_BEST" \
        || die "train_bam.py (Option B) failed"

    echo "  BAM Option B checkpoints → $BAM_B_CKPT_DIR/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — FIND BEST OPTION A EPOCH (for comparison in eval)
# ─────────────────────────────────────────────────────────────────────────────
if should_run find_bam_a; then
    log "STEP 2/4 — FIND BEST BAM OPTION A EPOCH (BSR)"
    [[ -d "$BAM_A_CKPT_DIR/epoch_0" ]] \
        || die "No BAM-A epoch checkpoints at $BAM_A_CKPT_DIR — run optionA-working_pipeline.sh first"

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
# STEP 3 — FIND BEST OPTION B EPOCH
# ─────────────────────────────────────────────────────────────────────────────
if should_run find_bam_b; then
    log "STEP 3/4 — FIND BEST BAM OPTION B EPOCH (BSR)"
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
# STEP 4 — FULL EVAL: Option B vs Option A vs MRL Baseline
# ─────────────────────────────────────────────────────────────────────────────
if should_run eval_compare; then
    log "STEP 4/4 — FULL EVALUATION (Option B vs Option A vs MRL)"

    BAM_A_BEST="$BAM_A_CKPT_DIR/best_bsr"
    BAM_B_BEST="$BAM_B_CKPT_DIR/best_bsr"

    [[ -f "$BAM_A_BEST/checkpoint.pt" ]] \
        || die "Option A best_bsr not found at $BAM_A_BEST — run find_bam_a first"
    [[ -f "$BAM_B_BEST/checkpoint.pt" ]] \
        || die "Option B best_bsr not found at $BAM_B_BEST — run find_bam_b first"

    # eval_bam.py loads Option A as primary (--checkpoint) and Option B via --checkpoint_v4
    python3 scripts/eval_bam.py \
        --config          "$BAM_A_CONFIG" \
        --checkpoint      "$BAM_A_BEST" \
        --baseline        "$MRL_BEST" \
        --checkpoint_v4   "$BAM_B_BEST" \
        --config_v4       "$BAM_B_CONFIG" \
        --output_dir      "$RESULTS_DIR/" \
        || die "eval_bam.py failed"

    echo "  Full results → $RESULTS_DIR/results.json"
fi

# ─────────────────────────────────────────────────────────────────────────────
log "PIPELINE COMPLETE"
echo ""
echo "  BAM-B checkpoints : $BAM_B_CKPT_DIR/"
echo "  BAM-B best (BSR)  : $BAM_B_CKPT_DIR/best_bsr/"
echo "  BSR table         : $RESULTS_DIR/optionB_bsr/epoch_results_bsr.json"
echo "  Full eval results : $RESULTS_DIR/results.json"
echo ""
echo "  Key metrics to compare:"
echo "    - recall@10, NDCG@10 (overall retrieval quality)"
echo "    - bloom_*_recall@10  (per-Bloom-level performance)"
echo "    - avg_active_dims    (efficiency — lower is better)"
echo "    - sparse_ratio       (fraction of zeroed dims)"
echo "    NOTE: Option B dims are scattered (not contiguous) — cannot use FAISS sub-index."
