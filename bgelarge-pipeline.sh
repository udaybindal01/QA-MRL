#!/usr/bin/env bash
# =============================================================================
# BAM BGE-large Pipeline — MRL → Option A → Option B
#
# Backbone: BAAI/bge-large-en-v1.5 (335M params, 1024-dim, encoder-only)
# Fits on any GPU with ~5 GB total training memory — no memory tricks needed.
#
# Prerequisites:
#   ./data/real/ must exist (run optionA-working_pipeline.sh or build_real_data.py)
#   train_curriculum.jsonl must exist (run curriculum_negatives.py)
#
# Steps:
#   1. patch_configs    — point all configs at existing train_curriculum.jsonl
#   2. train_mrl        — MRL baseline (BGE-large, cold start)
#   3. find_mrl         — select best MRL epoch by val NDCG
#   4. train_bam_a      — BAM Option A (prefix router, MRL warm-start)
#   5. train_bam_b      — BAM Option B (scattered mask, MRL warm-start)
#   6. find_bam_a       — BSR epoch selection for Option A
#   7. find_bam_b       — BSR epoch selection for Option B
#   8. eval_compare     — full eval: Option A vs Option B vs MRL
#
# Usage:
#   chmod +x bgelarge-pipeline.sh
#   ./bgelarge-pipeline.sh                        # run all steps
#   ./bgelarge-pipeline.sh --from train_bam_a     # resume from a step
#   REMINE=1 ./bgelarge-pipeline.sh               # re-mine negatives before training
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MRL_CKPT_DIR="/tmp/mrl-bgelarge-ckpts"
BAM_A_CKPT_DIR="/tmp/bam-a-bgelarge-ckpts"
BAM_B_CKPT_DIR="/tmp/bam-b-bgelarge-ckpts"
RESULTS_DIR="./results/bam_bgelarge"

MRL_CONFIG="configs/mrl_bgelarge.yaml"
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

ALL_STEPS=(patch_configs train_mrl find_mrl train_bam_a train_bam_b find_bam_a find_bam_b eval_compare)

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
echo "  Backbone   : BAAI/bge-large-en-v1.5 (1024-dim, ~5 GB training memory)"

mkdir -p "$MRL_CKPT_DIR" "$BAM_A_CKPT_DIR" "$BAM_B_CKPT_DIR" "$RESULTS_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — PATCH CONFIGS: point train_path at existing curriculum
# ─────────────────────────────────────────────────────────────────────────────
if should_run patch_configs; then
    log "STEP 1/8 — PATCH CONFIGS (train_path → $CURRICULUM)"

    CURRICULUM_ESC=$(echo "$CURRICULUM" | sed 's|/|\\/|g')
    for cfg in "$MRL_CONFIG" "$BAM_A_CONFIG" "$BAM_B_CONFIG"; do
        cp "$cfg" "${cfg}.bak"
        sed -i.bak "s|train_path:.*|train_path: \"$CURRICULUM_ESC\"|" "$cfg"
        echo "  Patched: $cfg"
    done
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — TRAIN MRL BASELINE
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_mrl; then
    log "STEP 2/8 — TRAIN MRL BASELINE (BGE-large cold start)"
    echo "  Config  : $MRL_CONFIG"
    echo "  Output  : $MRL_CKPT_DIR/"

    if [[ "$REMINE" == "1" ]]; then
        NUM_NEG=$(python3 -c "
import yaml
with open('$MRL_CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg['data']['num_hard_negatives'])
")
        echo "  Re-mining $NUM_NEG hard negatives before training..."
        python3 data/curriculum_negatives.py \
            --pairs  "$CURRICULUM" \
            --corpus "./data/real/corpus.jsonl" \
            --output "$CURRICULUM" \
            --num_neg "$NUM_NEG" \
            --stage  0.7 \
            || die "curriculum_negatives.py failed"
    fi

    python3 scripts/train_baseline_mrl.py \
        --config "$MRL_CONFIG" \
        || die "train_baseline_mrl.py failed"

    echo "  MRL checkpoints → $MRL_CKPT_DIR/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — SELECT BEST MRL EPOCH
# ─────────────────────────────────────────────────────────────────────────────
if should_run find_mrl; then
    log "STEP 3/8 — FIND BEST MRL EPOCH (val NDCG)"
    [[ -d "$MRL_CKPT_DIR/epoch_0" ]] \
        || die "No MRL epoch checkpoints at $MRL_CKPT_DIR — run train_mrl first"

    python3 scripts/find_best_epoch.py \
        --checkpoint_dir "$MRL_CKPT_DIR" \
        --config         "$MRL_CONFIG" \
        || die "find_best_epoch.py failed"

    # Resolve path to best checkpoint for downstream init_encoder
    MRL_BEST="$MRL_CKPT_DIR/best"
    BEST_FILE="$RESULTS_DIR/../best_epochs/mrl_bgelarge/best_checkpoint_path.txt"
    if [[ -f "$BEST_FILE" ]]; then
        MRL_BEST=$(cat "$BEST_FILE")
    fi
    echo "$MRL_BEST" > "$RESULTS_DIR/mrl_best_path.txt"
    echo "  MRL best → $MRL_BEST"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — TRAIN BAM OPTION A
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_bam_a; then
    log "STEP 4/8 — TRAIN BAM OPTION A (prefix router, MRL warm-start)"
    echo "  Config  : $BAM_A_CONFIG"
    echo "  Output  : $BAM_A_CKPT_DIR/"

    MRL_BEST="$MRL_CKPT_DIR/best"
    if [[ -f "$RESULTS_DIR/mrl_best_path.txt" ]]; then
        MRL_BEST=$(cat "$RESULTS_DIR/mrl_best_path.txt")
    fi
    [[ -f "$MRL_BEST/checkpoint.pt" ]] \
        || die "MRL best checkpoint not found at $MRL_BEST — run find_mrl first"

    python3 scripts/train_bam.py \
        --config       "$BAM_A_CONFIG" \
        --init_encoder "$MRL_BEST" \
        || die "train_bam.py (Option A) failed"

    echo "  BAM Option A checkpoints → $BAM_A_CKPT_DIR/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — TRAIN BAM OPTION B
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_bam_b; then
    log "STEP 5/8 — TRAIN BAM OPTION B (scattered mask, MRL warm-start)"
    echo "  Config  : $BAM_B_CONFIG"
    echo "  Output  : $BAM_B_CKPT_DIR/"

    MRL_BEST="$MRL_CKPT_DIR/best"
    if [[ -f "$RESULTS_DIR/mrl_best_path.txt" ]]; then
        MRL_BEST=$(cat "$RESULTS_DIR/mrl_best_path.txt")
    fi
    [[ -f "$MRL_BEST/checkpoint.pt" ]] \
        || die "MRL best checkpoint not found at $MRL_BEST — run find_mrl first"

    python3 scripts/train_bam.py \
        --config       "$BAM_B_CONFIG" \
        --init_encoder "$MRL_BEST" \
        || die "train_bam.py (Option B) failed"

    echo "  BAM Option B checkpoints → $BAM_B_CKPT_DIR/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — BSR EPOCH SELECTION — OPTION A
# ─────────────────────────────────────────────────────────────────────────────
if should_run find_bam_a; then
    log "STEP 6/8 — FIND BEST BAM OPTION A EPOCH (BSR)"
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
# STEP 7 — BSR EPOCH SELECTION — OPTION B
# ─────────────────────────────────────────────────────────────────────────────
if should_run find_bam_b; then
    log "STEP 7/8 — FIND BEST BAM OPTION B EPOCH (BSR)"
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
# STEP 8 — FULL EVAL: Option A vs Option B vs MRL
# ─────────────────────────────────────────────────────────────────────────────
if should_run eval_compare; then
    log "STEP 8/8 — FULL EVALUATION (Option A vs Option B vs MRL)"

    MRL_BEST="$MRL_CKPT_DIR/best"
    if [[ -f "$RESULTS_DIR/mrl_best_path.txt" ]]; then
        MRL_BEST=$(cat "$RESULTS_DIR/mrl_best_path.txt")
    fi
    BAM_A_BEST="$BAM_A_CKPT_DIR/best_bsr"
    BAM_B_BEST="$BAM_B_CKPT_DIR/best_bsr"

    [[ -f "$MRL_BEST/checkpoint.pt" ]]   || die "MRL best not found at $MRL_BEST"
    [[ -f "$BAM_A_BEST/checkpoint.pt" ]] || die "Option A best_bsr not found at $BAM_A_BEST — run find_bam_a first"
    [[ -f "$BAM_B_BEST/checkpoint.pt" ]] || die "Option B best_bsr not found at $BAM_B_BEST — run find_bam_b first"

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
echo "  MRL checkpoints    : $MRL_CKPT_DIR/"
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
