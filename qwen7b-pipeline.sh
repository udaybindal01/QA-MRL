#!/usr/bin/env bash
# =============================================================================
# Qwen2.5-7B Full Pipeline — MRL → Option A → Option B → BSR → Eval → Ablations
#
# Prerequisite: ./data/real/ must exist with corpus/val/test JSONL files.
#   If train_curriculum.jsonl already exists (mined by the BGE pipeline), it is
#   reused via sed-substitution into the Qwen configs — no re-mining needed.
#   Otherwise, set REMINE=1 to re-mine fresh hard negatives.
#
# Steps:
#   1. patch_configs     — sed-patch all three Qwen configs to use the mined curriculum
#   2. train_mrl         — train MRL baseline from Qwen2.5-7B base weights
#   3. find_mrl_best     — pick best MRL epoch (plain R@10)
#   4. train_option_a    — train BAM Option A (prefix router, MRL warm-start)
#   5. train_option_b    — train BAM Option B (scatter mask, MRL warm-start)
#   6. find_bsr_a        — BSR best-epoch selection for Option A
#   7. find_bsr_b        — BSR best-epoch selection for Option B
#   8. eval_compare      — full eval: Option B vs Option A vs MRL baseline
#   9. ablations         — ablation study
#
# Usage:
#   chmod +x qwen7b-pipeline.sh
#   ./qwen7b-pipeline.sh                         # run all steps
#   ./qwen7b-pipeline.sh --from train_option_a   # resume from a step
#   REMINE=1 ./qwen7b-pipeline.sh                # re-mine negatives in step 1
#   CUDA_A=0 CUDA_B=1 ./qwen7b-pipeline.sh       # run A/B training on separate GPUs
# =============================================================================

set -euo pipefail

# Reduce CUDA memory fragmentation — important for 7B model at batch_size=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MRL_CONFIG="configs/mrl_qwen7b.yaml"
BAM_A_CONFIG="configs/bam_optionA_qwen7b.yaml"
BAM_B_CONFIG="configs/bam_optionb_qwen7b.yaml"

MRL_CKPT_DIR="/tmp/mrl-qwen7b-ckpts"
BAM_A_CKPT_DIR="/tmp/bam-a-qwen7b-ckpts"
BAM_B_CKPT_DIR="/tmp/bam-b-qwen7b-ckpts"
RESULTS_DIR="./results/qwen7b"

CURRICULUM="./data/real/train_curriculum.jsonl"
CORPUS="./data/real/corpus.jsonl"
BSR_ALPHA="0.5"

# GPU assignment for parallel A/B training (override via env vars)
CUDA_A="${CUDA_A:-0}"
CUDA_B="${CUDA_B:-0}"   # set to 1 if you have a second GPU

# Set REMINE=1 to re-mine hard negatives even if curriculum already exists
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

ALL_STEPS=(patch_configs train_mrl find_mrl_best train_option_a train_option_b find_bsr_a find_bsr_b eval_compare ablations)

SKIP_STEPS=()
if [[ -n "$FROM_STEP" ]]; then
    found=0
    for s in "${ALL_STEPS[@]}"; do
        [[ "$s" == "$FROM_STEP" ]] && found=1
        [[ $found -eq 0 ]] && SKIP_STEPS+=("$s")
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

# ── Prereq check ──────────────────────────────────────────────────────────────
log "PREREQ CHECK"

[[ -f "./data/real/corpus.jsonl" ]]  || die "corpus.jsonl not found — run data prep first."
[[ -f "./data/real/val.jsonl" ]]     || die "val.jsonl not found — run data prep first."
[[ -f "./data/real/test.jsonl" ]]    || die "test.jsonl not found — run data prep first."

mkdir -p "$MRL_CKPT_DIR" "$BAM_A_CKPT_DIR" "$BAM_B_CKPT_DIR" "$RESULTS_DIR"

echo "  MRL config     : $MRL_CONFIG"
echo "  Option A config: $BAM_A_CONFIG"
echo "  Option B config: $BAM_B_CONFIG"
echo "  Results dir    : $RESULTS_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — PATCH CONFIGS + CURRICULUM
# ─────────────────────────────────────────────────────────────────────────────
if should_run patch_configs; then
    log "STEP 1/9 — PATCH CONFIGS WITH MINED CURRICULUM"

    if [[ "$REMINE" == "1" ]] || [[ ! -f "$CURRICULUM" ]]; then
        # Re-mine (or first-time mine) hard negatives using Qwen config's num_hard_negatives
        NUM_NEG=$(python3 -c "
import yaml
with open('$BAM_B_CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg['data']['num_hard_negatives'])
")
        echo "  Mining $NUM_NEG hard negatives → $CURRICULUM"
        python3 data/curriculum_negatives.py \
            --pairs  "$CURRICULUM" \
            --corpus "$CORPUS" \
            --output "$CURRICULUM" \
            --num_neg "$NUM_NEG" \
            --stage  0.7 \
            || die "curriculum_negatives.py failed"
        echo "  Mined $NUM_NEG negatives → $CURRICULUM"
    else
        echo "  Reusing existing mined curriculum: $CURRICULUM"
        echo "  (set REMINE=1 to force re-mine)"
    fi

    # sed-patch all three configs to point train_path at the mined curriculum.
    # This ensures all models train on the same hard-negative enriched dataset
    # regardless of what path was written in the YAML at commit time.
    CURRICULUM_ESC=$(echo "$CURRICULUM" | sed 's|/|\\/|g')
    for cfg in "$MRL_CONFIG" "$BAM_A_CONFIG" "$BAM_B_CONFIG"; do
        sed -i.bak "s|train_path:.*|train_path: \"$CURRICULUM_ESC\"|" "$cfg" \
            && echo "  Patched train_path → $CURRICULUM in $cfg"
    done
    echo "  Backup originals saved as *.bak"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — TRAIN MRL BASELINE
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_mrl; then
    log "STEP 2/9 — TRAIN MRL BASELINE (Qwen2.5-7B, from base weights)"
    echo "  Config  : $MRL_CONFIG"
    echo "  Output  : $MRL_CKPT_DIR/"

    python3 scripts/train_baseline_mrl.py \
        --config "$MRL_CONFIG" \
        || die "train_baseline_mrl.py failed"

    echo "  MRL checkpoints → $MRL_CKPT_DIR/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — FIND BEST MRL EPOCH
# ─────────────────────────────────────────────────────────────────────────────
if should_run find_mrl_best; then
    log "STEP 3/9 — FIND BEST MRL EPOCH"
    [[ -d "$MRL_CKPT_DIR/epoch_0" ]] \
        || die "No MRL epoch checkpoints at $MRL_CKPT_DIR — run train_mrl first"

    mkdir -p "$RESULTS_DIR/mrl_best"
    python3 scripts/find_best_epoch.py \
        --checkpoint_dir "$MRL_CKPT_DIR" \
        --config         "$MRL_CONFIG" \
        --output_dir     "$RESULTS_DIR/mrl_best/" \
        || die "find_best_epoch (MRL) failed"

    # Resolve best checkpoint: prefer saved path file, fall back to /best
    MRL_BEST_FILE="$RESULTS_DIR/mrl_best/best_checkpoint_path.txt"
    if [[ -f "$MRL_BEST_FILE" ]]; then
        MRL_BEST=$(cat "$MRL_BEST_FILE")
    else
        MRL_BEST="$MRL_CKPT_DIR/best"
    fi
    echo "  MRL best checkpoint : $MRL_BEST"

    # Inject the resolved path into a sentinel file so later steps can find it
    # even when --from skips this step.
    echo "$MRL_BEST" > "$RESULTS_DIR/mrl_best/resolved_path.txt"
fi

# Resolve MRL best path for downstream steps (needed when --from skips step 3)
MRL_BEST="$MRL_CKPT_DIR/best"
if [[ -f "$RESULTS_DIR/mrl_best/resolved_path.txt" ]]; then
    MRL_BEST=$(cat "$RESULTS_DIR/mrl_best/resolved_path.txt")
fi
[[ -f "$MRL_BEST/checkpoint.pt" ]] \
    || die "MRL best checkpoint not found at $MRL_BEST — run find_mrl_best first"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — TRAIN BAM OPTION A  (prefix router, MRL warm-start)
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_option_a; then
    log "STEP 4/9 — TRAIN BAM OPTION A (BloomDimRouter, prefix mask)"
    echo "  Config      : $BAM_A_CONFIG"
    echo "  Init encoder: $MRL_BEST"
    echo "  Output      : $BAM_A_CKPT_DIR/"
    echo "  GPU         : $CUDA_A"

    CUDA_VISIBLE_DEVICES="$CUDA_A" python3 scripts/train_bam.py \
        --config       "$BAM_A_CONFIG" \
        --init_encoder "$MRL_BEST" \
        || die "train_bam.py (Option A) failed"

    echo "  Option A checkpoints → $BAM_A_CKPT_DIR/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — TRAIN BAM OPTION B  (scatter mask, MRL warm-start)
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_option_b; then
    log "STEP 5/9 — TRAIN BAM OPTION B (BloomMaskHead, scatter mask)"
    echo "  Config      : $BAM_B_CONFIG"
    echo "  Init encoder: $MRL_BEST"
    echo "  Output      : $BAM_B_CKPT_DIR/"
    echo "  GPU         : $CUDA_B"
    echo ""
    echo "  NOTE: If two GPUs are available, run steps 4 and 5 in parallel:"
    echo "    CUDA_A=0 CUDA_B=1 ./qwen7b-pipeline.sh --from train_option_a &"
    echo "    (step 5 runs automatically after step 4 in a single-GPU run)"

    CUDA_VISIBLE_DEVICES="$CUDA_B" python3 scripts/train_bam.py \
        --config       "$BAM_B_CONFIG" \
        --init_encoder "$MRL_BEST" \
        || die "train_bam.py (Option B) failed"

    echo "  Option B checkpoints → $BAM_B_CKPT_DIR/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — BSR BEST EPOCH FOR OPTION A
# ─────────────────────────────────────────────────────────────────────────────
if should_run find_bsr_a; then
    log "STEP 6/9 — FIND BEST OPTION A EPOCH (BSR)"
    [[ -d "$BAM_A_CKPT_DIR/epoch_0" ]] \
        || die "No Option A checkpoints at $BAM_A_CKPT_DIR — run train_option_a first"

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
# STEP 7 — BSR BEST EPOCH FOR OPTION B
# ─────────────────────────────────────────────────────────────────────────────
if should_run find_bsr_b; then
    log "STEP 7/9 — FIND BEST OPTION B EPOCH (BSR)"
    [[ -d "$BAM_B_CKPT_DIR/epoch_0" ]] \
        || die "No Option B checkpoints at $BAM_B_CKPT_DIR — run train_option_b first"

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
# STEP 8 — FULL EVAL: Option B vs Option A vs MRL Baseline
# ─────────────────────────────────────────────────────────────────────────────
if should_run eval_compare; then
    log "STEP 8/9 — FULL EVALUATION (Option B vs Option A vs MRL)"

    BAM_A_BEST="$BAM_A_CKPT_DIR/best_bsr"
    BAM_B_BEST="$BAM_B_CKPT_DIR/best_bsr"

    [[ -f "$BAM_A_BEST/checkpoint.pt" ]] \
        || die "Option A best_bsr not found at $BAM_A_BEST — run find_bsr_a first"
    [[ -f "$BAM_B_BEST/checkpoint.pt" ]] \
        || die "Option B best_bsr not found at $BAM_B_BEST — run find_bsr_b first"

    mkdir -p "$RESULTS_DIR/eval"
    python3 scripts/eval_bam.py \
        --config        "$BAM_A_CONFIG" \
        --checkpoint    "$BAM_A_BEST" \
        --baseline      "$MRL_BEST" \
        --checkpoint_v4 "$BAM_B_BEST" \
        --config_v4     "$BAM_B_CONFIG" \
        --output_dir    "$RESULTS_DIR/eval/" \
        || die "eval_bam.py failed"

    echo "  Full results → $RESULTS_DIR/eval/results.json"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — ABLATIONS
# ─────────────────────────────────────────────────────────────────────────────
if should_run ablations; then
    log "STEP 9/9 — ABLATIONS"

    BAM_A_BEST="$BAM_A_CKPT_DIR/best_bsr"
    BAM_B_BEST="$BAM_B_CKPT_DIR/best_bsr"

    [[ -f "$BAM_A_BEST/checkpoint.pt" ]] \
        || die "Option A best_bsr not found — run find_bsr_a first"
    [[ -f "$BAM_B_BEST/checkpoint.pt" ]] \
        || die "Option B best_bsr not found — run find_bsr_b first"

    mkdir -p "$RESULTS_DIR/ablations"
    python3 scripts/run_ablations.py \
        --config        "$BAM_A_CONFIG" \
        --checkpoint    "$BAM_A_BEST" \
        --baseline      "$MRL_BEST" \
        --checkpoint_v4 "$BAM_B_BEST" \
        --config_v4     "$BAM_B_CONFIG" \
        --output_dir    "$RESULTS_DIR/ablations/" \
        || die "run_ablations.py failed"

    echo "  Ablations → $RESULTS_DIR/ablations/"
fi

# ─────────────────────────────────────────────────────────────────────────────
log "PIPELINE COMPLETE"
echo ""
echo "  MRL best           : $MRL_BEST"
echo "  Option A best (BSR): $BAM_A_CKPT_DIR/best_bsr/"
echo "  Option B best (BSR): $BAM_B_CKPT_DIR/best_bsr/"
echo ""
echo "  Results:"
echo "    Eval       : $RESULTS_DIR/eval/results.json"
echo "    Ablations  : $RESULTS_DIR/ablations/"
echo "    BSR tables : $RESULTS_DIR/optionA_bsr/epoch_results_bsr.json"
echo "                 $RESULTS_DIR/optionB_bsr/epoch_results_bsr.json"
echo ""
echo "  Key metrics to compare:"
echo "    - recall@10, NDCG@10        (overall retrieval quality)"
echo "    - bloom_*_recall@10         (per-Bloom-level performance)"
echo "    - avg_active_dims           (efficiency — lower is better)"
echo "    NOTE: Option B dims are scattered — cannot use FAISS sub-index."
