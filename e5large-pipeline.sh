#!/usr/bin/env bash
# =============================================================================
# BAM E5-large Pipeline  —  MRL → Option A → Option B
# =============================================================================
#
# Backbone: intfloat/e5-large-v2
#   - 335M params, 1024-dim, encoder-only, CLS pooling
#   - Contrastive-trained for retrieval but NOT MRL pre-trained
#   - Fits on any GPU with ~5 GB training memory (no memory tricks needed)
#
# Why this pipeline exists:
#   BAM training works best when the encoder already understands multi-resolution
#   structure. We cannot borrow that from e5-large's pre-training (it has none),
#   so we train it ourselves in Step 2 (MRL baseline). That checkpoint then
#   warm-starts both BAM variants, giving them a domain-adapted encoder that
#   already compresses well at small dims before routing is introduced.
#
# Pipeline overview:
#
#   Step 1  patch_configs      — rewrite train_path in all 3 configs to point at
#                                the already-mined curriculum file so we don't
#                                accidentally train on the raw (no negatives) pairs
#
#   Step 2  train_mrl          — train MRL baseline on e5-large from scratch.
#                                Loss = InfoNCE(full) + 0.5 * MRL(truncated dims).
#                                Teaches the encoder to compress well at all 6
#                                MRL dims. 15 epochs, batch 16 + accum 8 = 128.
#
#   Step 3  find_mrl           — pick the best MRL epoch by val-set NDCG (in-batch).
#                                Saves best checkpoint to $MRL_CKPT_DIR/best.
#                                NOTE: this is an approximation; true best requires
#                                corpus-level eval (run find_best_epoch.py separately).
#
#   Step 4  train_bam_a        — train BAM Option A on e5-large with MRL warm-start.
#                                BloomDimRouter: 6 learned scalars → prefix binary mask.
#                                Encoder initialized from Step 3 best checkpoint.
#                                router_lr = 10× encoder_lr for fast routing convergence.
#
#   Step 5  train_bam_b        — train BAM Option B on e5-large with MRL warm-start.
#                                BloomMaskHead: Embedding(6,1024) → Gumbel-STE → scattered mask.
#                                Same MRL warm-start as Option A — scattered mask training
#                                then freely reorganizes which dims are active per level.
#
#   Step 6  find_bam_a         — BSR (Bloom Stratified Recall) epoch selection for Option A.
#                                BSR = quality × (1 + α × efficiency). Picks the epoch that
#                                best balances retrieval quality and dimension compression.
#
#   Step 7  find_bam_b         — BSR epoch selection for Option B (same metric).
#
#   Step 8  eval_compare       — full corpus-level eval: Option A vs Option B vs MRL baseline.
#                                Outputs recall@K, NDCG@10, per-Bloom breakdown, avg_active_dims.
#
# Prerequisites:
#   ./data/real/train_curriculum.jsonl  — must exist (run optionA-working_pipeline.sh
#                                         or curriculum_negatives.py first)
#   ./data/real/{val,test,corpus}.jsonl — must exist (run build_real_data.py first)
#
# Usage:
#   chmod +x e5large-pipeline.sh
#   ./e5large-pipeline.sh                          # run all steps
#   ./e5large-pipeline.sh --from train_bam_a       # skip to a specific step
#   REMINE=1 ./e5large-pipeline.sh                 # re-mine hard negatives before MRL training
#
# Resume tip: after train_mrl completes, the MRL best path is written to
#   $RESULTS_DIR/mrl_best_path.txt. Steps 4 and 5 read that file automatically,
#   so --from train_bam_a just works without extra flags.
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MRL_CKPT_DIR="/tmp/mrl-e5large-ckpts"
BAM_A_CKPT_DIR="/tmp/bam-a-e5large-ckpts1"
BAM_B_CKPT_DIR="/tmp/bam-b-e5large-ckpts1"
RESULTS_DIR="./results/bam_e5large1"

MRL_CONFIG="configs/mrl_e5large.yaml"
BAM_A_CONFIG="configs/bam_optionA_e5large.yaml"
BAM_B_CONFIG="configs/bam_optionb_e5large.yaml"

CURRICULUM="./data/real/train_curriculum.jsonl"
BSR_ALPHA="0.5"
REMINE="${REMINE:-0}"          # set REMINE=1 to re-mine hard negatives before training
# ─────────────────────────────────────────────────────────────────────────────

# ── Argument parsing ──────────────────────────────────────────────────────────
FROM_STEP=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM_STEP="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ALL_STEPS=(patch_configs remine_negatives train_mrl find_mrl train_bam_a train_bam_b find_bam_a find_bam_b eval_compare)

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

echo "  Backbone   : intfloat/e5-large-v2 (1024-dim, NOT MRL pre-trained)"
echo "  Curriculum : $CURRICULUM"
echo "  Results    : $RESULTS_DIR/"

mkdir -p "$MRL_CKPT_DIR" "$BAM_A_CKPT_DIR" "$BAM_B_CKPT_DIR" "$RESULTS_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — PATCH CONFIGS
# Point train_path in all 3 configs at the already-mined curriculum file.
# The configs in git have a placeholder path; this sed overwrites it in-place.
# .bak backups are created so the originals can be restored if needed.
# ─────────────────────────────────────────────────────────────────────────────
if should_run patch_configs; then
    log "STEP 1/9 — PATCH CONFIGS (train_path → $CURRICULUM)"

    CURRICULUM_ESC=$(echo "$CURRICULUM" | sed 's|/|\\/|g')
    for cfg in "$MRL_CONFIG" "$BAM_A_CONFIG" "$BAM_B_CONFIG"; do
        cp "$cfg" "${cfg}.bak"
        sed -i.bak "s|train_path:.*|train_path: \"$CURRICULUM_ESC\"|" "$cfg"
        echo "  Patched: $cfg"
    done
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — RE-MINE HARD NEGATIVES  (optional, set REMINE=1 to enable)
# Uses curriculum_negatives.py to re-mine hard negatives against the current
# corpus using BM25 + in-batch hard mining. Updates train_curriculum.jsonl
# in-place. Recommended before the first training run if negatives are stale.
# ─────────────────────────────────────────────────────────────────────────────
if should_run remine_negatives; then
    if [[ "$REMINE" == "1" ]]; then
        log "STEP 2/9 — RE-MINE HARD NEGATIVES"
        NUM_NEG=$(python3 -c "
import yaml
with open('$MRL_CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg['data']['num_hard_negatives'])
")
        echo "  num_hard_negatives : $NUM_NEG"
        python3 data/curriculum_negatives.py \
            --pairs  "$CURRICULUM" \
            --corpus "./data/real/corpus.jsonl" \
            --output "$CURRICULUM" \
            --num_neg "$NUM_NEG" \
            --stage  0.7 \
            || die "curriculum_negatives.py failed"
        echo "  Done → $CURRICULUM"
    else
        log "STEP 2/9 — REMINE_NEGATIVES (skipped — set REMINE=1 to enable)"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TRAIN MRL BASELINE
# Fine-tunes e5-large with Matryoshka loss on our educational domain data.
# This is the only step that teaches multi-resolution structure — e5-large has
# none built in. The checkpoint becomes the shared warm-start for both BAM variants.
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_mrl; then
    log "STEP 3/9 — TRAIN MRL BASELINE (e5-large cold start)"
    echo "  Config     : $MRL_CONFIG"
    echo "  Init       : intfloat/e5-large-v2 pretrained weights"
    echo "  Output     : $MRL_CKPT_DIR/"
    echo "  Epochs     : 15  |  Batch 16 × accum 8 = 128 effective"

    python3 scripts/train_baseline_mrl.py \
        --config "$MRL_CONFIG" \
        || die "train_baseline_mrl.py failed"

    echo "  MRL checkpoints → $MRL_CKPT_DIR/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — SELECT BEST MRL EPOCH
# find_best_epoch.py scores each saved epoch checkpoint on val-set pairwise NDCG
# and copies the best to $MRL_CKPT_DIR/best/. Steps 5 and 6 read that path.
# ─────────────────────────────────────────────────────────────────────────────
if should_run find_mrl; then
    log "STEP 4/9 — FIND BEST MRL EPOCH (val NDCG)"
    [[ -d "$MRL_CKPT_DIR/epoch_0" ]] \
        || die "No MRL epoch checkpoints at $MRL_CKPT_DIR — run train_mrl first"

    python3 scripts/find_best_epoch.py \
        --checkpoint_dir "$MRL_CKPT_DIR" \
        --config         "$MRL_CONFIG" \
        --model_type     mrl \
        || die "find_best_epoch.py failed"

    # Resolve best path — fall back to /best if the script didn't write a .txt
    MRL_BEST="$MRL_CKPT_DIR/best"
    BEST_FILE="$RESULTS_DIR/../best_epochs/mrl_e5large/best_checkpoint_path.txt"
    if [[ -f "$BEST_FILE" ]]; then
        MRL_BEST=$(cat "$BEST_FILE")
    fi
    echo "$MRL_BEST" > "$RESULTS_DIR/mrl_best_path.txt"
    echo "  MRL best → $MRL_BEST"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — TRAIN BAM OPTION A  (prefix router, MRL warm-start)
# BloomDimRouter learns one truncation dim per Bloom level (0=Remember…5=Create).
# Efficiency loss pushes lower-complexity levels to fewer dims. With MRL warm-start
# the encoder already handles compression well, so routing converges faster.
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_bam_a; then
    log "STEP 5/9 — TRAIN BAM OPTION A (prefix router, MRL warm-start)"

    MRL_BEST="$MRL_CKPT_DIR/best"
    [[ -f "$RESULTS_DIR/mrl_best_path.txt" ]] && MRL_BEST=$(cat "$RESULTS_DIR/mrl_best_path.txt")
    [[ -f "$MRL_BEST/checkpoint.pt" ]] \
        || die "MRL best checkpoint not found at $MRL_BEST — run find_mrl first"

    echo "  Config     : $BAM_A_CONFIG"
    echo "  Init       : $MRL_BEST"
    echo "  Output     : $BAM_A_CKPT_DIR/"

    python3 scripts/train_bam.py \
        --config       "$BAM_A_CONFIG" \
        --init_encoder "$MRL_BEST" \
        || die "train_bam.py (Option A) failed"

    echo "  BAM Option A checkpoints → $BAM_A_CKPT_DIR/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — TRAIN BAM OPTION B  (scattered mask, raw e5-large init)
# BloomMaskHead learns one scattered binary mask per Bloom level.
#
# Intentionally NO --init_encoder here. MRL training teaches PREFIX structure
# (early dims are more important), which is correct for Option A's prefix mask
# but wrong for Option B's scattered mask. e5-large is already retrieval-tuned
# (contrastive) with no prefix bias — the best starting point for a model that
# needs to freely select any subset of 1024 dims per Bloom level.
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_bam_b; then
    log "STEP 6/9 — TRAIN BAM OPTION B (scattered mask, raw e5-large init)"

    echo "  Config     : $BAM_B_CONFIG"
    echo "  Init       : intfloat/e5-large-v2 pretrained weights (no MRL warm-start)"
    echo "  Output     : $BAM_B_CKPT_DIR/"

    # No --init_encoder: BloomAlignedMRL loads raw e5-large HuggingFace weights
    python3 scripts/train_bam.py \
        --config "$BAM_B_CONFIG" \
        || die "train_bam.py (Option B) failed"

    echo "  BAM Option B checkpoints → $BAM_B_CKPT_DIR/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — BSR EPOCH SELECTION — OPTION A
# BSR = bloom_consistent_recall@10 × (1 + α × sparse_ratio)
# Picks the checkpoint that best trades off retrieval quality against efficiency.
# α=0.5 means a 10% quality drop is acceptable for a 20% compression gain.
# ─────────────────────────────────────────────────────────────────────────────
if should_run find_bam_a; then
    log "STEP 7/9 — FIND BEST BAM OPTION A EPOCH (BSR, α=$BSR_ALPHA)"
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
# STEP 8 — BSR EPOCH SELECTION — OPTION B
# Same BSR metric as Option A. Option B's scattered dims mean avg_active_dims
# is a raw count (not a contiguous prefix), so sparse_ratio is computed differently
# but the selection criterion is identical.
# ─────────────────────────────────────────────────────────────────────────────
if should_run find_bam_b; then
    log "STEP 8/9 — FIND BEST BAM OPTION B EPOCH (BSR, α=$BSR_ALPHA)"
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
# STEP 9 — FULL EVALUATION
# eval_bam.py encodes the full corpus with each model, runs FAISS retrieval,
# and reports recall@K, NDCG@10, per-Bloom-level breakdown, avg_active_dims,
# latency. Option A is primary; Option B and MRL baseline are comparisons.
# ─────────────────────────────────────────────────────────────────────────────
if should_run eval_compare; then
    log "STEP 9/9 — FULL EVALUATION (Option A vs Option B vs MRL)"

    MRL_BEST="$MRL_CKPT_DIR/best"
    [[ -f "$RESULTS_DIR/mrl_best_path.txt" ]] && MRL_BEST=$(cat "$RESULTS_DIR/mrl_best_path.txt")
    BAM_A_BEST="$BAM_A_CKPT_DIR/best_bsr"
    BAM_B_BEST="$BAM_B_CKPT_DIR/best_bsr"

    [[ -f "$MRL_BEST/checkpoint.pt" ]]   || die "MRL best not found at $MRL_BEST — run find_mrl first"
    [[ -f "$BAM_A_BEST/checkpoint.pt" ]] || die "Option A best_bsr not found — run find_bam_a first"
    [[ -f "$BAM_B_BEST/checkpoint.pt" ]] || die "Option B best_bsr not found — run find_bam_b first"

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
echo "  MRL baseline       : $MRL_CKPT_DIR/best/"
echo "  Option A best (BSR): $BAM_A_CKPT_DIR/best_bsr/"
echo "  Option B best (BSR): $BAM_B_CKPT_DIR/best_bsr/"
echo "  BSR tables         : $RESULTS_DIR/optionA_bsr/  $RESULTS_DIR/optionB_bsr/"
echo "  Full eval results  : $RESULTS_DIR/results.json"
echo ""
echo "  Key metrics to compare:"
echo "    - recall@10, NDCG@10         (overall retrieval quality)"
echo "    - bloom_*_recall@10          (per-Bloom-level performance)"
echo "    - avg_active_dims            (efficiency — lower is better)"
echo "    - bloom_consistent_recall@10 (quality that the BSR step optimised for)"
echo "    NOTE: Option B dims are scattered — cannot use FAISS sub-index."
