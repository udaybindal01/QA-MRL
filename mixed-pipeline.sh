#!/usr/bin/env bash
# =============================================================================
# BAM Mixed-Domain Pipeline — Educational + BEIR Training
# =============================================================================
#
# Trains on educational data + BEIR train splits jointly, then evaluates on:
#   - Educational test set  (primary in-domain claim)
#   - BEIR test sets        (generalization — same datasets, held-out test split)
#
# This directly supports the paper claim:
#   "Bloom-level routing improves over standard MRL across domains — the model
#    learns that cognitively complex queries need more embedding dimensions
#    regardless of whether they come from educational or general IR corpora."
#
# Why this is legitimate:
#   - BEIR train and test splits are strictly separated (no leakage)
#   - Educational test set is never seen during training
#   - Mixed training prevents domain-specific overfitting
#   - BAM should outperform MRL on both test sets because:
#       * Educational: Bloom routing is semantically meaningful
#       * BEIR: SciFact (Analyze) gets more dims, NFCorpus (Remember) fewer
#
# Steps:
#   1  build_beir_data      — download BEIR train splits, annotate Bloom levels
#   2  mix_datasets         — combine BEIR train + educational curriculum
#   3  train_mrl            — MRL baseline on mixed data
#   4  find_mrl             — select best MRL epoch
#   5  train_bam_a          — BAM Option A (prefix routing)
#   6  train_bam_b          — BAM Option B (scattered mask, reverse two-stage)
#   7  find_bam_a           — BSR epoch selection for Option A
#   8  find_bam_b           — BSR epoch selection for Option B
#   9  eval_edu             — in-domain eval on educational test set
#   10 beir_mrl             — BEIR eval for MRL baseline
#   11 beir_bam_a           — BEIR eval for BAM Option A
#   12 beir_bam_b           — BEIR eval for BAM Option B
#   13 beir_compare         — print BEIR comparison table
#
# Prerequisites:
#   ./data/real/train_curriculum.jsonl  (run optionA-working_pipeline.sh first)
#   ./data/real/{val,test,corpus}.jsonl
#
# Usage:
#   chmod +x mixed-pipeline.sh
#   ./mixed-pipeline.sh                        # run all steps
#   ./mixed-pipeline.sh --from train_bam_b     # resume from a step
#   BEIR_DATASETS="scifact nfcorpus fiqa" ./mixed-pipeline.sh
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
EDU_CURRICULUM="./data/real/train_curriculum.jsonl"
BEIR_TRAIN_DIR="/tmp/data/beir_train"
MIXED_DATA_DIR="/tmp/data/mixed"

MRL_CKPT_DIR="/tmp/mrl-e5large-mixed-ckpts"
BAM_A_CKPT_DIR="/tmp/bam_optionA-e5large-mixed-ckpts"
BAM_B_CKPT_DIR="/tmp/bam_optionb-e5large-mixed-ckpts"
RESULTS_DIR="./results/mixed"

MRL_CONFIG="configs/mrl_e5large_mixed.yaml"
BAM_A_CONFIG="configs/bam_optionA_e5large_mixed.yaml"
BAM_B_CONFIG="configs/bam_optionb_e5large_mixed.yaml"

BSR_ALPHA="0.5"
BEIR_DATASETS="${BEIR_DATASETS:-scifact nfcorpus}"
BEIR_SPLIT="test"
# ─────────────────────────────────────────────────────────────────────────────

# ── Argument parsing ─────────────────────────────────────────────────────────
FROM_STEP=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM_STEP="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ALL_STEPS=(build_beir_data mix_datasets train_mrl find_mrl
           train_bam_a train_bam_b find_bam_a find_bam_b
           eval_edu beir_mrl beir_bam_a beir_bam_b beir_compare)

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

log "PREREQ CHECK"
[[ -f "$EDU_CURRICULUM" ]] \
    || die "train_curriculum.jsonl not found. Run optionA-working_pipeline.sh first."
[[ -f "./data/real/corpus.jsonl" ]] || die "corpus.jsonl not found."

mkdir -p "$BEIR_TRAIN_DIR" "$MIXED_DATA_DIR" \
         "$MRL_CKPT_DIR" "$BAM_A_CKPT_DIR" "$BAM_B_CKPT_DIR" "$RESULTS_DIR"

echo "  Educational curriculum : $EDU_CURRICULUM"
echo "  BEIR datasets          : $BEIR_DATASETS"
echo "  Mixed data             : $MIXED_DATA_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — BUILD BEIR TRAINING DATA
# Downloads train splits from HuggingFace, annotates with Bloom levels,
# mines hard negatives, writes JSONL in same format as educational data.
# ─────────────────────────────────────────────────────────────────────────────
if should_run build_beir_data; then
    log "STEP 1/13 — BUILD BEIR TRAINING DATA ($BEIR_DATASETS)"
    if [[ -f "$BEIR_TRAIN_DIR/combined_train.jsonl" ]]; then
        echo "  BEIR training data already exists — skipping."
    else
        python3 data/build_beir_training_data.py \
            --datasets $BEIR_DATASETS \
            --output_dir "$BEIR_TRAIN_DIR" \
            --num_neg 7 \
            || die "build_beir_training_data.py failed"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — MIX EDUCATIONAL + BEIR TRAINING DATA
# Concatenates both datasets into a single curriculum file.
# Educational data keeps its corpus; BEIR passages used only for training.
# ─────────────────────────────────────────────────────────────────────────────
if should_run mix_datasets; then
    log "STEP 2/13 — MIX DATASETS (educational + BEIR train)"
    if [[ -f "$MIXED_DATA_DIR/train_curriculum.jsonl" ]]; then
        echo "  Mixed curriculum already exists — skipping."
    else
        cat "$EDU_CURRICULUM" "$BEIR_TRAIN_DIR/combined_train.jsonl" \
            > "$MIXED_DATA_DIR/train_curriculum.jsonl"
        EDU_LINES=$(wc -l < "$EDU_CURRICULUM")
        BEIR_LINES=$(wc -l < "$BEIR_TRAIN_DIR/combined_train.jsonl")
        TOTAL=$(wc -l < "$MIXED_DATA_DIR/train_curriculum.jsonl")
        echo "  Educational : $EDU_LINES queries"
        echo "  BEIR train  : $BEIR_LINES queries"
        echo "  Combined    : $TOTAL queries → $MIXED_DATA_DIR/train_curriculum.jsonl"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TRAIN MRL BASELINE (mixed data)
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_mrl; then
    log "STEP 3/13 — TRAIN MRL BASELINE (mixed data, e5-large)"
    if [[ -f "$MRL_CKPT_DIR/best/checkpoint.pt" ]] || [[ -d "$MRL_CKPT_DIR/epoch_0" ]]; then
        echo "  MRL checkpoint already exists — skipping training."
    else
        python3 scripts/train_baseline_mrl.py \
            --config "$MRL_CONFIG" \
            || die "MRL training failed"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — SELECT BEST MRL EPOCH
# ─────────────────────────────────────────────────────────────────────────────
if should_run find_mrl; then
    log "STEP 4/13 — FIND BEST MRL EPOCH"
    if [[ -f "$MRL_CKPT_DIR/best/checkpoint.pt" ]]; then
        echo "  MRL best already exists."
    else
        [[ -d "$MRL_CKPT_DIR/epoch_0" ]] || die "No MRL epoch checkpoints — run train_mrl first"
        python3 scripts/find_best_epoch.py \
            --checkpoint_dir "$MRL_CKPT_DIR" \
            --config         "$MRL_CONFIG" \
            --model_type     mrl \
            || die "find_best_epoch.py failed"
    fi
    MRL_BEST="$MRL_CKPT_DIR/best"
    echo "$MRL_BEST" > "$RESULTS_DIR/mrl_best_path.txt"
    echo "  MRL best → $MRL_BEST"
fi

MRL_BEST="$MRL_CKPT_DIR/best"
[[ -f "$RESULTS_DIR/mrl_best_path.txt" ]] && MRL_BEST=$(cat "$RESULTS_DIR/mrl_best_path.txt")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — TRAIN BAM OPTION A (mixed data, prefix routing)
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_bam_a; then
    log "STEP 5/13 — TRAIN BAM OPTION A (mixed data, prefix routing)"
    [[ -f "$MRL_BEST/checkpoint.pt" ]] || die "MRL best not found — run find_mrl first"
    if [[ -f "$BAM_A_CKPT_DIR/best_bsr/checkpoint.pt" ]] || [[ -d "$BAM_A_CKPT_DIR/epoch_0" ]]; then
        echo "  Option A checkpoint already exists — skipping training."
    else
        python3 scripts/train_bam.py \
            --config       "$BAM_A_CONFIG" \
            --init_encoder "$MRL_BEST" \
            || die "Option A training failed"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — TRAIN BAM OPTION B (mixed data, reverse two-stage)
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_bam_b; then
    log "STEP 6/13 — TRAIN BAM OPTION B (mixed data, reverse two-stage)"
    [[ -f "$MRL_BEST/checkpoint.pt" ]] || die "MRL best not found — run find_mrl first"
    if [[ -f "$BAM_B_CKPT_DIR/best_bsr/checkpoint.pt" ]] || [[ -d "$BAM_B_CKPT_DIR/epoch_0" ]]; then
        echo "  Option B checkpoint already exists — skipping training."
    else
        python3 scripts/train_bam.py \
            --config       "$BAM_B_CONFIG" \
            --init_encoder "$MRL_BEST" \
            --freeze_encoder \
            || die "Option B training failed"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — BSR EPOCH SELECTION — OPTION A
# ─────────────────────────────────────────────────────────────────────────────
if should_run find_bam_a; then
    log "STEP 7/13 — FIND BEST BAM OPTION A EPOCH (BSR, α=$BSR_ALPHA)"
    [[ -d "$BAM_A_CKPT_DIR/epoch_0" ]] || die "No Option A checkpoints — run train_bam_a first"
    mkdir -p "$RESULTS_DIR/optionA_bsr"
    python3 scripts/find_best_epoch_bsr.py \
        --config         "$BAM_A_CONFIG" \
        --checkpoint_dir "$BAM_A_CKPT_DIR" \
        --output_dir     "$RESULTS_DIR/optionA_bsr/" \
        --alpha          "$BSR_ALPHA" \
        || die "BSR selection (A) failed"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — BSR EPOCH SELECTION — OPTION B
# ─────────────────────────────────────────────────────────────────────────────
if should_run find_bam_b; then
    log "STEP 8/13 — FIND BEST BAM OPTION B EPOCH (BSR, α=$BSR_ALPHA)"
    [[ -d "$BAM_B_CKPT_DIR/epoch_0" ]] || die "No Option B checkpoints — run train_bam_b first"
    mkdir -p "$RESULTS_DIR/optionB_bsr"
    python3 scripts/find_best_epoch_bsr.py \
        --config         "$BAM_B_CONFIG" \
        --checkpoint_dir "$BAM_B_CKPT_DIR" \
        --output_dir     "$RESULTS_DIR/optionB_bsr/" \
        --alpha          "$BSR_ALPHA" \
        || die "BSR selection (B) failed"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — IN-DOMAIN EVALUATION (educational test set)
# Val/test sets are always educational-only — never contaminated with BEIR.
# ─────────────────────────────────────────────────────────────────────────────
if should_run eval_edu; then
    log "STEP 9/13 — EDUCATIONAL IN-DOMAIN EVALUATION"
    BAM_A_BEST="$BAM_A_CKPT_DIR/best_bsr"
    BAM_B_BEST="$BAM_B_CKPT_DIR/best_bsr"

    [[ -f "$MRL_BEST/checkpoint.pt" ]]   || die "MRL best not found"
    [[ -f "$BAM_A_BEST/checkpoint.pt" ]] || die "Option A best_bsr not found — run find_bam_a first"
    [[ -f "$BAM_B_BEST/checkpoint.pt" ]] || die "Option B best_bsr not found — run find_bam_b first"

    python3 scripts/eval_bam.py \
        --config        "$BAM_A_CONFIG" \
        --checkpoint    "$BAM_A_BEST" \
        --baseline      "$MRL_BEST" \
        --checkpoint_v4 "$BAM_B_BEST" \
        --config_v4     "$BAM_B_CONFIG" \
        --output_dir    "$RESULTS_DIR/" \
        || die "eval_bam.py failed"
    echo "  Educational results → $RESULTS_DIR/results.json"
fi

# ─────────────────────────────────────────────────────────────────────────────
# BEIR EVALUATION (Steps 10-13)
# Test splits only — never seen during training. Shows generalization.
# ─────────────────────────────────────────────────────────────────────────────

BEIR_RESULTS="$RESULTS_DIR/beir"
mkdir -p "$BEIR_RESULTS"

if should_run beir_mrl; then
    log "STEP 10/13 — BEIR: MRL BASELINE ($BEIR_DATASETS)"
    python3 scripts/eval_beir.py \
        --config      "$MRL_CONFIG" \
        --checkpoint  "$MRL_BEST" \
        --model_type  mrl \
        --datasets    $BEIR_DATASETS \
        --split       "$BEIR_SPLIT" \
        --output_dir  "$BEIR_RESULTS/mrl/" \
        || die "BEIR (MRL) failed"
fi

if should_run beir_bam_a; then
    log "STEP 11/13 — BEIR: BAM OPTION A ($BEIR_DATASETS)"
    python3 scripts/eval_beir.py \
        --config      "$BAM_A_CONFIG" \
        --checkpoint  "$BAM_A_CKPT_DIR/best_bsr" \
        --model_type  bam \
        --datasets    $BEIR_DATASETS \
        --split       "$BEIR_SPLIT" \
        --output_dir  "$BEIR_RESULTS/bam_a/" \
        || die "BEIR (Option A) failed"
fi

if should_run beir_bam_b; then
    log "STEP 12/13 — BEIR: BAM OPTION B ($BEIR_DATASETS)"
    python3 scripts/eval_beir.py \
        --config      "$BAM_B_CONFIG" \
        --checkpoint  "$BAM_B_CKPT_DIR/best_bsr" \
        --model_type  bam \
        --datasets    $BEIR_DATASETS \
        --split       "$BEIR_SPLIT" \
        --output_dir  "$BEIR_RESULTS/bam_b_dense/" \
        || die "BEIR (Option B) failed"
fi

if should_run beir_compare; then
    log "STEP 13/13 — BEIR COMPARISON TABLE"
    python3 -c "
import json, os, sys
results_dir = '$BEIR_RESULTS'
datasets = '$BEIR_DATASETS'.split()
models = [
    ('MRL Baseline',         'mrl/beir_results.json'),
    ('BAM Option A',         'bam_a/beir_results.json'),
    ('BAM Option B (dense)', 'bam_b_dense/beir_results.json'),
]
data = {}
for label, path in models:
    fpath = os.path.join(results_dir, path)
    if os.path.exists(fpath):
        with open(fpath) as f:
            raw = json.load(f)
        for model_key, ds_results in raw.items():
            data[label] = ds_results
if not data:
    print('No BEIR results found.')
    sys.exit(0)
for ds in datasets:
    print(f'\n  Dataset: {ds}')
    print(f'  {\"Model\":30s} {\"NDCG@10\":>10s} {\"R@10\":>10s} {\"R@100\":>10s} {\"AvgDims\":>10s}')
    print('  ' + '-' * 64)
    for label in [m[0] for m in models]:
        if label not in data or ds not in data[label]:
            continue
        m = data[label][ds]
        dims = m.get('avg_active_dims', '-')
        dims_str = f'{dims:.0f}' if isinstance(dims, (int, float)) else dims
        print(f'  {label:30s} {m.get(\"ndcg@10\",0):>10.4f} {m.get(\"recall@10\",0):>10.4f} '
              f'{m.get(\"recall@100\",0):>10.4f} {dims_str:>10s}')
print()
" || echo "  (comparison table failed)"
fi

# ─────────────────────────────────────────────────────────────────────────────
log "MIXED PIPELINE COMPLETE"
echo ""
echo "  Mixed training data  : $MIXED_DATA_DIR/train_curriculum.jsonl"
echo "  MRL best             : $MRL_CKPT_DIR/best/"
echo "  Option A best (BSR)  : $BAM_A_CKPT_DIR/best_bsr/"
echo "  Option B best (BSR)  : $BAM_B_CKPT_DIR/best_bsr/"
echo "  Educational eval     : $RESULTS_DIR/results.json"
echo "  BEIR eval            : $BEIR_RESULTS/"
echo ""
echo "  Paper tables:"
echo "    Table 1 (in-domain): $RESULTS_DIR/results.json"
echo "    Table 2 (BEIR):      $BEIR_RESULTS/"
