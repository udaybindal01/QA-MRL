#!/usr/bin/env bash
# =============================================================================
# BAM MS MARCO Pipeline — General IR Claim
# =============================================================================
#
# Trains and evaluates MRL, BAM Option A, and BAM Option B on MS MARCO.
#
# Why this pipeline:
#   Training on MS MARCO (general IR) lets us claim Bloom-adaptive routing
#   improves upon standard MRL for ANY query corpus, not just educational IR.
#   Even general web queries vary in cognitive complexity:
#     "What is GDP?"                    → Remember  (few dims needed)
#     "Why did the 2008 crisis spread?" → Analyze   (more dims needed)
#   Bloom routing assigns dimensions based on this complexity.
#   BEIR evaluation then shows BAM beats or matches MRL out-of-domain.
#
# Pipeline overview:
#   Step 1  build_data           — download MS MARCO, annotate Bloom levels
#   Step 2  curriculum_negatives — mine hard negatives for training
#   Step 3  train_mrl            — MRL baseline on MS MARCO
#   Step 4  find_mrl             — select best MRL epoch (val NDCG)
#   Step 5  train_bam_a          — BAM Option A (prefix routing)
#   Step 6  train_bam_b          — BAM Option B (scattered mask, reverse two-stage)
#   Step 7  find_bam_a           — BSR epoch selection for Option A
#   Step 8  find_bam_b           — BSR epoch selection for Option B
#   Step 9  eval_compare         — in-domain eval (MS MARCO test set)
#   Step 10 beir_mrl             — BEIR eval for MRL baseline
#   Step 11 beir_bam_a           — BEIR eval for BAM Option A
#   Step 12 beir_bam_b           — BEIR eval for BAM Option B
#   Step 13 beir_compare         — print comparison table
#
# Usage:
#   chmod +x msmarco-pipeline.sh
#   ./msmarco-pipeline.sh                       # run all steps
#   ./msmarco-pipeline.sh --from train_bam_a    # resume from a step
#   MAX_TRAIN=50000 ./msmarco-pipeline.sh       # smaller dataset for fast runs
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR="/tmp/data/msmarco"
MRL_CKPT_DIR="/tmp/mrl-e5large-msmarco-ckpts"
BAM_A_CKPT_DIR="/tmp/bam-a-e5large-msmarco-ckpts"
BAM_B_CKPT_DIR="/tmp/bam-b-e5large-msmarco-ckpts"
RESULTS_DIR="./results/msmarco"

MRL_CONFIG="configs/mrl_e5large_msmarco.yaml"
BAM_A_CONFIG="configs/bam_optionA_e5large_msmarco.yaml"
BAM_B_CONFIG="configs/bam_optionb_e5large_msmarco.yaml"

BSR_ALPHA="0.5"
BEIR_DATASETS="scifact nfcorpus fiqa"   # OOD from MS MARCO — shows generalization
BEIR_SPLIT="test"
MAX_TRAIN="${MAX_TRAIN:-100000}"         # set MAX_TRAIN=50000 for faster iteration
# ─────────────────────────────────────────────────────────────────────────────

# ── Argument parsing ─────────────────────────────────────────────────────────
FROM_STEP=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM_STEP="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ALL_STEPS=(build_data curriculum_negatives train_mrl find_mrl
           train_bam_a train_bam_b find_bam_a find_bam_b
           eval_compare beir_mrl beir_bam_a beir_bam_b beir_compare)

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
mkdir -p "$DATA_DIR" "$MRL_CKPT_DIR" "$BAM_A_CKPT_DIR" "$BAM_B_CKPT_DIR" "$RESULTS_DIR"
echo "  Data dir   : $DATA_DIR"
echo "  Results    : $RESULTS_DIR"
echo "  MAX_TRAIN  : $MAX_TRAIN"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — BUILD MS MARCO DATA
# Downloads passages + queries from HuggingFace, annotates with Bloom levels,
# writes corpus/train/val/test JSONL files.
# ─────────────────────────────────────────────────────────────────────────────
if should_run build_data; then
    log "STEP 1/13 — BUILD MS MARCO DATA (max_train=$MAX_TRAIN)"
    if [[ -f "$DATA_DIR/corpus.jsonl" ]] && [[ -f "$DATA_DIR/train.jsonl" ]]; then
        echo "  MS MARCO data already exists at $DATA_DIR — skipping download."
    else
        python3 data/build_msmarco_data.py \
            --output_dir "$DATA_DIR" \
            --max_train  "$MAX_TRAIN" \
            --num_neg    7 \
            || die "build_msmarco_data.py failed"
        echo "  MS MARCO data → $DATA_DIR/"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — MINE HARD NEGATIVES
# ─────────────────────────────────────────────────────────────────────────────
if should_run curriculum_negatives; then
    log "STEP 2/13 — MINE HARD NEGATIVES"
    if [[ -f "$DATA_DIR/train_curriculum.jsonl" ]]; then
        echo "  Curriculum already exists — skipping."
    else
        python3 data/curriculum_negatives.py \
            --pairs  "$DATA_DIR/train.jsonl" \
            --corpus "$DATA_DIR/corpus.jsonl" \
            --output "$DATA_DIR/train_curriculum.jsonl" \
            --num_neg 7 \
            --stage  0.7 \
            || die "curriculum_negatives.py failed"
        echo "  Curriculum → $DATA_DIR/train_curriculum.jsonl"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TRAIN MRL BASELINE (MS MARCO)
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_mrl; then
    log "STEP 3/13 — TRAIN MRL BASELINE (MS MARCO, e5-large)"
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
# STEP 5 — TRAIN BAM OPTION A (MS MARCO, prefix routing)
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_bam_a; then
    log "STEP 5/13 — TRAIN BAM OPTION A (MS MARCO, prefix routing)"
    [[ -f "$MRL_BEST/checkpoint.pt" ]] || die "MRL best not found at $MRL_BEST — run find_mrl first"
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
# STEP 6 — TRAIN BAM OPTION B (MS MARCO, reverse two-stage)
# Stage 1: encoder frozen — mask learns which dims to activate per Bloom level
# Stage 2: encoder unfreezes at 1e-6 LR — gentle adaptation
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_bam_b; then
    log "STEP 6/13 — TRAIN BAM OPTION B (MS MARCO, reverse two-stage)"
    [[ -f "$MRL_BEST/checkpoint.pt" ]] || die "MRL best not found at $MRL_BEST — run find_mrl first"
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
    echo "  Option A best → $BAM_A_CKPT_DIR/best_bsr/"
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
    echo "  Option B best → $BAM_B_CKPT_DIR/best_bsr/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — FULL IN-DOMAIN EVALUATION (MS MARCO test set)
# ─────────────────────────────────────────────────────────────────────────────
if should_run eval_compare; then
    log "STEP 9/13 — FULL EVALUATION (MS MARCO: Option A vs Option B vs MRL)"
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
    echo "  In-domain results → $RESULTS_DIR/results.json"
fi

# ─────────────────────────────────────────────────────────────────────────────
# BEIR EVALUATION (Steps 10-13)
# MS MARCO → BEIR is a realistic OOD setting. BAM should beat MRL here because:
#   - Model trained on diverse general queries (not just educational)
#   - Bloom routing is meaningful for general queries (vary in complexity)
#   - Complex BEIR queries (SciFact = claim verification = Analyze) get more dims
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
" || echo "  (comparison table failed — check individual JSON files)"
fi

# ─────────────────────────────────────────────────────────────────────────────
log "MS MARCO PIPELINE COMPLETE"
echo ""
echo "  Data               : $DATA_DIR/"
echo "  MRL best           : $MRL_CKPT_DIR/best/"
echo "  Option A best (BSR): $BAM_A_CKPT_DIR/best_bsr/"
echo "  Option B best (BSR): $BAM_B_CKPT_DIR/best_bsr/"
echo "  In-domain eval     : $RESULTS_DIR/results.json"
echo "  BEIR eval          : $BEIR_RESULTS/"
echo ""
echo "  Paper framing:"
echo "    BAM is a general improvement over MRL — Bloom-level routing adapts"
echo "    embedding dimensionality to query cognitive complexity for any corpus."
