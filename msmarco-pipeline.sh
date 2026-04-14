#!/usr/bin/env bash
# =============================================================================
# MS MARCO Pipeline — Train + Evaluate BAM on MS MARCO
# =============================================================================
#
# Full end-to-end pipeline: download MS MARCO → mine negatives → train MRL →
# train BAM Option A → train BAM Option B → in-domain eval → BEIR eval.
#
# Backbone: intfloat/e5-large-v2 (335M params, 1024-dim, NOT MRL pre-trained)
#
# Pipeline overview (14 steps):
#
#   Step 1   build_data         — download MS MARCO, build corpus + pairs + Bloom annotation
#   Step 2   mine_negatives     — BM25 hard negative mining
#   Step 3   train_mrl          — MRL baseline (teaches multi-resolution structure)
#   Step 4   find_mrl           — find best MRL epoch
#   Step 5   train_bam_a        — BAM Option A (prefix router, MRL warm-start)
#   Step 6   train_bam_b        — BAM Option B (scattered mask, MRL warm-start)
#   Step 7   find_bam_a         — BSR epoch selection for Option A
#   Step 8   find_bam_b         — BSR epoch selection for Option B
#   Step 9   eval_compare       — in-domain eval (Option A vs B vs MRL)
#   Step 10  beir_mrl           — BEIR eval: MRL baseline
#   Step 11  beir_bam_a         — BEIR eval: BAM Option A (Bloom-annotated)
#   Step 12  beir_bam_b         — BEIR eval: BAM Option B dense + sparse
#   Step 13  beir_compare       — BEIR comparison table
#
# Usage:
#   chmod +x msmarco-pipeline.sh
#   ./msmarco-pipeline.sh                          # run all steps
#   ./msmarco-pipeline.sh --from train_bam_a       # resume from a step
#   MAX_TRAIN=50000 ./msmarco-pipeline.sh          # smaller training set
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR="/tmp/data/msmarco"
MRL_CKPT_DIR="/tmp/mrl-e5large-msmarco-ckpts"
BAM_A_CKPT_DIR="/tmp/bam-a-e5large-msmarco-ckpts"
BAM_B_CKPT_DIR="/tmp/bam-b-e5large-msmarco-ckpts"
RESULTS_DIR="./results/bam_e5large_msmarco"

MRL_CONFIG="configs/mrl_e5large_msmarco.yaml"
BAM_A_CONFIG="configs/bam_optionA_e5large_msmarco.yaml"
BAM_B_CONFIG="configs/bam_optionb_e5large_msmarco.yaml"

MAX_TRAIN="${MAX_TRAIN:-100000}"   # Training pairs (default 100k)
NUM_NEG=7
BSR_ALPHA="0.5"

# BEIR evaluation datasets (MS MARCO dev + optional others)
BEIR_DATASETS="${BEIR_DATASETS:-msmarco}"
BEIR_SPLIT="dev"
# ─────────────────────────────────────────────────────────────────────────────

# ── Argument parsing ──────────────────────────────────────────────────────────
FROM_STEP=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM_STEP="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ALL_STEPS=(build_data mine_negatives train_mrl find_mrl train_bam_a train_bam_b find_bam_a find_bam_b eval_compare beir_mrl beir_bam_a beir_bam_b beir_compare)

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
echo "  Backbone   : intfloat/e5-large-v2 (1024-dim, NOT MRL pre-trained)"
echo "  Data       : $DATA_DIR/"
echo "  Max train  : $MAX_TRAIN"
echo "  Results    : $RESULTS_DIR/"

mkdir -p "$DATA_DIR" "$MRL_CKPT_DIR" "$BAM_A_CKPT_DIR" "$BAM_B_CKPT_DIR" "$RESULTS_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — BUILD MS MARCO DATA
# Downloads MS MARCO, extracts passages and query-passage pairs,
# annotates queries with Bloom taxonomy levels.
# ─────────────────────────────────────────────────────────────────────────────
if should_run build_data; then
    log "STEP 1/13 — BUILD MS MARCO DATA (max_train=$MAX_TRAIN)"

    python3 data/build_msmarco_data.py \
        --output_dir "$DATA_DIR" \
        --max_train  "$MAX_TRAIN" \
        --num_neg    "$NUM_NEG" \
        || die "build_msmarco_data.py failed"

    echo "  train: $(wc -l < "$DATA_DIR/train.jsonl") pairs"
    echo "  val:   $(wc -l < "$DATA_DIR/val.jsonl") pairs"
    echo "  test:  $(wc -l < "$DATA_DIR/test.jsonl") pairs"
    echo "  corpus: $(wc -l < "$DATA_DIR/corpus.jsonl") passages"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — MINE HARD NEGATIVES
# BM25-based hard negative mining against the MS MARCO corpus.
# ─────────────────────────────────────────────────────────────────────────────
if should_run mine_negatives; then
    log "STEP 2/13 — MINE HARD NEGATIVES (num_neg=$NUM_NEG)"

    [[ -f "$DATA_DIR/train.jsonl" ]] \
        || die "train.jsonl not found — run build_data first"

    python3 data/curriculum_negatives.py \
        --pairs  "$DATA_DIR/train.jsonl" \
        --corpus "$DATA_DIR/corpus.jsonl" \
        --output "$DATA_DIR/train_curriculum.jsonl" \
        --num_neg "$NUM_NEG" \
        --stage  0.7 \
        || die "curriculum_negatives.py failed"

    echo "  Curriculum negatives → $DATA_DIR/train_curriculum.jsonl"
    echo "  $(wc -l < "$DATA_DIR/train_curriculum.jsonl") pairs"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TRAIN MRL BASELINE
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_mrl; then
    log "STEP 3/13 — TRAIN MRL BASELINE (e5-large on MS MARCO)"

    [[ -f "$DATA_DIR/train_curriculum.jsonl" ]] \
        || die "train_curriculum.jsonl not found — run mine_negatives first"

    echo "  Config     : $MRL_CONFIG"
    echo "  Output     : $MRL_CKPT_DIR/"

    python3 scripts/train_baseline_mrl.py \
        --config "$MRL_CONFIG" \
        || die "train_baseline_mrl.py failed"

    echo "  MRL checkpoints → $MRL_CKPT_DIR/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — FIND BEST MRL EPOCH
# ─────────────────────────────────────────────────────────────────────────────
if should_run find_mrl; then
    log "STEP 4/13 — FIND BEST MRL EPOCH (val NDCG)"

    [[ -d "$MRL_CKPT_DIR/epoch_0" ]] \
        || die "No MRL epoch checkpoints at $MRL_CKPT_DIR — run train_mrl first"

    python3 scripts/find_best_epoch.py \
        --checkpoint_dir "$MRL_CKPT_DIR" \
        --config         "$MRL_CONFIG" \
        --model_type     mrl \
        || die "find_best_epoch.py failed"

    MRL_BEST="$MRL_CKPT_DIR/best"
    echo "$MRL_BEST" > "$RESULTS_DIR/mrl_best_path.txt"
    echo "  MRL best → $MRL_BEST"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — TRAIN BAM OPTION A (prefix router, MRL warm-start)
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_bam_a; then
    log "STEP 5/13 — TRAIN BAM OPTION A (prefix router, MRL warm-start)"

    MRL_BEST="$MRL_CKPT_DIR/best"
    [[ -f "$RESULTS_DIR/mrl_best_path.txt" ]] && MRL_BEST=$(cat "$RESULTS_DIR/mrl_best_path.txt")
    [[ -f "$MRL_BEST/checkpoint.pt" ]] \
        || die "MRL best not found at $MRL_BEST — run find_mrl first"

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
# STEP 6 — TRAIN BAM OPTION B (scattered mask, MRL warm-start)
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_bam_b; then
    log "STEP 6/13 — TRAIN BAM OPTION B (scattered mask, MRL warm-start)"

    MRL_BEST="$MRL_CKPT_DIR/best"
    [[ -f "$RESULTS_DIR/mrl_best_path.txt" ]] && MRL_BEST=$(cat "$RESULTS_DIR/mrl_best_path.txt")
    [[ -f "$MRL_BEST/checkpoint.pt" ]] \
        || die "MRL best not found at $MRL_BEST — run find_mrl first"

    echo "  Config     : $BAM_B_CONFIG"
    echo "  Init       : $MRL_BEST"
    echo "  Output     : $BAM_B_CKPT_DIR/"

    python3 scripts/train_bam.py \
        --config       "$BAM_B_CONFIG" \
        --init_encoder "$MRL_BEST" \
        || die "train_bam.py (Option B) failed"

    echo "  BAM Option B checkpoints → $BAM_B_CKPT_DIR/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — BSR EPOCH SELECTION — OPTION A
# ─────────────────────────────────────────────────────────────────────────────
if should_run find_bam_a; then
    log "STEP 7/13 — FIND BEST BAM OPTION A EPOCH (BSR, α=$BSR_ALPHA)"

    [[ -d "$BAM_A_CKPT_DIR/epoch_0" ]] \
        || die "No BAM-A checkpoints at $BAM_A_CKPT_DIR — run train_bam_a first"

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
# ─────────────────────────────────────────────────────────────────────────────
if should_run find_bam_b; then
    log "STEP 8/13 — FIND BEST BAM OPTION B EPOCH (BSR, α=$BSR_ALPHA)"

    [[ -d "$BAM_B_CKPT_DIR/epoch_0" ]] \
        || die "No BAM-B checkpoints at $BAM_B_CKPT_DIR — run train_bam_b first"

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
# STEP 9 — IN-DOMAIN EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
if should_run eval_compare; then
    log "STEP 9/13 — IN-DOMAIN EVALUATION (Option A vs Option B vs MRL)"

    MRL_BEST="$MRL_CKPT_DIR/best"
    [[ -f "$RESULTS_DIR/mrl_best_path.txt" ]] && MRL_BEST=$(cat "$RESULTS_DIR/mrl_best_path.txt")
    BAM_A_BEST="$BAM_A_CKPT_DIR/best_bsr"
    BAM_B_BEST="$BAM_B_CKPT_DIR/best_bsr"

    [[ -f "$MRL_BEST/checkpoint.pt" ]]   || die "MRL best not found — run find_mrl first"
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

    echo "  In-domain results → $RESULTS_DIR/results.json"
fi

# ─────────────────────────────────────────────────────────────────────────────
# BEIR EVALUATION (Steps 10-13)
# Evaluate on BEIR MS MARCO dev split (and optionally other BEIR datasets).
# Queries auto-annotated with Bloom levels for per-query BAM routing.
# ─────────────────────────────────────────────────────────────────────────────

BEIR_RESULTS="$RESULTS_DIR/beir"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — BEIR: MRL BASELINE
# ─────────────────────────────────────────────────────────────────────────────
if should_run beir_mrl; then
    log "STEP 10/13 — BEIR: MRL BASELINE ($BEIR_DATASETS, split=$BEIR_SPLIT)"

    MRL_BEST="$MRL_CKPT_DIR/best"
    [[ -f "$RESULTS_DIR/mrl_best_path.txt" ]] && MRL_BEST=$(cat "$RESULTS_DIR/mrl_best_path.txt")

    python3 scripts/eval_beir.py \
        --config      "$MRL_CONFIG" \
        --checkpoint  "$MRL_BEST" \
        --model_type  mrl \
        --datasets    $BEIR_DATASETS \
        --split       "$BEIR_SPLIT" \
        --output_dir  "$BEIR_RESULTS/mrl/" \
        || die "eval_beir.py (MRL) failed"

    echo "  MRL BEIR results → $BEIR_RESULTS/mrl/beir_results.json"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — BEIR: BAM OPTION A (Bloom-annotated queries)
# ─────────────────────────────────────────────────────────────────────────────
if should_run beir_bam_a; then
    log "STEP 11/13 — BEIR: BAM OPTION A ($BEIR_DATASETS, split=$BEIR_SPLIT)"

    BAM_A_BEST="$BAM_A_CKPT_DIR/best_bsr"

    python3 scripts/eval_beir.py \
        --config      "$BAM_A_CONFIG" \
        --checkpoint  "$BAM_A_BEST" \
        --model_type  bam \
        --datasets    $BEIR_DATASETS \
        --split       "$BEIR_SPLIT" \
        --output_dir  "$BEIR_RESULTS/bam_a/" \
        || die "eval_beir.py (Option A) failed"

    echo "  Option A BEIR results → $BEIR_RESULTS/bam_a/beir_results.json"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 12 — BEIR: BAM OPTION B (dense + sparse, Bloom-annotated queries)
# ─────────────────────────────────────────────────────────────────────────────
if should_run beir_bam_b; then
    log "STEP 12/13 — BEIR: BAM OPTION B ($BEIR_DATASETS, split=$BEIR_SPLIT)"

    BAM_B_BEST="$BAM_B_CKPT_DIR/best_bsr"

    # Dense retrieval
    python3 scripts/eval_beir.py \
        --config      "$BAM_B_CONFIG" \
        --checkpoint  "$BAM_B_BEST" \
        --model_type  bam \
        --datasets    $BEIR_DATASETS \
        --split       "$BEIR_SPLIT" \
        --output_dir  "$BEIR_RESULTS/bam_b_dense/" \
        || die "eval_beir.py (Option B dense) failed"

    echo "  Option B (dense) → $BEIR_RESULTS/bam_b_dense/beir_results.json"

    # Sparse retrieval (true efficiency)
    python3 scripts/eval_beir.py \
        --config      "$BAM_B_CONFIG" \
        --checkpoint  "$BAM_B_BEST" \
        --model_type  bam \
        --sparse \
        --datasets    $BEIR_DATASETS \
        --split       "$BEIR_SPLIT" \
        --output_dir  "$BEIR_RESULTS/bam_b_sparse/" \
        || die "eval_beir.py (Option B sparse) failed"

    echo "  Option B (sparse) → $BEIR_RESULTS/bam_b_sparse/beir_results.json"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 13 — BEIR: COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
if should_run beir_compare; then
    log "STEP 13/13 — BEIR COMPARISON TABLE"

    python3 -c "
import json, os, sys

results_dir = '$BEIR_RESULTS'
datasets = '$BEIR_DATASETS'.split()

models = [
    ('MRL Baseline',            'mrl/beir_results.json'),
    ('BAM Option A',            'bam_a/beir_results.json'),
    ('BAM Option B (dense)',    'bam_b_dense/beir_results.json'),
    ('BAM Option B (sparse)',   'bam_b_sparse/beir_results.json'),
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
    print('No BEIR results found. Run steps 10-12 first.')
    sys.exit(0)

metrics = ['ndcg@10', 'recall@10', 'recall@100', 'map']
for ds in datasets:
    print(f'\n  Dataset: {ds}')
    print(f'  {\"Model\":30s} {\"NDCG@10\":>10s} {\"R@10\":>10s} {\"R@100\":>10s} {\"MAP\":>10s} {\"AvgDims\":>10s}')
    print('  ' + '-' * 82)
    for label in [m[0] for m in models]:
        if label not in data or ds not in data[label]:
            continue
        m = data[label][ds]
        dims = m.get('avg_active_dims', '-')
        dims_str = f'{dims:.0f}' if isinstance(dims, (int, float)) else dims
        print(f'  {label:30s} {m.get(\"ndcg@10\",0):>10.4f} {m.get(\"recall@10\",0):>10.4f} '
              f'{m.get(\"recall@100\",0):>10.4f} {m.get(\"map\",0):>10.4f} {dims_str:>10s}')
print()
" || echo "  (comparison script failed — check individual JSON files)"
fi

# ─────────────────────────────────────────────────────────────────────────────
log "PIPELINE COMPLETE"
echo ""
echo "  Data               : $DATA_DIR/"
echo "  MRL baseline       : $MRL_CKPT_DIR/best/"
echo "  Option A best (BSR): $BAM_A_CKPT_DIR/best_bsr/"
echo "  Option B best (BSR): $BAM_B_CKPT_DIR/best_bsr/"
echo "  In-domain eval     : $RESULTS_DIR/results.json"
echo "  BEIR eval          : $BEIR_RESULTS/"
echo ""
echo "  Key metrics:"
echo "    In-domain: recall@10, NDCG@10, bloom_*_recall@10, avg_active_dims"
echo "    BEIR:      NDCG@10, R@10, R@100, MAP, avg_active_dims"
