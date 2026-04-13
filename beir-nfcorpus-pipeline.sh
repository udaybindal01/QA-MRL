#!/usr/bin/env bash
# =============================================================================
# BEIR/NFCorpus Evaluation Pipeline — E5-large BAM models
# =============================================================================
#
# Evaluates all three trained e5-large models on the BEIR NFCorpus benchmark
# (biomedical information retrieval). This is an OUT-OF-DOMAIN evaluation:
# the models were trained on educational science data, but NFCorpus queries
# are biomedical. Good performance here demonstrates generalization.
#
# What this pipeline tests:
#   - Does BAM routing degrade out-of-domain retrieval quality?
#   - Does Option B's scattered mask provide real efficiency gains?
#   - How does MRL truncation compare to BAM adaptive masking on unseen data?
#
# Since NFCorpus has no Bloom labels, the BAM models use their default
# routing behavior:
#   Option A: defaults to Bloom level 5 (Create) → highest dim = full dims
#   Option B: defaults to Bloom level 5 (Create) mask → ~276 active dims (27%)
#             This shows true efficiency: 73% dimension reduction even without
#             Bloom-specific routing.
#
# Pipeline overview:
#
#   Step 1  eval_mrl        — evaluate MRL baseline on NFCorpus
#                             Also runs MRL truncation comparisons at each
#                             dim in [64, 128, 256, 512, 768, 1024]
#
#   Step 2  eval_bam_a      — evaluate BAM Option A (prefix router)
#                             Without Bloom labels: uses full-dim prefix mask
#
#   Step 3  eval_bam_b      — evaluate BAM Option B (scattered mask)
#                             Without Bloom labels: uses Create-level mask (~276 dims)
#                             Runs both dense (full-emb dot product) and sparse
#                             (mask-active-only dot product) retrieval
#
#   Step 4  compare         — print side-by-side NDCG@10 / R@10 / R@100 / MAP
#
# Prerequisites:
#   Trained checkpoints from e5large-pipeline.sh:
#     /tmp/mrl-e5large-ckpts/best/checkpoint.pt
#     /tmp/bam-a-e5large-ckpts1/best_bsr/checkpoint.pt
#     /tmp/bam-b-e5large-ckpts2/best_bsr/checkpoint.pt
#
#   NFCorpus data: auto-downloaded to data/beir/nfcorpus/ on first run (~50 MB).
#
#   Required packages: beir (pip install beir), faiss-gpu or faiss-cpu
#
# Usage:
#   chmod +x beir-nfcorpus-pipeline.sh
#   ./beir-nfcorpus-pipeline.sh                     # run all steps
#   ./beir-nfcorpus-pipeline.sh --from eval_bam_b   # resume from a step
#   DATASETS="nfcorpus scifact fiqa" ./beir-nfcorpus-pipeline.sh  # multiple datasets
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MRL_CKPT_DIR="/tmp/mrl-e5large-ckpts"
BAM_A_CKPT_DIR="/tmp/bam-a-e5large-ckpts1"
BAM_B_CKPT_DIR="/tmp/bam-b-e5large-ckpts2"
RESULTS_DIR="./results/beir_e5large"

MRL_CONFIG="configs/mrl_e5large.yaml"
BAM_A_CONFIG="configs/bam_optionA_e5large.yaml"
BAM_B_CONFIG="configs/bam_optionb_e5large.yaml"

# Which BEIR datasets to evaluate on. Override with env var for more:
#   DATASETS="nfcorpus scifact fiqa arguana scidocs" ./beir-nfcorpus-pipeline.sh
DATASETS="${DATASETS:-nfcorpus}"
# ─────────────────────────────────────────────────────────────────────────────

# ── Argument parsing ──────────────────────────────────────────────────────────
FROM_STEP=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM_STEP="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ALL_STEPS=(eval_mrl eval_bam_a eval_bam_b compare)

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
# Try resolved path from training pipeline
TRAINING_RESULTS="./results/bam_e5large2"
if [[ -f "$TRAINING_RESULTS/mrl_best_path.txt" ]]; then
    MRL_BEST=$(cat "$TRAINING_RESULTS/mrl_best_path.txt")
fi

BAM_A_BEST="$BAM_A_CKPT_DIR/best_bsr"
BAM_B_BEST="$BAM_B_CKPT_DIR/best_bsr"

[[ -f "$MRL_BEST/checkpoint.pt" ]]   || die "MRL best not found at $MRL_BEST. Run e5large-pipeline.sh first."
[[ -f "$BAM_A_BEST/checkpoint.pt" ]] || die "Option A best_bsr not found at $BAM_A_BEST. Run e5large-pipeline.sh first."
[[ -f "$BAM_B_BEST/checkpoint.pt" ]] || die "Option B best_bsr not found at $BAM_B_BEST. Run e5large-pipeline.sh first."

echo "  Datasets   : $DATASETS"
echo "  MRL        : $MRL_BEST"
echo "  Option A   : $BAM_A_BEST"
echo "  Option B   : $BAM_B_BEST"
echo "  Results    : $RESULTS_DIR/"

mkdir -p "$RESULTS_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — EVALUATE MRL BASELINE
# Runs full-dimensional retrieval + MRL truncation comparisons at each
# configured mrl_dim. This shows what fixed-dim truncation can achieve
# without any routing — the baseline that BAM needs to beat.
# ─────────────────────────────────────────────────────────────────────────────
if should_run eval_mrl; then
    log "STEP 1/4 — EVALUATE MRL BASELINE ON BEIR ($DATASETS)"

    python3 scripts/eval_beir.py \
        --config      "$MRL_CONFIG" \
        --checkpoint  "$MRL_BEST" \
        --model_type  mrl \
        --datasets    $DATASETS \
        --output_dir  "$RESULTS_DIR/mrl/" \
        || die "eval_beir.py (MRL) failed"

    echo "  MRL results → $RESULTS_DIR/mrl/beir_results.json"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — EVALUATE BAM OPTION A (prefix router)
# Without Bloom labels, encode_queries defaults to level 5 (Create) which
# maps to the highest prefix dim. This is effectively full-dimensional
# retrieval for Option A — the interesting metric is whether the encoder
# quality was preserved or degraded by BAM training.
# ─────────────────────────────────────────────────────────────────────────────
if should_run eval_bam_a; then
    log "STEP 2/4 — EVALUATE BAM OPTION A ON BEIR ($DATASETS)"

    python3 scripts/eval_beir.py \
        --config      "$BAM_A_CONFIG" \
        --checkpoint  "$BAM_A_BEST" \
        --model_type  bam \
        --datasets    $DATASETS \
        --output_dir  "$RESULTS_DIR/bam_a/" \
        || die "eval_beir.py (Option A) failed"

    echo "  Option A results → $RESULTS_DIR/bam_a/beir_results.json"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — EVALUATE BAM OPTION B (scattered mask)
# Two modes:
#   Dense:  normalized(full_emb * mask) · normalized(full_emb_doc)
#           Standard dot product — same FAISS index as MRL/Option A
#   Sparse: only compute dot product over mask-active dims per query
#           Shows real computational savings, but slower to evaluate
#           (builds per-pattern FAISS sub-index)
#
# Without Bloom labels, uses Create-level mask (~27% of 1024 = ~276 dims).
# This is the key efficiency result: 73% dimension reduction on unseen data.
# ─────────────────────────────────────────────────────────────────────────────
if should_run eval_bam_b; then
    log "STEP 3/4 — EVALUATE BAM OPTION B ON BEIR ($DATASETS)"

    # Dense retrieval (comparable to MRL/Option A)
    python3 scripts/eval_beir.py \
        --config      "$BAM_B_CONFIG" \
        --checkpoint  "$BAM_B_BEST" \
        --model_type  bam \
        --datasets    $DATASETS \
        --output_dir  "$RESULTS_DIR/bam_b_dense/" \
        || die "eval_beir.py (Option B dense) failed"

    echo "  Option B (dense) results → $RESULTS_DIR/bam_b_dense/beir_results.json"

    # Sparse retrieval (true efficiency — only active dims)
    python3 scripts/eval_beir.py \
        --config      "$BAM_B_CONFIG" \
        --checkpoint  "$BAM_B_BEST" \
        --model_type  bam \
        --sparse \
        --datasets    $DATASETS \
        --output_dir  "$RESULTS_DIR/bam_b_sparse/" \
        || die "eval_beir.py (Option B sparse) failed"

    echo "  Option B (sparse) results → $RESULTS_DIR/bam_b_sparse/beir_results.json"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — COMPARISON TABLE
# Reads the JSON results from steps 1-3 and prints a side-by-side comparison.
# ─────────────────────────────────────────────────────────────────────────────
if should_run compare; then
    log "STEP 4/4 — COMPARISON TABLE"

    python3 -c "
import json, os, sys

results_dir = '$RESULTS_DIR'
datasets = '$DATASETS'.split()

models = [
    ('MRL Baseline',      'mrl/beir_results.json'),
    ('BAM Option A',      'bam_a/beir_results.json'),
    ('BAM Option B (dense)', 'bam_b_dense/beir_results.json'),
    ('BAM Option B (sparse)', 'bam_b_sparse/beir_results.json'),
]

data = {}
for label, path in models:
    fpath = os.path.join(results_dir, path)
    if os.path.exists(fpath):
        with open(fpath) as f:
            raw = json.load(f)
        # The JSON has {model_label: {dataset: {metric: value}}}
        for model_key, ds_results in raw.items():
            data[label] = ds_results

if not data:
    print('No results found. Run steps 1-3 first.')
    sys.exit(0)

# Print table
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
echo "  Results directory: $RESULTS_DIR/"
echo "    mrl/beir_results.json          — MRL baseline + truncation comparisons"
echo "    bam_a/beir_results.json        — BAM Option A (prefix router)"
echo "    bam_b_dense/beir_results.json  — BAM Option B (dense retrieval)"
echo "    bam_b_sparse/beir_results.json — BAM Option B (sparse, true efficiency)"
echo ""
echo "  Key questions this answers:"
echo "    1. Does BAM routing hurt out-of-domain quality? (compare MRL vs BAM A)"
echo "    2. Does scattered masking work? (compare BAM B dense vs sparse)"
echo "    3. What's the quality/efficiency tradeoff? (BAM B dims vs NDCG drop)"
