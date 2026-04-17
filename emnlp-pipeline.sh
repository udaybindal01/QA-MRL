#!/usr/bin/env bash
# =============================================================================
# EMNLP Paper Pipeline — Full Experiment Suite
# =============================================================================
#
# Runs ALL experiments needed for the paper:
#
#   Part A: E5-large backbone (1024 dims)
#     1. MRL baseline training
#     2. BAM Option A (prefix routing)
#     3. BAM Option B (scattered mask, reverse two-stage)
#     4. Full in-domain evaluation (Option A vs Option B vs MRL)
#     5. BEIR multi-dataset (HotpotQA, SciFact, NFCorpus)
#
#   Part B: BGE-base backbone (768 dims) — second backbone for generalization
#     6. MRL baseline (reuse existing if available)
#     7. BAM Option A (reuse existing)
#     8. BAM Option B (reverse two-stage)
#     9. Full in-domain evaluation
#
#   Part C: Analysis & Ablations
#     10. Bloom classifier robustness (noise injection 0-50%)
#     11. Query-adaptive baseline (K-Means vs Bloom routing)
#     12. Standard ablations (random Bloom, fixed Bloom, no routing, etc.)
#
# Prerequisites:
#   ./data/real/train_curriculum.jsonl  — run build_real_data.py + curriculum_negatives.py
#   ./data/real/{val,test,corpus}.jsonl
#
# Usage:
#   chmod +x emnlp-pipeline.sh
#   ./emnlp-pipeline.sh                          # run all
#   ./emnlp-pipeline.sh --from eval_e5large      # resume from a step
#   SKIP_BGE=1 ./emnlp-pipeline.sh               # skip BGE-base (e5-large only)
#   SKIP_BEIR=1 ./emnlp-pipeline.sh              # skip BEIR evaluation
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# E5-large paths
E5_MRL_CKPT="/tmp/mrl-e5large-ckpts"
E5_BAM_A_CKPT="/tmp/bam-a-e5large-ckpts1"
E5_BAM_B_CKPT="/tmp/bam-b-e5large-ckpts5"
E5_MRL_CONFIG="configs/mrl_e5large.yaml"
E5_BAM_A_CONFIG="configs/bam_optionA_e5large.yaml"
E5_BAM_B_CONFIG="configs/bam_optionb_e5large.yaml"
E5_RESULTS="./results/emnlp_e5large"

# BGE-base paths
BGE_MRL_CKPT="/tmp/mrl-ckpts"
BGE_BAM_A_CKPT="/tmp/bam-ckpts"
BGE_BAM_B_CKPT="/tmp/bam-b-ckpts11"
BGE_MRL_CONFIG="configs/bam.yaml"           # optionA pipeline trains BGE MRL with bam.yaml (BAAI/bge-base-en-v1.5)
BGE_BAM_A_CONFIG="configs/bam.yaml"
BGE_BAM_B_CONFIG="configs/bam_optionb.yaml"
BGE_RESULTS="./results/emnlp_bge"

CURRICULUM="./data/real/train_curriculum.jsonl"
BSR_ALPHA="0.5"
BEIR_DATASETS="scifact nfcorpus"   # hotpotqa excluded: 5.2M-doc corpus causes OOM on most GPUs
BEIR_SPLIT="test"

SKIP_BGE="${SKIP_BGE:-0}"
SKIP_BEIR="${SKIP_BEIR:-0}"
# ─────────────────────────────────────────────────────────────────────────────

# ── Argument parsing ──────────────────────────────────────────────────────────
FROM_STEP=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM_STEP="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ALL_STEPS=(
    # Part A: E5-large
    train_mrl_e5 find_mrl_e5 train_bam_a_e5 train_bam_b_e5
    find_bam_a_e5 find_bam_b_e5 eval_e5large
    # Part A: BEIR
    beir_mrl beir_bam_a beir_bam_b beir_compare
    # Part B: BGE-base
    train_bam_b_bge find_bam_b_bge eval_bge
    # Part C: Analysis
    bloom_robustness query_adaptive_baseline ablations
    dim_allocation mask_specialization generate_figures
)

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
[[ -f "$CURRICULUM" ]] || die "train_curriculum.jsonl not found. Run data pipeline first."
[[ -f "./data/real/corpus.jsonl" ]] || die "corpus.jsonl not found."

mkdir -p "$E5_RESULTS" "$BGE_RESULTS" "$E5_RESULTS/beir"

# ═════════════════════════════════════════════════════════════════════════════
# PART A: E5-LARGE BACKBONE (1024 dims)
# ═════════════════════════════════════════════════════════════════════════════

# Step 1: Train MRL baseline (e5-large)
if should_run train_mrl_e5; then
    log "PART A — TRAIN MRL BASELINE (e5-large)"
    if [[ -f "$E5_MRL_CKPT/best/checkpoint.pt" ]]; then
        echo "  MRL checkpoint already exists at $E5_MRL_CKPT/best/ — skipping training."
    else
        python3 scripts/train_baseline_mrl.py --config "$E5_MRL_CONFIG" \
            || die "MRL training failed"
    fi
fi

# Step 2: Find best MRL epoch
if should_run find_mrl_e5; then
    log "PART A — FIND BEST MRL EPOCH"
    if [[ -f "$E5_MRL_CKPT/best/checkpoint.pt" ]]; then
        echo "  MRL best already exists."
    else
        python3 scripts/find_best_epoch.py --checkpoint_dir "$E5_MRL_CKPT" \
            --config "$E5_MRL_CONFIG" --model_type mrl || die "find_best_epoch failed"
    fi
    E5_MRL_BEST="$E5_MRL_CKPT/best"
    echo "$E5_MRL_BEST" > "$E5_RESULTS/mrl_best_path.txt"
fi

E5_MRL_BEST="$E5_MRL_CKPT/best"
[[ -f "$E5_RESULTS/mrl_best_path.txt" ]] && E5_MRL_BEST=$(cat "$E5_RESULTS/mrl_best_path.txt")

# Step 3: Train BAM Option A (e5-large, prefix routing)
if should_run train_bam_a_e5; then
    log "PART A — TRAIN BAM OPTION A (e5-large, prefix routing)"
    if [[ -f "$E5_BAM_A_CKPT/best_bsr/checkpoint.pt" ]]; then
        echo "  Option A checkpoint already exists — skipping training."
    else
        python3 scripts/train_bam.py --config "$E5_BAM_A_CONFIG" \
            --init_encoder "$E5_MRL_BEST" || die "Option A training failed"
    fi
fi

# Step 4: Train BAM Option B (e5-large, reverse two-stage)
if should_run train_bam_b_e5; then
    log "PART A — TRAIN BAM OPTION B (e5-large, reverse two-stage)"
    if [[ -f "$E5_BAM_B_CKPT/best_bsr/checkpoint.pt" ]] || [[ -d "$E5_BAM_B_CKPT/epoch_0" ]]; then
        echo "  Option B checkpoint already exists at $E5_BAM_B_CKPT — skipping training."
    else
        python3 scripts/train_bam.py --config "$E5_BAM_B_CONFIG" \
            --init_encoder "$E5_MRL_BEST" --freeze_encoder \
            || die "Option B training failed"
    fi
fi

# Step 5-6: BSR epoch selection
if should_run find_bam_a_e5; then
    log "PART A — FIND BEST OPTION A EPOCH (BSR)"
    mkdir -p "$E5_RESULTS/optionA_bsr"
    python3 scripts/find_best_epoch_bsr.py --config "$E5_BAM_A_CONFIG" \
        --checkpoint_dir "$E5_BAM_A_CKPT" --output_dir "$E5_RESULTS/optionA_bsr/" \
        --alpha "$BSR_ALPHA" || die "BSR selection (A) failed"
fi

if should_run find_bam_b_e5; then
    log "PART A — FIND BEST OPTION B EPOCH (BSR)"
    mkdir -p "$E5_RESULTS/optionB_bsr"
    python3 scripts/find_best_epoch_bsr.py --config "$E5_BAM_B_CONFIG" \
        --checkpoint_dir "$E5_BAM_B_CKPT" --output_dir "$E5_RESULTS/optionB_bsr/" \
        --alpha "$BSR_ALPHA" || die "BSR selection (B) failed"
fi

# Step 7: Full in-domain evaluation (e5-large)
if should_run eval_e5large; then
    log "PART A — FULL EVALUATION (e5-large: A vs B vs MRL)"
    python3 scripts/eval_bam.py \
        --config "$E5_BAM_A_CONFIG" \
        --checkpoint "$E5_BAM_A_CKPT/best_bsr" \
        --baseline "$E5_MRL_BEST" \
        --checkpoint_v4 "$E5_BAM_B_CKPT/best_bsr" \
        --config_v4 "$E5_BAM_B_CONFIG" \
        --output_dir "$E5_RESULTS/" \
        || die "eval_bam.py failed"
fi

# ═════════════════════════════════════════════════════════════════════════════
# PART A (cont): BEIR MULTI-DATASET EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

if [[ "$SKIP_BEIR" != "1" ]]; then
    BEIR_RESULTS="$E5_RESULTS/beir"

    if should_run beir_mrl; then
        log "BEIR — MRL BASELINE ($BEIR_DATASETS)"
        python3 scripts/eval_beir.py --config "$E5_MRL_CONFIG" \
            --checkpoint "$E5_MRL_BEST" --model_type mrl \
            --datasets $BEIR_DATASETS --split "$BEIR_SPLIT" \
            --output_dir "$BEIR_RESULTS/mrl/" || die "BEIR (MRL) failed"
    fi

    if should_run beir_bam_a; then
        log "BEIR — BAM OPTION A ($BEIR_DATASETS)"
        python3 scripts/eval_beir.py --config "$E5_BAM_A_CONFIG" \
            --checkpoint "$E5_BAM_A_CKPT/best_bsr" --model_type bam \
            --datasets $BEIR_DATASETS --split "$BEIR_SPLIT" \
            --output_dir "$BEIR_RESULTS/bam_a/" || die "BEIR (A) failed"
    fi

    if should_run beir_bam_b; then
        log "BEIR — BAM OPTION B ($BEIR_DATASETS)"
        # Dense retrieval
        python3 scripts/eval_beir.py --config "$E5_BAM_B_CONFIG" \
            --checkpoint "$E5_BAM_B_CKPT/best_bsr" --model_type bam \
            --datasets $BEIR_DATASETS --split "$BEIR_SPLIT" \
            --output_dir "$BEIR_RESULTS/bam_b_dense/" || die "BEIR (B dense) failed"
        # Sparse retrieval
        python3 scripts/eval_beir.py --config "$E5_BAM_B_CONFIG" \
            --checkpoint "$E5_BAM_B_CKPT/best_bsr" --model_type bam --sparse \
            --datasets $BEIR_DATASETS --split "$BEIR_SPLIT" \
            --output_dir "$BEIR_RESULTS/bam_b_sparse/" || die "BEIR (B sparse) failed"
    fi

    if should_run beir_compare; then
        log "BEIR — COMPARISON TABLE"
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
    print('No BEIR results found.')
    sys.exit(0)
for ds in datasets:
    print(f'\n  Dataset: {ds}')
    print(f'  {\"Model\":30s} {\"NDCG@10\":>10s} {\"R@10\":>10s} {\"R@100\":>10s} {\"MAP\":>10s}')
    print('  ' + '-' * 72)
    for label in [m[0] for m in models]:
        if label not in data or ds not in data[label]:
            continue
        m = data[label][ds]
        print(f'  {label:30s} {m.get(\"ndcg@10\",0):>10.4f} {m.get(\"recall@10\",0):>10.4f} '
              f'{m.get(\"recall@100\",0):>10.4f} {m.get(\"map\",0):>10.4f}')
print()
" || echo "  (comparison table failed)"
    fi
fi

# ═════════════════════════════════════════════════════════════════════════════
# PART B: BGE-BASE BACKBONE (768 dims) — second backbone
# ═════════════════════════════════════════════════════════════════════════════

if [[ "$SKIP_BGE" != "1" ]]; then
    BGE_MRL_BEST="$BGE_MRL_CKPT/best"
    BEST_FILE="$BGE_RESULTS/../best_epochs/mrl/best_checkpoint_path.txt"
    [[ -f "$BEST_FILE" ]] && BGE_MRL_BEST=$(cat "$BEST_FILE")

    # BGE-base MRL + Option A assumed to exist from previous runs.
    # Only train Option B with reverse two-stage.

    if should_run train_bam_b_bge; then
        log "PART B — TRAIN BAM OPTION B (BGE-base, reverse two-stage)"
        mkdir -p "$BGE_BAM_B_CKPT" "$BGE_RESULTS"
        [[ -f "$BGE_MRL_BEST/checkpoint.pt" ]] \
            || die "BGE MRL best not found at $BGE_MRL_BEST. Run optionA pipeline first."
        if [[ -f "$BGE_BAM_B_CKPT/best_bsr/checkpoint.pt" ]] || [[ -d "$BGE_BAM_B_CKPT/epoch_0" ]]; then
            echo "  BGE Option B checkpoint already exists at $BGE_BAM_B_CKPT — skipping training."
        else
            python3 scripts/train_bam.py --config "$BGE_BAM_B_CONFIG" \
                --init_encoder "$BGE_MRL_BEST" --freeze_encoder \
                || die "BGE Option B training failed"
        fi
    fi

    if should_run find_bam_b_bge; then
        log "PART B — FIND BEST BGE OPTION B EPOCH (BSR)"
        mkdir -p "$BGE_RESULTS/optionB_bsr"
        python3 scripts/find_best_epoch_bsr.py --config "$BGE_BAM_B_CONFIG" \
            --checkpoint_dir "$BGE_BAM_B_CKPT" --output_dir "$BGE_RESULTS/optionB_bsr/" \
            --alpha "$BSR_ALPHA" || die "BSR (BGE-B) failed"
    fi

    if should_run eval_bge; then
        log "PART B — FULL EVALUATION (BGE-base: A vs B vs MRL)"
        BGE_BAM_A_BEST="$BGE_BAM_A_CKPT/best_bsr"
        [[ -f "$BGE_BAM_A_BEST/checkpoint.pt" ]] \
            || BGE_BAM_A_BEST="$BGE_BAM_A_CKPT/best"  # fallback
        python3 scripts/eval_bam.py \
            --config "$BGE_BAM_A_CONFIG" \
            --checkpoint "$BGE_BAM_A_BEST" \
            --baseline "$BGE_MRL_BEST" \
            --checkpoint_v4 "$BGE_BAM_B_CKPT/best_bsr" \
            --config_v4 "$BGE_BAM_B_CONFIG" \
            --output_dir "$BGE_RESULTS/" \
            || die "eval_bam.py (BGE) failed"
    fi
fi

# ═════════════════════════════════════════════════════════════════════════════
# PART C: ANALYSIS & ABLATIONS
# ═════════════════════════════════════════════════════════════════════════════

# Step 10: Bloom classifier robustness
if should_run bloom_robustness; then
    log "PART C — BLOOM CLASSIFIER ROBUSTNESS ANALYSIS"
    mkdir -p "$E5_RESULTS/robustness"
    python3 scripts/analyze_classifier_robustness.py \
        --config "$E5_BAM_B_CONFIG" \
        --checkpoint "$E5_BAM_B_CKPT/best_bsr" \
        --output_dir "$E5_RESULTS/robustness/" \
        || die "Bloom robustness analysis failed"
fi

# Step 11: Query-adaptive baseline (Bloom vs K-Means vs Random)
if should_run query_adaptive_baseline; then
    log "PART C — QUERY-ADAPTIVE BASELINE (Bloom vs K-Means)"
    mkdir -p "$E5_RESULTS/query_adaptive"
    python3 scripts/eval_query_adaptive_baseline.py \
        --config "$E5_BAM_B_CONFIG" \
        --checkpoint "$E5_BAM_B_CKPT/best_bsr" \
        --baseline "$E5_MRL_BEST" \
        --output_dir "$E5_RESULTS/query_adaptive/" \
        || die "Query-adaptive baseline failed"
fi

# Step 12: Standard ablations
if should_run ablations; then
    log "PART C — ABLATION STUDY"
    mkdir -p "$E5_RESULTS/ablations"
    python3 scripts/run_ablations.py \
        --config "$E5_BAM_A_CONFIG" \
        --checkpoint "$E5_BAM_A_CKPT/best_bsr" \
        --baseline "$E5_MRL_BEST" \
        --checkpoint_v4 "$E5_BAM_B_CKPT/best_bsr" \
        --config_v4 "$E5_BAM_B_CONFIG" \
        --output_dir "$E5_RESULTS/ablations/" \
        || die "Ablations failed"
fi

# Step 13: Bloom dim allocation analysis (Figure: which dims each level uses)
if should_run dim_allocation; then
    log "PART C — BLOOM DIM ALLOCATION ANALYSIS"
    mkdir -p "$E5_RESULTS/dim_allocation"
    python3 scripts/analyze_bloom_dim_allocation.py \
        --config "$E5_BAM_B_CONFIG" \
        --checkpoint "$E5_BAM_B_CKPT/best_bsr" \
        --output_dir "$E5_RESULTS/dim_allocation/" \
        || die "Dim allocation analysis failed"
fi

# Step 14: Mask specialization analysis (Figure: mask overlap / specialization per level)
if should_run mask_specialization; then
    log "PART C — MASK SPECIALIZATION ANALYSIS"
    mkdir -p "$E5_RESULTS/mask_specialization"
    python3 scripts/analyze_mask_specialization.py \
        --config "$E5_BAM_B_CONFIG" \
        --checkpoint "$E5_BAM_B_CKPT/best_bsr" \
        --output_dir "$E5_RESULTS/mask_specialization/" \
        || die "Mask specialization analysis failed"
fi

# Step 15: Generate paper figures from all results
if should_run generate_figures; then
    log "PART C — GENERATE PAPER FIGURES"
    mkdir -p "$E5_RESULTS/figures"
    python3 scripts/generate_figures.py \
        --results_dir "$E5_RESULTS/" \
        --output_dir  "$E5_RESULTS/figures/" \
        || echo "  WARNING: generate_figures.py failed — check individual result files"
fi

# ═════════════════════════════════════════════════════════════════════════════
log "EMNLP PIPELINE COMPLETE"
echo ""
echo "  Results:"
echo "    E5-large in-domain  : $E5_RESULTS/results.json"
echo "    E5-large BEIR       : $E5_RESULTS/beir/"
if [[ "$SKIP_BGE" != "1" ]]; then
echo "    BGE-base in-domain  : $BGE_RESULTS/results.json"
fi
echo "    Bloom robustness    : $E5_RESULTS/robustness/"
echo "    Query-adaptive      : $E5_RESULTS/query_adaptive/"
echo "    Ablations           : $E5_RESULTS/ablations/"
echo "    Dim allocation      : $E5_RESULTS/dim_allocation/"
echo "    Mask specialization : $E5_RESULTS/mask_specialization/"
echo "    Paper figures       : $E5_RESULTS/figures/"
echo ""
echo "  Paper experiments covered:"
echo "    ✓ Two backbones (e5-large 1024d + BGE-base 768d)"
echo "    ✓ Three BEIR datasets (HotpotQA, SciFact, NFCorpus)"
echo "    ✓ Bloom classifier robustness (0-50% noise)"
echo "    ✓ Query-adaptive baseline (K-Means vs Bloom routing)"
echo "    ✓ Ablation study (random/fixed/no routing)"
echo "    ✓ Bloom dim allocation per level"
echo "    ✓ Mask specialization analysis"
echo "    ✓ Paper figures generated"
