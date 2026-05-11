#!/usr/bin/env bash
# Loss ablation study for BAM-PQ (e5-large backbone).
#
# All 7 variants warm-start from the same MRL checkpoint so comparisons are fair.
# Init order matches the real pipeline (bam_council_pipeline.sh):
#   e5-large BAM-PQ normally inits from BAM-B, but for ablations we init from
#   the MRL baseline directly — same init for all 7 variants, controlled experiment.
#
# Set MRL_INIT to override the default MRL checkpoint path.
#
# Usage:
#   bash scripts/run_loss_ablations.sh              # all 7 variants
#   bash scripts/run_loss_ablations.sh full          # only full model
#   bash scripts/run_loss_ablations.sh no_sparsity   # single ablation
#   MRL_INIT=/my/ckpt bash scripts/run_loss_ablations.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# MRL checkpoint used to warm-start all ablation variants.
# Matches the backbone-matched MRL from the real pipeline.
MRL_INIT="${MRL_INIT:-/tmp/uday/multi-domain/educational/mrl_e5large/best}"

if [[ ! -f "${MRL_INIT}/checkpoint.pt" ]]; then
    echo "ERROR: MRL init checkpoint not found at ${MRL_INIT}/checkpoint.pt"
    echo "  Set MRL_INIT=/path/to/mrl/best to override."
    exit 1
fi
echo "  MRL init: ${MRL_INIT}"

ABLATIONS=(
    full
    no_sparsity
    no_diversity
    no_variance
    no_distill
    no_query_div
    contrastive_only
)

DESCRIPTIONS=(
    "Full model (all losses)"
    "No mask sparsity"
    "No mask diversity"
    "No mask variance"
    "No mask distillation"
    "No query routing diversity"
    "Contrastive only (lower bound)"
)

run_variant() {
    local name="$1"
    local cfg="configs/ablations/abl_${name}.yaml"
    local ckpt_dir="/tmp/bam-pq-ckpts/abl_${name}"
    local best_dir="${ckpt_dir}/best_bsr"

    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  Training: ${name}"
    echo "══════════════════════════════════════════════════"

    # Train — warm-start from MRL (same init for all variants = controlled ablation)
    python scripts/train_bam.py \
        --config         "${cfg}" \
        --checkpoint_dir "${ckpt_dir}" \
        --init_encoder   "${MRL_INIT}" \
        --freeze_encoder

    echo ""
    echo "  Selecting best checkpoint for ${name} ..."
    python scripts/find_best_epoch_bsr.py \
        --checkpoint_dir "${ckpt_dir}" \
        --config "${cfg}" \
        --output_dir "${best_dir}"

    echo ""
    echo "  Evaluating ${name} ..."
    python scripts/evaluate.py \
        --corpus_path ./data/real/corpus.jsonl \
        --test_path   ./data/real/test.jsonl \
        --model_type  bam_pq \
        --config      "${cfg}" \
        --checkpoint  "${best_dir}" \
        --output_dir  "results/ablations/"

    echo "  Done: ${name}"
}

# If argument given, run only that variant
if [[ $# -gt 0 ]]; then
    run_variant "$1"
    exit 0
fi

# Otherwise run all
for name in "${ABLATIONS[@]}"; do
    run_variant "$name"
done

echo ""
echo "══════════════════════════════════════════════════"
echo "  All ablations complete. Collecting results ..."
echo "══════════════════════════════════════════════════"
python scripts/collect_ablation_results.py
