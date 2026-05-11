#!/usr/bin/env bash
# Loss ablation study for BAM-PQ (e5-large backbone).
#
# Step 0: Train the MRL baseline (e5-large) on the educational dataset.
#         Finds best epoch and saves to /tmp/bam-pq-ckpts/abl_mrl/best
# Step 1: Train all 7 BAM-PQ ablation variants, each warm-started from
#         the same MRL checkpoint — controlled experiment.
# Step 2: Collect results and print table.
#
# Usage:
#   bash scripts/run_loss_ablations.sh              # MRL + all 7 variants
#   bash scripts/run_loss_ablations.sh full          # MRL (if needed) + full model only
#   bash scripts/run_loss_ablations.sh no_sparsity   # MRL (if needed) + single ablation
#   SKIP_MRL=1 bash scripts/run_loss_ablations.sh   # skip MRL training (reuse existing)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MRL_CKPT_DIR="/tmp/bam-pq-ckpts/abl_mrl"
MRL_BEST="${MRL_CKPT_DIR}/best"
MRL_CFG="configs/mrl_e5large.yaml"
SKIP_MRL="${SKIP_MRL:-0}"

# ── Step 0: Train MRL baseline ───────────────────────────────────────────────
if [[ "$SKIP_MRL" == "1" ]] && [[ -f "${MRL_BEST}/checkpoint.pt" ]]; then
    echo "  SKIP_MRL=1 — reusing MRL checkpoint at ${MRL_BEST}"
elif [[ -f "${MRL_BEST}/checkpoint.pt" ]] && [[ "$SKIP_MRL" != "0" ]]; then
    echo "  MRL checkpoint already exists at ${MRL_BEST} — skipping."
else
    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  Step 0: Training MRL baseline (e5-large)"
    echo "══════════════════════════════════════════════════"
    mkdir -p "${MRL_CKPT_DIR}"
    python scripts/train_baseline_mrl.py \
        --config         "${MRL_CFG}" \
        --checkpoint_dir "${MRL_CKPT_DIR}"

    echo ""
    echo "  Selecting best MRL epoch ..."
    python scripts/find_best_epoch.py \
        --config         "${MRL_CFG}" \
        --checkpoint_dir "${MRL_CKPT_DIR}" \
        --model_type     mrl \
        --output_dir     "${MRL_BEST}"

    echo "  MRL baseline ready → ${MRL_BEST}"
fi

if [[ ! -f "${MRL_BEST}/checkpoint.pt" ]]; then
    echo "ERROR: MRL checkpoint not found at ${MRL_BEST}/checkpoint.pt"
    echo "  MRL training may have failed. Check logs above."
    exit 1
fi

echo ""
echo "  Using MRL init: ${MRL_BEST}"

ABLATIONS=(
    full
    no_sparsity
    no_diversity
    no_variance
    no_distill
    no_query_div
    contrastive_only
)

run_variant() {
    local name="$1"
    local cfg="configs/ablations/abl_${name}.yaml"
    local ckpt_dir="/tmp/bam-pq-ckpts/abl_${name}"
    local best_dir="${ckpt_dir}/best_bsr"

    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  Training ablation: ${name}"
    echo "══════════════════════════════════════════════════"

    # Train — warm-start from the MRL checkpoint trained in Step 0
    python scripts/train_bam.py \
        --config         "${cfg}" \
        --checkpoint_dir "${ckpt_dir}" \
        --init_encoder   "${MRL_BEST}" \
        --freeze_encoder

    echo ""
    echo "  Selecting best checkpoint for ${name} ..."
    python scripts/find_best_epoch_bsr.py \
        --checkpoint_dir "${ckpt_dir}" \
        --config         "${cfg}" \
        --output_dir     "${best_dir}"

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

# If argument given, run only that variant (MRL already handled above)
if [[ $# -gt 0 ]]; then
    run_variant "$1"
    exit 0
fi

# Otherwise run all 7 variants
for name in "${ABLATIONS[@]}"; do
    run_variant "$name"
done

echo ""
echo "══════════════════════════════════════════════════"
echo "  All ablations complete. Collecting results ..."
echo "══════════════════════════════════════════════════"
python scripts/collect_ablation_results.py
