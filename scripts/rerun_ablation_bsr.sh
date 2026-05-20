#!/usr/bin/env bash
# Re-run find_best_epoch_bsr.py for all ablation variants with the fixed evaluator
# (masked-query vs full-corpus, matching training forward pass).
#
# Usage:
#   bash scripts/rerun_ablation_bsr.sh
#   bash scripts/rerun_ablation_bsr.sh full          # single variant
#   bash scripts/rerun_ablation_bsr.sh no_sparsity

set -euo pipefail

CKPT_ROOT="${CKPT_ROOT:-/tmp/bam-pq-ckpts}"
ALPHA="${ALPHA:-0.5}"
SINGLE="${1:-}"

ABLATIONS=(full no_sparsity no_diversity no_variance no_distill no_query_div contrastive_only)

log() { echo ""; echo "══════════════════════════════════════════════════"; echo "  $*"; echo "══════════════════════════════════════════════════"; }

for name in "${ABLATIONS[@]}"; do
    [[ -n "$SINGLE" && "$name" != "$SINGLE" ]] && continue

    cfg="configs/ablations/abl_${name}.yaml"
    ckpt_dir="${CKPT_ROOT}/abl_${name}"
    best_dir="${ckpt_dir}/best_bsr"

    if [[ ! -f "$cfg" ]]; then
        echo "  [$name] config not found: $cfg — skipping"
        continue
    fi
    if [[ ! -d "$ckpt_dir" ]]; then
        echo "  [$name] checkpoint dir not found: $ckpt_dir — skipping"
        continue
    fi

    log "[$name] re-selecting best BSR epoch"
    python scripts/find_best_epoch_bsr.py \
        --config         "$cfg"      \
        --checkpoint_dir "$ckpt_dir" \
        --output_dir     "$best_dir" \
        --alpha          "$ALPHA"    \
        || { echo "  [$name] FAILED"; continue; }

    echo "  [$name] done → $best_dir"
done

log "ALL DONE"
echo "  Results in: ${CKPT_ROOT}/abl_<variant>/best_bsr/epoch_results_bsr.json"
echo ""
echo "  To collect all into a table:"
echo "    python scripts/collect_ablation_bsr.py --ckpt_root $CKPT_ROOT"
