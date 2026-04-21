#!/usr/bin/env bash
# =============================================================================
# Statistical significance testing: MRL vs BAM-B vs BAM-PQ
#
# Uses already-trained checkpoints — no training.
# Runs paired Wilcoxon signed-rank tests with Bonferroni correction.
# Reports: Δ mean, 95% bootstrap CI, p-value, significance stars.
#
# Usage:
#   chmod +x scripts/run_significance_tests.sh
#   ./scripts/run_significance_tests.sh
#
#   # Subset of datasets:
#   DATASETS="educational scifact" ./scripts/run_significance_tests.sh
# =============================================================================

set -euo pipefail

DATASETS="${DATASETS:-educational scifact nfcorpus fiqa}"
CKPT_ROOT="${CKPT_ROOT:-/tmp/multi-domain}"
CFG_ROOT="${CFG_ROOT:-./results/multi_domain}"
BEIR_ROOT="${BEIR_ROOT:-/tmp/data/beir}"
OUTPUT_DIR="${OUTPUT_DIR:-results/significance}"

log() { echo ""; echo "══════════════════════════════════════════════════════"; \
        echo "  [$(date '+%H:%M:%S')]  $*"; \
        echo "══════════════════════════════════════════════════════"; }

log "SIGNIFICANCE TESTING"
echo "  Datasets : $DATASETS"
echo "  Ckpt root: $CKPT_ROOT"
echo "  Output   : $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# Install scipy if missing (needed for Wilcoxon test)
python3 -c "import scipy" 2>/dev/null || {
    echo "  Installing scipy ..."
    pip install -q scipy
}

python3 scripts/run_significance_tests.py \
    --datasets  $DATASETS    \
    --ckpt_root "$CKPT_ROOT" \
    --cfg_root  "$CFG_ROOT"  \
    --beir_root "$BEIR_ROOT" \
    --output_dir "$OUTPUT_DIR"

log "DONE"
echo ""
echo "  Results: $OUTPUT_DIR/significance_results.json"
