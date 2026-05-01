#!/usr/bin/env bash
# =============================================================================
# BAM Council Pipeline — SHARED STEPS (run once before backbone scripts)
#
# Runs everything up to and including BAM-B:
#   Step 0 — Train Bloom council classifier
#   Step 1 — Build data + curriculum negatives (all 5 datasets)
#   Step 2 — Annotate with Bloom council
#   Step 3 — Train MRL baseline (e5-large, all datasets)
#   Step 4 — Find best MRL epoch
#   Step 5 — Train BAM-B (e5-large, all datasets)
#   Step 6 — Find best BAM-B epoch
#
# Usage:
#   ./pipeline_shared.sh                    # run shared steps
#   ./pipeline_shared.sh --force            # wipe and retrain everything
#   ./pipeline_shared.sh --from build       # skip council training
#   ./pipeline_shared.sh --datasets "educational msmarco"   # subset
#
# After this completes, run the 3 backbone scripts in parallel:
#   ./pipeline_backbones_1.sh &    # e5large + bge
#   ./pipeline_backbones_2.sh &    # qwen06b + qwen4b
#   ./pipeline_backbones_3.sh &    # llm2vec + gritlm
#   wait
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$SCRIPT_DIR/bam_council_pipeline.sh" \
    --until find_bam_b \
    "$@"
