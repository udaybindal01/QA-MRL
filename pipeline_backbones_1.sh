#!/usr/bin/env bash
# =============================================================================
# BAM-PQ Backbone Group 1 — e5-large + BGE-large
#
# Runs BAM-PQ train → best-epoch selection → eval for these two backbones
# across all 5 datasets (educational, msmarco, scifact, nfcorpus, fiqa).
#
# MRL warm-start:
#   e5large — reuses shared MRL checkpoint from pipeline_shared.sh
#   bge     — trains its own bge MRL baseline first, then warm-starts BAM-PQ from it
# Both MRL checkpoints also serve as the evaluation baselines (BAM-PQ vs MRL).
#
# Run AFTER pipeline_shared.sh completes:
#   ./pipeline_backbones_1.sh &
#   ./pipeline_backbones_2.sh &
#   ./pipeline_backbones_3.sh &
#   wait
#
# Usage:
#   ./pipeline_backbones_1.sh                        # all 5 datasets
#   ./pipeline_backbones_1.sh --datasets "educational msmarco"
#   ./pipeline_backbones_1.sh --force                # wipe and retrain
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BACKBONES_TO_RUN="e5large bge" \
REUSE_TRAINED_MODELS=1 \
exec bash "$SCRIPT_DIR/bam_council_pipeline.sh" \
    --from build \
    "$@"
