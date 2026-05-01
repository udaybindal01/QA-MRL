#!/usr/bin/env bash
# =============================================================================
# BAM-PQ Backbone Group 2 — Qwen3-Embedding-0.6B + Qwen3-Embedding-4B
#
# Runs BAM-PQ train → best-epoch selection → eval for these two backbones
# across all 5 datasets (educational, msmarco, scifact, nfcorpus, fiqa).
#
# MRL warm-start: NO for both (decoder-style architecture, last-token pooling;
#   incompatible with the e5-large MRL baseline weights).
#
# Memory notes:
#   qwen06b — ~1.2 GB; fp16 fine; batch 16
#   qwen4b  — ~8 GB bf16; gradient checkpointing enabled in config
#
# Run AFTER pipeline_shared.sh completes:
#   ./pipeline_backbones_1.sh &
#   ./pipeline_backbones_2.sh &
#   ./pipeline_backbones_3.sh &
#   wait
#
# Requires: pip install transformers>=4.40  (Qwen3-Embedding support)
#
# Usage:
#   ./pipeline_backbones_2.sh                        # all 5 datasets
#   ./pipeline_backbones_2.sh --datasets "educational msmarco"
#   ./pipeline_backbones_2.sh --force                # wipe and retrain
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BACKBONES_TO_RUN="qwen06b qwen4b" \
exec bash "$SCRIPT_DIR/bam_council_pipeline.sh" \
    --from train_mrl_bk \
    "$@"
