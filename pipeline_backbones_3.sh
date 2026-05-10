#!/usr/bin/env bash
# =============================================================================
# BAM-PQ Backbone Group 3 — LLM2Vec-Mistral-7B + GritLM-7B
#
# Runs BAM-PQ train → best-epoch selection → eval for these two backbones
# across all 5 datasets (educational, msmarco, scifact, nfcorpus, fiqa).
#
# MRL warm-start: NO for both (7B decoder models; encoder stays FROZEN
#   throughout training — encoder_unfreeze_after_epochs: null in configs).
#
# Memory notes:
#   llm2vec — ~14 GB bf16 (frozen); requires: pip install llm2vec
#   gritlm  — ~14 GB bf16 (frozen); requires: pip install gritlm
#   Both need a GPU with ≥16 GB VRAM (A100/H100 recommended).
#   Run on a separate GPU node from groups 1 and 2 if possible.
#
# Run AFTER pipeline_shared.sh completes:
#   ./pipeline_backbones_1.sh &
#   ./pipeline_backbones_2.sh &
#   ./pipeline_backbones_3.sh &
#   wait
#
# Requires: pip install llm2vec gritlm
#
# Usage:
#   ./pipeline_backbones_3.sh                        # all 5 datasets
#   ./pipeline_backbones_3.sh --datasets "educational msmarco"
#   ./pipeline_backbones_3.sh --force                # wipe and retrain
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BACKBONES_TO_RUN="phi3mini" \
REUSE_TRAINED_MODELS=1 \
exec bash "$SCRIPT_DIR/bam_council_pipeline.sh" \
    --from train_mrl_bk \
    "$@"