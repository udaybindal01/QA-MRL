#!/usr/bin/env bash
# =============================================================================
# BAM-PQ Backbone Group 4 — Qwen3-Embedding-8B + LLM2Vec-LLaMA-3-8B
#
# Runs BAM-PQ train → best-epoch selection → eval for these two backbones
# across all 5 datasets (educational, msmarco, scifact, nfcorpus, fiqa).
#
# MRL warm-start: NO for both (8B decoder models; encoder stays FROZEN
#   throughout training — encoder_unfreeze_after_epochs: null in configs).
#
# Memory notes:
#   qwen8b   — ~16 GB bf16 (frozen); gradient_checkpointing enabled
#   llama8b  — ~16 GB bf16 (frozen); requires: pip install llm2vec
#   Both need a GPU with ≥24 GB VRAM (A100/H100 recommended).
#   Run on a separate GPU node from groups 1–3 if possible.
#
# Run AFTER pipeline_shared.sh completes:
#   ./pipeline_backbones_1.sh &
#   ./pipeline_backbones_2.sh &
#   ./pipeline_backbones_3.sh &
#   ./pipeline_backbones_4.sh &
#   wait
#
# Requires:
#   pip install transformers>=4.40   (Qwen3-Embedding-8B support)
#   pip install llm2vec              (LLM2Vec-LLaMA-3-8B support)
#
# Usage:
#   ./pipeline_backbones_4.sh                        # all 5 datasets
#   ./pipeline_backbones_4.sh --datasets "educational msmarco"
#   ./pipeline_backbones_4.sh --force                # wipe and retrain
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BACKBONES_TO_RUN="qwen8b llama8b" \
exec bash "$SCRIPT_DIR/bam_council_pipeline.sh" \
    --from eval_pretrained \
    "$@"
