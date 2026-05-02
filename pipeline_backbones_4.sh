#!/usr/bin/env bash
# =============================================================================
# BAM-PQ Backbone Group 4 — Qwen3-Embedding-8B + LLM2Vec-LLaMA-3-8B
#                         + Llama-3.2-1B + Llama-3.2-3B
#
# Runs BAM-PQ train → best-epoch selection → eval for these backbones
# across all 5 datasets (educational, msmarco, scifact, nfcorpus, fiqa).
#
# MRL warm-start:
#   qwen8b / llama8b — encoder stays FROZEN throughout (8B models)
#   llama1b / llama3b — reverse two-stage; encoder unfreezes at epoch 8
#
# Memory notes:
#   qwen8b   — ~16 GB bf16 (frozen); batch=1, gradient_checkpointing enabled
#   llama8b  — ~16 GB bf16 (frozen); requires: pip install llm2vec
#   llama1b  — ~2 GB bf16; batch=32, no memory concerns
#   llama3b  — ~6 GB bf16; batch=16, no memory concerns
#   Run qwen8b/llama8b on a separate GPU node from groups 1–3 if possible.
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

BACKBONES_TO_RUN="llama1b llama3b" \
exec bash "$SCRIPT_DIR/bam_council_pipeline.sh" \
    --from eval_pretrained \
    "$@"
