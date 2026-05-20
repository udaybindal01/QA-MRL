#!/bin/bash
#SBATCH --job-name=bam-emgemma
#SBATCH --output=bam-emgemma.out
#SBATCH --error=bam-emgemma.err
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=25G
#SBATCH --time=96:00:00
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -w gnode067
# --- Environment setup ---
source ~/.bashrc
conda activate pyg

# --- HuggingFace authentication (required for gated google/embeddinggemma-300m) ---
# Reads ~/.cache/huggingface/token (created by `huggingface-cli login`) if present.
if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

# --- Redirect all caches to scratch ---
export SCRATCH=/scratch/ishaan.karan
mkdir -p $SCRATCH/{hf,torch,pip,bampq-checkpoints,bampq-data/beir,bampq-data/bloom-council}
export HF_HOME=$SCRATCH/hf
export HF_HUB_CACHE=$SCRATCH/hf/hub
export TRANSFORMERS_CACHE=$SCRATCH/hf/transformers
export HF_DATASETS_CACHE=$SCRATCH/hf/datasets
export SENTENCE_TRANSFORMERS_HOME=$SCRATCH/hf/sentence-transformers
export TORCH_HOME=$SCRATCH/torch
export PIP_CACHE_DIR=$SCRATCH/pip
export CKPT_ROOT=$SCRATCH/bampq-checkpoints
export BEIR_DATA_ROOT=$SCRATCH/bampq-data/beir
export COUNCIL_DIR=$SCRATCH/bampq-data/bloom-council

./bam_council_pipeline.sh --backbone emgemma --datasets "educational scifact nfcorpus fiqa"
