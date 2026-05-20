#!/bin/bash
#SBATCH --job-name=bge
#SBATCH --output=bge.out
#SBATCH --error=bge.err
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH -A raja
#SBATCH -w node10

# --- Environment ---
source ~/.bashrc
conda activate pyg

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

./bam_council_pipeline.sh \
    --backbone bge \
    --datasets "educational scifact nfcorpus fiqa"
