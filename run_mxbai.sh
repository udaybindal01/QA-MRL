#!/bin/bash
#SBATCH --job-name=bam-mxbai
#SBATCH --output=bam-mxbai.out
#SBATCH --error=bam-mxbai.err
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

./bam_council_pipeline.sh --backbone mxbai --datasets "educational scifact nfcorpus fiqa"
