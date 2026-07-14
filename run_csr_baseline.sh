#!/bin/bash
#SBATCH -A raja
#SBATCH --qos=medium
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=1024
#SBATCH --output=./QA-MRL/csr_baseline.log
#SBATCH --nodelist=node08
# =============================================================================
# CSR retrieval baseline — sparse dictionary coding on top of pretrained
# BAAI/bge-base-en-v1.5 (no MRL fine-tune). Sweeps active-atom budgets and
# reports R@1 / R@10 / R@50 / NDCG@10 comparable to Table 3 in the paper.
#
# Runtime: ~5-10 min including model download the first time.
# =============================================================================

# Load the CUDA module so the hardware is visible
module load cuda/12.4 2>/dev/null || true

echo "Running on host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

# Navigate + activate
cd ~/QA-MRL
source ~/myenv/bin/activate
git fetch origin
git checkout BAM-PQ
git pull origin BAM-PQ

# ── Output paths (edit if you want them elsewhere) ────────────────────────
export OUT_ROOT=${OUT_ROOT:-$HOME/bampq_cluster_baseline/results}
mkdir -p "$OUT_ROOT"

# ── HuggingFace cache (avoids re-downloading models to $HOME) ──────────────
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
mkdir -p "$HF_HOME"

# ── PyTorch memory hint ────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ── Run CSR sweep on pretrained bge-base ──────────────────────────────────
python scripts/eval_csr_baseline.py \
    --config     configs/mrl_bge_base.yaml \
    --output_dir "$OUT_ROOT/csr_bge_base_pretrained" \
    --n_atoms 512 \
    --n_nonzero_sweep 20 40 80 160 320

echo "=========================================="
echo "Done. Results in $OUT_ROOT/csr_bge_base_pretrained/"
