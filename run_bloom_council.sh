#!/bin/bash
#SBATCH -A raja
#SBATCH --qos=medium
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --output=./QA-MRL/bloom_council_training.log
#SBATCH --nodelist=node08
# =============================================================================
# Train the 4-model Bloom Council on 3 Kaggle datasets.
#
# Models:
#   - SVM (TF-IDF + verb-count features)  ~1 min
#   - BERT-base-uncased                   ~10 min
#   - RoBERTa-large                       ~20 min
#   - DeBERTa-v3-large                    ~40 min
#
# Data: 800 train + 100 val + 100 test per Bloom level (balanced).
# Pulls from Kaggle via kagglehub, so KAGGLE_USERNAME + KAGGLE_KEY must be set.
#
# Runtime: ~1.5-2 hours on one A100.
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

# ── Where the trained council lands ────────────────────────────────────────
export COUNCIL_DIR=${COUNCIL_DIR:-$HOME/bloom-council}
mkdir -p "$COUNCIL_DIR"

# ── HuggingFace cache (large models — DeBERTa-v3-large is ~1.7 GB) ────────
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
mkdir -p "$HF_HOME"

# ── Bloom training data source ─────────────────────────────────────────────
#   Default: read local CSVs from $HOME/bloom-data (BLOOM_DATA_DIR override
#   supported). The loader (data/annotate_bloom_council.py) globs **/*.csv
#   there and auto-detects text + label columns, so no Kaggle credentials
#   are needed. If the directory is missing, print a clear error.
export BLOOM_DATA_DIR=${BLOOM_DATA_DIR:-$HOME/bloom-data}
if [[ ! -d "$BLOOM_DATA_DIR" ]]; then
    echo "ERROR: BLOOM_DATA_DIR=$BLOOM_DATA_DIR does not exist."
    echo "  Copy the Bloom training CSVs to that directory:"
    echo "    scp <your-mac>:/path/to/blooms_taxonomy_dataset.csv \\"
    echo "        <your-mac>:/path/to/collected_dataset.csv         \\"
    echo "        <your-mac>:/path/to/q_bt_human_label.csv          \\"
    echo "        ${USER}@${HOSTNAME}:$BLOOM_DATA_DIR/"
    exit 1
fi
echo "  BLOOM_DATA_DIR = $BLOOM_DATA_DIR"

# ── PyTorch memory hint ────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "  COUNCIL_DIR = $COUNCIL_DIR"
echo "  HF_HOME     = $HF_HOME"
echo "=========================================="

# ── Train the council ──────────────────────────────────────────────────────
python data/train_bloom_council.py \
    --output_dir "$COUNCIL_DIR" \
    --n_train 800 \
    --n_val   100 \
    --n_test  100

# ── Sanity: council_info.json + council_weights.json should exist ─────────
if [[ ! -f "$COUNCIL_DIR/council_weights.json" ]] \
   && [[ ! -f "$COUNCIL_DIR/council_info.json" ]]; then
    echo "ERROR: Council training completed but council_weights.json /"
    echo "       council_info.json not written to $COUNCIL_DIR."
    exit 1
fi

echo "=========================================="
echo "Done. Council saved to $COUNCIL_DIR"
ls -la "$COUNCIL_DIR"
