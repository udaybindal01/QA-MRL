#!/bin/bash
#SBATCH -A raja
#SBATCH --qos=medium
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --mem-per-cpu=1024
#SBATCH --output=./QA-MRL/cluster_bampq_baseline.log
#SBATCH --nodelist=node08
# =============================================================================
# Cluster-routed BAM-PQ — full pipeline in one job:
#   [1] Train MRL baseline for bge_base
#   [2] Fit k-means (k=6) on training query embeddings, relabel data with
#       cluster IDs (replacing predicted Bloom labels via new bloom_cache.json
#       sidecars pointing at symlinked jsonl files)
#   [3] Retrain BAM-PQ from the same MRL warm-start but with cluster IDs as
#       the routing signal (no code changes; the trainer just reads the new
#       cache). Uses the paper's reverse two-stage schedule (frozen 0-7,
#       unfrozen 8-19) inherited from configs/bam_pq_bge_base.yaml.
#   [4] Evaluate the cluster-routed BAM-PQ against the freshly trained MRL.
#
# The paper argument this supports: BAM-PQ's Bloom supervision outperforms
# unsupervised cluster-based routing at matched architecture, warm-start,
# and training budget. Runtime: ~3-5 hours on a single A100 for bge_base.
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

# Single-threaded git to avoid 'unable to create thread' on constrained
# clusters (ulimit -u trips index-pack's default multi-threaded unpack).
git -c pack.threads=1 -c core.preloadIndex=false fetch origin
git checkout BAM-PQ
git -c pack.threads=1 -c core.preloadIndex=false pull origin BAM-PQ

# ── Output paths (edit if you want them elsewhere) ────────────────────────
export CKPT_ROOT=${CKPT_ROOT:-$HOME/bampq_cluster_baseline/ckpts}
export OUT_ROOT=${OUT_ROOT:-$HOME/bampq_cluster_baseline/results}
export DATA_ROOT=${DATA_ROOT:-$HOME/bampq_cluster_baseline/data}
export INPUT_DATA=${INPUT_DATA:-data/real}

# ── Clustering hyperparameters ────────────────────────────────────────────
export K=${K:-6}                                 # matches Bloom levels

# ── Skip flags (set to 1 to reuse existing outputs) ───────────────────────
export SKIP_TRAIN_MRL=${SKIP_TRAIN_MRL:-0}
export SKIP_TRAIN_BAM=${SKIP_TRAIN_BAM:-0}

# ── HuggingFace cache (avoids re-downloading models to $HOME) ──────────────
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
mkdir -p "$HF_HOME"

# ── PyTorch memory hint ────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ── Thread caps — this cluster has ulimit -u = 200, so pin every framework
#     to single-digit threads to avoid 'unable to create thread' failures.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export FAISS_NUM_THREADS=${FAISS_NUM_THREADS:-4}

# ── Create output dirs so nothing fails silently ──────────────────────────
mkdir -p "$CKPT_ROOT" "$OUT_ROOT" "$DATA_ROOT"

echo "  CKPT_ROOT       = $CKPT_ROOT"
echo "  OUT_ROOT        = $OUT_ROOT"
echo "  DATA_ROOT       = $DATA_ROOT"
echo "  INPUT_DATA      = $INPUT_DATA"
echo "  K               = $K"
echo "  SKIP_TRAIN_MRL  = $SKIP_TRAIN_MRL"
echo "  SKIP_TRAIN_BAM  = $SKIP_TRAIN_BAM"
echo "  HF_HOME         = $HF_HOME"
echo "=========================================="

# ── Sanity: verify input data exists ──────────────────────────────────────
for f in train_curriculum.jsonl val.jsonl test.jsonl corpus.jsonl; do
    if [[ ! -f "$INPUT_DATA/$f" ]]; then
        echo "ERROR: missing $INPUT_DATA/$f"
        echo "  Run data/build_real_data.py + data/curriculum_negatives.py first."
        exit 1
    fi
done

# ── Full pipeline: MRL train -> relabel -> BAM-PQ retrain -> eval ────────
./scripts/run_cluster_baseline.sh bge_base

echo "=========================================="
echo "Done."
echo "  MRL ckpt              : $CKPT_ROOT/mrl_bge_base/best/checkpoint.pt"
echo "  Cluster-routed BAM-PQ : $CKPT_ROOT/bam_pq_cluster_bge_base/best_bsr/checkpoint.pt"
echo "  Clustered data        : $DATA_ROOT/bge_base_k6/"
echo "  Results               : $OUT_ROOT/bge_base/results.json"
