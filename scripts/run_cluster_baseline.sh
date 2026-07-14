#!/usr/bin/env bash
# =============================================================================
# run_cluster_baseline.sh — full "cluster-routed BAM-PQ" pipeline
#
# This is the unsupervised counterpart to BAM-PQ, run end-to-end in one job:
#
#   1. Train MRL baseline for the requested backbone.
#   2. Fit k-means (k=6) on training query embeddings, relabel train / val /
#      test data with cluster IDs (replacing predicted Bloom labels via new
#      .bloom_cache.json sidecars pointing at the same jsonl files).
#   3. Train BAM-PQ from the same MRL warm-start but with cluster IDs as the
#      routing signal (no code changes; the trainer just reads the new cache).
#   4. Evaluate the cluster-routed BAM-PQ against the MRL baseline.
#
# The paper argument this supports: BAM-PQ's Bloom supervision outperforms
# unsupervised cluster-based routing at matched architecture, warm-start,
# and training budget. Any advantage BAM-PQ retains here comes from the
# supervised Bloom prior, not the routing mechanism itself.
#
# Usage:
#     ./scripts/run_cluster_baseline.sh <backbone>
#
# where <backbone> ∈ {bge_base, bge_large, e5large, arctic}.
#
# Env overrides:
#     CKPT_ROOT    base dir for MRL + BAM-PQ-cluster checkpoints
#                  (default: /scratch/$USER/cluster_baseline/ckpts)
#     OUT_ROOT     base dir for eval outputs
#                  (default: /scratch/$USER/cluster_baseline/results)
#     DATA_ROOT    base dir for clustered data mirrors
#                  (default: /scratch/$USER/cluster_baseline/data)
#     INPUT_DATA   source data dir (default: data/real)
#     K            number of clusters (default: 6)
#     SKIP_TRAIN_MRL   set to 1 to reuse existing MRL checkpoint
#     SKIP_TRAIN_BAM   set to 1 to reuse existing BAM-PQ-cluster checkpoint
# =============================================================================
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <backbone>   (bge_base | bge_large | e5large | arctic)"
    exit 1
fi

BB="$1"
MRL_CFG="configs/mrl_${BB}.yaml"

# BAM-PQ config selection — e5-large uses the bam_optionb_e5large.yaml
# variant (paper's canonical config for e5), others use bam_pq_<bb>.yaml.
if [[ "$BB" == "e5large" ]]; then
    BAM_BASE_CFG="configs/bam_optionb_e5large.yaml"
else
    BAM_BASE_CFG="configs/bam_pq_${BB}.yaml"
fi

for cfg in "$MRL_CFG" "$BAM_BASE_CFG"; do
    [[ -f "$cfg" ]] || { echo "ERROR: config not found: $cfg"; exit 1; }
done

CKPT_ROOT="${CKPT_ROOT:-/scratch/${USER}/cluster_baseline/ckpts}"
OUT_ROOT="${OUT_ROOT:-/scratch/${USER}/cluster_baseline/results}"
DATA_ROOT="${DATA_ROOT:-/scratch/${USER}/cluster_baseline/data}"
INPUT_DATA="${INPUT_DATA:-data/real}"
K="${K:-6}"
SKIP_TRAIN_MRL="${SKIP_TRAIN_MRL:-0}"
SKIP_TRAIN_BAM="${SKIP_TRAIN_BAM:-0}"

MRL_CKPT_DIR="${CKPT_ROOT}/mrl_${BB}"
MRL_BEST="${MRL_CKPT_DIR}/best"

CLUSTERED_DATA="${DATA_ROOT}/${BB}_k${K}"

BAM_CKPT_DIR="${CKPT_ROOT}/bam_pq_cluster_${BB}"
BAM_CFG_OUT="${BAM_CKPT_DIR}/bam_pq_cluster_${BB}.yaml"

OUT_DIR="${OUT_ROOT}/${BB}"

mkdir -p "$MRL_CKPT_DIR" "$CLUSTERED_DATA" "$BAM_CKPT_DIR" "$OUT_DIR"

echo "======================================================================"
echo "  backbone           : $BB"
echo "  MRL config         : $MRL_CFG"
echo "  BAM-PQ base config : $BAM_BASE_CFG"
echo "  MRL checkpoint dir : $MRL_CKPT_DIR"
echo "  Clustered data dir : $CLUSTERED_DATA"
echo "  BAM ckpt dir       : $BAM_CKPT_DIR"
echo "  Output dir         : $OUT_DIR"
echo "  k                  : $K"
echo "  SKIP_TRAIN_MRL     : $SKIP_TRAIN_MRL"
echo "  SKIP_TRAIN_BAM     : $SKIP_TRAIN_BAM"
echo "======================================================================"

# ── Step 1: Train MRL baseline ─────────────────────────────────────────────
if [[ "$SKIP_TRAIN_MRL" == "1" ]] && [[ -f "$MRL_BEST/checkpoint.pt" ]]; then
    echo "[1/4] SKIP_TRAIN_MRL=1 and MRL ckpt exists — reusing."
else
    echo "[1/4] Training MRL baseline ..."
    python3 scripts/train_baseline_mrl.py \
        --config         "$MRL_CFG" \
        --checkpoint_dir "$MRL_CKPT_DIR"
    if [[ ! -f "$MRL_BEST/checkpoint.pt" ]]; then
        # Some train scripts save 'final' but not 'best'; fall back.
        if [[ -f "$MRL_CKPT_DIR/final/checkpoint.pt" ]]; then
            ln -sfn "$MRL_CKPT_DIR/final" "$MRL_BEST"
        else
            echo "ERROR: no MRL checkpoint at $MRL_BEST or $MRL_CKPT_DIR/final"
            exit 1
        fi
    fi
fi

# ── Step 2: Relabel data with k-means cluster IDs ──────────────────────────
echo ""
echo "[2/4] Relabelling data with k-means cluster IDs ..."
python3 scripts/relabel_with_clusters.py \
    --config     "$MRL_CFG" \
    --checkpoint "$MRL_BEST" \
    --input_dir  "$INPUT_DATA" \
    --output_dir "$CLUSTERED_DATA" \
    --k          "$K"

# ── Step 3: Generate a BAM-PQ config that points at the clustered data ────
echo ""
echo "[3/4] Generating BAM-PQ config with clustered data paths ..."
python3 - <<PYEOF
import os, yaml
with open("$BAM_BASE_CFG") as f:
    cfg = yaml.safe_load(f)
cfg["data"]["train_path"]         = os.path.abspath("$CLUSTERED_DATA/train_curriculum.jsonl")
cfg["data"]["val_path"]           = os.path.abspath("$CLUSTERED_DATA/val.jsonl")
cfg["data"]["test_path"]          = os.path.abspath("$CLUSTERED_DATA/test.jsonl")
cfg["data"]["corpus_path"]        = os.path.abspath("$CLUSTERED_DATA/corpus.jsonl")
cfg["training"]["checkpoint_dir"] = "$BAM_CKPT_DIR"
with open("$BAM_CFG_OUT", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
print(f"  Wrote {'$BAM_CFG_OUT'}")
PYEOF

# ── Step 4: Train "cluster-routed" BAM-PQ (encoder frozen, warm-start MRL) ─
BAM_BEST_BSR="${BAM_CKPT_DIR}/best_bsr"
if [[ "$SKIP_TRAIN_BAM" == "1" ]] && [[ -f "$BAM_BEST_BSR/checkpoint.pt" ]]; then
    echo "[4/4] SKIP_TRAIN_BAM=1 and BAM-PQ-cluster ckpt exists — reusing."
else
    echo "[4/4] Training BAM-PQ with cluster IDs as routing signal ..."
    python3 scripts/train_bam.py \
        --config         "$BAM_CFG_OUT" \
        --checkpoint_dir "$BAM_CKPT_DIR" \
        --freeze_encoder \
        --init_encoder   "$MRL_BEST"
    # Best-BSR selection (mirrors bam_council_pipeline.sh)
    if [[ ! -f "$BAM_BEST_BSR/checkpoint.pt" ]]; then
        python3 scripts/find_best_epoch_bsr.py \
            --config         "$BAM_CFG_OUT" \
            --checkpoint_dir "$BAM_CKPT_DIR" \
            --output_dir     "$OUT_DIR/bsr/" \
            --alpha          0.5 \
            || {
                # Fall back to 'final' if BSR selection isn't available
                if [[ -f "$BAM_CKPT_DIR/final/checkpoint.pt" ]]; then
                    ln -sfn "$BAM_CKPT_DIR/final" "$BAM_BEST_BSR"
                fi
            }
    fi
fi

# ── Step 5: Evaluate cluster-routed BAM-PQ vs MRL ──────────────────────────
echo ""
echo "[eval] Evaluating cluster-routed BAM-PQ ..."
python3 scripts/eval_bam.py \
    --config     "$BAM_CFG_OUT" \
    --checkpoint "$BAM_BEST_BSR" \
    --baseline   "$MRL_BEST" \
    --output_dir "$OUT_DIR/"

echo ""
echo "======================================================================"
echo "  DONE.  MRL ckpt              : $MRL_BEST/checkpoint.pt"
echo "         Cluster-routed BAM-PQ : $BAM_BEST_BSR/checkpoint.pt"
echo "         Clustered data        : $CLUSTERED_DATA"
echo "         Results               : $OUT_DIR/results.json"
echo "======================================================================"
