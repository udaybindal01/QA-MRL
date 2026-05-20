#!/usr/bin/env bash
# =============================================================================
# Zero-shot BEIR evaluation — uses educational BAM-PQ + MRL checkpoints,
# evaluates on scifact / nfcorpus / fiqa WITHOUT any BEIR-specific training.
#
# Checkpoint paths follow the council pipeline convention:
#   BAM-PQ : $CKPT_ROOT/educational/bam_pq_<backbone>/best_bsr/
#   MRL    : $CKPT_ROOT/educational/mrl_<backbone>/best/
#
# Usage:
#   ./scripts/run_zero_shot_beir.sh
#   BACKBONES="e5large bge" ./scripts/run_zero_shot_beir.sh
#   DATASETS="scifact fiqa" ./scripts/run_zero_shot_beir.sh
# =============================================================================

set -euo pipefail

CKPT_ROOT="${CKPT_ROOT:-/tmp/multi-domain}"
RESULTS_ROOT="${RESULTS_ROOT:-./results/zero_shot_beir}"
DATASETS="${DATASETS:-scifact nfcorpus fiqa}"
BACKBONES="${BACKBONES:-e5large bge arctic mxbai bge_base phi3mini}"

# ── Config registry (mirrors bam_council_pipeline.sh) ────────────────────────
declare -A BAM_CFG=(
    [e5large]="configs/bam_pq.yaml"
    [bge]="configs/bam_pq_bge_large.yaml"
    [arctic]="configs/bam_pq_arctic.yaml"
    [mxbai]="configs/bam_pq_mxbai.yaml"
    [bge_base]="configs/bam_pq_bge_base.yaml"
    [phi3mini]="configs/bam_pq_phi3mini.yaml"
    [roberta]="configs/bam_pq_roberta.yaml"
)
declare -A MRL_CFG=(
    [e5large]="configs/mrl_e5large.yaml"
    [bge]="configs/mrl_bge_large.yaml"
    [arctic]="configs/mrl_arctic.yaml"
    [mxbai]="configs/mrl_mxbai.yaml"
    [bge_base]="configs/mrl_bge_base.yaml"
    [phi3mini]="configs/mrl_phi3mini.yaml"
    [roberta]="configs/mrl_roberta.yaml"
)

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo ""; echo "══════════════════════════════════════════════════════════"; \
        echo "  [$(date '+%H:%M:%S')]  $*"; \
        echo "══════════════════════════════════════════════════════════"; }

mkdir -p "$RESULTS_ROOT"

# ── Per-backbone eval ─────────────────────────────────────────────────────────
for BK in $BACKBONES; do
    BK_BAM_CFG="${BAM_CFG[$BK]:-}"
    BK_MRL_CFG="${MRL_CFG[$BK]:-}"

    [[ -n "$BK_BAM_CFG" && -f "$BK_BAM_CFG" ]] \
        || { echo "  [$BK] BAM config not found ($BK_BAM_CFG) — skipping."; continue; }
    [[ -n "$BK_MRL_CFG" && -f "$BK_MRL_CFG" ]] \
        || { echo "  [$BK] MRL config not found ($BK_MRL_CFG) — skipping."; continue; }

    BK_BAM_CKPT="$CKPT_ROOT/educational/bam_pq_$BK/best_bsr"
    BK_MRL_CKPT="$CKPT_ROOT/educational/mrl_$BK/best"

    if [[ ! -f "$BK_BAM_CKPT/checkpoint.pt" ]]; then
        echo "  [$BK] BAM-PQ checkpoint not found at $BK_BAM_CKPT — skipping."
        continue
    fi
    if [[ ! -f "$BK_MRL_CKPT/checkpoint.pt" ]]; then
        echo "  [$BK] MRL checkpoint not found at $BK_MRL_CKPT — skipping."
        continue
    fi

    BK_OUT="$RESULTS_ROOT/$BK"
    mkdir -p "$BK_OUT"

    log "[$BK] zero-shot on: $DATASETS"
    echo "  BAM-PQ : $BK_BAM_CKPT"
    echo "  MRL    : $BK_MRL_CKPT"
    echo "  Output : $BK_OUT"

    python3 scripts/eval_beir.py \
        --config     "$BK_BAM_CFG"   \
        --checkpoint "$BK_BAM_CKPT"  \
        --baseline   "$BK_MRL_CKPT"  \
        --model_type bam              \
        --datasets   $DATASETS        \
        --output_dir "$BK_OUT/"       \
        || die "[$BK] eval_beir.py failed"

    echo "  [$BK] done → $BK_OUT/beir_results.json"
done

# ── Summary table ─────────────────────────────────────────────────────────────
log "ZERO-SHOT SUMMARY (NDCG@10)"
python3 - <<'PYEOF'
import json, os, glob

results_root = os.environ.get("RESULTS_ROOT", "./results/zero_shot_beir")
datasets = os.environ.get("DATASETS", "scifact nfcorpus fiqa").split()

rows = []
for bk_dir in sorted(glob.glob(f"{results_root}/*")):
    bk = os.path.basename(bk_dir)
    jf = f"{bk_dir}/beir_results.json"
    if not os.path.exists(jf):
        continue
    data = json.load(open(jf))
    row = {"backbone": bk}
    for model in ("BAM", "MRL Baseline"):
        for ds in datasets:
            ndcg = data.get(model, {}).get(ds, {}).get("ndcg@10", None)
            r10  = data.get(model, {}).get(ds, {}).get("recall@10", None)
            key  = f"{model}_{ds}"
            row[f"{key}_ndcg"] = f"{ndcg:.4f}" if ndcg is not None else "--"
            row[f"{key}_r10"]  = f"{r10:.4f}"  if r10  is not None else "--"
    rows.append(row)

if not rows:
    print("  No results found yet.")
else:
    print(f"\n  {'Backbone':<12}", end="")
    for ds in datasets:
        print(f"  {ds[:8]:>16}", end="")
    print()
    print("  " + "-" * (12 + 18 * len(datasets)))
    for row in rows:
        print(f"  {row['backbone']:<12}", end="")
        for ds in datasets:
            bam  = row.get(f"BAM_{ds}_ndcg", "--")
            mrl  = row.get(f"MRL Baseline_{ds}_ndcg", "--")
            print(f"  {bam:>7}/{mrl:>7}", end="")
        print()
    print(f"\n  Format: BAM-PQ / MRL (NDCG@10)")
PYEOF

log "ALL DONE — results in $RESULTS_ROOT"
