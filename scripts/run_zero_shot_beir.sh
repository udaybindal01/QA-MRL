#!/usr/bin/env bash
# =============================================================================
# Zero-shot BEIR evaluation — uses educational BAM-PQ / MRL / Standard-FT
# checkpoints, evaluates on scifact / nfcorpus / fiqa WITHOUT BEIR fine-tuning.
#
# Checkpoint paths follow the council pipeline convention:
#   BAM-PQ      : $CKPT_ROOT/educational/bam_pq_<backbone>/best_bsr/
#   MRL         : $CKPT_ROOT/educational/mrl_<backbone>/best/
#   Standard FT : $CKPT_ROOT/educational/standard_ft_<backbone>/best/
#
# Usage:
#   ./scripts/run_zero_shot_beir.sh
#   CKPT_ROOT=/tmp/ishaan.karan/bampq-checkpoints ./scripts/run_zero_shot_beir.sh
#   BACKBONES="e5large bge" DATASETS="scifact fiqa" ./scripts/run_zero_shot_beir.sh
#   SKIP_SFT=1 ./scripts/run_zero_shot_beir.sh   # skip standard FT eval
# =============================================================================

set -euo pipefail

CKPT_ROOT="${CKPT_ROOT:-/tmp/multi-domain}"
RESULTS_ROOT="${RESULTS_ROOT:-./results/zero_shot_beir}"
DATASETS="${DATASETS:-scifact nfcorpus fiqa}"
BACKBONES="${BACKBONES:-e5large bge arctic mxbai bge_base phi3mini}"
SKIP_SFT="${SKIP_SFT:-0}"   # set to 1 to skip standard FT evaluation

# ── Config registry ───────────────────────────────────────────────────────────
declare -A BAM_CFG=(
    [e5large]="configs/bam_pq.yaml"
    [bge]="configs/bam_pq_bge_large.yaml"
    [arctic]="configs/bam_pq_arctic.yaml"
    [mxbai]="configs/bam_pq_mxbai.yaml"
    [bge_base]="configs/bam_pq_bge_base.yaml"
    [phi3mini]="configs/bam_pq_phi3mini.yaml"
    [roberta]="configs/bam_pq_roberta.yaml"
    [qwen06b]="configs/bam_pq_qwen06b.yaml"
)
declare -A MRL_CFG=(
    [e5large]="configs/mrl_e5large.yaml"
    [bge]="configs/mrl_bge_large.yaml"
    [arctic]="configs/mrl_arctic.yaml"
    [mxbai]="configs/mrl_mxbai.yaml"
    [bge_base]="configs/mrl_bge_base.yaml"
    [phi3mini]="configs/mrl_phi3mini.yaml"
    [roberta]="configs/mrl_roberta.yaml"
    [qwen06b]="configs/mrl_qwen06b.yaml"
)
declare -A SFT_CFG=(
    [e5large]="configs/standard_ft_e5large.yaml"
    [bge]="configs/standard_ft_bge.yaml"
    [arctic]="configs/standard_ft_arctic.yaml"
    [mxbai]="configs/standard_ft_mxbai.yaml"
    [bge_base]="configs/standard_ft_bge_base.yaml"
    [phi3mini]="configs/standard_ft_phi3mini.yaml"
    [roberta]="configs/standard_ft_roberta.yaml"
    [qwen06b]="configs/standard_ft_qwen06b.yaml"
)

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo ""; echo "══════════════════════════════════════════════════════════"; \
        echo "  [$(date '+%H:%M:%S')]  $*"; \
        echo "══════════════════════════════════════════════════════════"; }

mkdir -p "$RESULTS_ROOT"

log "Zero-shot BEIR eval"
echo "  CKPT_ROOT   : $CKPT_ROOT"
echo "  RESULTS_ROOT: $RESULTS_ROOT"
echo "  DATASETS    : $DATASETS"
echo "  BACKBONES   : $BACKBONES"
echo "  SKIP_SFT    : $SKIP_SFT"

# ── Per-backbone eval ─────────────────────────────────────────────────────────
for BK in $BACKBONES; do
    BK_BAM_CFG="${BAM_CFG[$BK]:-}"
    BK_MRL_CFG="${MRL_CFG[$BK]:-}"
    BK_SFT_CFG="${SFT_CFG[$BK]:-}"

    BK_BAM_CKPT="$CKPT_ROOT/educational/bam_pq_$BK/best_bsr"
    BK_MRL_CKPT="$CKPT_ROOT/educational/mrl_$BK/best"
    BK_SFT_CKPT="$CKPT_ROOT/educational/standard_ft_$BK/best"

    BK_OUT="$RESULTS_ROOT/$BK"
    mkdir -p "$BK_OUT"

    log "[$BK]"

    # ── BAM-PQ + MRL (evaluated in one call) ──────────────────────────────────
    if [[ -n "$BK_BAM_CFG" && -f "$BK_BAM_CFG" && -f "$BK_BAM_CKPT/checkpoint.pt" ]]; then
        MRL_ARGS=""
        if [[ -n "$BK_MRL_CFG" && -f "$BK_MRL_CFG" && -f "$BK_MRL_CKPT/checkpoint.pt" ]]; then
            MRL_ARGS="--baseline $BK_MRL_CKPT"
            echo "  BAM-PQ : $BK_BAM_CKPT  +  MRL baseline: $BK_MRL_CKPT"
        else
            echo "  BAM-PQ : $BK_BAM_CKPT  (MRL checkpoint not found — skipping MRL)"
        fi

        python3 scripts/eval_beir.py \
            --config     "$BK_BAM_CFG"  \
            --checkpoint "$BK_BAM_CKPT" \
            $MRL_ARGS                   \
            --model_type bam            \
            --datasets   $DATASETS      \
            --output_dir "$BK_OUT/"     \
            || die "[$BK] eval_beir.py (BAM-PQ+MRL) failed"

        echo "  Saved → $BK_OUT/beir_results.json"
    else
        echo "  [$BK] BAM-PQ config or checkpoint missing — skipping BAM-PQ+MRL."
    fi

    # ── Standard FT (separate call, --model_type mrl with SFT checkpoint) ────
    if [[ "$SKIP_SFT" == "0" ]]; then
        if [[ -n "$BK_SFT_CFG" && -f "$BK_SFT_CFG" && -f "$BK_SFT_CKPT/checkpoint.pt" ]]; then
            echo "  Standard FT: $BK_SFT_CKPT"

            python3 scripts/eval_beir.py \
                --config     "$BK_SFT_CFG"  \
                --checkpoint "$BK_SFT_CKPT" \
                --model_type mrl            \
                --datasets   $DATASETS      \
                --output_dir "$BK_OUT/sft/" \
                || die "[$BK] eval_beir.py (Standard FT) failed"

            echo "  Saved → $BK_OUT/sft/beir_results.json"
        else
            echo "  [$BK] Standard FT config or checkpoint missing — skipping SFT."
        fi
    fi

done

# ── Summary table ─────────────────────────────────────────────────────────────
log "ZERO-SHOT SUMMARY (NDCG@10)  format: BAM / MRL / SFT"
RESULTS_ROOT="$RESULTS_ROOT" DATASETS="$DATASETS" python3 - <<'PYEOF'
import json, os, glob

results_root = os.environ.get("RESULTS_ROOT", "./results/zero_shot_beir")
datasets = os.environ.get("DATASETS", "scifact nfcorpus fiqa").split()

rows = []
for bk_dir in sorted(glob.glob(f"{results_root}/*")):
    if not os.path.isdir(bk_dir):
        continue
    bk = os.path.basename(bk_dir)
    jf     = f"{bk_dir}/beir_results.json"
    sft_jf = f"{bk_dir}/sft/beir_results.json"
    if not os.path.exists(jf):
        continue

    data     = json.load(open(jf))
    sft_data = json.load(open(sft_jf)) if os.path.exists(sft_jf) else {}

    row = {"backbone": bk}
    for ds in datasets:
        bam = data.get("BAM",          {}).get(ds, {}).get("ndcg@10")
        mrl = data.get("MRL Baseline", {}).get(ds, {}).get("ndcg@10")
        sft = sft_data.get("MRL",      {}).get(ds, {}).get("ndcg@10")
        row[ds] = (
            f"{bam:.4f}" if bam is not None else "--",
            f"{mrl:.4f}" if mrl is not None else "--",
            f"{sft:.4f}" if sft is not None else "--",
        )
    rows.append(row)

if not rows:
    print("  No results found yet.")
else:
    w = max(len(ds) for ds in datasets) + 2
    header = f"  {'Backbone':<14}" + "".join(f"  {ds:>{w*3+4}}" for ds in datasets)
    sub    = f"  {'':14}" + "".join(f"  {'BAM':>8} {'MRL':>8} {'SFT':>8}" for _ in datasets)
    print(header)
    print(sub)
    print("  " + "-" * len(header))
    for row in rows:
        line = f"  {row['backbone']:<14}"
        for ds in datasets:
            bam, mrl, sft = row.get(ds, ("--", "--", "--"))
            line += f"  {bam:>8} {mrl:>8} {sft:>8}"
        print(line)
PYEOF

log "ALL DONE — results in $RESULTS_ROOT"
