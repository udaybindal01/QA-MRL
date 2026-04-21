#!/usr/bin/env bash
# =============================================================================
# Evaluate BAM-PQ using already-trained checkpoints.
#
# Runs eval_bam_pq + fair_cmp_pq for each dataset.
# Skips any dataset whose BAM-PQ best_bsr checkpoint is missing.
#
# Usage:
#   chmod +x scripts/run_eval_bam_pq.sh
#   ./scripts/run_eval_bam_pq.sh
#
#   # Subset of datasets:
#   DATASETS="educational scifact" ./scripts/run_eval_bam_pq.sh
#
#   # Override checkpoint root:
#   CKPT_ROOT=/my/path ./scripts/run_eval_bam_pq.sh
# =============================================================================

set -euo pipefail

DATASETS="${DATASETS:-educational scifact nfcorpus fiqa}"
CKPT_ROOT="${CKPT_ROOT:-/tmp/multi-domain}"
RESULTS_ROOT="${RESULTS_ROOT:-./results/multi_domain}"

log() { echo ""; echo "══════════════════════════════════════════════════════"; \
        echo "  [$(date '+%H:%M:%S')]  $*"; \
        echo "══════════════════════════════════════════════════════"; }
die() { echo "ERROR: $*" >&2; exit 1; }

log "BAM-PQ EVALUATION"
echo "  Datasets : $DATASETS"
echo "  Ckpt root: $CKPT_ROOT"
echo "  Results  : $RESULTS_ROOT"

for DS in $DATASETS; do
    log "━━━━  DATASET: $DS  ━━━━"

    BAM_PQ_BEST="$CKPT_ROOT/$DS/bam_pq/best_bsr"
    MRL_BEST="$CKPT_ROOT/$DS/mrl/best"
    BAM_PQ_CFG="$RESULTS_ROOT/$DS/configs/bam_pq.yaml"
    DS_RESULTS="$RESULTS_ROOT/$DS"

    # Fallback config
    [[ -f "$BAM_PQ_CFG" ]] || BAM_PQ_CFG="configs/bam_pq.yaml"

    if [[ ! -f "$BAM_PQ_BEST/checkpoint.pt" ]]; then
        echo "  BAM-PQ checkpoint not found at $BAM_PQ_BEST — skipping $DS."
        continue
    fi
    [[ -f "$MRL_BEST/checkpoint.pt" ]] || die "[$DS] MRL best not found at $MRL_BEST"
    [[ -f "$BAM_PQ_CFG" ]]             || die "[$DS] BAM-PQ config not found at $BAM_PQ_CFG"

    # ── Standard eval ────────────────────────────────────────────────────────
    log "[$DS] EVAL BAM-PQ vs MRL"
    mkdir -p "$DS_RESULTS/bam_pq"
    python3 scripts/eval_bam.py \
        --config     "$BAM_PQ_CFG" \
        --checkpoint "$BAM_PQ_BEST" \
        --baseline   "$MRL_BEST" \
        --output_dir "$DS_RESULTS/bam_pq/" \
        || die "[$DS] eval_bam.py (BAM-PQ) failed"
    echo "  Results → $DS_RESULTS/bam_pq/results.json"

    # ── Fair comparison ───────────────────────────────────────────────────────
    log "[$DS] FAIR COMPARISON BAM-PQ vs MRL (same per-Bloom dim budget)"
    mkdir -p "$DS_RESULTS/bam_pq/fair_comparison"
    python3 scripts/eval_fair_comparison.py \
        --config         "$BAM_PQ_CFG" \
        --bam_checkpoint "$BAM_PQ_BEST" \
        --mrl_checkpoint "$MRL_BEST" \
        --bam_results    "$DS_RESULTS/bam_pq/results.json" \
        --output_dir     "$DS_RESULTS/bam_pq/fair_comparison/" \
        || die "[$DS] eval_fair_comparison.py (BAM-PQ) failed"
    echo "  Fair cmp → $DS_RESULTS/bam_pq/fair_comparison/fair_comparison.json"

done

# ── Summary ───────────────────────────────────────────────────────────────────
log "SUMMARY"
python3 - <<PYEOF
import json, os, math

datasets = "$DATASETS".split()
results_root = "$RESULTS_ROOT"

print()
print("  BAM-PQ vs MRL — standard eval (R@10)")
print(f"  {'Dataset':14s} {'MRL R@10':>10s} {'BAM-PQ R@10':>13s} {'AvgDims':>8s} {'Δ':>8s}")
print("  " + "─" * 60)
for ds in datasets:
    path = os.path.join(results_root, ds, "bam_pq", "results.json")
    if not os.path.exists(path):
        print(f"  {ds:14s}  (skipped)")
        continue
    with open(path) as f:
        r = json.load(f)
    mrl_r10 = r.get("MRL Baseline", {}).get("recall@10", 0)
    pq_r10  = next((r[k].get("recall@10", 0) for k in r
                    if k not in ("MRL Baseline",) and isinstance(r[k], dict)
                    and "recall@10" in r[k]), 0)
    dims    = next((r[k].get("avg_active_dims", 0) for k in r
                    if k not in ("MRL Baseline",) and isinstance(r[k], dict)
                    and "avg_active_dims" in r[k]), 0)
    delta   = pq_r10 - mrl_r10
    sign    = "+" if delta >= 0 else ""
    print(f"  {ds:14s} {mrl_r10:>10.4f} {pq_r10:>13.4f} {dims:>8.0f} {sign}{delta*100:>6.2f}%")

print()
print("  BAM-PQ vs MRL — fair comparison (same per-Bloom dim budget)")
print(f"  {'Dataset':14s} {'Avg Δ':>10s} {'BAM wins':>10s}")
print("  " + "─" * 38)
for ds in datasets:
    path = os.path.join(results_root, ds, "bam_pq", "fair_comparison", "fair_comparison.json")
    if not os.path.exists(path):
        print(f"  {ds:14s}  (skipped)")
        continue
    with open(path) as f:
        fc = json.load(f)
    avg_d = fc.get("avg_delta_bam_minus_mrl_trunc", math.nan)
    wins  = fc.get("bam_wins", 0)
    total = fc.get("total_levels", 6)
    sign  = "+" if avg_d >= 0 else ""
    print(f"  {ds:14s} {sign}{avg_d*100:>8.2f}%  {wins}/{total}")
PYEOF

log "DONE"
echo ""
echo "  Results : $RESULTS_ROOT/{dataset}/bam_pq/results.json"
echo "  Fair cmp: $RESULTS_ROOT/{dataset}/bam_pq/fair_comparison/fair_comparison.json"
