#!/usr/bin/env bash
# Run fair comparison (BAM-B vs MRL @ same per-Bloom dim budget) for all datasets.
# Uses checkpoints trained by multi-domain-pipeline.sh.
#
# Usage:
#   ./scripts/run_fair_comparison_all.sh
#   ./scripts/run_fair_comparison_all.sh --datasets "educational scifact"

set -euo pipefail

DATASETS="${1:-educational scifact nfcorpus fiqa}"
CKPT_ROOT="/tmp/multi-domain"
RESULTS_ROOT="./results/multi_domain"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets) DATASETS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

for DS in $DATASETS; do
    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  Fair comparison: $DS"
    echo "══════════════════════════════════════════════════════"

    MRL_BEST="$CKPT_ROOT/$DS/mrl/best"
    BAM_B_BEST="$CKPT_ROOT/$DS/bam_b/best_bsr"
    BAM_B_CFG="$RESULTS_ROOT/$DS/configs/bam_b.yaml"
    DS_RESULTS="$RESULTS_ROOT/$DS"
    BAM_RESULTS="$DS_RESULTS/results.json"
    OUT_DIR="$DS_RESULTS/fair_comparison"

    # Validate
    [[ -f "$MRL_BEST/checkpoint.pt" ]]   || { echo "  ERROR: MRL checkpoint missing: $MRL_BEST"; continue; }
    [[ -f "$BAM_B_BEST/checkpoint.pt" ]] || { echo "  ERROR: BAM-B checkpoint missing: $BAM_B_BEST"; continue; }
    [[ -f "$BAM_B_CFG" ]]                || { echo "  ERROR: config missing: $BAM_B_CFG"; continue; }

    mkdir -p "$OUT_DIR"

    BAM_RESULTS_ARG=""
    [[ -f "$BAM_RESULTS" ]] && BAM_RESULTS_ARG="--bam_results $BAM_RESULTS"

    python scripts/eval_fair_comparison.py \
        --config         "$BAM_B_CFG" \
        --bam_checkpoint "$BAM_B_BEST" \
        --mrl_checkpoint "$MRL_BEST" \
        $BAM_RESULTS_ARG \
        --output_dir     "$OUT_DIR/" \
        && echo "  Saved → $OUT_DIR/fair_comparison.json" \
        || echo "  FAILED for $DS"
done

echo ""
echo "══════════════════════════════════════════════════════"
echo "  SUMMARY ACROSS ALL DATASETS"
echo "══════════════════════════════════════════════════════"

python3 - <<'PYEOF'
import json, os, math

datasets = ["educational", "scifact", "nfcorpus", "fiqa"]
results_root = "./results/multi_domain"
BLOOM_NAMES = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

rows = []
for ds in datasets:
    path = os.path.join(results_root, ds, "fair_comparison", "fair_comparison.json")
    if not os.path.exists(path):
        print(f"  {ds}: no results yet")
        continue
    with open(path) as f:
        fc = json.load(f)
    rows.append((ds, fc))

if not rows:
    print("  No results found.")
    exit()

# Per-level table
print(f"\n  {'Dataset':14s}  {'Level':12s}  {'N':>5}  {'Budget':>7}  "
      f"{'MRL-full':>9}  {'MRL-trunc':>10}  {'BAM-B':>7}  {'Δ':>7}")
print("  " + "-" * 80)
for ds, fc in rows:
    for name in BLOOM_NAMES:
        lv = fc["per_level"].get(name)
        if not lv:
            continue
        n      = lv["n"]
        budget = lv.get("budget_dims", 0)
        r_full = lv.get("mrl_full_recall@10", 0)
        r_trunc= lv.get("mrl_truncated_recall@10", 0)
        r_bam  = lv.get("bam_recall@10", 0)
        delta  = lv.get("delta_bam_minus_mrl_trunc", math.nan)
        sign   = "+" if delta >= 0 else ""
        print(f"  {ds:14s}  {name:12s}  {n:5d}  {budget:7d}  "
              f"{r_full:9.4f}  {r_trunc:10.4f}  {r_bam:7.4f}  {sign}{delta*100:.1f}%")
    print()

# Dataset-level summary
print(f"\n  {'Dataset':14s}  {'Avg Δ (BAM−MRL_trunc)':>22s}  {'BAM wins':>10s}")
print("  " + "-" * 55)
for ds, fc in rows:
    avg_d = fc.get("avg_delta_bam_minus_mrl_trunc", math.nan)
    wins  = fc.get("bam_wins", 0)
    total = fc.get("total_levels", 6)
    sign  = "+" if avg_d >= 0 else ""
    print(f"  {ds:14s}  {sign}{avg_d*100:>20.2f}%  {wins}/{total:>8}")
PYEOF
