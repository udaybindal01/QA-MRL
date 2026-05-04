#!/usr/bin/env bash
# =============================================================================
# run_analysis.sh — Efficiency, Correlation & Significance analysis
#
# Runs scripts/analyze_bampq_complete.py for every completed BAM-PQ backbone
# on the educational dataset.  Skips any backbone whose checkpoint is missing.
#
# Usage:
#   ./run_analysis.sh                          # all backbones
#   ./run_analysis.sh --backbones "e5large bge"
#   ./run_analysis.sh --dataset msmarco        # once msmarco evals are done
#
# Output per backbone: results/analysis/{backbone}/bampq_analysis.json
# Cross-backbone:      results/analysis/summary_all.json
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ─────────────────────────────────────────────────────────────────
DATASET="${DATASET:-educational}"
CKPT_ROOT="${CKPT_ROOT:-/tmp/uday/multi-domain}"
CFG_ROOT="${CFG_ROOT:-results/multi_domain}"
OUT_ROOT="${OUT_ROOT:-results/analysis}"
N_DIM_SAMPLES="${N_DIM_SAMPLES:-2000}"
CORPUS_BATCH="${CORPUS_BATCH:-32}"
BACKBONES="e5large bge arctic roberta qwen06b"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backbones)  BACKBONES="$2";  shift 2 ;;
        --dataset)    DATASET="$2";    shift 2 ;;
        --ckpt_root)  CKPT_ROOT="$2";  shift 2 ;;
        --out)        OUT_ROOT="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

DS="$DATASET"
CFG_DIR="$CFG_ROOT/$DS/configs"
DS_CKPT="$CKPT_ROOT/$DS"

echo "============================================================"
echo "  BAM-PQ Analysis: $DS"
echo "  Backbones : $BACKBONES"
echo "  Ckpt root : $DS_CKPT"
echo "  Output    : $OUT_ROOT"
echo "============================================================"

mkdir -p "$OUT_ROOT"

BAM_B_CKPT="$DS_CKPT/bam_b/best_bsr"
BAM_B_CFG="$CFG_DIR/bam_b.yaml"

FAILED=()
DONE=()

for BK in $BACKBONES; do

    echo ""
    echo "──────────────────────────────────────────────────────"
    echo "  Backbone: $BK"
    echo "──────────────────────────────────────────────────────"

    BAMPQ_CKPT="$DS_CKPT/bam_pq_$BK/best_bsr"
    BAMPQ_CFG="$CFG_DIR/bam_pq_${BK}.yaml"

    # e5large shares the shared MRL checkpoint
    if [[ "$BK" == "e5large" ]]; then
        MRL_CKPT="$DS_CKPT/mrl/best"
        MRL_CFG="$CFG_DIR/mrl.yaml"
    else
        MRL_CKPT="$DS_CKPT/mrl_$BK/best"
        MRL_CFG="$CFG_DIR/mrl_${BK}.yaml"
    fi

    OUT_DIR="$OUT_ROOT/$BK"
    mkdir -p "$OUT_DIR"

    # Check required files
    SKIP=0
    for f in "$BAMPQ_CKPT/checkpoint.pt" "$MRL_CKPT/checkpoint.pt" \
              "$BAMPQ_CFG" "$MRL_CFG"; do
        if [[ ! -f "$f" ]]; then
            echo "  SKIP — missing: $f"
            SKIP=1
        fi
    done
    [[ $SKIP -eq 1 ]] && { FAILED+=("$BK"); continue; }

    # BAM-B optional inclusion
    BAM_B_ARGS=""
    if [[ -f "$BAM_B_CKPT/checkpoint.pt" && -f "$BAM_B_CFG" ]]; then
        BAM_B_ARGS="--bam_b_ckpt $BAM_B_CKPT --bam_b_config $BAM_B_CFG"
        echo "  BAM-B: found (will include in significance tests)"
    fi

    echo "  BAM-PQ : $BAMPQ_CKPT"
    echo "  MRL    : $MRL_CKPT"
    echo "  Out    : $OUT_DIR"
    echo ""

    if python3 scripts/analyze_bampq_complete.py \
            --config        "$BAMPQ_CFG" \
            --mrl_config    "$MRL_CFG" \
            --bampq_ckpt    "$BAMPQ_CKPT" \
            --mrl_ckpt      "$MRL_CKPT" \
            $BAM_B_ARGS \
            --output_dir    "$OUT_DIR" \
            --n_dim_samples "$N_DIM_SAMPLES" \
            --corpus_batch  "$CORPUS_BATCH" \
            --k             10; then
        echo "  ✓  $BK done → $OUT_DIR/bampq_analysis.json"
        DONE+=("$BK")
    else
        echo "  ✗  $BK FAILED"
        FAILED+=("$BK")
    fi

done

# ── Cross-backbone summary ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  DONE   : ${DONE[*]:-none}"
echo "  FAILED : ${FAILED[*]:-none}"
echo "============================================================"

export OUT_ROOT

python3 - <<'PYEOF'
import json, os

out_root = os.environ.get("OUT_ROOT", "results/analysis")
summary  = {}

for bk in sorted(os.listdir(out_root)):
    p = os.path.join(out_root, bk, "bampq_analysis.json")
    if not os.path.isfile(p):
        continue
    with open(p) as f:
        d = json.load(f)

    means   = d.get("means", {})
    dim_ana = d.get("dim_analysis", {})
    sig     = d.get("significance", {})
    eff     = d.get("efficiency", {})

    pq_r10  = means.get("BAM-PQ", {}).get("r10")
    mrl_r10 = means.get("MRL",    {}).get("r10")
    pq_nd   = means.get("BAM-PQ", {}).get("ndcg10")
    mrl_nd  = means.get("MRL",    {}).get("ndcg10")
    bb_r10  = means.get("BAM-B",  {}).get("r10")

    summary[bk] = {
        "BAM-PQ_r10":          pq_r10,
        "MRL_r10":             mrl_r10,
        "BAM-B_r10":           bb_r10,
        "delta_pq_mrl_r10":    round(pq_r10 - mrl_r10, 4) if (pq_r10 and mrl_r10) else None,
        "BAM-PQ_ndcg10":       pq_nd,
        "MRL_ndcg10":          mrl_nd,
        "delta_pq_mrl_ndcg":   round(pq_nd  - mrl_nd,  4) if (pq_nd  and mrl_nd)  else None,
        "kendall_tau":         dim_ana.get("kendall_tau"),
        "kendall_p":           dim_ana.get("kendall_p"),
        "spread_dims":         dim_ana.get("spread_dims"),
        "supports_cog_hyp":    dim_ana.get("supports_cognitive_hypothesis"),
        "per_bloom_mean_dims": dim_ana.get("per_bloom_mean_dims"),
        "sig_pq_vs_mrl_r10":   sig.get("BAM-PQ_vs_MRL_r10", {}).get("stars"),
        "p_bonferroni_r10":    sig.get("BAM-PQ_vs_MRL_r10", {}).get("p_bonferroni"),
        "ci_lo_r10":           sig.get("BAM-PQ_vs_MRL_r10", {}).get("ci_lo"),
        "ci_hi_r10":           sig.get("BAM-PQ_vs_MRL_r10", {}).get("ci_hi"),
    }

# ── Print table ──────────────────────────────────────────────────────────────
cols = [
    ("BAM-PQ_r10",        "PQ R@10",  ".4f"),
    ("MRL_r10",           "MRL R@10", ".4f"),
    ("delta_pq_mrl_r10",  "Δ R@10",   "+.4f"),
    ("BAM-PQ_ndcg10",     "PQ NDCG",  ".4f"),
    ("delta_pq_mrl_ndcg", "Δ NDCG",   "+.4f"),
    ("kendall_tau",        "τ (dims)", "+.3f"),
    ("kendall_p",          "τ p-val",  ".4f"),
    ("sig_pq_vs_mrl_r10",  "sig",      "s"),
    ("p_bonferroni_r10",   "p(adj)",   ".4f"),
]

print()
print(f"  {'Backbone':<14}", end="")
for _, hdr, _ in cols:
    print(f"  {hdr:>10}", end="")
print()
print("  " + "-" * (14 + 12 * len(cols)))

for bk, v in summary.items():
    print(f"  {bk:<14}", end="")
    for key, _, fmt in cols:
        val = v.get(key)
        if val is None:
            print(f"  {'—':>10}", end="")
        elif fmt == "s":
            print(f"  {str(val):>10}", end="")
        else:
            try:
                print(f"  {val:{fmt}s if fmt=='s' else fmt:>10}", end="")
            except Exception:
                print(f"  {str(val):>10}", end="")
    print()

# ── Per-Bloom dims table ──────────────────────────────────────────────────────
print()
print(f"\n  Per-Bloom Mean Active Dims")
bloom_order = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
print(f"  {'Backbone':<14}", end="")
for b in bloom_order:
    print(f"  {b:>10}", end="")
print()
print("  " + "-" * (14 + 12 * 6))
for bk, v in summary.items():
    dims = v.get("per_bloom_mean_dims") or {}
    print(f"  {bk:<14}", end="")
    for b in bloom_order:
        d = dims.get(b)
        print(f"  {d:>10.1f}" if d else f"  {'—':>10}", end="")
    print()

# Save
out_path = os.path.join(out_root, "summary_all.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n  Full JSON → {out_path}")
PYEOF
