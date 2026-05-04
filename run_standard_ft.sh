#!/usr/bin/env bash
# =============================================================================
# run_standard_ft.sh  —  Train + eval standard fine-tuning baselines
#                         for all 5 completed BAM-PQ backbones
#
# Usage:
#   ./run_standard_ft.sh                          # all backbones
#   ./run_standard_ft.sh --backbones "bge arctic" # subset
#   ./run_standard_ft.sh --skip_train             # eval only (ckpts exist)
#
# Output:
#   Checkpoints : /tmp/uday/standard-ft/{backbone}/
#   Results     : results/standard_ft/educational/{backbone}/eval.json
#   Summary     : results/standard_ft/educational/summary.json
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ──────────────────────────────────────────────────────────────────
BACKBONES="${BACKBONES:-e5large bge arctic roberta qwen06b}"
CKPT_ROOT="${CKPT_ROOT:-/tmp/uday/standard-ft}"
OUT_ROOT="${OUT_ROOT:-results/standard_ft/educational}"
SKIP_TRAIN=0

# Config map: backbone → config file
declare -A CFG_MAP=(
    [e5large]="configs/standard_ft_e5large.yaml"
    [bge]="configs/standard_ft_bge.yaml"
    [arctic]="configs/standard_ft_arctic.yaml"
    [roberta]="configs/standard_ft_roberta.yaml"
    [qwen06b]="configs/standard_ft_qwen06b.yaml"
)

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backbones)   BACKBONES="$2";  shift 2 ;;
        --skip_train)  SKIP_TRAIN=1;    shift   ;;
        --ckpt_root)   CKPT_ROOT="$2";  shift 2 ;;
        --out)         OUT_ROOT="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUT_ROOT"

echo "============================================================"
echo "  Standard FT baseline — Educational dataset"
echo "  Backbones  : $BACKBONES"
echo "  Ckpt root  : $CKPT_ROOT"
echo "  Output     : $OUT_ROOT"
echo "  Skip train : $SKIP_TRAIN"
echo "============================================================"

FAILED=()
DONE=()

for BK in $BACKBONES; do
    echo ""
    echo "──────────────────────────────────────────────────────"
    echo "  Backbone: $BK"
    echo "──────────────────────────────────────────────────────"

    CFG="${CFG_MAP[$BK]:-}"
    if [[ -z "$CFG" || ! -f "$CFG" ]]; then
        echo "  SKIP — no config found for $BK (looked for ${CFG_MAP[$BK]:-?})"
        FAILED+=("$BK")
        continue
    fi

    CKPT_DIR="$CKPT_ROOT/$BK"
    OUT_DIR="$OUT_ROOT/$BK"
    mkdir -p "$CKPT_DIR" "$OUT_DIR"

    # ── 1. Train ──────────────────────────────────────────────────────────────
    if [[ $SKIP_TRAIN -eq 0 ]]; then
        echo "  [1/3] Training standard FT ..."
        if ! python3 scripts/train_baseline_mrl.py \
                --config "$CFG" \
                --checkpoint_dir "$CKPT_DIR"; then
            echo "  ✗  $BK training FAILED"
            FAILED+=("$BK")
            continue
        fi
        echo "  ✓  Training done"
    else
        echo "  [1/3] Skipping training (--skip_train)"
    fi

    # ── 2. Find best epoch ────────────────────────────────────────────────────
    BEST_DIR="$CKPT_DIR/best"
    if [[ ! -f "$BEST_DIR/checkpoint.pt" ]]; then
        echo "  [2/3] Selecting best epoch ..."
        if ! python3 scripts/find_best_epoch.py \
                --config "$CFG" \
                --checkpoint_dir "$CKPT_DIR" \
                --model_type mrl \
                --metric "recall@10"; then
            echo "  ✗  $BK find_best_epoch FAILED"
            FAILED+=("$BK")
            continue
        fi
        echo "  ✓  Best checkpoint → $BEST_DIR"
    else
        echo "  [2/3] Best checkpoint exists, skipping selection"
    fi

    # ── 3. Evaluate ───────────────────────────────────────────────────────────
    echo "  [3/3] Evaluating ..."
    if python3 scripts/eval_edu_baselines.py \
            --config         "$CFG" \
            --checkpoint     "$BEST_DIR" \
            --model_type     mrl \
            --output_dir     "$OUT_DIR" \
            --bloom_stratified; then
        echo "  ✓  $BK done → $OUT_DIR/eval.json"
        DONE+=("$BK")
    else
        echo "  ✗  $BK eval FAILED"
        FAILED+=("$BK")
    fi
done

# ── Summary JSON ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  DONE   : ${DONE[*]:-none}"
echo "  FAILED : ${FAILED[*]:-none}"
echo "============================================================"

export OUT_ROOT

python3 - <<'PYEOF'
import json, os

out_root = os.environ.get("OUT_ROOT", "results/standard_ft/educational")
summary  = {}

bloom_names = ["Remember","Understand","Apply","Analyze","Evaluate","Create"]

for bk in sorted(os.listdir(out_root)):
    p = os.path.join(out_root, bk, "eval.json")
    if not os.path.isfile(p):
        continue
    with open(p) as f:
        d = json.load(f)

    overall = d.get("overall", {})
    per_bloom = d.get("per_bloom", {})

    summary[bk] = {
        "r10":    overall.get("recall@10"),
        "ndcg10": overall.get("ndcg@10"),
        "mrr":    overall.get("mrr"),
        "per_bloom_r10": {
            bl: per_bloom.get(bl, {}).get("recall@10")
            for bl in bloom_names
        }
    }

# Print table
print()
print(f"  {'Backbone':<14}  {'R@10':>8}  {'NDCG@10':>10}  {'MRR':>8}")
print("  " + "-"*46)
for bk, v in summary.items():
    r   = f"{v['r10']:.4f}"  if v['r10']  else "—"
    nd  = f"{v['ndcg10']:.4f}" if v['ndcg10'] else "—"
    mrr = f"{v['mrr']:.4f}"  if v['mrr']  else "—"
    print(f"  {bk:<14}  {r:>8}  {nd:>10}  {mrr:>8}")

print()
print(f"  Per-Bloom R@10")
print(f"  {'Backbone':<14}", end="")
for b in bloom_names:
    print(f"  {b:>10}", end="")
print()
print("  " + "-"*80)
for bk, v in summary.items():
    print(f"  {bk:<14}", end="")
    for b in bloom_names:
        val = v["per_bloom_r10"].get(b)
        print(f"  {val:>10.4f}" if val else f"  {'—':>10}", end="")
    print()

out_path = os.path.join(out_root, "summary.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n  Saved → {out_path}")
PYEOF
