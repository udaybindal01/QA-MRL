#!/usr/bin/env bash
# =============================================================================
# pipeline_cutoff_norm.sh
#
# Evaluate Cutoff and Norm baselines on educational MRL checkpoints.
#
# Both methods are post-hoc (no training) — they run directly on the best/
# checkpoint produced by find_mrl_bk for each backbone.
#
#   Cutoff  — fixed prefix truncation at each MRL dim budget.
#             All queries use the same d dimensions.
#
#   Norm    — per-query top-d dimension selection by embedding magnitude |e_q[i]|.
#             Each query picks its own dims; corpus projected to match.
#
# Checkpoint path expected: $CKPT_ROOT/educational/mrl_$BACKBONE/best/checkpoint.pt
#
# Usage:
#   ./pipeline_cutoff_norm.sh                         # all backbones
#   ./pipeline_cutoff_norm.sh --backbone "bge qwen06b"
#   ./pipeline_cutoff_norm.sh --add_30pct              # also evaluate at 30% of dim
#   ./pipeline_cutoff_norm.sh --force                  # re-run even if output exists
#   CKPT_ROOT=/my/ckpts ./pipeline_cutoff_norm.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Paths ─────────────────────────────────────────────────────────────────────
CKPT_ROOT="${CKPT_ROOT:-/tmp/multi-domain}"
RESULTS_ROOT="${RESULTS_ROOT:-./results/cutoff_norm}"
EDU_DATA_DIR="${EDU_DATA_DIR:-./data/real}"
DATASET="educational"

# ── Defaults ──────────────────────────────────────────────────────────────────
ALL_BACKBONES="e5large bge qwen06b qwen4b qwen8b llm2vec llama8b gritlm llama1b llama3b arctic roberta phi3mini"
BACKBONES="${BACKBONES:-$ALL_BACKBONES}"
ADD_30PCT=""
FORCE=0
BATCH_SIZE=128
K=10

# ── Backbone → MRL config map (educational) ───────────────────────────────────
declare -A MRL_CFG=(
    [e5large]="configs/mrl_e5large.yaml"
    [bge]="configs/mrl_bge_large.yaml"
    [qwen06b]="configs/mrl_qwen06b.yaml"
    [qwen4b]="configs/mrl_qwen4b.yaml"
    [qwen8b]="configs/mrl_qwen8b.yaml"
    [llm2vec]="configs/mrl_llm2vec_mistral7b.yaml"
    [llama8b]="configs/mrl_llm2vec_llama8b.yaml"
    [gritlm]="configs/mrl_gritlm7b.yaml"
    [llama1b]="configs/mrl_llama1b.yaml"
    [llama3b]="configs/mrl_llama3b.yaml"
    [arctic]="configs/mrl_arctic.yaml"
    [roberta]="configs/mrl_roberta.yaml"
    [phi3mini]="configs/mrl_phi3mini.yaml"
)

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backbone)   BACKBONES="$2";  shift 2 ;;
        --add_30pct)  ADD_30PCT="--add_30pct"; shift ;;
        --force)      FORCE=1;         shift ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --k)          K="$2";          shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

# ── Validate paths ────────────────────────────────────────────────────────────
[[ -d "$EDU_DATA_DIR" ]] || die "EDU_DATA_DIR not found: $EDU_DATA_DIR"
mkdir -p "$RESULTS_ROOT"

TEST_PATH="$EDU_DATA_DIR/test.jsonl"
CORPUS_PATH="$EDU_DATA_DIR/corpus.jsonl"
[[ -f "$TEST_PATH"   ]] || die "test.jsonl not found: $TEST_PATH"
[[ -f "$CORPUS_PATH" ]] || die "corpus.jsonl not found: $CORPUS_PATH"

echo "============================================================"
echo "  Cutoff + Norm Baseline Pipeline — Educational Dataset"
echo "============================================================"
echo "  CKPT_ROOT   : $CKPT_ROOT"
echo "  Results     : $RESULTS_ROOT"
echo "  Backbones   : $BACKBONES"
echo "  R@K         : $K"
echo "  Add 30%     : ${ADD_30PCT:-no}"
echo "============================================================"

# ── Track results for summary ─────────────────────────────────────────────────
declare -a DONE_BACKBONES=()
declare -a SKIP_BACKBONES=()
declare -a MISS_BACKBONES=()

# ── Main loop ─────────────────────────────────────────────────────────────────
for BK in $BACKBONES; do
    [[ -n "${MRL_CFG[$BK]+x}" ]] || { log "[$BK] Unknown backbone — skipping"; continue; }

    CKPT_DIR="$CKPT_ROOT/$DATASET/mrl_$BK/best"
    CKPT_FILE="$CKPT_DIR/checkpoint.pt"
    OUT_DIR="$RESULTS_ROOT/$BK"
    OUT_FILE="$OUT_DIR/cutoff_norm.json"
    CFG="${MRL_CFG[$BK]}"

    # Skip if already done and not forced
    if [[ -f "$OUT_FILE" && "$FORCE" -eq 0 ]]; then
        log "[$BK] Already done — skipping (use --force to re-run)"
        SKIP_BACKBONES+=("$BK")
        continue
    fi

    # Skip if checkpoint missing
    if [[ ! -f "$CKPT_FILE" ]]; then
        log "[$BK] No checkpoint at $CKPT_FILE — skipping"
        MISS_BACKBONES+=("$BK")
        continue
    fi

    # Skip if config missing
    if [[ ! -f "$CFG" ]]; then
        log "[$BK] Config not found: $CFG — skipping"
        MISS_BACKBONES+=("$BK")
        continue
    fi

    mkdir -p "$OUT_DIR"
    log "[$BK] Running Cutoff + Norm eval..."
    log "[$BK]   checkpoint : $CKPT_DIR"
    log "[$BK]   config     : $CFG"
    log "[$BK]   output     : $OUT_FILE"

    python "$SCRIPT_DIR/scripts/eval_cutoff_norm.py" \
        --config      "$CFG" \
        --checkpoint  "$CKPT_DIR" \
        --test_path   "$TEST_PATH" \
        --corpus_path "$CORPUS_PATH" \
        --output_dir  "$OUT_DIR" \
        --batch_size  "$BATCH_SIZE" \
        --k           "$K" \
        $ADD_30PCT

    log "[$BK] Done → $OUT_FILE"
    DONE_BACKBONES+=("$BK")
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"
echo "  Completed  (${#DONE_BACKBONES[@]}): ${DONE_BACKBONES[*]:-none}"
echo "  Skipped    (${#SKIP_BACKBONES[@]}): ${SKIP_BACKBONES[*]:-none}"
echo "  Missing ck (${#MISS_BACKBONES[@]}): ${MISS_BACKBONES[*]:-none}"

if [[ ${#DONE_BACKBONES[@]} -eq 0 && ${#SKIP_BACKBONES[@]} -eq 0 ]]; then
    echo ""
    echo "  No backbones ran. Check that MRL training has completed and"
    echo "  find_mrl_bk has been run for each backbone."
    echo "  Expected checkpoint path: $CKPT_ROOT/educational/mrl_<backbone>/best/"
    exit 1
fi

# ── Aggregate results table ───────────────────────────────────────────────────
ALL_DONE=( "${DONE_BACKBONES[@]:-}" "${SKIP_BACKBONES[@]:-}" )

if [[ ${#ALL_DONE[@]} -gt 0 ]]; then
    echo ""
    echo "  Aggregated R@${K} across backbones"
    echo "  (run python scripts/aggregate_cutoff_norm.py for full table)"
    python - <<PYEOF
import json, os, sys

results_root = "$RESULTS_ROOT"
k            = $K
backbones    = "$BACKBONES".split()

rows = []
for bk in backbones:
    path = os.path.join(results_root, bk, "cutoff_norm.json")
    if not os.path.exists(path):
        continue
    with open(path) as f:
        data = json.load(f)

    dims = data["dims_evaluated"]
    row  = {"backbone": bk, "embedding_dim": data["embedding_dim"]}
    for d in dims:
        s  = str(d)
        r_c = data["cutoff"].get(s, {}).get(f"recall@{k}", float("nan"))
        r_n = data["norm"].get(s,   {}).get(f"recall@{k}", float("nan"))
        row[f"cut_{d}"] = r_c
        row[f"nrm_{d}"] = r_n
    rows.append(row)

if not rows:
    print("  No results to aggregate yet.")
    sys.exit(0)

# Print header
all_dims = sorted({d for row in rows for key in row
                   if key.startswith("cut_")
                   for d in [int(key.split("_")[1])]})

hdr = f"  {'Backbone':14s}  {'Dim':>5}"
for d in all_dims:
    hdr += f"  {'Cut'+str(d):>8}  {'Nrm'+str(d):>8}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))

for row in rows:
    line = f"  {row['backbone']:14s}  {row['embedding_dim']:>5}"
    for d in all_dims:
        r_c = row.get(f"cut_{d}", float("nan"))
        r_n = row.get(f"nrm_{d}", float("nan"))
        line += f"  {r_c:>8.4f}  {r_n:>8.4f}"
    print(line)
PYEOF
fi

echo ""
echo "  Results saved in $RESULTS_ROOT/<backbone>/cutoff_norm.json"
