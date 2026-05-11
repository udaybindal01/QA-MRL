#!/usr/bin/env bash
# =============================================================================
# pipeline_cutoff_norm.sh
#
# Evaluate Cutoff and Norm baselines on educational Standard FT and MRL
# checkpoints for all backbones.
#
# Both methods are post-hoc (no training) — they run directly on the best/
# checkpoint produced by find_standard_ft / find_mrl_bk for each backbone.
#
#   Cutoff  — fixed prefix truncation at each dim budget.
#             All queries use the same d dimensions.
#
#   Norm    — per-query top-d dimension selection by embedding magnitude |e_q[i]|.
#             Each query picks its own dims; corpus projected to match.
#
# Runs both Standard FT and MRL checkpoints so results can be compared side
# by side. Standard FT shows how badly random-structured dims degrade under
# Cutoff; MRL shows the benefit of Matryoshka training for prefix truncation.
#
# Checkpoint paths expected:
#   Standard FT : $CKPT_ROOT/educational/standard_ft_$BK/best/checkpoint.pt
#   MRL         : $CKPT_ROOT/educational/mrl_$BK/best/checkpoint.pt
#
# Usage:
#   ./pipeline_cutoff_norm.sh                            # all backbones, both models
#   ./pipeline_cutoff_norm.sh --backbone "bge qwen06b"
#   ./pipeline_cutoff_norm.sh --model mrl                # MRL only
#   ./pipeline_cutoff_norm.sh --model standard_ft        # Standard FT only
#   ./pipeline_cutoff_norm.sh --add_30pct                # also eval at 30% of dim
#   ./pipeline_cutoff_norm.sh --force                    # re-run even if output exists
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
RUN_MODELS="standard_ft mrl"   # which model types to evaluate
ADD_30PCT=""
FORCE=0
BATCH_SIZE=128
K=10

# ── Backbone → Standard FT config map (educational) ──────────────────────────
declare -A SFT_CFG=(
    [e5large]="configs/standard_ft_e5large.yaml"
    [bge]="configs/standard_ft_bge.yaml"
    [qwen06b]="configs/standard_ft_qwen06b.yaml"
    [qwen4b]="configs/standard_ft_qwen4b.yaml"
    [qwen8b]="configs/standard_ft_qwen8b.yaml"
    [llm2vec]="configs/standard_ft_llm2vec.yaml"
    [llama8b]="configs/standard_ft_llm2vec_llama8b.yaml"
    [gritlm]="configs/standard_ft_gritlm.yaml"
    [llama1b]="configs/standard_ft_llama1b.yaml"
    [llama3b]="configs/standard_ft_llama3b.yaml"
    [arctic]="configs/standard_ft_arctic.yaml"
    [roberta]="configs/standard_ft_roberta.yaml"
    [phi3mini]="configs/standard_ft_phi3mini.yaml"
)

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
        --backbone)    BACKBONES="$2";    shift 2 ;;
        --model)       RUN_MODELS="$2";   shift 2 ;;
        --add_30pct)   ADD_30PCT="--add_30pct"; shift ;;
        --force)       FORCE=1;           shift ;;
        --batch_size)  BATCH_SIZE="$2";   shift 2 ;;
        --k)           K="$2";            shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

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
echo "  Models      : $RUN_MODELS"
echo "  R@K         : $K"
echo "  Add 30%     : ${ADD_30PCT:-no}"
echo "============================================================"

# ── Helper: run one backbone + model type ─────────────────────────────────────
run_eval() {
    local BK="$1"
    local MODEL_TYPE="$2"   # "standard_ft" or "mrl"
    local CKPT_SUBDIR       # e.g. standard_ft_bge or mrl_bge
    local CFG

    if [[ "$MODEL_TYPE" == "standard_ft" ]]; then
        [[ -n "${SFT_CFG[$BK]+x}" ]] || { log "[$BK][sft] No SFT config — skipping"; return; }
        CFG="${SFT_CFG[$BK]}"
        CKPT_SUBDIR="standard_ft_$BK"
    else
        [[ -n "${MRL_CFG[$BK]+x}" ]] || { log "[$BK][mrl] No MRL config — skipping"; return; }
        CFG="${MRL_CFG[$BK]}"
        CKPT_SUBDIR="mrl_$BK"
    fi

    local CKPT_DIR="$CKPT_ROOT/$DATASET/$CKPT_SUBDIR/best"
    local CKPT_FILE="$CKPT_DIR/checkpoint.pt"
    local OUT_DIR="$RESULTS_ROOT/$MODEL_TYPE/$BK"
    local OUT_FILE="$OUT_DIR/cutoff_norm.json"

    if [[ -f "$OUT_FILE" && "$FORCE" -eq 0 ]]; then
        log "[$BK][$MODEL_TYPE] Already done — skipping (use --force to re-run)"
        return 0
    fi

    if [[ ! -f "$CKPT_FILE" ]]; then
        log "[$BK][$MODEL_TYPE] No checkpoint at $CKPT_FILE — skipping"
        return 1
    fi

    if [[ ! -f "$CFG" ]]; then
        log "[$BK][$MODEL_TYPE] Config not found: $CFG — skipping"
        return 1
    fi

    mkdir -p "$OUT_DIR"
    log "[$BK][$MODEL_TYPE] Running Cutoff + Norm eval..."

    python "$SCRIPT_DIR/scripts/eval_cutoff_norm.py" \
        --config      "$CFG" \
        --checkpoint  "$CKPT_DIR" \
        --test_path   "$TEST_PATH" \
        --corpus_path "$CORPUS_PATH" \
        --output_dir  "$OUT_DIR" \
        --batch_size  "$BATCH_SIZE" \
        --k           "$K" \
        $ADD_30PCT

    log "[$BK][$MODEL_TYPE] Done → $OUT_FILE"
    return 0
}

# ── Main loop ─────────────────────────────────────────────────────────────────
for BK in $BACKBONES; do
    for MODEL_TYPE in $RUN_MODELS; do
        run_eval "$BK" "$MODEL_TYPE" || true
    done
done

# ── Summary table ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Aggregated R@${K}  —  Standard FT vs MRL  (Cutoff | Norm)"
echo "============================================================"

python - <<PYEOF
import json, os, sys

results_root = "$RESULTS_ROOT"
k            = $K
backbones    = "$BACKBONES".split()
models       = "$RUN_MODELS".split()

# Collect results for each (model_type, backbone)
all_data = {}
all_dims = set()
for mt in models:
    for bk in backbones:
        path = os.path.join(results_root, mt, bk, "cutoff_norm.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        all_data[(mt, bk)] = d
        all_dims.update(int(x) for x in d["cutoff"].keys())

if not all_data:
    print("  No results found yet.")
    sys.exit(0)

all_dims = sorted(all_dims)

# Print one row per backbone, columns = (model_type × dim × cutoff/norm)
# Header: Backbone | SFT Cut-D  SFT Nrm-D  MRL Cut-D  MRL Nrm-D  ...
print(f"\n  Standard FT = model trained with standard contrastive loss (no MRL structure)")
print(f"  MRL         = model trained with Matryoshka multi-resolution loss\n")

# Show a compact table: for each backbone, per-dim recall for both model types
for mt in models:
    label = "Standard FT" if mt == "standard_ft" else "MRL"
    print(f"  [{label}]  R@{k} by dim budget")
    hdr = f"  {'Backbone':14s}  {'EmbDim':>7}"
    for d in all_dims:
        hdr += f"  {'Cut@'+str(d):>9}  {'Nrm@'+str(d):>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for bk in backbones:
        entry = all_data.get((mt, bk))
        if entry is None:
            print(f"  {bk:14s}  {'—':>7}  (no checkpoint)")
            continue
        emb_dim = entry["embedding_dim"]
        line = f"  {bk:14s}  {emb_dim:>7}"
        for d in all_dims:
            r_c = entry["cutoff"].get(str(d), {}).get(f"recall@{k}", float("nan"))
            r_n = entry["norm"].get(str(d),   {}).get(f"recall@{k}", float("nan"))
            line += f"  {r_c:>9.4f}  {r_n:>9.4f}"
        print(line)
    print()

# Side-by-side diff: MRL Cutoff - SFT Cutoff at each dim
if "standard_ft" in models and "mrl" in models:
    print(f"  [Delta: MRL Cutoff − SFT Cutoff]  (positive = MRL structure helps)")
    hdr = f"  {'Backbone':14s}"
    for d in all_dims:
        hdr += f"  {'Δ Cut@'+str(d):>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for bk in backbones:
        sft = all_data.get(("standard_ft", bk))
        mrl = all_data.get(("mrl", bk))
        if sft is None or mrl is None:
            continue
        line = f"  {bk:14s}"
        for d in all_dims:
            r_sft = sft["cutoff"].get(str(d), {}).get(f"recall@{k}", float("nan"))
            r_mrl = mrl["cutoff"].get(str(d), {}).get(f"recall@{k}", float("nan"))
            delta = r_mrl - r_sft
            line += f"  {delta:>+10.4f}"
        print(line)
    print()
PYEOF

echo "  Results saved in $RESULTS_ROOT/{standard_ft,mrl}/<backbone>/cutoff_norm.json"
