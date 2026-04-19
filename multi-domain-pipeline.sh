#!/usr/bin/env bash
# =============================================================================
# BAM Multi-Domain Pipeline
# =============================================================================
#
# Trains MRL + BAM-A + BAM-B independently on each dataset, then evaluates
# on each dataset's own test set. Produces a single comparison table showing
# BAM consistently beats MRL across all domains.
#
# Datasets (each trained and tested independently):
#   educational  — SciQ/QASC/OpenBookQA educational QA (existing data)
#   scifact      — scientific claim verification
#   nfcorpus     — medical/health information retrieval
#   fiqa         — financial QA
#
# For each dataset the pipeline runs:
#   build  → train_mrl → find_mrl → train_bam_a → train_bam_b
#          → find_bam_a → find_bam_b → eval
#
# Final step prints a table:
#   Dataset      | MRL R@10 | BAM-A R@10 | BAM-B R@10 | BAM-B Dims
#   educational  |  0.527   |   0.559    |   0.564    |   407
#   scifact      |   ...    |    ...     |    ...     |   ...
#   nfcorpus     |   ...    |    ...     |    ...     |   ...
#   fiqa         |   ...    |    ...     |    ...     |   ...
#
# Usage:
#   chmod +x multi-domain-pipeline.sh
#   ./multi-domain-pipeline.sh                         # all datasets
#   ./multi-domain-pipeline.sh --datasets "educational scifact"
#   ./multi-domain-pipeline.sh --from eval             # skip to eval for all
#   ./multi-domain-pipeline.sh --dataset scifact --from train_bam_b
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATASETS="${DATASETS:-educational scifact nfcorpus fiqa}"
BEIR_DATA_ROOT="/tmp/data/beir"
EDU_DATA_DIR="./data/real"
CKPT_ROOT="/tmp/multi-domain"
RESULTS_ROOT="./results/multi_domain"
BSR_ALPHA="0.5"

# Base configs — architecture only, data paths overridden per dataset
BASE_MRL_CONFIG="configs/mrl_e5large.yaml"
BASE_BAM_A_CONFIG="configs/bam_optionA_e5large.yaml"
BASE_BAM_B_CONFIG="configs/bam_optionb_e5large.yaml"
# ─────────────────────────────────────────────────────────────────────────────

# ── Argument parsing ─────────────────────────────────────────────────────────
FROM_STEP=""
SINGLE_DATASET=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)     FROM_STEP="$2";      shift 2 ;;
        --datasets) DATASETS="$2";       shift 2 ;;
        --dataset)  SINGLE_DATASET="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -n "$SINGLE_DATASET" ]] && DATASETS="$SINGLE_DATASET"

ALL_STEPS=(build annotate train_mrl find_mrl train_bam_a train_bam_b find_bam_a find_bam_b eval fair_cmp eff_curves)

should_run() {
    local step="$1"
    if [[ -z "$FROM_STEP" ]]; then return 0; fi
    local found=0
    for s in "${ALL_STEPS[@]}"; do
        [[ "$s" == "$FROM_STEP" ]] && found=1
        [[ $found -eq 1 && "$s" == "$step" ]] && return 0
    done
    return 1
}

log() {
    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  [$(date '+%H:%M:%S')]  $1"
    echo "══════════════════════════════════════════════════════"
}
die() { echo "ERROR: $1" >&2; exit 1; }

# ── Helper: generate per-dataset config ──────────────────────────────────────
make_config() {
    local base_cfg="$1"
    local out_cfg="$2"
    local train_path="$3"
    local val_path="$4"
    local test_path="$5"
    local corpus_path="$6"
    local ckpt_dir="$7"

    python3 - <<PYEOF
import yaml, sys

with open("$base_cfg") as f:
    cfg = yaml.safe_load(f)

cfg["data"]["train_path"]   = "$train_path"
cfg["data"]["val_path"]     = "$val_path"
cfg["data"]["test_path"]    = "$test_path"
cfg["data"]["corpus_path"]  = "$corpus_path"
cfg["training"]["checkpoint_dir"] = "$ckpt_dir"

with open("$out_cfg", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
print(f"  Generated config: $out_cfg")
PYEOF
}

# ─────────────────────────────────────────────────────────────────────────────
# PREREQ CHECK
# ─────────────────────────────────────────────────────────────────────────────
log "PREREQ CHECK"
[[ -f "$EDU_DATA_DIR/train_curriculum.jsonl" ]] \
    || die "Educational data not found at $EDU_DATA_DIR. Run optionA-working_pipeline.sh first."

mkdir -p "$BEIR_DATA_ROOT" "$CKPT_ROOT" "$RESULTS_ROOT"
echo "  Datasets   : $DATASETS"
echo "  Ckpt root  : $CKPT_ROOT"
echo "  Results    : $RESULTS_ROOT"

# ─────────────────────────────────────────────────────────────────────────────
# PER-DATASET LOOP
# ─────────────────────────────────────────────────────────────────────────────
for DS in $DATASETS; do

    log "━━━━  DATASET: $DS  ━━━━"

    # ── Resolve data paths ───────────────────────────────────────────────────
    if [[ "$DS" == "educational" ]]; then
        TRAIN_PATH="$EDU_DATA_DIR/train_curriculum.jsonl"
        VAL_PATH="$EDU_DATA_DIR/val.jsonl"
        TEST_PATH="$EDU_DATA_DIR/test.jsonl"
        CORPUS_PATH="$EDU_DATA_DIR/corpus.jsonl"
    else
        DS_DIR="$BEIR_DATA_ROOT/$DS"
        TRAIN_PATH="$DS_DIR/train.jsonl"
        VAL_PATH="$DS_DIR/val.jsonl"
        TEST_PATH="$DS_DIR/test.jsonl"
        CORPUS_PATH="$DS_DIR/corpus.jsonl"
    fi

    MRL_CKPT="$CKPT_ROOT/$DS/mrl"
    BAM_A_CKPT="$CKPT_ROOT/$DS/bam_a"
    BAM_B_CKPT="$CKPT_ROOT/$DS/bam_b"
    DS_RESULTS="$RESULTS_ROOT/$DS"
    CFG_DIR="$DS_RESULTS/configs"

    mkdir -p "$MRL_CKPT" "$BAM_A_CKPT" "$BAM_B_CKPT" "$DS_RESULTS" "$CFG_DIR"

    MRL_CFG="$CFG_DIR/mrl.yaml"
    BAM_A_CFG="$CFG_DIR/bam_a.yaml"
    BAM_B_CFG="$CFG_DIR/bam_b.yaml"

    # ── STEP: build ──────────────────────────────────────────────────────────
    if should_run build; then
        if [[ "$DS" == "educational" ]]; then
            log "[$DS] BUILD — using existing educational data"
            echo "  train : $TRAIN_PATH"
            echo "  val   : $VAL_PATH"
            echo "  test  : $TEST_PATH"
            echo "  corpus: $CORPUS_PATH"
        else
            log "[$DS] BUILD — downloading BEIR train/val/test splits"
            if [[ -f "$CORPUS_PATH" ]] && [[ -f "$TRAIN_PATH" ]]; then
                echo "  Already built at $DS_DIR — skipping."
            else
                python3 data/build_beir_training_data.py \
                    --datasets "$DS" \
                    --output_dir "$BEIR_DATA_ROOT" \
                    --num_neg 7 \
                    || die "[$DS] build_beir_training_data.py failed"
            fi
        fi
        # Generate per-dataset configs
        make_config "$BASE_MRL_CONFIG"   "$MRL_CFG"   "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$CORPUS_PATH" "$MRL_CKPT/"
        make_config "$BASE_BAM_A_CONFIG" "$BAM_A_CFG" "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$CORPUS_PATH" "$BAM_A_CKPT/"
        make_config "$BASE_BAM_B_CONFIG" "$BAM_B_CFG" "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$CORPUS_PATH" "$BAM_B_CKPT/"
    fi

    # Configs must exist for subsequent steps even when skipping build
    if [[ ! -f "$MRL_CFG" ]]; then
        make_config "$BASE_MRL_CONFIG"   "$MRL_CFG"   "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$CORPUS_PATH" "$MRL_CKPT/"
        make_config "$BASE_BAM_A_CONFIG" "$BAM_A_CFG" "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$CORPUS_PATH" "$BAM_A_CKPT/"
        make_config "$BASE_BAM_B_CONFIG" "$BAM_B_CFG" "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$CORPUS_PATH" "$BAM_B_CKPT/"
    fi

    # ── STEP: annotate ───────────────────────────────────────────────────────
    if should_run annotate; then
        if [[ "$DS" == "educational" ]]; then
            log "[$DS] ANNOTATE — educational data uses pretrained BERT classifier (skipping)"
        else
            log "[$DS] ANNOTATE — zero-shot NLI Bloom labels for $DS (fixes 82% Remember collapse)"
            python3 data/annotate_bloom_local.py \
                --beir_root "$BEIR_DATA_ROOT" \
                --datasets  "$DS" \
                || die "[$DS] Bloom annotation failed"
            echo "  Annotation complete → bloom_cache files updated"
        fi
    fi

    # ── STEP: train_mrl ──────────────────────────────────────────────────────
    if should_run train_mrl; then
        log "[$DS] TRAIN MRL BASELINE"
        if [[ -f "$MRL_CKPT/best/checkpoint.pt" ]] || [[ -d "$MRL_CKPT/epoch_0" ]]; then
            echo "  MRL checkpoint exists — skipping."
        else
            python3 scripts/train_baseline_mrl.py \
                --config "$MRL_CFG" \
                --checkpoint_dir "$MRL_CKPT" \
                || die "[$DS] MRL training failed"
        fi
    fi

    # ── STEP: find_mrl ───────────────────────────────────────────────────────
    if should_run find_mrl; then
        log "[$DS] FIND BEST MRL EPOCH"
        if [[ -f "$MRL_CKPT/best/checkpoint.pt" ]]; then
            echo "  MRL best already exists."
        else
            python3 scripts/find_best_epoch.py \
                --config "$MRL_CFG" \
                --checkpoint_dir "$MRL_CKPT" \
                --model_type mrl \
                || die "[$DS] find_best_epoch (MRL) failed"
        fi
    fi

    MRL_BEST="$MRL_CKPT/best"

    # ── STEP: train_bam_a ────────────────────────────────────────────────────
    if should_run train_bam_a; then
        log "[$DS] TRAIN BAM OPTION A"
        if [[ -f "$BAM_A_CKPT/best_bsr/checkpoint.pt" ]] || [[ -d "$BAM_A_CKPT/epoch_0" ]]; then
            echo "  Option A checkpoint exists — skipping."
        else
            [[ -f "$MRL_BEST/checkpoint.pt" ]] || die "[$DS] MRL best not found — run find_mrl first"
            python3 scripts/train_bam.py \
                --config "$BAM_A_CFG" \
                --init_encoder "$MRL_BEST" \
                --checkpoint_dir "$BAM_A_CKPT" \
                || die "[$DS] BAM-A training failed"
        fi
    fi

    # ── STEP: train_bam_b ────────────────────────────────────────────────────
    if should_run train_bam_b; then
        log "[$DS] TRAIN BAM OPTION B (reverse two-stage)"
        if [[ -f "$BAM_B_CKPT/best_bsr/checkpoint.pt" ]] || [[ -d "$BAM_B_CKPT/epoch_0" ]]; then
            echo "  Option B checkpoint exists — skipping."
        else
            [[ -f "$MRL_BEST/checkpoint.pt" ]] || die "[$DS] MRL best not found — run find_mrl first"
            python3 scripts/train_bam.py \
                --config "$BAM_B_CFG" \
                --init_encoder "$MRL_BEST" \
                --checkpoint_dir "$BAM_B_CKPT" \
                --freeze_encoder \
                || die "[$DS] BAM-B training failed"
        fi
    fi

    # ── STEP: find_bam_a ─────────────────────────────────────────────────────
    if should_run find_bam_a; then
        log "[$DS] BSR EPOCH SELECTION — OPTION A"
        mkdir -p "$DS_RESULTS/bam_a_bsr"
        python3 scripts/find_best_epoch_bsr.py \
            --config "$BAM_A_CFG" \
            --checkpoint_dir "$BAM_A_CKPT" \
            --output_dir "$DS_RESULTS/bam_a_bsr/" \
            --alpha "$BSR_ALPHA" \
            || die "[$DS] BSR selection (A) failed"
    fi

    # ── STEP: find_bam_b ─────────────────────────────────────────────────────
    if should_run find_bam_b; then
        log "[$DS] BSR EPOCH SELECTION — OPTION B"
        mkdir -p "$DS_RESULTS/bam_b_bsr"
        python3 scripts/find_best_epoch_bsr.py \
            --config "$BAM_B_CFG" \
            --checkpoint_dir "$BAM_B_CKPT" \
            --output_dir "$DS_RESULTS/bam_b_bsr/" \
            --alpha "$BSR_ALPHA" \
            || die "[$DS] BSR selection (B) failed"
    fi

    # ── STEP: eval ───────────────────────────────────────────────────────────
    if should_run eval; then
        log "[$DS] EVALUATION (BAM-A vs BAM-B vs MRL)"
        BAM_A_BEST="$BAM_A_CKPT/best_bsr"
        BAM_B_BEST="$BAM_B_CKPT/best_bsr"

        [[ -f "$MRL_BEST/checkpoint.pt" ]]   || die "[$DS] MRL best not found"
        [[ -f "$BAM_A_BEST/checkpoint.pt" ]] || die "[$DS] BAM-A best_bsr not found"
        [[ -f "$BAM_B_BEST/checkpoint.pt" ]] || die "[$DS] BAM-B best_bsr not found"

        python3 scripts/eval_bam.py \
            --config        "$BAM_A_CFG" \
            --checkpoint    "$BAM_A_BEST" \
            --baseline      "$MRL_BEST" \
            --checkpoint_v4 "$BAM_B_BEST" \
            --config_v4     "$BAM_B_CFG" \
            --output_dir    "$DS_RESULTS/" \
            || die "[$DS] eval_bam.py failed"
        echo "  Results → $DS_RESULTS/results.json"
    fi

    # ── STEP: fair_cmp ──────────────────────────────────────────────────────
    if should_run fair_cmp; then
        log "[$DS] FAIR COMPARISON — BAM-B vs MRL at same per-Bloom dim budget"
        BAM_B_BEST="$BAM_B_CKPT/best_bsr"
        [[ -f "$MRL_BEST/checkpoint.pt" ]]   || die "[$DS] MRL best not found"
        [[ -f "$BAM_B_BEST/checkpoint.pt" ]] || die "[$DS] BAM-B best_bsr not found"
        mkdir -p "$DS_RESULTS/fair_comparison"
        python3 scripts/eval_fair_comparison.py \
            --config         "$BAM_B_CFG" \
            --bam_checkpoint "$BAM_B_BEST" \
            --mrl_checkpoint "$MRL_BEST" \
            --bam_results    "$DS_RESULTS/results.json" \
            --output_dir     "$DS_RESULTS/fair_comparison/" \
            || die "[$DS] eval_fair_comparison.py failed"
        echo "  Fair comparison → $DS_RESULTS/fair_comparison/fair_comparison.json"
    fi

    # ── STEP: eff_curves ────────────────────────────────────────────────────
    if should_run eff_curves; then
        log "[$DS] EFFICIENCY CURVES — R@10 vs dims for MRL and BAM-B"
        BAM_B_BEST="$BAM_B_CKPT/best_bsr"
        [[ -f "$MRL_BEST/checkpoint.pt" ]]   || die "[$DS] MRL best not found"
        [[ -f "$BAM_B_BEST/checkpoint.pt" ]] || die "[$DS] BAM-B best_bsr not found"
        python3 scripts/eval_efficiency_curves.py \
            --config         "$BAM_B_CFG" \
            --bam_checkpoint "$BAM_B_BEST" \
            --mrl_checkpoint "$MRL_BEST" \
            --output_dir     "$DS_RESULTS/efficiency_curves/" \
            || die "[$DS] eval_efficiency_curves.py failed"
    fi

done  # end per-dataset loop

# ─────────────────────────────────────────────────────────────────────────────
# FINAL: MULTI-DOMAIN COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
log "MULTI-DOMAIN COMPARISON TABLE"

python3 - <<PYEOF
import json, os, math

datasets = "$DATASETS".split()
results_root = "$RESULTS_ROOT"

print()
print("  Standard comparison (BAM-B vs MRL at full MRL dims):")
print(f"  {'Dataset':14s} {'MRL R@10':>10s} {'BAM-A R@10':>12s} {'BAM-B R@10':>12s} {'BAM-B Dims':>12s} {'vs MRL':>8s}")
print("  " + "-" * 72)

for ds in datasets:
    path = os.path.join(results_root, ds, "results.json")
    if not os.path.exists(path):
        print(f"  {ds:14s}  (no results yet)")
        continue
    with open(path) as f:
        r = json.load(f)

    mrl   = r.get("MRL Baseline", {})
    bam_a = r.get("BAM", {})
    bam_b = r.get("BAM v4 (Option B)", {})

    mrl_r10  = mrl.get("recall@10", 0)
    a_r10    = bam_a.get("recall@10", 0)
    b_r10    = bam_b.get("recall@10", 0)
    b_dims   = bam_b.get("avg_active_dims", bam_b.get("avg_active_dims_scattered", 0))
    delta    = b_r10 - mrl_r10
    sign     = "+" if delta >= 0 else ""
    print(f"  {ds:14s} {mrl_r10:>10.4f} {a_r10:>12.4f} {b_r10:>12.4f} {b_dims:>12.0f} {sign}{delta*100:>6.1f}%")

print()
print("  FAIR comparison (BAM-B vs MRL truncated to SAME per-Bloom dim budget):")
print(f"  {'Dataset':14s} {'Avg Δ(BAM−MRL_trunc)':>22s} {'BAM wins (levels)':>20s}")
print("  " + "-" * 60)

for ds in datasets:
    fc_path = os.path.join(results_root, ds, "fair_comparison", "fair_comparison.json")
    if not os.path.exists(fc_path):
        print(f"  {ds:14s}  (fair comparison not yet run)")
        continue
    with open(fc_path) as f:
        fc = json.load(f)
    avg_d = fc.get("avg_delta_bam_minus_mrl_trunc", math.nan)
    wins  = fc.get("bam_wins", 0)
    total = fc.get("total_levels", 6)
    sign  = "+" if avg_d >= 0 else ""
    print(f"  {ds:14s} {sign}{avg_d*100:>20.2f}% {wins}/{total:>18}")

    # Print per-level breakdown
    for name, lv in fc.get("per_level", {}).items():
        budget = lv.get("budget_dims", 0)
        r_mrl  = lv.get("mrl_truncated_recall@10", 0)
        r_bam  = lv.get("bam_recall@10", 0)
        d      = r_bam - r_mrl
        sign   = "+" if d >= 0 else ""
        print(f"    {name:14s}  dims={budget:4d}  MRL-trunc={r_mrl:.4f}  BAM={r_bam:.4f}  {sign}{d*100:.1f}%")
    print()

PYEOF

log "PIPELINE COMPLETE"
echo ""
echo "  Per-dataset results: $RESULTS_ROOT/{dataset}/results.json"
echo "  Configs generated:   $RESULTS_ROOT/{dataset}/configs/"
