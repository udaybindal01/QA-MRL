#!/usr/bin/env bash
# =============================================================================
# Full BAM Pipeline — from raw data to EMNLP-ready results
# =============================================================================
#
# Runs every stage in order:
#   1. build        — download/prepare data for all 4 datasets
#   2. annotate     — zero-shot NLI Bloom labels on every dataset (educational +
#                     each BEIR set): train/val/test JSONLs + .bloom_cache.json
#                     (default: --overwrite so NLI_MODEL is always applied)
#   3. train_mrl    — train MRL baseline for each dataset
#   4. find_mrl     — select best MRL checkpoint (corpus-level, not in-batch)
#   5. train_bam_b  — reverse two-stage BAM-B: frozen mask → gentle encoder tune
#   6. find_bam_b   — BSR epoch selection for BAM-B
#   7. eval         — standard evaluation (BAM-B vs MRL full dims)
#   8. fair_cmp     — per-Bloom fair comparison at same dim budget
#   9. eff_curves   — efficiency-quality curves for paper Figure 2
#
# Usage:
#   chmod +x run_full_pipeline.sh
#   ./run_full_pipeline.sh                            # all datasets, all steps
#   ./run_full_pipeline.sh --from annotate            # skip build, start at annotate
#   ./run_full_pipeline.sh --from train_bam_b         # skip to BAM-B training
#   ./run_full_pipeline.sh --datasets "scifact fiqa bioasq"  # subset of datasets
#   ./run_full_pipeline.sh --from eval                # re-eval only
#   ./run_full_pipeline.sh --force                    # same as default annotate + also wipe any
#                                                       leftover ckpts before train (belt-and-suspenders)
#   FORCE_PIPELINE=1 ./run_full_pipeline.sh           # same as --force (env)
#   REUSE_BLOOM_CACHE=1 ./run_full_pipeline.sh        # skip --overwrite on NLI (faster if labels unchanged)
#   REUSE_TRAINED_MODELS=1 ./run_full_pipeline.sh     # skip MRL/BAM-B training if epoch ckpts already exist
#
# Requirements:
#   pip install transformers torch sentence-transformers faiss-gpu pyyaml
#
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit paths here if needed
# ─────────────────────────────────────────────────────────────────────────────
DATASETS="${DATASETS:-educational scifact nfcorpus fiqa bioasq}"
BEIR_DATA_ROOT="/tmp/data/beir"
EDU_DATA_DIR="./data/real"
CKPT_ROOT="/tmp/multi-domain"
RESULTS_ROOT="./results/multi_domain"
BSR_ALPHA="0.5"
NLI_MODEL="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
NLI_BATCH_SIZE="64"   # reduce to 32 if GPU OOM

# --force / FORCE_PIPELINE=1: extra checkpoint wipe before train (use if ckpts look stale).
FORCE="${FORCE_PIPELINE:-0}"
# Default 0: annotate step passes --overwrite so NLI_MODEL refreshes all splits; then ckpts are
# cleared and models retrained. Set REUSE_BLOOM_CACHE=1 to keep existing .bloom_cache.json behavior.
REUSE_BLOOM_CACHE="${REUSE_BLOOM_CACHE:-0}"
# Default 0: after a Bloom refresh, always retrain. Set REUSE_TRAINED_MODELS=1 to skip train when ckpts exist.
REUSE_TRAINED_MODELS="${REUSE_TRAINED_MODELS:-0}"

BASE_MRL_CONFIG="configs/mrl_e5large.yaml"
BASE_BAM_B_CONFIG="configs/bam_optionb_e5large.yaml"
BASE_BAM_PQ_CONFIG="configs/bam_pq.yaml"
# ─────────────────────────────────────────────────────────────────────────────

# ── Argument parsing ─────────────────────────────────────────────────────────
FROM_STEP=""
SINGLE_DATASET=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)      FROM_STEP="$2";      shift 2 ;;
        --datasets)  DATASETS="$2";       shift 2 ;;
        --dataset)   SINGLE_DATASET="$2"; shift 2 ;;
        --nli_model) NLI_MODEL="$2";      shift 2 ;;
        --force)     FORCE=1;             shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -n "$SINGLE_DATASET" ]] && DATASETS="$SINGLE_DATASET"

ALL_STEPS=(build annotate train_mrl find_mrl train_bam_b find_bam_b train_bam_pq find_bam_pq eval fair_cmp eff_curves eval_bam_pq fair_cmp_pq)

should_run() {
    local step="$1"
    [[ -z "$FROM_STEP" ]] && return 0
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
    echo "  [$(date '+%H:%M:%S')]  $*"
    echo "══════════════════════════════════════════════════════"
}
die() { echo "ERROR: $*" >&2; exit 1; }

# ── Config generator (injects per-dataset data paths into base YAML) ──────────
make_config() {
    local base_cfg="$1" out_cfg="$2" train="$3" val="$4" test="$5" corpus="$6" ckpt="$7"
    python3 - <<PYEOF
import yaml
with open("$base_cfg") as f:
    cfg = yaml.safe_load(f)
cfg["data"]["train_path"]         = "$train"
cfg["data"]["val_path"]           = "$val"
cfg["data"]["test_path"]          = "$test"
cfg["data"]["corpus_path"]        = "$corpus"
cfg["training"]["checkpoint_dir"] = "$ckpt"
with open("$out_cfg", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
print(f"  Config written: $out_cfg")
PYEOF
}

# ─────────────────────────────────────────────────────────────────────────────
# PREREQ CHECK
# ─────────────────────────────────────────────────────────────────────────────
log "PREREQ CHECK"
[[ -f "$EDU_DATA_DIR/train_curriculum.jsonl" ]] \
    || die "Educational data missing at $EDU_DATA_DIR — run data/build_real_data.py first."
mkdir -p "$BEIR_DATA_ROOT" "$CKPT_ROOT" "$RESULTS_ROOT"
echo "  Datasets  : $DATASETS"
echo "  CKPT root : $CKPT_ROOT"
echo "  Results   : $RESULTS_ROOT"
echo "  NLI model : $NLI_MODEL"
echo "  Bloom cache: REUSE_BLOOM_CACHE=$REUSE_BLOOM_CACHE  (1 = no --overwrite on annotate)"
echo "  Train skip : REUSE_TRAINED_MODELS=$REUSE_TRAINED_MODELS  (1 = keep ckpts if present)"
echo "  Force      : $FORCE  (1 = always wipe ckpts before train)"

# ─────────────────────────────────────────────────────────────────────────────
# PER-DATASET LOOP
# ─────────────────────────────────────────────────────────────────────────────
for DS in $DATASETS; do
    # Set when this run executed NLI with --overwrite so downstream training matches new Bloom labels.
    BLOOM_NLI_REFRESHED=0

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
    BAM_B_CKPT="$CKPT_ROOT/$DS/bam_b"
    BAM_PQ_CKPT="$CKPT_ROOT/$DS/bam_pq"
    DS_RESULTS="$RESULTS_ROOT/$DS"
    CFG_DIR="$DS_RESULTS/configs"
    MRL_CFG="$CFG_DIR/mrl.yaml"
    BAM_B_CFG="$CFG_DIR/bam_b.yaml"
    BAM_PQ_CFG="$CFG_DIR/bam_pq.yaml"
    MRL_BEST="$MRL_CKPT/best"
    BAM_B_BEST="$BAM_B_CKPT/best_bsr"
    BAM_PQ_BEST="$BAM_PQ_CKPT/best_bsr"

    mkdir -p "$MRL_CKPT" "$BAM_B_CKPT" "$BAM_PQ_CKPT" "$DS_RESULTS" "$CFG_DIR"

    # ── Always regenerate configs if missing (safe to re-run) ────────────────
    if [[ ! -f "$MRL_CFG" ]]; then
        make_config "$BASE_MRL_CONFIG"    "$MRL_CFG"    "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$CORPUS_PATH" "$MRL_CKPT/"
        make_config "$BASE_BAM_B_CONFIG"  "$BAM_B_CFG"  "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$CORPUS_PATH" "$BAM_B_CKPT/"
        make_config "$BASE_BAM_PQ_CONFIG" "$BAM_PQ_CFG" "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$CORPUS_PATH" "$BAM_PQ_CKPT/"
    fi

    # ── STEP 1: build ────────────────────────────────────────────────────────
    if should_run build; then
        if [[ "$DS" == "educational" ]]; then
            log "[$DS] BUILD — using existing educational data"
        else
            log "[$DS] BUILD — downloading BEIR dataset"
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
        make_config "$BASE_MRL_CONFIG"    "$MRL_CFG"    "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$CORPUS_PATH" "$MRL_CKPT/"
        make_config "$BASE_BAM_B_CONFIG"  "$BAM_B_CFG"  "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$CORPUS_PATH" "$BAM_B_CKPT/"
        make_config "$BASE_BAM_PQ_CONFIG" "$BAM_PQ_CFG" "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$CORPUS_PATH" "$BAM_PQ_CKPT/"
    fi

    # ── STEP 2: annotate ─────────────────────────────────────────────────────
    # Every dataset: same NLI path on train/val/test JSONLs. Default --overwrite so NLI_MODEL
    # always rewrites bloom_level + .bloom_cache.json (set REUSE_BLOOM_CACHE=1 to skip overwrite).
    if should_run annotate; then
        OVERWRITE_FLAG=(--overwrite)
        [[ "$REUSE_BLOOM_CACHE" == "1" ]] && OVERWRITE_FLAG=()

        ANNOTATE_JSONL=()
        if [[ "$DS" == "educational" ]]; then
            ANNOTATE_JSONL+=(
                "$EDU_DATA_DIR/train_curriculum.jsonl"
                "$EDU_DATA_DIR/val.jsonl"
                "$EDU_DATA_DIR/test.jsonl"
            )
        else
            for split in train val test; do
                p="$BEIR_DATA_ROOT/$DS/${split}.jsonl"
                [[ -f "$p" ]] && ANNOTATE_JSONL+=("$p")
            done
            if [[ ${#ANNOTATE_JSONL[@]} -eq 0 ]]; then
                die "[$DS] No train/val/test.jsonl under $BEIR_DATA_ROOT/$DS — run build first"
            fi
        fi

        log "[$DS] ANNOTATE — NLI Bloom (${#ANNOTATE_JSONL[@]} splits, model=$NLI_MODEL)"
        if [[ ${#OVERWRITE_FLAG[@]} -gt 0 ]]; then
            echo "  --overwrite on each file (fresh labels for this NLI run)"
            BLOOM_NLI_REFRESHED=1
        else
            echo "  REUSE_BLOOM_CACHE=1 — existing cache kept if size matches"
        fi
        for jsonl in "${ANNOTATE_JSONL[@]}"; do
            [[ -f "$jsonl" ]] || die "Missing $jsonl"
            python3 data/annotate_bloom_local.py \
                --input      "$jsonl" \
                --model      "$NLI_MODEL" \
                --batch_size "$NLI_BATCH_SIZE" \
                "${OVERWRITE_FLAG[@]}" \
                || die "[$DS] Bloom annotation failed for $jsonl"
        done
        echo "  Annotation complete for $DS"
    fi

    # ── STEP 3: train_mrl ────────────────────────────────────────────────────
    if should_run train_mrl; then
        log "[$DS] TRAIN MRL BASELINE"
        if [[ "$FORCE" == "1" ]] || [[ "$BLOOM_NLI_REFRESHED" == "1" ]]; then
            log "[$DS] Clearing MRL checkpoints (Bloom labels refreshed or --force)"
            rm -rf "$MRL_CKPT"/epoch_* "$MRL_CKPT"/inbatch_best "$MRL_CKPT"/best "$MRL_CKPT"/final 2>/dev/null || true
        fi
        if [[ "$REUSE_TRAINED_MODELS" == "1" ]] && [[ "$FORCE" != "1" ]] && [[ "$BLOOM_NLI_REFRESHED" != "1" ]] \
            && { [[ -f "$MRL_BEST/checkpoint.pt" ]] || ls "$MRL_CKPT"/epoch_* &>/dev/null 2>&1; }; then
            echo "  MRL checkpoint exists — skipping (REUSE_TRAINED_MODELS=1). Delete $MRL_CKPT or unset to retrain."
        else
            python3 scripts/train_baseline_mrl.py \
                --config "$MRL_CFG" \
                --checkpoint_dir "$MRL_CKPT" \
                || die "[$DS] MRL training failed"
        fi
    fi

    # ── STEP 4: find_mrl ─────────────────────────────────────────────────────
    if should_run find_mrl; then
        log "[$DS] SELECT BEST MRL EPOCH (corpus-level, not in-batch)"
        if [[ "$FORCE" != "1" ]] && [[ "$BLOOM_NLI_REFRESHED" != "1" ]] && [[ -f "$MRL_BEST/checkpoint.pt" ]]; then
            echo "  MRL best already selected."
        else
            python3 scripts/find_best_epoch.py \
                --config "$MRL_CFG" \
                --checkpoint_dir "$MRL_CKPT" \
                --model_type mrl \
                || die "[$DS] find_best_epoch (MRL) failed"
        fi
    fi

    # ── STEP 5: train_bam_b ──────────────────────────────────────────────────
    if should_run train_bam_b; then
        log "[$DS] TRAIN BAM-B (reverse two-stage: frozen mask → encoder fine-tune)"
        if [[ "$FORCE" == "1" ]] || [[ "$BLOOM_NLI_REFRESHED" == "1" ]]; then
            log "[$DS] Clearing BAM-B checkpoints (Bloom labels refreshed or --force)"
            rm -rf "$BAM_B_CKPT"/epoch_* "$BAM_B_CKPT"/inbatch_best "$BAM_B_CKPT"/best_bsr "$BAM_B_CKPT"/final 2>/dev/null || true
        fi
        if [[ "$REUSE_TRAINED_MODELS" == "1" ]] && [[ "$FORCE" != "1" ]] && [[ "$BLOOM_NLI_REFRESHED" != "1" ]] \
            && { [[ -f "$BAM_B_BEST/checkpoint.pt" ]] || ls "$BAM_B_CKPT"/epoch_* &>/dev/null 2>&1; }; then
            echo "  BAM-B checkpoint exists — skipping (REUSE_TRAINED_MODELS=1). Delete $BAM_B_CKPT or unset to retrain."
        else
            [[ -f "$MRL_BEST/checkpoint.pt" ]] || die "[$DS] MRL best not found — run find_mrl first"
            python3 scripts/train_bam.py \
                --config          "$BAM_B_CFG" \
                --init_encoder    "$MRL_BEST" \
                --checkpoint_dir  "$BAM_B_CKPT" \
                --freeze_encoder \
                || die "[$DS] BAM-B training failed"
        fi
    fi

    # ── STEP 6: find_bam_b ───────────────────────────────────────────────────
    if should_run find_bam_b; then
        log "[$DS] BSR EPOCH SELECTION — BAM-B"
        mkdir -p "$DS_RESULTS/bam_b_bsr"
        python3 scripts/find_best_epoch_bsr.py \
            --config         "$BAM_B_CFG" \
            --checkpoint_dir "$BAM_B_CKPT" \
            --output_dir     "$DS_RESULTS/bam_b_bsr/" \
            --alpha          "$BSR_ALPHA" \
            || die "[$DS] BSR selection (BAM-B) failed"
    fi

    # ── STEP 5b: train_bam_pq ────────────────────────────────────────────────
    if should_run train_bam_pq; then
        log "[$DS] TRAIN BAM-PQ (Bloom anchor + per-query MLP residual)"
        if [[ "$FORCE" == "1" ]] || [[ "$BLOOM_NLI_REFRESHED" == "1" ]]; then
            rm -rf "$BAM_PQ_CKPT"/epoch_* "$BAM_PQ_CKPT"/inbatch_best "$BAM_PQ_CKPT"/best_bsr "$BAM_PQ_CKPT"/final 2>/dev/null || true
        fi
        if [[ "$REUSE_TRAINED_MODELS" == "1" ]] && [[ "$FORCE" != "1" ]] && [[ "$BLOOM_NLI_REFRESHED" != "1" ]] \
            && { [[ -f "$BAM_PQ_BEST/checkpoint.pt" ]] || ls "$BAM_PQ_CKPT"/epoch_* &>/dev/null 2>&1; }; then
            echo "  BAM-PQ checkpoint exists — skipping (REUSE_TRAINED_MODELS=1)."
        else
            [[ -f "$MRL_BEST/checkpoint.pt" ]] || die "[$DS] MRL best not found — run find_mrl first"
            python3 scripts/train_bam.py \
                --config          "$BAM_PQ_CFG" \
                --init_encoder    "$MRL_BEST" \
                --checkpoint_dir  "$BAM_PQ_CKPT" \
                --freeze_encoder \
                || die "[$DS] BAM-PQ training failed"
        fi
    fi

    # ── STEP 6b: find_bam_pq ─────────────────────────────────────────────────
    if should_run find_bam_pq; then
        log "[$DS] BSR EPOCH SELECTION — BAM-PQ"
        mkdir -p "$DS_RESULTS/bam_pq_bsr"
        python3 scripts/find_best_epoch_bsr.py \
            --config         "$BAM_PQ_CFG" \
            --checkpoint_dir "$BAM_PQ_CKPT" \
            --output_dir     "$DS_RESULTS/bam_pq_bsr/" \
            --alpha          "$BSR_ALPHA" \
            || die "[$DS] BSR selection (BAM-PQ) failed"
    fi

    # ── STEP 7: eval ─────────────────────────────────────────────────────────
    if should_run eval; then
        log "[$DS] STANDARD EVALUATION — BAM-B vs MRL"
        [[ -f "$MRL_BEST/checkpoint.pt" ]]   || die "[$DS] MRL best not found"
        [[ -f "$BAM_B_BEST/checkpoint.pt" ]] || die "[$DS] BAM-B best_bsr not found"
        python3 scripts/eval_bam.py \
            --config        "$BAM_B_CFG" \
            --checkpoint    "$BAM_B_BEST" \
            --baseline      "$MRL_BEST" \
            --output_dir    "$DS_RESULTS/" \
            || die "[$DS] eval_bam.py failed"
        echo "  Results → $DS_RESULTS/results.json"
    fi

    # ── STEP 8: fair_cmp ─────────────────────────────────────────────────────
    if should_run fair_cmp; then
        log "[$DS] FAIR COMPARISON — BAM-B vs MRL at same per-Bloom dim budget"
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
        echo "  Results → $DS_RESULTS/fair_comparison/fair_comparison.json"
    fi

    # ── STEP 9: eff_curves ───────────────────────────────────────────────────
    if should_run eff_curves; then
        log "[$DS] EFFICIENCY CURVES — R@10 vs dims (paper Figure 2)"
        [[ -f "$MRL_BEST/checkpoint.pt" ]]   || die "[$DS] MRL best not found"
        [[ -f "$BAM_B_BEST/checkpoint.pt" ]] || die "[$DS] BAM-B best_bsr not found"
        mkdir -p "$DS_RESULTS/efficiency_curves"
        python3 scripts/eval_efficiency_curves.py \
            --config         "$BAM_B_CFG" \
            --bam_checkpoint "$BAM_B_BEST" \
            --mrl_checkpoint "$MRL_BEST" \
            --output_dir     "$DS_RESULTS/efficiency_curves/" \
            || die "[$DS] eval_efficiency_curves.py failed"
        echo "  Curves → $DS_RESULTS/efficiency_curves/"
    fi

    # ── STEP 10: eval_bam_pq ─────────────────────────────────────────────────
    if should_run eval_bam_pq; then
        log "[$DS] STANDARD EVALUATION — BAM-PQ vs MRL"
        if [[ ! -f "$BAM_PQ_BEST/checkpoint.pt" ]]; then
            echo "  BAM-PQ best_bsr not found at $BAM_PQ_BEST — skipping eval_bam_pq."
        else
            [[ -f "$MRL_BEST/checkpoint.pt" ]] || die "[$DS] MRL best not found"
            mkdir -p "$DS_RESULTS/bam_pq"
            python3 scripts/eval_bam.py \
                --config        "$BAM_PQ_CFG" \
                --checkpoint    "$BAM_PQ_BEST" \
                --baseline      "$MRL_BEST" \
                --output_dir    "$DS_RESULTS/bam_pq/" \
                || die "[$DS] eval_bam.py (BAM-PQ) failed"
            echo "  Results → $DS_RESULTS/bam_pq/results.json"
        fi
    fi

    # ── STEP 11: fair_cmp_pq ─────────────────────────────────────────────────
    if should_run fair_cmp_pq; then
        log "[$DS] FAIR COMPARISON — BAM-PQ vs MRL at same per-Bloom dim budget"
        if [[ ! -f "$BAM_PQ_BEST/checkpoint.pt" ]]; then
            echo "  BAM-PQ best_bsr not found at $BAM_PQ_BEST — skipping fair_cmp_pq."
        else
            [[ -f "$MRL_BEST/checkpoint.pt" ]] || die "[$DS] MRL best not found"
            mkdir -p "$DS_RESULTS/bam_pq/fair_comparison"
            python3 scripts/eval_fair_comparison.py \
                --config         "$BAM_PQ_CFG" \
                --bam_checkpoint "$BAM_PQ_BEST" \
                --mrl_checkpoint "$MRL_BEST" \
                --bam_results    "$DS_RESULTS/bam_pq/results.json" \
                --output_dir     "$DS_RESULTS/bam_pq/fair_comparison/" \
                || die "[$DS] eval_fair_comparison.py (BAM-PQ) failed"
            echo "  Results → $DS_RESULTS/bam_pq/fair_comparison/fair_comparison.json"
        fi
    fi

done  # end per-dataset loop

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
log "RESULTS SUMMARY"

python3 - <<PYEOF
import json, os, math

datasets = "$DATASETS".split()
results_root = "$RESULTS_ROOT"

print()
print("  Standard: BAM-B vs MRL at full dims")
print(f"  {'Dataset':14s} {'MRL R@10':>10s} {'BAM-B R@10':>12s} {'Dims':>8s} {'Δ':>8s}")
print("  " + "─" * 58)
for ds in datasets:
    path = os.path.join(results_root, ds, "results.json")
    if not os.path.exists(path):
        print(f"  {ds:14s}  (no results)")
        continue
    with open(path) as f:
        r = json.load(f)
    mrl_r10 = r.get("MRL Baseline", {}).get("recall@10", 0)
    bam_r10 = r.get("BAM v4 (Option B)", {}).get("recall@10", 0)
    dims    = r.get("BAM v4 (Option B)", {}).get("avg_active_dims", 0)
    delta   = bam_r10 - mrl_r10
    sign    = "+" if delta >= 0 else ""
    print(f"  {ds:14s} {mrl_r10:>10.4f} {bam_r10:>12.4f} {dims:>8.0f} {sign}{delta*100:>6.2f}%")

print()
print("  Standard: BAM-PQ vs MRL at full dims")
print(f"  {'Dataset':14s} {'MRL R@10':>10s} {'BAM-PQ R@10':>13s} {'Dims':>8s} {'Δ':>8s}")
print("  " + "─" * 60)
for ds in datasets:
    path = os.path.join(results_root, ds, "bam_pq", "results.json")
    if not os.path.exists(path):
        print(f"  {ds:14s}  (not run)")
        continue
    with open(path) as f:
        r = json.load(f)
    mrl_r10 = r.get("MRL Baseline", {}).get("recall@10", 0)
    pq_r10  = r.get("BAM v4 (Option B)", {}).get("recall@10",
              r.get("BAM-PQ", {}).get("recall@10", 0))
    dims    = r.get("BAM v4 (Option B)", {}).get("avg_active_dims",
              r.get("BAM-PQ", {}).get("avg_active_dims", 0))
    delta   = pq_r10 - mrl_r10
    sign    = "+" if delta >= 0 else ""
    print(f"  {ds:14s} {mrl_r10:>10.4f} {pq_r10:>13.4f} {dims:>8.0f} {sign}{delta*100:>6.2f}%")

print()
print("  Fair: BAM-B vs MRL truncated to same per-Bloom budget")
print(f"  {'Dataset':14s} {'Avg Δ':>10s} {'BAM wins':>10s}")
print("  " + "─" * 38)
for ds in datasets:
    fc_path = os.path.join(results_root, ds, "fair_comparison", "fair_comparison.json")
    if not os.path.exists(fc_path):
        print(f"  {ds:14s}  (not run)")
        continue
    with open(fc_path) as f:
        fc = json.load(f)
    avg_d = fc.get("avg_delta_bam_minus_mrl_trunc", math.nan)
    wins  = fc.get("bam_wins", 0)
    total = fc.get("total_levels", 6)
    sign  = "+" if avg_d >= 0 else ""
    print(f"  {ds:14s} {sign}{avg_d*100:>8.2f}%  {wins}/{total}")

print()
print("  Fair: BAM-PQ vs MRL truncated to same per-Bloom budget")
print(f"  {'Dataset':14s} {'Avg Δ':>10s} {'BAM wins':>10s}")
print("  " + "─" * 38)
for ds in datasets:
    fc_path = os.path.join(results_root, ds, "bam_pq", "fair_comparison", "fair_comparison.json")
    if not os.path.exists(fc_path):
        print(f"  {ds:14s}  (not run)")
        continue
    with open(fc_path) as f:
        fc = json.load(f)
    avg_d = fc.get("avg_delta_bam_minus_mrl_trunc", math.nan)
    wins  = fc.get("bam_wins", 0)
    total = fc.get("total_levels", 6)
    sign  = "+" if avg_d >= 0 else ""
    print(f"  {ds:14s} {sign}{avg_d*100:>8.2f}%  {wins}/{total}")

PYEOF

log "PIPELINE COMPLETE"
echo ""
echo "  Checkpoints : $CKPT_ROOT/{dataset}/{mrl,bam_b,bam_pq}/"
echo "  Results     : $RESULTS_ROOT/{dataset}/results.json           (BAM-B)"
echo "  Results PQ  : $RESULTS_ROOT/{dataset}/bam_pq/results.json   (BAM-PQ)"
echo "  Fair cmp    : $RESULTS_ROOT/{dataset}/fair_comparison/"
echo "  Fair cmp PQ : $RESULTS_ROOT/{dataset}/bam_pq/fair_comparison/"
echo "  Eff curves  : $RESULTS_ROOT/{dataset}/efficiency_curves/"
