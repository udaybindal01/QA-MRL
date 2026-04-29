#!/usr/bin/env bash
# =============================================================================
# BAM Council Pipeline — same as run_full_pipeline.sh but uses a trained
# 4-model Bloom council (DeBERTa + RoBERTa + BERT + LinearSVC, ~88% acc)
# instead of zero-shot NLI for annotation.
#
# Extra steps vs run_full_pipeline.sh:
#   0. train_classifier — trains the 4-model council on Kaggle Bloom data
#                         (skipped if /tmp/bloom-council/council_weights.json
#                          already exists; set FORCE_CLASSIFIER=1 to retrain)
#
# Annotate step changes:
#   - educational: SKIPPED — build_real_data.py already writes bloom_level
#                  via the same council (bloom_classifier.py); re-running
#                  would waste time without changing labels.
#   - BEIR datasets: uses data/annotate_with_council.py (council) instead of
#                    annotate_bloom_local.py (NLI).
#   - msmarco:       same council annotation on train/val/test splits.
#
# MS MARCO special handling:
#   The standard BEIR evaluation trains on MS MARCO (~500k queries) and
#   evaluates zero-shot on BEIR datasets. When msmarco is in DATASETS:
#     - build:    downloads MS MARCO, mines BM25 hard negatives (default 100k
#                 train queries; override with MSMARCO_MAX_TRAIN=200000)
#     - annotate: council labels all splits
#     - train:    MRL + BAM-B + BAM-PQ trained on MS MARCO
#     - eval:     zero-shot evaluation on scifact/nfcorpus/fiqa via
#                 eval_zero_shot.py (NOT in-domain MS MARCO eval)
#
# All other steps are identical to run_full_pipeline.sh.
#
# Usage:
#   chmod +x bam_council_pipeline.sh
#   ./bam_council_pipeline.sh                               # all datasets, all steps
#   ./bam_council_pipeline.sh --edu-only                    # educational corpus only
#   ./bam_council_pipeline.sh --msmarco-only                # MS MARCO → BEIR zero-shot
#   ./bam_council_pipeline.sh --from annotate               # skip build+classifier
#   ./bam_council_pipeline.sh --from train_bam_b            # skip to BAM-B training
#   ./bam_council_pipeline.sh --from train_classifier       # retrain council + rest
#   ./bam_council_pipeline.sh --datasets "scifact fiqa"     # subset of datasets
#   ./bam_council_pipeline.sh --datasets "msmarco scifact"  # msmarco + in-domain
#   ./bam_council_pipeline.sh --force                       # wipe ckpts before train
#   FORCE_CLASSIFIER=1 ./bam_council_pipeline.sh            # force retrain council
#   REUSE_BLOOM_CACHE=1 ./bam_council_pipeline.sh           # skip --overwrite on annotate
#   REUSE_TRAINED_MODELS=1 ./bam_council_pipeline.sh        # skip MRL/BAM training if ckpts exist
#   MSMARCO_MAX_TRAIN=200000 ./bam_council_pipeline.sh      # larger MS MARCO train set
#
# Requirements:
#   pip install transformers torch sentence-transformers faiss-gpu pyyaml scikit-learn
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATASETS="${DATASETS:-educational scifact nfcorpus fiqa}"
BEIR_DATA_ROOT="/tmp/data/beir"
MSMARCO_DATA_DIR="/tmp/data/msmarco"
MSMARCO_MAX_TRAIN="${MSMARCO_MAX_TRAIN:-100000}"
ZERO_SHOT_DATASETS="${ZERO_SHOT_DATASETS:-scifact nfcorpus fiqa}"
EDU_DATA_DIR="./data/real"
CKPT_ROOT="/tmp/multi-domain"
RESULTS_ROOT="./results/multi_domain"
BSR_ALPHA="0.5"
COUNCIL_DIR="/tmp/bloom-council"
COUNCIL_WEIGHTS="$COUNCIL_DIR/council_weights.json"
ANNOTATE_BATCH_SIZE="64"

FORCE="${FORCE_PIPELINE:-0}"
FORCE_CLASSIFIER="${FORCE_CLASSIFIER:-0}"
REUSE_BLOOM_CACHE="${REUSE_BLOOM_CACHE:-0}"
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
        --from)           FROM_STEP="$2";           shift 2 ;;
        --datasets)       DATASETS="$2";            shift 2 ;;
        --dataset)        SINGLE_DATASET="$2";      shift 2 ;;
        --edu-only)       DATASETS="educational";   shift ;;
        --msmarco-only)   DATASETS="msmarco";       shift ;;
        --force)          FORCE=1;                  shift ;;
        --force_classifier) FORCE_CLASSIFIER=1;     shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -n "$SINGLE_DATASET" ]] && DATASETS="$SINGLE_DATASET"

ALL_STEPS=(train_classifier build annotate train_mrl find_mrl train_bam_b find_bam_b train_bam_pq find_bam_pq eval fair_cmp eff_curves eval_bam_pq fair_cmp_pq)

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

# ── Config generator ─────────────────────────────────────────────────────────
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
if [[ ! -f "$EDU_DATA_DIR/train_curriculum.jsonl" ]]; then
    echo "  Educational data not found at $EDU_DATA_DIR — will be built in the 'build' step."
fi
mkdir -p "$BEIR_DATA_ROOT" "$CKPT_ROOT" "$RESULTS_ROOT" "$EDU_DATA_DIR"
echo "  Datasets          : $DATASETS"
echo "  CKPT root         : $CKPT_ROOT"
echo "  Results           : $RESULTS_ROOT"
echo "  Council dir       : $COUNCIL_DIR"
echo "  Bloom cache reuse : REUSE_BLOOM_CACHE=$REUSE_BLOOM_CACHE"
echo "  Train skip        : REUSE_TRAINED_MODELS=$REUSE_TRAINED_MODELS"
echo "  Force             : $FORCE"
echo "  Force classifier  : $FORCE_CLASSIFIER"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0: TRAIN CLASSIFIER (once, before per-dataset loop)
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_classifier; then
    log "TRAIN BLOOM COUNCIL (DeBERTa + RoBERTa + BERT + SVM)"
    if [[ "$FORCE_CLASSIFIER" == "1" ]]; then
        echo "  FORCE_CLASSIFIER=1 — removing existing council and retraining."
        rm -rf "$COUNCIL_DIR"
    fi
    if [[ -f "$COUNCIL_WEIGHTS" ]]; then
        echo "  Council already trained at $COUNCIL_WEIGHTS — skipping."
        echo "  (Set FORCE_CLASSIFIER=1 or --force_classifier to retrain.)"
    else
        mkdir -p "$COUNCIL_DIR"
        python3 data/train_bloom_council.py \
            --output_dir "$COUNCIL_DIR" \
            || die "train_bloom_council.py failed"
        [[ -f "$COUNCIL_WEIGHTS" ]] \
            || die "train_bloom_council.py completed but $COUNCIL_WEIGHTS not found"
        echo "  Council trained and saved to $COUNCIL_WEIGHTS"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# PER-DATASET LOOP
# ─────────────────────────────────────────────────────────────────────────────
for DS in $DATASETS; do
    BLOOM_COUNCIL_REFRESHED=0

    log "━━━━  DATASET: $DS  ━━━━"

    # ── Resolve data paths ───────────────────────────────────────────────────
    if [[ "$DS" == "educational" ]]; then
        TRAIN_PATH="$EDU_DATA_DIR/train_curriculum.jsonl"
        VAL_PATH="$EDU_DATA_DIR/val.jsonl"
        TEST_PATH="$EDU_DATA_DIR/test.jsonl"
        CORPUS_PATH="$EDU_DATA_DIR/corpus.jsonl"
    elif [[ "$DS" == "msmarco" ]]; then
        TRAIN_PATH="$MSMARCO_DATA_DIR/train.jsonl"
        VAL_PATH="$MSMARCO_DATA_DIR/val.jsonl"
        TEST_PATH="$MSMARCO_DATA_DIR/test.jsonl"
        CORPUS_PATH="$MSMARCO_DATA_DIR/corpus.jsonl"
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

    if [[ ! -f "$MRL_CFG" ]]; then
        make_config "$BASE_MRL_CONFIG"    "$MRL_CFG"    "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$CORPUS_PATH" "$MRL_CKPT/"
        make_config "$BASE_BAM_B_CONFIG"  "$BAM_B_CFG"  "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$CORPUS_PATH" "$BAM_B_CKPT/"
        make_config "$BASE_BAM_PQ_CONFIG" "$BAM_PQ_CFG" "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$CORPUS_PATH" "$BAM_PQ_CKPT/"
    fi

    # ── STEP 1: build ────────────────────────────────────────────────────────
    if should_run build; then
        if [[ "$DS" == "educational" ]]; then
            log "[$DS] BUILD — educational data (SciQ/ARC/OpenBookQA/QASC)"
            if [[ -f "$CORPUS_PATH" ]] && [[ -f "$TRAIN_PATH" ]]; then
                echo "  Already built at $EDU_DATA_DIR — skipping."
            else
                echo "  Running build_real_data.py → $EDU_DATA_DIR ..."
                python3 data/build_real_data.py \
                    --config configs/real_data.yaml \
                    --output_dir "$EDU_DATA_DIR" \
                    || die "[$DS] build_real_data.py failed"

                # curriculum_negatives re-mines hard negatives with BM25 ordering
                # output is train_curriculum.jsonl (used as TRAIN_PATH)
                echo "  Running curriculum_negatives.py ..."
                python3 data/curriculum_negatives.py \
                    --pairs  "$EDU_DATA_DIR/train.jsonl" \
                    --corpus "$CORPUS_PATH" \
                    --output "$EDU_DATA_DIR/train_curriculum.jsonl" \
                    --num_neg 7 \
                    || die "[$DS] curriculum_negatives.py failed"

                echo "  Educational data built at $EDU_DATA_DIR"
            fi
        elif [[ "$DS" == "msmarco" ]]; then
            log "[$DS] BUILD — MS MARCO (max_train=$MSMARCO_MAX_TRAIN, BM25 hard negatives)"
            if [[ -f "$CORPUS_PATH" ]] && [[ -f "$TRAIN_PATH" ]]; then
                echo "  Already built at $MSMARCO_DATA_DIR — skipping."
            else
                mkdir -p "$MSMARCO_DATA_DIR"
                python3 data/build_msmarco_data.py \
                    --output_dir "$MSMARCO_DATA_DIR" \
                    --max_train  "$MSMARCO_MAX_TRAIN" \
                    --num_neg    7 \
                    --skip_bloom_annotation \
                    || die "[$DS] build_msmarco_data.py failed"
            fi
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
    # Educational: SKIP — bloom_level already written by build_real_data.py via
    #   the council (bloom_classifier.py). Re-annotating wastes GPU time without
    #   changing labels.
    # BEIR: annotate train/val/test with the trained council instead of NLI.
    #   --no_overwrite keeps rows that already have a bloom_level (fast resume).
    #   Set REUSE_BLOOM_CACHE=1 to skip entirely if files look annotated.
    if should_run annotate; then
        if [[ "$DS" == "educational" ]]; then
            log "[$DS] ANNOTATE — SKIPPED (educational data already has council labels from build step)"
        else
            [[ -f "$COUNCIL_WEIGHTS" ]] \
                || die "[$DS] Council not trained — run train_classifier step first (or: python3 data/train_bloom_council.py)"

            # Resolve which directory holds the JSONL splits
            if [[ "$DS" == "msmarco" ]]; then
                ANNOTATE_BASE="$MSMARCO_DATA_DIR"
            else
                ANNOTATE_BASE="$BEIR_DATA_ROOT/$DS"
            fi

            ANNOTATE_JSONL=()
            for split in train val test; do
                p="$ANNOTATE_BASE/${split}.jsonl"
                [[ -f "$p" ]] && ANNOTATE_JSONL+=("$p")
            done
            [[ ${#ANNOTATE_JSONL[@]} -gt 0 ]] \
                || die "[$DS] No train/val/test.jsonl under $ANNOTATE_BASE — run build first"

            if [[ "$REUSE_BLOOM_CACHE" == "1" ]]; then
                log "[$DS] ANNOTATE — REUSE_BLOOM_CACHE=1, skipping (${#ANNOTATE_JSONL[@]} splits)"
            else
                log "[$DS] ANNOTATE — trained council (${#ANNOTATE_JSONL[@]} splits, $MSMARCO_MAX_TRAIN train queries)"
                OVERWRITE_FLAG="--overwrite"
                python3 data/annotate_with_council.py \
                    --input "${ANNOTATE_JSONL[@]}" \
                    $OVERWRITE_FLAG \
                    --batch_size "$ANNOTATE_BATCH_SIZE" \
                    || die "[$DS] annotate_with_council.py failed"
                BLOOM_COUNCIL_REFRESHED=1
                echo "  Annotation complete for $DS"
            fi
        fi
    fi

    # ── STEP 3: train_mrl ────────────────────────────────────────────────────
    if should_run train_mrl; then
        log "[$DS] TRAIN MRL BASELINE"
        if [[ "$FORCE" == "1" ]] || [[ "$BLOOM_COUNCIL_REFRESHED" == "1" ]]; then
            log "[$DS] Clearing MRL checkpoints (Bloom labels refreshed or --force)"
            rm -rf "$MRL_CKPT"/epoch_* "$MRL_CKPT"/inbatch_best "$MRL_CKPT"/best "$MRL_CKPT"/final 2>/dev/null || true
        fi
        if [[ "$REUSE_TRAINED_MODELS" == "1" ]] && [[ "$FORCE" != "1" ]] && [[ "$BLOOM_COUNCIL_REFRESHED" != "1" ]] \
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
        if [[ "$FORCE" != "1" ]] && [[ "$BLOOM_COUNCIL_REFRESHED" != "1" ]] && [[ -f "$MRL_BEST/checkpoint.pt" ]]; then
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
        if [[ "$FORCE" == "1" ]] || [[ "$BLOOM_COUNCIL_REFRESHED" == "1" ]]; then
            log "[$DS] Clearing BAM-B checkpoints (Bloom labels refreshed or --force)"
            rm -rf "$BAM_B_CKPT"/epoch_* "$BAM_B_CKPT"/inbatch_best "$BAM_B_CKPT"/best_bsr "$BAM_B_CKPT"/final 2>/dev/null || true
        fi
        if [[ "$REUSE_TRAINED_MODELS" == "1" ]] && [[ "$FORCE" != "1" ]] && [[ "$BLOOM_COUNCIL_REFRESHED" != "1" ]] \
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

    # ── STEP 7: train_bam_pq ─────────────────────────────────────────────────
    if should_run train_bam_pq; then
        log "[$DS] TRAIN BAM-PQ (Bloom anchor + per-query MLP residual)"
        if [[ "$FORCE" == "1" ]] || [[ "$BLOOM_COUNCIL_REFRESHED" == "1" ]]; then
            rm -rf "$BAM_PQ_CKPT"/epoch_* "$BAM_PQ_CKPT"/inbatch_best "$BAM_PQ_CKPT"/best_bsr "$BAM_PQ_CKPT"/final 2>/dev/null || true
        fi
        if [[ "$REUSE_TRAINED_MODELS" == "1" ]] && [[ "$FORCE" != "1" ]] && [[ "$BLOOM_COUNCIL_REFRESHED" != "1" ]] \
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

    # ── STEP 8: find_bam_pq ──────────────────────────────────────────────────
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

    # ── STEPS 9–13: eval  ────────────────────────────────────────────────────
    # MS MARCO: trained on MS MARCO → evaluate zero-shot on BEIR datasets.
    # All other datasets: standard in-domain eval.
    if [[ "$DS" == "msmarco" ]]; then

        if should_run eval || should_run eval_bam_pq; then
            log "[$DS] ZERO-SHOT BEIR EVAL — MRL + BAM-B + BAM-PQ on $ZERO_SHOT_DATASETS"
            [[ -f "$MRL_BEST/checkpoint.pt" ]]   || die "[$DS] MRL best not found"
            [[ -f "$BAM_B_BEST/checkpoint.pt" ]] || die "[$DS] BAM-B best_bsr not found"

            ZERO_SHOT_OUT="$DS_RESULTS/zero_shot"
            mkdir -p "$ZERO_SHOT_OUT"

            BAM_PQ_ARGS=""
            if [[ -f "$BAM_PQ_BEST/checkpoint.pt" ]]; then
                BAM_PQ_ARGS="--bam_pq_checkpoint $BAM_PQ_BEST --bam_pq_config $BAM_PQ_CFG"
                echo "  BAM-PQ checkpoint found — including in zero-shot eval."
            else
                echo "  BAM-PQ best_bsr not found — evaluating MRL + BAM-B only."
            fi

            python3 scripts/eval_zero_shot.py \
                --mrl_checkpoint   "$MRL_BEST"   \
                --mrl_config       "$MRL_CFG"    \
                --bam_b_checkpoint "$BAM_B_BEST" \
                --bam_b_config     "$BAM_B_CFG"  \
                $BAM_PQ_ARGS                     \
                --datasets         $ZERO_SHOT_DATASETS \
                --output_dir       "$ZERO_SHOT_OUT" \
                || die "[$DS] eval_zero_shot.py failed"
            echo "  Zero-shot results → $ZERO_SHOT_OUT/zero_shot_results.json"
        fi

        # skip fair_cmp / eff_curves / fair_cmp_pq for msmarco
        # (they compare at same dim budget — only meaningful in-domain)

    else

        # ── STEP 9: eval ─────────────────────────────────────────────────────
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

        # ── STEP 10: fair_cmp ────────────────────────────────────────────────
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

        # ── STEP 11: eff_curves ──────────────────────────────────────────────
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

        # ── STEP 12: eval_bam_pq ─────────────────────────────────────────────
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

        # ── STEP 13: fair_cmp_pq ─────────────────────────────────────────────
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

    fi  # end msmarco vs other datasets

done  # end per-dataset loop

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
log "RESULTS SUMMARY"

python3 - <<PYEOF
import json, os, math

datasets = "$DATASETS".split()
results_root = "$RESULTS_ROOT"
zero_shot_datasets = "$ZERO_SHOT_DATASETS".split()

# ── MS MARCO zero-shot summary ────────────────────────────────────────────────
if "msmarco" in datasets:
    zs_path = os.path.join(results_root, "msmarco", "zero_shot", "zero_shot_results.json")
    if os.path.exists(zs_path):
        with open(zs_path) as f:
            zs = json.load(f)
        print()
        print("  MS MARCO → BEIR Zero-Shot Transfer (NDCG@10 / R@10)")
        models_seen = []
        for ds_res in zs.values():
            for k in ds_res:
                if not k.startswith("_") and k not in models_seen:
                    models_seen.append(k)
        header = f"  {'Model':<16}" + "".join(f"  {ds[:12]:>14}" for ds in zero_shot_datasets)
        print(header)
        print("  " + "─" * len(header))
        for model_name in models_seen:
            row = f"  {model_name:<16}"
            for ds in zero_shot_datasets:
                m = zs.get(ds, {}).get(model_name, {})
                n10 = m.get("ndcg@10", float("nan"))
                r10 = m.get("recall@10", float("nan"))
                row += f"  {n10:.4f}/{r10:.4f}  "
            print(row)
    else:
        print()
        print("  MS MARCO zero-shot results not found (run eval step).")

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
echo "  Council         : $COUNCIL_WEIGHTS"
echo "  Checkpoints     : $CKPT_ROOT/{dataset}/{mrl,bam_b,bam_pq}/"
echo "  Results         : $RESULTS_ROOT/{dataset}/results.json            (BAM-B, in-domain)"
echo "  Results PQ      : $RESULTS_ROOT/{dataset}/bam_pq/results.json    (BAM-PQ, in-domain)"
echo "  Fair cmp        : $RESULTS_ROOT/{dataset}/fair_comparison/"
echo "  Fair cmp PQ     : $RESULTS_ROOT/{dataset}/bam_pq/fair_comparison/"
echo "  Eff curves      : $RESULTS_ROOT/{dataset}/efficiency_curves/"
echo "  MS MARCO→BEIR   : $RESULTS_ROOT/msmarco/zero_shot/zero_shot_results.json"
