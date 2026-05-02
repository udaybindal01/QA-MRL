#!/usr/bin/env bash
# =============================================================================
# BAM Council Pipeline — full end-to-end: build → annotate → train → eval
#
# Models trained per dataset:
#   MRL baseline    — intfloat/e5-large-v2
#   BAM-B           — intfloat/e5-large-v2
#   BAM-PQ ×6       — e5-large | BGE-large | Qwen-0.6B | Qwen-4B | LLM2Vec-7B | GritLM-7B
#
# Datasets:
#   educational     — SciQ / ARC / OpenBookQA / QASC  (small, curriculum negatives)
#   msmarco         — 50k train queries (Wu et al. protocol); eval in-domain + BEIR
#   scifact         — BEIR, train on its own qrels
#   nfcorpus        — BEIR
#   fiqa            — BEIR
#
# Usage:
#   ./bam_council_pipeline.sh                                 # all models, all datasets
#   ./bam_council_pipeline.sh --edu-only                      # educational only
#   ./bam_council_pipeline.sh --msmarco-only                  # MS MARCO → BEIR zero-shot
#   ./bam_council_pipeline.sh --from train_mrl                # skip build+annotate
#   ./bam_council_pipeline.sh --from train_bam_pq             # skip to BAM-PQ
#   ./bam_council_pipeline.sh --until find_bam_b              # stop after BAM-B (shared steps only)
#   ./bam_council_pipeline.sh --datasets "scifact fiqa"       # subset of datasets
#   ./bam_council_pipeline.sh --backbone "e5large qwen06b"    # subset of BAM-PQ backbones
#   ./bam_council_pipeline.sh --force                         # wipe checkpoints and retrain
#
# Env overrides:
#   BACKBONES_TO_RUN="e5large qwen06b"     # which BAM-PQ backbones to run
#   DATASETS="educational msmarco"         # datasets to process
#   MSMARCO_MAX_TRAIN=50000                # Wu et al. 50k default
#   FORCE_CLASSIFIER=1                     # retrain Bloom council
#   REUSE_BLOOM_CACHE=1                    # skip re-annotation
#   REUSE_TRAINED_MODELS=1                 # skip training if checkpoints exist
#
# Requirements:
#   pip install transformers torch sentence-transformers faiss-gpu pyyaml scikit-learn
#   pip install llm2vec      # for LLM2Vec-Mistral-7B backbone
#   pip install gritlm       # for GritLM-7B backbone
# =============================================================================

set -euo pipefail

# Reduce CUDA memory fragmentation (helps large models on 40/48 GB GPUs)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATASETS="${DATASETS:-educational msmarco scifact nfcorpus fiqa}"
BEIR_DATA_ROOT="${BEIR_DATA_ROOT:-/tmp/data/beir}"
MSMARCO_DATA_DIR="${MSMARCO_DATA_DIR:-/tmp/data/msmarco}"
MSMARCO_MAX_TRAIN="${MSMARCO_MAX_TRAIN:-50000}"
MSMARCO_EVAL_CORPUS_SIZE="${MSMARCO_EVAL_CORPUS_SIZE:-500000}"
ZERO_SHOT_DATASETS="${ZERO_SHOT_DATASETS:-scifact nfcorpus fiqa}"
EDU_DATA_DIR="${EDU_DATA_DIR:-./data/real}"
CKPT_ROOT="${CKPT_ROOT:-/tmp/multi-domain}"
RESULTS_ROOT="${RESULTS_ROOT:-./results/multi_domain}"
BSR_ALPHA="0.5"
COUNCIL_DIR="${COUNCIL_DIR:-/tmp/bloom-council}"
COUNCIL_WEIGHTS="$COUNCIL_DIR/council_weights.json"
ANNOTATE_BATCH_SIZE="64"
NUM_NEG=15          # hard negatives per query (was 7; 15 matches modern IR practice)

FORCE="${FORCE_PIPELINE:-0}"
FORCE_CLASSIFIER="${FORCE_CLASSIFIER:-0}"
REUSE_BLOOM_CACHE="${REUSE_BLOOM_CACHE:-0}"
REUSE_TRAINED_MODELS="${REUSE_TRAINED_MODELS:-0}"

# ── Backbone registry ─────────────────────────────────────────────────────────
# Format: "name:edu_base_config:msmarco_base_config"
#   edu_base_config    — used for educational + BEIR datasets (small corpus)
#   msmarco_base_config — used for MS MARCO (different batch/dtype/freeze settings)
#
# MRL and BAM-B always use e5-large.
# BAM-PQ loops over BACKBONES_TO_RUN.
BASE_MRL_CONFIG="configs/mrl_e5large.yaml"
BASE_BAM_B_CONFIG="configs/bam_optionb_e5large.yaml"

# MRL base configs per backbone (educational and MS MARCO variants)
# e5large MRL is the shared baseline also used for BAM-B warm-start.
# All other backbones get their own backbone-matched MRL baseline.
declare -A BACKBONE_MRL_EDU_CFG=(
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
declare -A BACKBONE_MRL_MSMARCO_CFG=(
    [e5large]="configs/mrl_e5large_msmarco.yaml"
    [bge]="configs/mrl_bge_large_msmarco.yaml"
    [qwen06b]="configs/mrl_qwen06b_msmarco.yaml"
    [qwen4b]="configs/mrl_qwen4b_msmarco.yaml"
    [qwen8b]="configs/mrl_qwen8b_msmarco.yaml"
    [llm2vec]="configs/mrl_llm2vec_mistral7b_msmarco.yaml"
    [llama8b]="configs/mrl_llm2vec_llama8b_msmarco.yaml"
    [gritlm]="configs/mrl_gritlm7b_msmarco.yaml"
    [llama1b]="configs/mrl_llama1b_msmarco.yaml"
    [llama3b]="configs/mrl_llama3b_msmarco.yaml"
    [arctic]="configs/mrl_arctic_msmarco.yaml"
    [roberta]="configs/mrl_roberta_msmarco.yaml"
    [phi3mini]="configs/mrl_phi3mini_msmarco.yaml"
)

declare -A BACKBONE_STANDARD_FT_EDU_CFG=(
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
declare -A BACKBONE_STANDARD_FT_MSMARCO_CFG=(
    [e5large]="configs/standard_ft_e5large_msmarco.yaml"
    [bge]="configs/standard_ft_bge_msmarco.yaml"
    [qwen06b]="configs/standard_ft_qwen06b_msmarco.yaml"
    [qwen4b]="configs/standard_ft_qwen4b_msmarco.yaml"
    [qwen8b]="configs/standard_ft_qwen8b_msmarco.yaml"
    [llm2vec]="configs/standard_ft_llm2vec_msmarco.yaml"
    [llama8b]="configs/standard_ft_llm2vec_llama8b_msmarco.yaml"
    [gritlm]="configs/standard_ft_gritlm_msmarco.yaml"
    [llama1b]="configs/standard_ft_llama1b_msmarco.yaml"
    [llama3b]="configs/standard_ft_llama3b_msmarco.yaml"
)

declare -A BACKBONE_EDU_CFG=(
    [e5large]="configs/bam_pq.yaml"
    [bge]="configs/bam_pq_bge_large.yaml"
    [qwen06b]="configs/bam_pq_qwen06b.yaml"
    [qwen4b]="configs/bam_pq_qwen4b.yaml"
    [qwen8b]="configs/bam_pq_qwen8b.yaml"
    [llm2vec]="configs/bam_pq_llm2vec_mistral7b.yaml"
    [llama8b]="configs/bam_pq_llm2vec_llama8b.yaml"
    [gritlm]="configs/bam_pq_gritlm7b.yaml"
    [llama1b]="configs/bam_pq_llama1b.yaml"
    [llama3b]="configs/bam_pq_llama3b.yaml"
    [arctic]="configs/bam_pq_arctic.yaml"
    [roberta]="configs/bam_pq_roberta.yaml"
    [phi3mini]="configs/bam_pq_phi3mini.yaml"
)
declare -A BACKBONE_MSMARCO_CFG=(
    [e5large]="configs/bam_pq_msmarco.yaml"
    [bge]="configs/bam_pq_bge_large_msmarco.yaml"
    [qwen06b]="configs/bam_pq_qwen06b_msmarco.yaml"
    [qwen4b]="configs/bam_pq_qwen4b_msmarco.yaml"
    [qwen8b]="configs/bam_pq_qwen8b_msmarco.yaml"
    [llm2vec]="configs/bam_pq_llm2vec_mistral7b_msmarco.yaml"
    [llama8b]="configs/bam_pq_llm2vec_llama8b_msmarco.yaml"
    [gritlm]="configs/bam_pq_gritlm7b_msmarco.yaml"
    [llama1b]="configs/bam_pq_llama1b_msmarco.yaml"
    [llama3b]="configs/bam_pq_llama3b_msmarco.yaml"
    [arctic]="configs/bam_pq_arctic_msmarco.yaml"
    [roberta]="configs/bam_pq_roberta_msmarco.yaml"
    [phi3mini]="configs/bam_pq_phi3mini_msmarco.yaml"
)
# All backbones warm-start BAM-PQ from their own backbone-matched MRL checkpoint.
# e5large MRL is also used for BAM-B (e5large only model).
BACKBONE_USE_MRL_INIT="e5large bge qwen06b qwen4b qwen8b llm2vec llama8b gritlm llama1b llama3b arctic roberta phi3mini"

# Which backbones to run for BAM-PQ (override with --backbone or BACKBONES_TO_RUN)
BACKBONES_TO_RUN="${BACKBONES_TO_RUN:-e5large bge qwen06b qwen4b qwen8b llm2vec llama8b gritlm llama1b llama3b arctic roberta phi3mini}"

# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────
FROM_STEP=""
UNTIL_STEP=""
SINGLE_DATASET=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)           FROM_STEP="$2";                   shift 2 ;;
        --until)          UNTIL_STEP="$2";                  shift 2 ;;
        --datasets)       DATASETS="$2";                    shift 2 ;;
        --dataset)        SINGLE_DATASET="$2";              shift 2 ;;
        --backbone)       BACKBONES_TO_RUN="$2";            shift 2 ;;
        --edu-only)       DATASETS="educational";           shift ;;
        --msmarco-only)   DATASETS="msmarco";               shift ;;
        --force)          FORCE=1;                          shift ;;
        --force_classifier) FORCE_CLASSIFIER=1;             shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -n "$SINGLE_DATASET" ]] && DATASETS="$SINGLE_DATASET"

ALL_STEPS=(
    train_classifier
    build
    annotate
    train_mrl find_mrl
    train_bam_b find_bam_b
    eval_pretrained
    train_standard_ft find_standard_ft eval_standard_ft
    train_mrl_bk find_mrl_bk
    train_bam_pq find_bam_pq
    eval fair_cmp eff_curves
    eval_bam_pq fair_cmp_pq
)

should_run() {
    local step="$1"
    # --from: skip steps before FROM_STEP
    if [[ -n "$FROM_STEP" ]]; then
        local found=0
        for s in "${ALL_STEPS[@]}"; do
            [[ "$s" == "$FROM_STEP" ]] && found=1
            [[ $found -eq 1 && "$s" == "$step" ]] && { break; } || true
        done
        [[ $found -eq 0 ]] && return 1
        # check step is at or after FROM_STEP
        local after=0
        for s in "${ALL_STEPS[@]}"; do
            [[ "$s" == "$FROM_STEP" ]] && after=1
            [[ $after -eq 1 && "$s" == "$step" ]] && break
            [[ $after -eq 0 && "$s" == "$step" ]] && return 1
        done
    fi
    # --until: skip steps after UNTIL_STEP
    if [[ -n "$UNTIL_STEP" ]]; then
        local past=0
        for s in "${ALL_STEPS[@]}"; do
            [[ $past -eq 1 && "$s" == "$step" ]] && return 1
            [[ "$s" == "$UNTIL_STEP" ]] && past=1
        done
    fi
    return 0
}

# Minimum free disk space (MB) required to load 7B models from HF cache
LARGE_MODEL_MIN_FREE_MB=5000
HF_CACHE_DIR="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"

# Returns 0 if backbone can be loaded (cached or enough disk), 1 to skip.
check_backbone_loadable() {
    local bk="$1"
    if [[ "$bk" != "llm2vec" && "$bk" != "llama8b" && "$bk" != "gritlm" && "$bk" != "qwen8b" ]]; then
        return 0   # small models: always fine
    fi
    # Resolve Python: prefer PYTHON_EXEC env var, then active venv, then PATH.
    # This avoids false negatives when the user's venv is active but PYTHON_EXEC isn't exported.
    local pyexec
    if [[ -n "${PYTHON_EXEC:-}" ]]; then
        pyexec="$PYTHON_EXEC"
    elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python3" ]]; then
        pyexec="$VIRTUAL_ENV/bin/python3"
    else
        pyexec="$(which python3 2>/dev/null || echo python3)"
    fi
    if [[ "$bk" == "llm2vec" || "$bk" == "llama8b" ]]; then
        import_err=$("$pyexec" -c "import llm2vec" 2>&1) || {
            echo "  SKIP [$bk]: llm2vec import failed in $pyexec."
            echo "    Error: $import_err"
            echo "    Run: pip install llm2vec peft accelerate"
            return 1
        }
    elif [[ "$bk" == "gritlm" ]]; then
        import_err=$("$pyexec" -c "import gritlm" 2>&1) || {
            echo "  SKIP [$bk]: gritlm import failed in $pyexec."
            echo "    Error: $import_err"
            echo "    Run: pip install gritlm"
            return 1
        }
    fi
    # Check disk space — use df with 1K blocks (POSIX-portable) then convert
    local free_mb
    free_mb=$(df -k "$HF_CACHE_DIR" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1024}')
    if [[ -z "$free_mb" ]]; then return 0; fi
    if (( free_mb < LARGE_MODEL_MIN_FREE_MB )); then
        echo "  SKIP [$bk]: only ${free_mb} MB free in $HF_CACHE_DIR"
        echo "    Need ≥ ${LARGE_MODEL_MIN_FREE_MB} MB to load 7B model weights."
        echo "    Fix: export HF_HOME=/path/with/more/space  then re-run."
        return 1
    fi
    return 0
}

log() {
    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  [$(date '+%H:%M:%S')]  $*"
    echo "══════════════════════════════════════════════════════"
}
die() { echo "ERROR: $*" >&2; exit 1; }

# ── make_config: patch data paths + checkpoint_dir into a base YAML ──────────
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
mkdir -p "$BEIR_DATA_ROOT" "$CKPT_ROOT" "$RESULTS_ROOT" "$EDU_DATA_DIR"
echo "  Datasets          : $DATASETS"
echo "  BAM-PQ backbones  : $BACKBONES_TO_RUN"
echo "  CKPT root         : $CKPT_ROOT"
echo "  Results           : $RESULTS_ROOT"
echo "  Hard negatives    : $NUM_NEG"
echo "  Bloom cache reuse : REUSE_BLOOM_CACHE=$REUSE_BLOOM_CACHE"
echo "  Train skip        : REUSE_TRAINED_MODELS=$REUSE_TRAINED_MODELS"
echo "  Force             : $FORCE"

# Validate backbone registry
for BK in $BACKBONES_TO_RUN; do
    [[ -n "${BACKBONE_EDU_CFG[$BK]+x}" ]] || die "Unknown backbone: $BK"
done

# Warn when BEIR datasets are requested but BEIR_DATA_ROOT looks empty
for DS in $DATASETS; do
    case "$DS" in educational|msmarco) continue ;; esac
    if [[ ! -f "$BEIR_DATA_ROOT/$DS/corpus.jsonl" ]]; then
        echo "WARNING: BEIR corpus not found at $BEIR_DATA_ROOT/$DS/corpus.jsonl"
        echo "  If your BEIR data is elsewhere, set: export BEIR_DATA_ROOT=/path/to/beir"
        echo "  Otherwise the build step will download it."
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0: TRAIN BLOOM COUNCIL (once, shared across all datasets)
# ─────────────────────────────────────────────────────────────────────────────
if should_run train_classifier; then
    log "TRAIN BLOOM COUNCIL (DeBERTa + RoBERTa + BERT + SVM)"
    if [[ "$FORCE_CLASSIFIER" == "1" ]]; then
        echo "  FORCE_CLASSIFIER=1 — removing existing council and retraining."
        rm -rf "$COUNCIL_DIR"
    fi
    if [[ -f "$COUNCIL_WEIGHTS" ]]; then
        echo "  Council already trained at $COUNCIL_WEIGHTS — skipping."
        echo "  (Set FORCE_CLASSIFIER=1 to retrain.)"
    else
        mkdir -p "$COUNCIL_DIR"
        python3 data/train_bloom_council.py \
            --output_dir "$COUNCIL_DIR" \
            || die "train_bloom_council.py failed"
        [[ -f "$COUNCIL_WEIGHTS" ]] \
            || die "train_bloom_council.py completed but $COUNCIL_WEIGHTS not found"
        echo "  Council trained → $COUNCIL_WEIGHTS"
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
        # Educational: train_curriculum.jsonl produced by curriculum_negatives.py
        TRAIN_PATH="$EDU_DATA_DIR/train_curriculum.jsonl"
        VAL_PATH="$EDU_DATA_DIR/val.jsonl"
        TEST_PATH="$EDU_DATA_DIR/test.jsonl"
        CORPUS_PATH="$EDU_DATA_DIR/corpus.jsonl"
        IS_MSMARCO=0
    elif [[ "$DS" == "msmarco" ]]; then
        # MS MARCO: train.jsonl has BM25 negatives from HF (is_selected=0)
        # No curriculum_negatives needed — OOM on 2.9M corpus
        TRAIN_PATH="$MSMARCO_DATA_DIR/train.jsonl"
        VAL_PATH="$MSMARCO_DATA_DIR/val.jsonl"
        TEST_PATH="$MSMARCO_DATA_DIR/test.jsonl"
        CORPUS_PATH="$MSMARCO_DATA_DIR/corpus.jsonl"
        IS_MSMARCO=1
    else
        # BEIR: train.jsonl produced by build_beir_training_data.py
        DS_DIR="$BEIR_DATA_ROOT/$DS"
        TRAIN_PATH="$DS_DIR/train.jsonl"
        VAL_PATH="$DS_DIR/val.jsonl"
        TEST_PATH="$DS_DIR/test.jsonl"
        CORPUS_PATH="$DS_DIR/corpus.jsonl"
        IS_MSMARCO=0
    fi

    # ── Checkpoint dirs ──────────────────────────────────────────────────────
    MRL_CKPT="$CKPT_ROOT/$DS/mrl"
    BAM_B_CKPT="$CKPT_ROOT/$DS/bam_b"
    DS_RESULTS="$RESULTS_ROOT/$DS"
    CFG_DIR="$DS_RESULTS/configs"

    mkdir -p "$MRL_CKPT" "$BAM_B_CKPT" "$DS_RESULTS" "$CFG_DIR"

    MRL_CFG="$CFG_DIR/mrl.yaml"
    BAM_B_CFG="$CFG_DIR/bam_b.yaml"
    MRL_BEST="$MRL_CKPT/best"
    BAM_B_BEST="$BAM_B_CKPT/best_bsr"

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: BUILD
    # ─────────────────────────────────────────────────────────────────────────
    if should_run build; then
        if [[ "$DS" == "educational" ]]; then
            log "[$DS] BUILD — educational data (SciQ / ARC / OpenBookQA / QASC)"
            if [[ -f "$CORPUS_PATH" ]] && [[ -f "$EDU_DATA_DIR/train.jsonl" ]]; then
                echo "  Raw data already built at $EDU_DATA_DIR — skipping download."
            else
                python3 data/build_real_data.py \
                    --config configs/real_data.yaml \
                    --output_dir "$EDU_DATA_DIR" \
                    || die "[$DS] build_real_data.py failed"
            fi

            # Curriculum negatives: BM25 hard negative mining.
            # Regenerate if missing OR if NUM_NEG changed since last run
            # (tracked in .curriculum_num_neg marker file).
            CURR_NEG_MARKER="$EDU_DATA_DIR/.curriculum_num_neg"
            LAST_NUM_NEG="$(cat "$CURR_NEG_MARKER" 2>/dev/null || echo 0)"
            if [[ -f "$TRAIN_PATH" ]] && [[ "$LAST_NUM_NEG" == "$NUM_NEG" ]]; then
                echo "  train_curriculum.jsonl exists with num_neg=$NUM_NEG — skipping."
            else
                if [[ -f "$TRAIN_PATH" ]] && [[ "$LAST_NUM_NEG" != "$NUM_NEG" ]]; then
                    echo "  num_neg changed ($LAST_NUM_NEG → $NUM_NEG) — rebuilding curriculum negatives."
                else
                    echo "  train_curriculum.jsonl missing — building curriculum negatives."
                fi
                echo "  Mining BM25 curriculum hard negatives (num_neg=$NUM_NEG)..."
                python3 data/curriculum_negatives.py \
                    --pairs  "$EDU_DATA_DIR/train.jsonl" \
                    --corpus "$CORPUS_PATH" \
                    --output "$TRAIN_PATH" \
                    --num_neg "$NUM_NEG" \
                    || die "[$DS] curriculum_negatives.py failed"
                echo "$NUM_NEG" > "$CURR_NEG_MARKER"
                echo "  Curriculum negatives written → $TRAIN_PATH"
            fi

        elif [[ "$DS" == "msmarco" ]]; then
            log "[$DS] BUILD — MS MARCO (max_train=$MSMARCO_MAX_TRAIN)"
            if [[ -f "$CORPUS_PATH" ]] && [[ -f "$TRAIN_PATH" ]]; then
                echo "  Already built at $MSMARCO_DATA_DIR — skipping."
            else
                mkdir -p "$MSMARCO_DATA_DIR"
                # bloom_level=1 placeholder; council annotates in the next step
                python3 data/build_msmarco_data.py \
                    --output_dir "$MSMARCO_DATA_DIR" \
                    --max_train  "$MSMARCO_MAX_TRAIN" \
                    --num_neg    "$NUM_NEG" \
                    --skip_bloom_annotation \
                    || die "[$DS] build_msmarco_data.py failed"
                echo "  MS MARCO built → $MSMARCO_DATA_DIR"
                echo "  NOTE: HF is_selected=0 passages ARE BM25 hard negatives."
                echo "        curriculum_negatives.py skipped (would OOM on 2.9M corpus)."
            fi

        else
            log "[$DS] BUILD — BEIR dataset: $DS"
            if [[ -f "$CORPUS_PATH" ]] && [[ -f "$TRAIN_PATH" ]]; then
                echo "  Already built at $DS_DIR — skipping."
            else
                python3 data/build_beir_training_data.py \
                    --datasets "$DS" \
                    --output_dir "$BEIR_DATA_ROOT" \
                    --num_neg "$NUM_NEG" \
                    || die "[$DS] build_beir_training_data.py failed"
                echo "  BEIR $DS built → $DS_DIR"
            fi
        fi

        # Generate per-dataset configs for MRL and BAM-B (e5-large only)
        make_config "$BASE_MRL_CONFIG"   "$MRL_CFG"   "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$CORPUS_PATH" "$MRL_CKPT/"
        make_config "$BASE_BAM_B_CONFIG" "$BAM_B_CFG" "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$CORPUS_PATH" "$BAM_B_CKPT/"

        # Generate per-dataset configs for each BAM-PQ backbone
        for BK in $BACKBONES_TO_RUN; do
            BK_CKPT="$CKPT_ROOT/$DS/bam_pq_$BK"
            BK_CFG="$CFG_DIR/bam_pq_${BK}.yaml"
            mkdir -p "$BK_CKPT"
            if [[ "$IS_MSMARCO" == "1" ]]; then
                BASE_BK_CFG="${BACKBONE_MSMARCO_CFG[$BK]}"
            else
                BASE_BK_CFG="${BACKBONE_EDU_CFG[$BK]}"
            fi
            [[ -f "$BASE_BK_CFG" ]] || die "Base config not found: $BASE_BK_CFG"
            make_config "$BASE_BK_CFG" "$BK_CFG" "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$CORPUS_PATH" "$BK_CKPT/"
        done
    fi

    # Always regenerate configs — ensures changes to base configs propagate
    make_config "$BASE_MRL_CONFIG"   "$MRL_CFG"   "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$CORPUS_PATH" "$MRL_CKPT/"
    make_config "$BASE_BAM_B_CONFIG" "$BAM_B_CFG" "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$CORPUS_PATH" "$BAM_B_CKPT/"
    for BK in $BACKBONES_TO_RUN; do
        BK_CFG="$CFG_DIR/bam_pq_${BK}.yaml"
        BK_CKPT="$CKPT_ROOT/$DS/bam_pq_$BK"
        mkdir -p "$BK_CKPT"
        if [[ "$IS_MSMARCO" == "1" ]]; then
            BASE_BK_CFG="${BACKBONE_MSMARCO_CFG[$BK]}"
        else
            BASE_BK_CFG="${BACKBONE_EDU_CFG[$BK]}"
        fi
        make_config "$BASE_BK_CFG" "$BK_CFG" "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$CORPUS_PATH" "$BK_CKPT/"
    done

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: ANNOTATE
    # ─────────────────────────────────────────────────────────────────────────
    if should_run annotate; then
        if [[ "$DS" == "educational" ]]; then
            log "[$DS] ANNOTATE — SKIPPED (build_real_data.py already writes council labels)"
        else
            [[ -f "$COUNCIL_WEIGHTS" ]] \
                || die "[$DS] Council not found — run train_classifier step first"

            if [[ "$IS_MSMARCO" == "1" ]]; then
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
                || die "[$DS] No splits found under $ANNOTATE_BASE — run build first"

            if [[ "$REUSE_BLOOM_CACHE" == "1" ]]; then
                log "[$DS] ANNOTATE — skipped (REUSE_BLOOM_CACHE=1)"
            else
                log "[$DS] ANNOTATE — council Bloom labelling (${#ANNOTATE_JSONL[@]} splits)"
                python3 data/annotate_with_council.py \
                    --input "${ANNOTATE_JSONL[@]}" \
                    --overwrite \
                    --batch_size "$ANNOTATE_BATCH_SIZE" \
                    || die "[$DS] annotate_with_council.py failed"
                BLOOM_COUNCIL_REFRESHED=1
                echo "  Annotation complete for $DS"
            fi
        fi
    fi

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: TRAIN MRL (e5-large baseline, once per dataset)
    # ─────────────────────────────────────────────────────────────────────────
    if should_run train_mrl; then
        log "[$DS] TRAIN MRL BASELINE (e5-large)"
        if [[ "$FORCE" == "1" ]] || [[ "$BLOOM_COUNCIL_REFRESHED" == "1" ]]; then
            rm -rf "$MRL_CKPT"/epoch_* "$MRL_CKPT"/inbatch_best "$MRL_CKPT"/best "$MRL_CKPT"/final 2>/dev/null || true
        fi
        if [[ "$REUSE_TRAINED_MODELS" == "1" ]] && [[ "$FORCE" != "1" ]] \
            && [[ "$BLOOM_COUNCIL_REFRESHED" != "1" ]] \
            && { [[ -f "$MRL_BEST/checkpoint.pt" ]] || ls "$MRL_CKPT"/epoch_* &>/dev/null 2>&1; }; then
            echo "  MRL checkpoint exists — skipping (REUSE_TRAINED_MODELS=1)."
        else
            python3 scripts/train_baseline_mrl.py \
                --config "$MRL_CFG" \
                --checkpoint_dir "$MRL_CKPT" \
                || die "[$DS] MRL training failed"
        fi
    fi

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: FIND BEST MRL EPOCH
    # ─────────────────────────────────────────────────────────────────────────
    if should_run find_mrl; then
        if [[ "$IS_MSMARCO" == "1" ]]; then
            log "[$DS] SELECT BEST MRL — using final checkpoint (2.9M corpus OOMs sweep)"
            if [[ -f "$MRL_BEST/checkpoint.pt" ]]; then
                echo "  MRL best already linked."
            elif [[ -f "$MRL_CKPT/final/checkpoint.pt" ]]; then
                ln -sfn "$MRL_CKPT/final" "$MRL_BEST"
                echo "  Linked $MRL_BEST → final"
            else
                die "[$DS] MRL final checkpoint not found at $MRL_CKPT/final"
            fi
        else
            log "[$DS] SELECT BEST MRL EPOCH (corpus-level retrieval)"
            if [[ "$FORCE" != "1" ]] && [[ "$BLOOM_COUNCIL_REFRESHED" != "1" ]] \
                && [[ -f "$MRL_BEST/checkpoint.pt" ]]; then
                echo "  MRL best already selected."
            else
                python3 scripts/find_best_epoch.py \
                    --config "$MRL_CFG" \
                    --checkpoint_dir "$MRL_CKPT" \
                    --model_type mrl \
                    || die "[$DS] find_best_epoch (MRL) failed"
            fi
        fi
    fi

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: TRAIN BAM-B (e5-large, once per dataset)
    # ─────────────────────────────────────────────────────────────────────────
    if should_run train_bam_b; then
        log "[$DS] TRAIN BAM-B (e5-large, reverse two-stage)"
        if [[ "$FORCE" == "1" ]] || [[ "$BLOOM_COUNCIL_REFRESHED" == "1" ]]; then
            rm -rf "$BAM_B_CKPT"/epoch_* "$BAM_B_CKPT"/inbatch_best "$BAM_B_CKPT"/best_bsr "$BAM_B_CKPT"/final 2>/dev/null || true
        fi
        if [[ "$REUSE_TRAINED_MODELS" == "1" ]] && [[ "$FORCE" != "1" ]] \
            && [[ "$BLOOM_COUNCIL_REFRESHED" != "1" ]] \
            && { [[ -f "$BAM_B_BEST/checkpoint.pt" ]] || ls "$BAM_B_CKPT"/epoch_* &>/dev/null 2>&1; }; then
            echo "  BAM-B checkpoint exists — skipping (REUSE_TRAINED_MODELS=1)."
        else
            [[ -f "$MRL_BEST/checkpoint.pt" ]] || die "[$DS] MRL best not found — run find_mrl first"
            python3 scripts/train_bam.py \
                --config         "$BAM_B_CFG" \
                --init_encoder   "$MRL_BEST" \
                --checkpoint_dir "$BAM_B_CKPT" \
                --freeze_encoder \
                || die "[$DS] BAM-B training failed"
        fi
    fi

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: FIND BEST BAM-B EPOCH
    # ─────────────────────────────────────────────────────────────────────────
    if should_run find_bam_b; then
        if [[ "$IS_MSMARCO" == "1" ]]; then
            log "[$DS] BSR SELECTION BAM-B — using final checkpoint"
            if [[ -f "$BAM_B_BEST/checkpoint.pt" ]]; then
                echo "  BAM-B best already linked."
            elif [[ -f "$BAM_B_CKPT/final/checkpoint.pt" ]]; then
                ln -sfn "$BAM_B_CKPT/final" "$BAM_B_BEST"
                echo "  Linked $BAM_B_BEST → final"
            else
                die "[$DS] BAM-B final checkpoint not found"
            fi
        else
            log "[$DS] BSR SELECTION — BAM-B"
            mkdir -p "$DS_RESULTS/bam_b_bsr"
            python3 scripts/find_best_epoch_bsr.py \
                --config         "$BAM_B_CFG" \
                --checkpoint_dir "$BAM_B_CKPT" \
                --output_dir     "$DS_RESULTS/bam_b_bsr/" \
                --alpha          "$BSR_ALPHA" \
                || die "[$DS] BSR selection (BAM-B) failed"
        fi
    fi

    # ─────────────────────────────────────────────────────────────────────────
    # STEPS 7–12: Per-backbone MRL + BAM-PQ
    # Each backbone gets its own MRL baseline (fair comparison), then BAM-PQ
    # warm-starts from that backbone's MRL checkpoint.
    # e5large MRL is already done in the shared steps — skip train_mrl_bk for it.
    # ─────────────────────────────────────────────────────────────────────────
    for BK in $BACKBONES_TO_RUN; do
        # Skip 7B models when there isn't enough disk space in the HF cache dir
        check_backbone_loadable "$BK" || continue

        BK_CKPT="$CKPT_ROOT/$DS/bam_pq_$BK"
        BK_CFG="$CFG_DIR/bam_pq_${BK}.yaml"
        BK_BEST="$BK_CKPT/best_bsr"
        BK_RESULTS="$DS_RESULTS/bam_pq_$BK"
        mkdir -p "$BK_CKPT" "$BK_RESULTS"

        # Per-backbone MRL checkpoint location
        BK_MRL_CKPT="$CKPT_ROOT/$DS/mrl_$BK"
        BK_MRL_BEST="$BK_MRL_CKPT/best"
        BK_MRL_CFG="$CFG_DIR/mrl_${BK}.yaml"
        mkdir -p "$BK_MRL_CKPT"

        # Resolve base MRL config for this backbone + dataset and generate it
        # early — needed by eval_pretrained (no training) and train_mrl_bk.
        if [[ "$IS_MSMARCO" == "1" ]]; then
            BASE_BK_MRL_CFG="${BACKBONE_MRL_MSMARCO_CFG[$BK]}"
        else
            BASE_BK_MRL_CFG="${BACKBONE_MRL_EDU_CFG[$BK]}"
        fi
        if [[ "$BK" == "e5large" ]]; then
            BK_MRL_CFG="$MRL_CFG"
            BK_MRL_BEST="$MRL_BEST"
        else
            make_config "$BASE_BK_MRL_CFG" "$BK_MRL_CFG" \
                "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$CORPUS_PATH" "$BK_MRL_CKPT/"
        fi

        # ── STEP 7: eval_pretrained ──────────────────────────────────────────
        # Zero-shot truncation baseline — no training, just eval pretrained
        # weights at each MRL dim. Uses the MRL config for model architecture only.
        if should_run eval_pretrained; then
            log "[$DS][$BK] EVAL PRETRAINED TRUNCATION BASELINE"
            PRETRAINED_OUT="$BK_RESULTS/pretrained_truncation"
            mkdir -p "$PRETRAINED_OUT"
            if [[ -f "$PRETRAINED_OUT/pretrained_truncation.json" ]] && [[ "$FORCE" != "1" ]]; then
                echo "  Pretrained baseline already evaluated — skipping."
            else
                python3 scripts/eval_pretrained_truncation.py \
                    --config      "$BK_MRL_CFG" \
                    --test_path   "$TEST_PATH" \
                    --corpus_path "$CORPUS_PATH" \
                    --output_dir  "$PRETRAINED_OUT" \
                    || echo "  WARNING: pretrained truncation eval failed for $BK (non-fatal)"
                echo "  Results → $PRETRAINED_OUT/pretrained_truncation.json"
            fi
        fi

        # ── STEPS 7b–7d: train/find/eval standard FT ────────────────────────
        # Standard FT baseline: train with mrl_dims=[full_dim] only.
        # Single-dim MRL loss = standard contrastive InfoNCE, no Matryoshka.
        # Isolates whether the multi-resolution structure (MRL) adds value.
        BK_SFT_CKPT="$CKPT_ROOT/$DS/standard_ft_$BK"
        BK_SFT_CFG="$CFG_DIR/standard_ft_${BK}.yaml"
        BK_SFT_BEST="$BK_SFT_CKPT/best"
        mkdir -p "$BK_SFT_CKPT"

        if [[ "$IS_MSMARCO" == "1" ]]; then
            BASE_BK_SFT_CFG="${BACKBONE_STANDARD_FT_MSMARCO_CFG[$BK]}"
        else
            BASE_BK_SFT_CFG="${BACKBONE_STANDARD_FT_EDU_CFG[$BK]}"
        fi
        make_config "$BASE_BK_SFT_CFG" "$BK_SFT_CFG" \
            "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$CORPUS_PATH" "$BK_SFT_CKPT/"

        if should_run train_standard_ft; then
            log "[$DS][$BK] TRAIN STANDARD FT BASELINE (contrastive only, no Matryoshka)"
            if [[ "$FORCE" == "1" ]] || [[ "$BLOOM_COUNCIL_REFRESHED" == "1" ]]; then
                rm -rf "$BK_SFT_CKPT"/epoch_* "$BK_SFT_CKPT"/best "$BK_SFT_CKPT"/final 2>/dev/null || true
            fi
            if [[ "$REUSE_TRAINED_MODELS" == "1" ]] && [[ "$FORCE" != "1" ]] \
                && { [[ -f "$BK_SFT_BEST/checkpoint.pt" ]] || ls "$BK_SFT_CKPT"/epoch_* &>/dev/null 2>&1; }; then
                echo "  Standard FT ($BK) checkpoint exists — skipping."
            else
                python3 scripts/train_baseline_mrl.py \
                    --config         "$BK_SFT_CFG" \
                    --checkpoint_dir "$BK_SFT_CKPT" \
                    || die "[$DS][$BK] Standard FT training failed"
            fi
        fi

        if should_run find_standard_ft; then
            if [[ "$IS_MSMARCO" == "1" || "$BK" == "qwen8b" || "$BK" == "llama8b" || "$BK" == "qwen4b" ]]; then
                log "[$DS][$BK] SELECT BEST STANDARD FT — using final checkpoint"
                if [[ -f "$BK_SFT_BEST/checkpoint.pt" ]]; then
                    echo "  Standard FT ($BK) best already linked."
                elif [[ -f "$BK_SFT_CKPT/final/checkpoint.pt" ]]; then
                    ln -sfn "$BK_SFT_CKPT/final" "$BK_SFT_BEST"
                    echo "  Linked $BK_SFT_BEST → final"
                else
                    die "[$DS][$BK] Standard FT final checkpoint not found"
                fi
            else
                log "[$DS][$BK] SELECT BEST STANDARD FT EPOCH"
                if [[ "$FORCE" != "1" ]] && [[ -f "$BK_SFT_BEST/checkpoint.pt" ]]; then
                    echo "  Standard FT ($BK) best already selected."
                else
                    python3 scripts/find_best_epoch.py \
                        --config         "$BK_SFT_CFG" \
                        --checkpoint_dir "$BK_SFT_CKPT" \
                        --model_type     mrl \
                        || die "[$DS][$BK] find_best_epoch (standard FT) failed"
                fi
            fi
        fi

        if should_run eval_standard_ft; then
            log "[$DS][$BK] EVAL STANDARD FT BASELINE"
            SFT_OUT="$BK_RESULTS/standard_ft"
            mkdir -p "$SFT_OUT"
            if [[ -f "$SFT_OUT/pretrained_truncation.json" ]] && [[ "$FORCE" != "1" ]]; then
                echo "  Standard FT baseline already evaluated — skipping."
            elif [[ ! -f "$BK_SFT_BEST/checkpoint.pt" ]]; then
                echo "  Standard FT ($BK) checkpoint not found — skipping eval."
            else
                python3 scripts/eval_pretrained_truncation.py \
                    --config      "$BK_SFT_CFG" \
                    --test_path   "$TEST_PATH" \
                    --corpus_path "$CORPUS_PATH" \
                    --checkpoint  "$BK_SFT_BEST" \
                    --output_dir  "$SFT_OUT" \
                    || echo "  WARNING: standard FT eval failed for $BK (non-fatal)"
                echo "  Results → $SFT_OUT/pretrained_truncation.json"
            fi
        fi

        # ── STEP 8: train_mrl_bk ─────────────────────────────────────────────
        # e5large MRL was already trained in the shared step (train_mrl).
        # For e5large, just reuse $MRL_BEST as BK_MRL_BEST.
        if [[ "$BK" == "e5large" ]]; then
            : # BK_MRL_BEST and BK_MRL_CFG already set above
        else
            if should_run train_mrl_bk; then
                log "[$DS][$BK] TRAIN MRL BASELINE"
                if [[ "$FORCE" == "1" ]] || [[ "$BLOOM_COUNCIL_REFRESHED" == "1" ]]; then
                    rm -rf "$BK_MRL_CKPT"/epoch_* "$BK_MRL_CKPT"/best "$BK_MRL_CKPT"/final 2>/dev/null || true
                fi
                if [[ "$REUSE_TRAINED_MODELS" == "1" ]] && [[ "$FORCE" != "1" ]] \
                    && { [[ -f "$BK_MRL_BEST/checkpoint.pt" ]] || ls "$BK_MRL_CKPT"/epoch_* &>/dev/null 2>&1; }; then
                    echo "  MRL ($BK) checkpoint exists — skipping."
                else
                    python3 scripts/train_baseline_mrl.py \
                        --config         "$BK_MRL_CFG" \
                        --checkpoint_dir "$BK_MRL_CKPT" \
                        || die "[$DS][$BK] MRL training failed"
                fi
            fi

            # ── STEP 8: find_mrl_bk ──────────────────────────────────────────
            if should_run find_mrl_bk; then
                if [[ "$IS_MSMARCO" == "1" || "$BK" == "qwen8b" || "$BK" == "llama8b" || "$BK" == "qwen4b" ]]; then
                    log "[$DS][$BK] SELECT BEST MRL — using final checkpoint"
                    if [[ -f "$BK_MRL_BEST/checkpoint.pt" ]]; then
                        echo "  MRL ($BK) best already linked."
                    elif [[ -f "$BK_MRL_CKPT/final/checkpoint.pt" ]]; then
                        ln -sfn "$BK_MRL_CKPT/final" "$BK_MRL_BEST"
                        echo "  Linked $BK_MRL_BEST → final"
                    else
                        die "[$DS][$BK] MRL final checkpoint not found"
                    fi
                else
                    log "[$DS][$BK] SELECT BEST MRL EPOCH"
                    if [[ "$FORCE" != "1" ]] && [[ -f "$BK_MRL_BEST/checkpoint.pt" ]]; then
                        echo "  MRL ($BK) best already selected."
                    else
                        python3 scripts/find_best_epoch.py \
                            --config         "$BK_MRL_CFG" \
                            --checkpoint_dir "$BK_MRL_CKPT" \
                            --model_type     mrl \
                            || die "[$DS][$BK] find_best_epoch (MRL) failed"
                    fi
                fi
            fi
        fi  # end non-e5large MRL

        # e5large BAM-PQ warms from BAM-B (encoder already mask-adapted, bloom_logit pre-trained).
        # Other backbones have no BAM-B — warm from their own backbone-matched MRL.
        if [[ "$BK" == "e5large" ]] && [[ -f "$BAM_B_BEST/checkpoint.pt" ]]; then
            INIT_ENCODER_ARG="--init_encoder $BAM_B_BEST"
        else
            INIT_ENCODER_ARG="--init_encoder $BK_MRL_BEST"
        fi

        # ── STEP 9: train_bam_pq ─────────────────────────────────────────────
        if should_run train_bam_pq; then
            log "[$DS][$BK] TRAIN BAM-PQ"
            if [[ "$FORCE" == "1" ]] || [[ "$BLOOM_COUNCIL_REFRESHED" == "1" ]]; then
                rm -rf "$BK_CKPT"/epoch_* "$BK_CKPT"/inbatch_best "$BK_CKPT"/best_bsr "$BK_CKPT"/final 2>/dev/null || true
            fi
            if [[ "$REUSE_TRAINED_MODELS" == "1" ]] && [[ "$FORCE" != "1" ]] \
                && [[ "$BLOOM_COUNCIL_REFRESHED" != "1" ]] \
                && { [[ -f "$BK_BEST/checkpoint.pt" ]] || ls "$BK_CKPT"/epoch_* &>/dev/null 2>&1; }; then
                echo "  BAM-PQ ($BK) checkpoint exists — skipping."
            else
                if [[ "$BK" == "e5large" ]] && [[ -f "$BAM_B_BEST/checkpoint.pt" ]]; then
                    : # warm from BAM-B — no MRL prereq needed
                else
                    [[ -f "$BK_MRL_BEST/checkpoint.pt" ]] \
                        || die "[$DS][$BK] MRL best not found — run find_mrl_bk first"
                fi
                python3 scripts/train_bam.py \
                    --config         "$BK_CFG" \
                    --checkpoint_dir "$BK_CKPT" \
                    --freeze_encoder \
                    $INIT_ENCODER_ARG \
                    || die "[$DS][$BK] BAM-PQ training failed"
            fi
        fi

        # ── STEP 8: find_bam_pq ──────────────────────────────────────────────
        if should_run find_bam_pq; then
            if [[ "$IS_MSMARCO" == "1" || "$BK" == "qwen8b" || "$BK" == "llama8b" || "$BK" == "qwen4b" ]]; then
                log "[$DS][$BK] BSR SELECTION BAM-PQ — using final checkpoint"
                if [[ -f "$BK_BEST/checkpoint.pt" ]]; then
                    echo "  BAM-PQ ($BK) best already linked."
                elif [[ -f "$BK_CKPT/final/checkpoint.pt" ]]; then
                    ln -sfn "$BK_CKPT/final" "$BK_BEST"
                    echo "  Linked $BK_BEST → final"
                else
                    echo "  BAM-PQ ($BK) final not found — train step may not have run."
                fi
            else
                log "[$DS][$BK] BSR SELECTION — BAM-PQ"
                python3 scripts/find_best_epoch_bsr.py \
                    --config         "$BK_CFG" \
                    --checkpoint_dir "$BK_CKPT" \
                    --output_dir     "$BK_RESULTS/bsr/" \
                    --alpha          "$BSR_ALPHA" \
                    || die "[$DS][$BK] BSR selection (BAM-PQ) failed"
            fi
        fi

    done  # end backbone loop (train + find)

    # ─────────────────────────────────────────────────────────────────────────
    # STEPS 9–13: EVALUATION
    # MS MARCO: in-domain + zero-shot BEIR
    # Others:   standard in-domain eval
    # ─────────────────────────────────────────────────────────────────────────
    if [[ "$IS_MSMARCO" == "1" ]]; then

        if should_run eval || should_run eval_bam_pq; then
            if [[ ! -f "$MRL_BEST/checkpoint.pt" ]] || [[ ! -f "$BAM_B_BEST/checkpoint.pt" ]]; then
                echo "  SKIP [$DS] eval: shared checkpoints not found — run pipeline_shared.sh first."
            else

            # Build --bam_pq_checkpoint / --bam_pq_config args for eval_zero_shot.py
            # Use e5large BAM-PQ as the "primary" PQ model for the combined script;
            # other backbones are evaluated separately below.
            E5_PQ_BEST="$CKPT_ROOT/$DS/bam_pq_e5large/best_bsr"
            E5_PQ_CFG="$CFG_DIR/bam_pq_e5large.yaml"
            BAM_PQ_ARGS=""
            if [[ -f "$E5_PQ_BEST/checkpoint.pt" ]] && [[ -f "$E5_PQ_CFG" ]]; then
                BAM_PQ_ARGS="--bam_pq_checkpoint $E5_PQ_BEST --bam_pq_config $E5_PQ_CFG"
            fi

            # ── In-domain MS MARCO eval ───────────────────────────────────────
            log "[$DS] IN-DOMAIN EVAL — corpus capped at $MSMARCO_EVAL_CORPUS_SIZE"
            INDOMAIN_OUT="$DS_RESULTS/indomain"
            mkdir -p "$INDOMAIN_OUT"
            python3 scripts/eval_zero_shot.py \
                --mrl_checkpoint   "$MRL_BEST"   \
                --mrl_config       "$MRL_CFG"    \
                --bam_b_checkpoint "$BAM_B_BEST" \
                --bam_b_config     "$BAM_B_CFG"  \
                $BAM_PQ_ARGS                     \
                --datasets         msmarco        \
                --output_dir       "$INDOMAIN_OUT" \
                --max_corpus_size  "$MSMARCO_EVAL_CORPUS_SIZE" \
                || die "[$DS] in-domain eval failed"

            # ── Eval additional BAM-PQ backbones (in-domain) ─────────────────
            for BK in $BACKBONES_TO_RUN; do
                [[ "$BK" == "e5large" ]] && continue   # already included above
                BK_BEST="$CKPT_ROOT/$DS/bam_pq_$BK/best_bsr"
                BK_CFG_F="$CFG_DIR/bam_pq_${BK}.yaml"
                [[ -f "$BK_BEST/checkpoint.pt" ]] || { echo "  [$BK] no checkpoint — skipping in-domain eval."; continue; }
                INDOMAIN_BK="$DS_RESULTS/indomain_$BK"
                mkdir -p "$INDOMAIN_BK"
                python3 scripts/eval_zero_shot.py \
                    --mrl_checkpoint   "$MRL_BEST"   \
                    --mrl_config       "$MRL_CFG"    \
                    --bam_b_checkpoint "$BAM_B_BEST" \
                    --bam_b_config     "$BAM_B_CFG"  \
                    --bam_pq_checkpoint "$BK_BEST"   \
                    --bam_pq_config    "$BK_CFG_F"   \
                    --datasets         msmarco        \
                    --output_dir       "$INDOMAIN_BK" \
                    --max_corpus_size  "$MSMARCO_EVAL_CORPUS_SIZE" \
                    || echo "  WARNING: in-domain eval failed for backbone $BK"
                echo "  In-domain [$BK] → $INDOMAIN_BK/zero_shot_results.json"
            done

            # ── Zero-shot BEIR eval ───────────────────────────────────────────
            log "[$DS] ZERO-SHOT BEIR EVAL — $ZERO_SHOT_DATASETS"
            ZERO_SHOT_OUT="$DS_RESULTS/zero_shot"
            mkdir -p "$ZERO_SHOT_OUT"
            python3 scripts/eval_zero_shot.py \
                --mrl_checkpoint   "$MRL_BEST"   \
                --mrl_config       "$MRL_CFG"    \
                --bam_b_checkpoint "$BAM_B_BEST" \
                --bam_b_config     "$BAM_B_CFG"  \
                $BAM_PQ_ARGS                     \
                --datasets         $ZERO_SHOT_DATASETS \
                --output_dir       "$ZERO_SHOT_OUT" \
                || die "[$DS] zero-shot BEIR eval failed"

            # ── Zero-shot eval for additional backbones ───────────────────────
            for BK in $BACKBONES_TO_RUN; do
                [[ "$BK" == "e5large" ]] && continue
                BK_BEST="$CKPT_ROOT/$DS/bam_pq_$BK/best_bsr"
                BK_CFG_F="$CFG_DIR/bam_pq_${BK}.yaml"
                [[ -f "$BK_BEST/checkpoint.pt" ]] || { echo "  [$BK] no checkpoint — skipping zero-shot eval."; continue; }
                ZERO_SHOT_BK="$DS_RESULTS/zero_shot_$BK"
                mkdir -p "$ZERO_SHOT_BK"
                python3 scripts/eval_zero_shot.py \
                    --mrl_checkpoint   "$MRL_BEST"   \
                    --mrl_config       "$MRL_CFG"    \
                    --bam_b_checkpoint "$BAM_B_BEST" \
                    --bam_b_config     "$BAM_B_CFG"  \
                    --bam_pq_checkpoint "$BK_BEST"   \
                    --bam_pq_config    "$BK_CFG_F"   \
                    --datasets         $ZERO_SHOT_DATASETS \
                    --output_dir       "$ZERO_SHOT_BK" \
                    || echo "  WARNING: zero-shot eval failed for backbone $BK"
                echo "  Zero-shot [$BK] → $ZERO_SHOT_BK/zero_shot_results.json"
            done
            fi  # end shared-checkpoint guard
        fi

    else  # ── Non-MS MARCO datasets ──────────────────────────────────────────

        # STEP 9: eval BAM-B
        if should_run eval; then
            log "[$DS] EVAL — BAM-B vs MRL"
            if [[ ! -f "$MRL_BEST/checkpoint.pt" ]]; then
                echo "  SKIP: MRL best not found at $MRL_BEST — run pipeline_shared.sh first."
            elif [[ ! -f "$BAM_B_BEST/checkpoint.pt" ]]; then
                echo "  SKIP: BAM-B best not found at $BAM_B_BEST — run pipeline_shared.sh first."
            else
            python3 scripts/eval_bam.py \
                --config     "$BAM_B_CFG" \
                --checkpoint "$BAM_B_BEST" \
                --baseline   "$MRL_BEST" \
                --output_dir "$DS_RESULTS/" \
                || die "[$DS] eval_bam.py (BAM-B) failed"
            echo "  Results → $DS_RESULTS/results.json"
            fi
        fi

        # STEP 10: fair comparison BAM-B
        if should_run fair_cmp; then
            log "[$DS] FAIR COMPARISON — BAM-B vs MRL at same per-Bloom budget"
            if [[ ! -f "$MRL_BEST/checkpoint.pt" ]] || [[ ! -f "$BAM_B_BEST/checkpoint.pt" ]]; then
                echo "  SKIP: shared checkpoints not found — run pipeline_shared.sh first."
            else
            mkdir -p "$DS_RESULTS/fair_comparison"
            python3 scripts/eval_fair_comparison.py \
                --config         "$BAM_B_CFG" \
                --bam_checkpoint "$BAM_B_BEST" \
                --mrl_checkpoint "$MRL_BEST" \
                --bam_results    "$DS_RESULTS/results.json" \
                --output_dir     "$DS_RESULTS/fair_comparison/" \
                || die "[$DS] eval_fair_comparison.py (BAM-B) failed"
            fi
        fi

        # STEP 11: efficiency curves
        if should_run eff_curves; then
            log "[$DS] EFFICIENCY CURVES — R@10 vs dims (paper Figure 2)"
            mkdir -p "$DS_RESULTS/efficiency_curves"
            python3 scripts/eval_efficiency_curves.py \
                --config         "$BAM_B_CFG" \
                --bam_checkpoint "$BAM_B_BEST" \
                --mrl_checkpoint "$MRL_BEST" \
                --output_dir     "$DS_RESULTS/efficiency_curves/" \
                || die "[$DS] eval_efficiency_curves.py failed"
        fi

        # STEPS 12–13: BAM-PQ eval per backbone
        # Each backbone is evaluated against its own backbone-matched MRL baseline.
        for BK in $BACKBONES_TO_RUN; do
            BK_BEST="$CKPT_ROOT/$DS/bam_pq_$BK/best_bsr"
            BK_CFG_F="$CFG_DIR/bam_pq_${BK}.yaml"
            BK_RESULTS="$DS_RESULTS/bam_pq_$BK"
            mkdir -p "$BK_RESULTS"

            # Resolve this backbone's MRL baseline (e5large reuses shared MRL_BEST)
            if [[ "$BK" == "e5large" ]]; then
                BK_MRL_BASELINE="$MRL_BEST"
            else
                BK_MRL_BASELINE="$CKPT_ROOT/$DS/mrl_$BK/best"
            fi

            if should_run eval_bam_pq; then
                log "[$DS][$BK] EVAL — BAM-PQ vs MRL ($BK baseline)"
                if [[ ! -f "$BK_BEST/checkpoint.pt" ]]; then
                    echo "  [$BK] best_bsr not found — skipping eval."
                else
                    [[ -f "$BK_MRL_BASELINE/checkpoint.pt" ]] \
                        || die "[$DS][$BK] MRL baseline not found at $BK_MRL_BASELINE"
                    python3 scripts/eval_bam.py \
                        --config     "$BK_CFG_F" \
                        --checkpoint "$BK_BEST" \
                        --baseline   "$BK_MRL_BASELINE" \
                        --output_dir "$BK_RESULTS/" \
                        || die "[$DS][$BK] eval_bam.py (BAM-PQ) failed"
                    echo "  Results → $BK_RESULTS/results.json"
                fi
            fi

            if should_run fair_cmp_pq; then
                log "[$DS][$BK] FAIR COMPARISON — BAM-PQ vs MRL ($BK) at same per-Bloom budget"
                if [[ ! -f "$BK_BEST/checkpoint.pt" ]]; then
                    echo "  [$BK] best_bsr not found — skipping fair comparison."
                else
                    [[ -f "$BK_MRL_BASELINE/checkpoint.pt" ]] \
                        || die "[$DS][$BK] MRL baseline not found at $BK_MRL_BASELINE"
                    mkdir -p "$BK_RESULTS/fair_comparison"
                    python3 scripts/eval_fair_comparison.py \
                        --config         "$BK_CFG_F" \
                        --bam_checkpoint "$BK_BEST" \
                        --mrl_checkpoint "$BK_MRL_BASELINE" \
                        --bam_results    "$BK_RESULTS/results.json" \
                        --output_dir     "$BK_RESULTS/fair_comparison/" \
                        || die "[$DS][$BK] eval_fair_comparison.py (BAM-PQ) failed"
                fi
            fi
        done

    fi  # end msmarco vs other

done  # end per-dataset loop

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
log "RESULTS SUMMARY"

python3 - <<PYEOF
import json, os, math

datasets       = "$DATASETS".split()
backbones      = "$BACKBONES_TO_RUN".split()
results_root   = "$RESULTS_ROOT"
zero_shot_dss  = "$ZERO_SHOT_DATASETS".split()

def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else "  n/a "

# ── MS MARCO summary ──────────────────────────────────────────────────────────
if "msmarco" in datasets:
    print("\n  MS MARCO In-Domain (NDCG@10  /  R@10)")
    print(f"  {'Backbone':<16}  {'msmarco':>16}")
    print("  " + "─" * 36)
    for bk in ["e5large"] + [b for b in backbones if b != "e5large"]:
        if bk == "e5large":
            path = os.path.join(results_root, "msmarco", "indomain", "zero_shot_results.json")
        else:
            path = os.path.join(results_root, "msmarco", f"indomain_{bk}", "zero_shot_results.json")
        r = _load(path)
        if r is None:
            print(f"  {'BAM-PQ-'+bk:<16}  {'(not run)':>16}")
            continue
        ms = r.get("msmarco", {})
        pq = ms.get("bam_pq", ms.get("BAM-PQ", {}))
        n10 = fmt(pq.get("ndcg@10", float("nan")))
        r10 = fmt(pq.get("recall@10", float("nan")))
        print(f"  {'BAM-PQ-'+bk:<16}  {n10} / {r10}")

    print(f"\n  MS MARCO → BEIR Zero-Shot (NDCG@10)")
    header = f"  {'Backbone':<16}" + "".join(f"  {d[:10]:>12}" for d in zero_shot_dss)
    print(header)
    print("  " + "─" * len(header))
    for bk in ["e5large"] + [b for b in backbones if b != "e5large"]:
        if bk == "e5large":
            path = os.path.join(results_root, "msmarco", "zero_shot", "zero_shot_results.json")
        else:
            path = os.path.join(results_root, "msmarco", f"zero_shot_{bk}", "zero_shot_results.json")
        r = _load(path)
        row = f"  {'BAM-PQ-'+bk:<16}"
        for d in zero_shot_dss:
            if r is None:
                row += f"  {'(not run)':>12}"
            else:
                pq = r.get(d, {}).get("bam_pq", r.get(d, {}).get("BAM-PQ", {}))
                row += f"  {fmt(pq.get('ndcg@10', float('nan'))):>12}"
        print(row)

# ── In-domain summary (non-MS MARCO) ─────────────────────────────────────────
non_ms = [d for d in datasets if d != "msmarco"]
if non_ms:
    print(f"\n  Standard: BAM-B vs MRL (R@10)")
    print(f"  {'Dataset':<14}  {'MRL':>8}  {'BAM-B':>8}  {'Dims':>6}  {'Δ':>7}")
    print("  " + "─" * 52)
    for ds in non_ms:
        r = _load(os.path.join(results_root, ds, "results.json"))
        if r is None:
            print(f"  {ds:<14}  (not run)")
            continue
        mrl = r.get("MRL Baseline", {}).get("recall@10", float("nan"))
        bam = r.get("BAM v4 (Option B)", {}).get("recall@10", float("nan"))
        dim = r.get("BAM v4 (Option B)", {}).get("avg_active_dims", float("nan"))
        d   = bam - mrl if not (math.isnan(bam) or math.isnan(mrl)) else float("nan")
        print(f"  {ds:<14}  {fmt(mrl):>8}  {fmt(bam):>8}  {dim:>6.0f}  {'+' if d>=0 else ''}{d*100:>5.2f}%")

    print(f"\n  BAM-PQ backbones vs MRL (R@10)")
    print(f"  {'Dataset':<14}  {'Backbone':<12}  {'MRL':>8}  {'BAM-PQ':>8}  {'Dims':>6}  {'Δ':>7}")
    print("  " + "─" * 66)
    for ds in non_ms:
        for bk in backbones:
            r = _load(os.path.join(results_root, ds, f"bam_pq_{bk}", "results.json"))
            if r is None:
                print(f"  {ds:<14}  {bk:<12}  (not run)")
                continue
            mrl = r.get("MRL Baseline", {}).get("recall@10", float("nan"))
            pq  = r.get("BAM v4 (Option B)", r.get("BAM-PQ", {})).get("recall@10", float("nan"))
            dim = r.get("BAM v4 (Option B)", r.get("BAM-PQ", {})).get("avg_active_dims", float("nan"))
            d   = pq - mrl if not (math.isnan(pq) or math.isnan(mrl)) else float("nan")
            print(f"  {ds:<14}  {bk:<12}  {fmt(mrl):>8}  {fmt(pq):>8}  {dim:>6.0f}  {'+' if d>=0 else ''}{d*100:>5.2f}%")

    print(f"\n  Fair comparison: BAM-PQ vs MRL truncated to same per-Bloom budget")
    print(f"  {'Dataset':<14}  {'Backbone':<12}  {'Avg Δ':>8}  {'BAM wins':>10}")
    print("  " + "─" * 50)
    for ds in non_ms:
        for bk in backbones:
            fc = _load(os.path.join(results_root, ds, f"bam_pq_{bk}", "fair_comparison", "fair_comparison.json"))
            if fc is None:
                print(f"  {ds:<14}  {bk:<12}  (not run)")
                continue
            avg_d = fc.get("avg_delta_bam_minus_mrl_trunc", float("nan"))
            wins  = fc.get("bam_wins", 0)
            total = fc.get("total_levels", 6)
            print(f"  {ds:<14}  {bk:<12}  {'+' if avg_d>=0 else ''}{avg_d*100:>6.2f}%  {wins}/{total}")

PYEOF

log "PIPELINE COMPLETE"
echo ""
echo "  Council         : $COUNCIL_WEIGHTS"
echo "  Checkpoints     : $CKPT_ROOT/{dataset}/{mrl,bam_b,bam_pq_{backbone}}/"
echo "  Results         : $RESULTS_ROOT/{dataset}/bam_pq_{backbone}/results.json"
echo "  MS MARCO BEIR   : $RESULTS_ROOT/msmarco/zero_shot_{backbone}/zero_shot_results.json"
echo ""
echo "  To run a single backbone:    ./bam_council_pipeline.sh --backbone qwen06b"
echo "  To resume from training:     ./bam_council_pipeline.sh --from train_bam_pq"
echo "  To run MS MARCO only:        ./bam_council_pipeline.sh --msmarco-only"
