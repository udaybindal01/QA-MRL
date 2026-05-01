#!/usr/bin/env bash
# =============================================================================
# pipeline_baselines.sh — All baselines for all 6 BAM-PQ backbones
#
# Runs three baseline tiers per backbone per dataset:
#   1. Pretrained truncation  — zero-shot eval at each MRL dim (no training)
#   2. Standard FT            — contrastive InfoNCE only, mrl_dims=[full_dim]
#   3. MRL                    — full multi-resolution Matryoshka training
#      + MRL truncation eval  — same truncation sweep on the trained MRL model
#
# Together these give the ablation ladder:
#   Pretrained → Standard FT → MRL → BAM-PQ
#
# Run AFTER pipeline_shared.sh (data must already be built and annotated).
#
# Usage:
#   ./pipeline_baselines.sh                            # all 6 backbones, all datasets
#   ./pipeline_baselines.sh --datasets "educational"   # single dataset
#   ./pipeline_baselines.sh --backbone "bge qwen06b"   # subset of backbones
#   ./pipeline_baselines.sh --edu-only
#   ./pipeline_baselines.sh --msmarco-only
#   ./pipeline_baselines.sh --force                    # wipe and retrain
#   ./pipeline_baselines.sh --from train_standard_ft   # skip pretrained eval
#   ./pipeline_baselines.sh --until find_mrl_bk        # stop before MRL truncation eval
#
# Env overrides:
#   BACKBONES_TO_RUN="e5large bge"
#   DATASETS="educational msmarco"
#   REUSE_TRAINED_MODELS=1   # skip training if checkpoints exist
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ─────────────────────────────────────────────────────────────────
DATASETS="${DATASETS:-educational msmarco scifact nfcorpus fiqa}"
BACKBONES_TO_RUN="${BACKBONES_TO_RUN:-e5large bge qwen06b qwen4b llm2vec gritlm}"
CKPT_ROOT="/tmp/multi-domain"
RESULTS_ROOT="./results/multi_domain"
BEIR_DATA_ROOT="/tmp/data/beir"
MSMARCO_DATA_DIR="/tmp/data/msmarco"
EDU_DATA_DIR="./data/real"
FORCE="${FORCE_PIPELINE:-0}"

# ── Argument parsing ──────────────────────────────────────────────────────────
FORWARD_ARGS=()
UNTIL_STEP=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --until)          UNTIL_STEP="$2";                          shift 2 ;;
        --datasets)       DATASETS="$2";    FORWARD_ARGS+=("$1" "$2"); shift 2 ;;
        --dataset)        DATASETS="$2";    FORWARD_ARGS+=("$1" "$2"); shift 2 ;;
        --backbone)       BACKBONES_TO_RUN="$2"; FORWARD_ARGS+=("$1" "$2"); shift 2 ;;
        --edu-only)       DATASETS="educational"; FORWARD_ARGS+=("$1"); shift ;;
        --msmarco-only)   DATASETS="msmarco";     FORWARD_ARGS+=("$1"); shift ;;
        --force)          FORCE=1;          FORWARD_ARGS+=("$1");   shift ;;
        --from)           FORWARD_ARGS+=("$1" "$2");                shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Phase 1: eval_pretrained + standard FT + MRL (delegated to main pipeline) ─
# Runs steps: eval_pretrained → train_standard_ft → find_standard_ft →
#             eval_standard_ft → train_mrl_bk → find_mrl_bk
# ─────────────────────────────────────────────────────────────────────────────
STOP_AFTER="find_mrl_bk"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  BASELINES PHASE 1: pretrained eval + standard FT + MRL train"
echo "  Backbones : $BACKBONES_TO_RUN"
echo "  Datasets  : $DATASETS"
echo "══════════════════════════════════════════════════════"

BACKBONES_TO_RUN="$BACKBONES_TO_RUN" \
DATASETS="$DATASETS" \
FORCE_PIPELINE="$FORCE" \
bash "$SCRIPT_DIR/bam_council_pipeline.sh" \
    --from eval_pretrained \
    --until "$STOP_AFTER" \
    "${FORWARD_ARGS[@]}"

# ── Phase 2: MRL truncation eval ─────────────────────────────────────────────
# eval_pretrained_truncation.py --checkpoint lets us reuse the same script to
# sweep truncation dims on the trained MRL checkpoint.
# This produces results/{dataset}/bam_pq_{bk}/mrl_truncation/pretrained_truncation.json
# ─────────────────────────────────────────────────────────────────────────────

# Skip phase 2 if user stopped before it
if [[ "$UNTIL_STEP" == "find_mrl_bk" ]]; then
    echo "  --until find_mrl_bk: skipping MRL truncation eval."
    exit 0
fi

echo ""
echo "══════════════════════════════════════════════════════"
echo "  BASELINES PHASE 2: MRL truncation eval per backbone"
echo "══════════════════════════════════════════════════════"

for DS in $DATASETS; do
    # Resolve data paths
    if [[ "$DS" == "educational" ]]; then
        TEST_PATH="$EDU_DATA_DIR/test.jsonl"
        CORPUS_PATH="$EDU_DATA_DIR/corpus.jsonl"
        IS_MSMARCO=0
    elif [[ "$DS" == "msmarco" ]]; then
        TEST_PATH="$MSMARCO_DATA_DIR/test.jsonl"
        CORPUS_PATH="$MSMARCO_DATA_DIR/corpus.jsonl"
        IS_MSMARCO=1
    else
        TEST_PATH="$BEIR_DATA_ROOT/$DS/test.jsonl"
        CORPUS_PATH="$BEIR_DATA_ROOT/$DS/corpus.jsonl"
        IS_MSMARCO=0
    fi

    DS_RESULTS="$RESULTS_ROOT/$DS"
    CFG_DIR="$DS_RESULTS/configs"

    echo ""
    echo "  ── Dataset: $DS ──"

    for BK in $BACKBONES_TO_RUN; do
        BK_RESULTS="$DS_RESULTS/bam_pq_$BK"
        MRL_OUT="$BK_RESULTS/mrl_truncation"
        mkdir -p "$MRL_OUT"

        # Locate the MRL checkpoint and config for this backbone
        if [[ "$BK" == "e5large" ]]; then
            BK_MRL_BEST="$CKPT_ROOT/$DS/mrl/best"
            BK_MRL_CFG="$CFG_DIR/mrl.yaml"
        else
            BK_MRL_BEST="$CKPT_ROOT/$DS/mrl_$BK/best"
            BK_MRL_CFG="$CFG_DIR/mrl_${BK}.yaml"
        fi

        echo ""
        echo "  [$DS][$BK] MRL truncation eval"

        if [[ -f "$MRL_OUT/pretrained_truncation.json" ]] && [[ "$FORCE" != "1" ]]; then
            echo "    Already evaluated — skipping."
            continue
        fi

        if [[ ! -f "$BK_MRL_BEST/checkpoint.pt" ]]; then
            echo "    WARNING: MRL checkpoint not found at $BK_MRL_BEST — skipping."
            continue
        fi

        if [[ ! -f "$BK_MRL_CFG" ]]; then
            echo "    WARNING: MRL config not found at $BK_MRL_CFG — skipping."
            continue
        fi

        python3 scripts/eval_pretrained_truncation.py \
            --config      "$BK_MRL_CFG" \
            --test_path   "$TEST_PATH" \
            --corpus_path "$CORPUS_PATH" \
            --checkpoint  "$BK_MRL_BEST" \
            --output_dir  "$MRL_OUT" \
            || echo "    WARNING: MRL truncation eval failed for $BK/$DS (non-fatal)"

        echo "    Results → $MRL_OUT/pretrained_truncation.json"
    done
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  BASELINES COMPLETE"
echo ""
echo "  Output layout per backbone per dataset:"
echo "    results/multi_domain/{DS}/bam_pq_{BK}/"
echo "      pretrained_truncation/pretrained_truncation.json  ← zero-shot"
echo "      standard_ft/pretrained_truncation.json            ← standard FT"
echo "      mrl_truncation/pretrained_truncation.json         ← MRL trained"
echo ""
echo "  Compare all three JSON files to see the ablation ladder:"
echo "    Pretrained → Standard FT → MRL → (run BAM-PQ to complete)"
echo "══════════════════════════════════════════════════════"
