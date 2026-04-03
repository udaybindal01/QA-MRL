#!/usr/bin/env bash
# =============================================================================
# BAM v4 Full Pipeline
# =============================================================================
# Runs the complete pipeline from data prep → training → eval → analysis.
# Edit the variables below before running.
#
# Usage:
#   chmod +x pipeline.sh
#   ./pipeline.sh [--step <step_name>]
#
# Steps (run in order, or pass --step <name> to run a single step):
#   data        — build dataset and annotate Bloom labels
#   train_mrl   — train MRL baseline
#   train_bam   — train BAM v4-A (prefix routing, configs/bam.yaml)
#   train_v4    — train BAM v4 Option B (scattered mask, configs/bam_v4.yaml)
#   find_best   — find best epoch checkpoint for each model
#   eval        — run full evaluation (BAM + MRL + BAM v4 comparison)
#   ablations   — run ablation study
#   analysis    — run all analysis scripts (ambiguity, mask, failures, robustness)
#   efficiency  — run sub-index efficiency benchmark
# =============================================================================

set -euo pipefail

# ---- Configuration -----------------------------------------------------------
BAM_CKPT_DIR="/tmp/bam-ckpts"
BAM_V4_CKPT_DIR="/tmp/bam-v4-ckpts"
MRL_CKPT_DIR="/tmp/mrl-ckpts"
RESULTS_DIR="results"
DATA_DIR="./data/real"

BAM_CONFIG="configs/bam.yaml"
BAM_V4_CONFIG="configs/bam_v4.yaml"
MRL_CONFIG="configs/neurips.yaml"
# ------------------------------------------------------------------------------

STEP="${1:-all}"
if [ "${1:-}" = "--step" ]; then
    STEP="${2:-all}"
fi

log() { echo ""; echo "=========================================================="; echo "  $1"; echo "=========================================================="; }

# ---- STEP: data --------------------------------------------------------------
run_data() {
    log "DATA PREPARATION"
    python data/build_real_data.py
    python data/annotate_bloom_pretrained.py \
        --data_dir "$DATA_DIR" \
        --method pretrained
    python data/curriculum_negatives.py \
        --pairs "$DATA_DIR/train.jsonl" \
        --corpus "$DATA_DIR/corpus.jsonl" \
        --output "$DATA_DIR/train_curriculum.jsonl" \
        --stage 0.7 --num_neg 3
    echo "Data preparation complete. Corpus at $DATA_DIR/"
}

# ---- STEP: train_mrl ---------------------------------------------------------
run_train_mrl() {
    log "TRAINING: MRL Baseline"
    python scripts/train_baseline_mrl.py \
        --config "$MRL_CONFIG"
    echo "MRL training complete. Checkpoints at $MRL_CKPT_DIR/"
}

# ---- STEP: train_bam ---------------------------------------------------------
run_train_bam() {
    log "TRAINING: BAM v4-A (prefix routing)"
    python scripts/train_bam.py \
        --config "$BAM_CONFIG"
    echo "BAM v4-A training complete. Checkpoints at $BAM_CKPT_DIR/"
}

# ---- STEP: train_v4 ----------------------------------------------------------
run_train_v4() {
    log "TRAINING: BAM v4 Option B (scattered mask)"
    python scripts/train_bam.py \
        --config "$BAM_V4_CONFIG"
    echo "BAM v4 Option B training complete. Checkpoints at $BAM_V4_CKPT_DIR/"
}

# ---- STEP: find_best ---------------------------------------------------------
run_find_best() {
    log "FINDING BEST EPOCH CHECKPOINTS"
    echo "  BAM v4-A..."
    python scripts/find_best_epoch.py \
        --checkpoint_dir "$BAM_CKPT_DIR" \
        --config "$BAM_CONFIG" \
        --output_dir "$RESULTS_DIR/best_epochs/bam/"

    echo "  MRL Baseline..."
    python scripts/find_best_epoch.py \
        --checkpoint_dir "$MRL_CKPT_DIR" \
        --config "$MRL_CONFIG" \
        --output_dir "$RESULTS_DIR/best_epochs/mrl/"

    if [ -d "$BAM_V4_CKPT_DIR" ]; then
        echo "  BAM v4 Option B..."
        python scripts/find_best_epoch.py \
            --checkpoint_dir "$BAM_V4_CKPT_DIR" \
            --config "$BAM_V4_CONFIG" \
            --output_dir "$RESULTS_DIR/best_epochs/bam_v4/"
    fi
}

# ---- STEP: eval --------------------------------------------------------------
run_eval() {
    log "EVALUATION"
    mkdir -p "$RESULTS_DIR/bam_eval"

    BAM_BEST="$BAM_CKPT_DIR/best"
    MRL_BEST="$MRL_CKPT_DIR/best"

    # Use find_best_epoch output if available
    if [ -f "$RESULTS_DIR/best_epochs/bam/best_checkpoint_path.txt" ]; then
        BAM_BEST=$(cat "$RESULTS_DIR/best_epochs/bam/best_checkpoint_path.txt")
    fi
    if [ -f "$RESULTS_DIR/best_epochs/mrl/best_checkpoint_path.txt" ]; then
        MRL_BEST=$(cat "$RESULTS_DIR/best_epochs/mrl/best_checkpoint_path.txt")
    fi

    CMD="python scripts/eval_bam.py \
        --config $BAM_CONFIG \
        --checkpoint $BAM_BEST \
        --baseline $MRL_BEST \
        --output_dir $RESULTS_DIR/bam_eval/"

    # Add BAM v4 if checkpoint exists
    if [ -d "$BAM_V4_CKPT_DIR" ]; then
        BAM_V4_BEST="$BAM_V4_CKPT_DIR/best"
        if [ -f "$RESULTS_DIR/best_epochs/bam_v4/best_checkpoint_path.txt" ]; then
            BAM_V4_BEST=$(cat "$RESULTS_DIR/best_epochs/bam_v4/best_checkpoint_path.txt")
        fi
        CMD="$CMD --checkpoint_v4 $BAM_V4_BEST --config_v4 $BAM_V4_CONFIG"
    fi

    eval "$CMD"
    echo "Evaluation complete. Results at $RESULTS_DIR/bam_eval/"
}

# ---- STEP: ablations ---------------------------------------------------------
run_ablations() {
    log "ABLATIONS"
    mkdir -p "$RESULTS_DIR/ablations"

    BAM_BEST="$BAM_CKPT_DIR/best"
    if [ -f "$RESULTS_DIR/best_epochs/bam/best_checkpoint_path.txt" ]; then
        BAM_BEST=$(cat "$RESULTS_DIR/best_epochs/bam/best_checkpoint_path.txt")
    fi

    CMD="python scripts/run_ablations.py \
        --config $BAM_CONFIG \
        --checkpoint $BAM_BEST \
        --output_dir $RESULTS_DIR/ablations/"

    if [ -d "$MRL_CKPT_DIR" ]; then
        MRL_BEST="$MRL_CKPT_DIR/best"
        CMD="$CMD --baseline $MRL_BEST"
    fi

    if [ -d "$BAM_V4_CKPT_DIR" ]; then
        BAM_V4_BEST="$BAM_V4_CKPT_DIR/best"
        CMD="$CMD --checkpoint_v4 $BAM_V4_BEST --config_v4 $BAM_V4_CONFIG"
    fi

    eval "$CMD"
    echo "Ablations complete. Results at $RESULTS_DIR/ablations/"
}

# ---- STEP: analysis ----------------------------------------------------------
run_analysis() {
    log "ANALYSIS SCRIPTS"
    mkdir -p "$RESULTS_DIR/analysis"

    BAM_BEST="$BAM_CKPT_DIR/best"
    if [ -f "$RESULTS_DIR/best_epochs/bam/best_checkpoint_path.txt" ]; then
        BAM_BEST=$(cat "$RESULTS_DIR/best_epochs/bam/best_checkpoint_path.txt")
    fi

    echo "  [1/4] Routing ambiguity analysis (Challenge 2 + 7)..."
    python scripts/analyze_routing_ambiguity.py \
        --config "$BAM_CONFIG" \
        --checkpoint "$BAM_BEST" \
        --output_dir "$RESULTS_DIR/analysis/"

    echo "  [2/4] Evaluate-level failure analysis (Challenge 3)..."
    python scripts/analyze_evaluate_failures.py \
        --config "$BAM_CONFIG" \
        --checkpoint "$BAM_BEST" \
        --output_dir "$RESULTS_DIR/analysis/"

    echo "  [3/4] Classifier robustness (Challenge 4)..."
    python scripts/analyze_classifier_robustness.py \
        --config "$BAM_CONFIG" \
        --checkpoint "$BAM_BEST" \
        --output_dir "$RESULTS_DIR/analysis/"

    echo "  [4/4] Mask specialization (Option B qualitative)..."
    if [ -d "$BAM_V4_CKPT_DIR" ]; then
        BAM_V4_BEST="$BAM_V4_CKPT_DIR/best"
        if [ -f "$RESULTS_DIR/best_epochs/bam_v4/best_checkpoint_path.txt" ]; then
            BAM_V4_BEST=$(cat "$RESULTS_DIR/best_epochs/bam_v4/best_checkpoint_path.txt")
        fi
        python scripts/analyze_mask_specialization.py \
            --config "$BAM_V4_CONFIG" \
            --checkpoint "$BAM_V4_BEST" \
            --output_dir "$RESULTS_DIR/analysis/"
    else
        echo "    Skipping mask specialization — BAM v4 checkpoint not found."
        echo "    Run: ./pipeline.sh --step train_v4  first."
    fi

    echo "Analysis complete. Results at $RESULTS_DIR/analysis/"
}

# ---- STEP: efficiency --------------------------------------------------------
run_efficiency() {
    log "EFFICIENCY BENCHMARK"
    mkdir -p "$RESULTS_DIR/efficiency"

    BAM_BEST="$BAM_CKPT_DIR/best"
    if [ -f "$RESULTS_DIR/best_epochs/bam/best_checkpoint_path.txt" ]; then
        BAM_BEST=$(cat "$RESULTS_DIR/best_epochs/bam/best_checkpoint_path.txt")
    fi

    python -m evaluation.bloom_subindex \
        --config "$BAM_CONFIG" \
        --checkpoint "$BAM_BEST" \
        --output_dir "$RESULTS_DIR/efficiency/"

    echo "Efficiency benchmark complete. Results at $RESULTS_DIR/efficiency/"
}

# ---- MAIN --------------------------------------------------------------------
case "$STEP" in
    all)
        run_data
        run_train_mrl
        run_train_bam
        run_train_v4
        run_find_best
        run_eval
        run_ablations
        run_analysis
        run_efficiency
        log "PIPELINE COMPLETE"
        echo "Results saved to $RESULTS_DIR/"
        ;;
    data)        run_data ;;
    train_mrl)   run_train_mrl ;;
    train_bam)   run_train_bam ;;
    train_v4)    run_train_v4 ;;
    find_best)   run_find_best ;;
    eval)        run_eval ;;
    ablations)   run_ablations ;;
    analysis)    run_analysis ;;
    efficiency)  run_efficiency ;;
    *)
        echo "Unknown step: $STEP"
        echo "Valid steps: all, data, train_mrl, train_bam, train_v4, find_best, eval, ablations, analysis, efficiency"
        exit 1
        ;;
esac
