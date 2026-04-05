#!/usr/bin/env bash
# =============================================================================
# BAM v4 — End-to-End Pipeline
# =============================================================================
#
# Usage:
#   chmod +x pipeline.sh
#   ./pipeline.sh                    # run all steps in order
#   ./pipeline.sh --step <name>      # run one step
#   ./pipeline.sh --from <name>      # run from a step to the end
#
# Steps (always executed in this order):
#   1.  data          build dataset + Bloom annotation + curriculum negatives
#   2.  train_mrl     train MRL baseline (15 epochs)
#   3.  find_mrl      find best MRL epoch (corpus-level recall@10)
#   4.  train_bam     train BAM Option A (prefix routing, warm-starts from MRL)
#   5.  train_v4      train BAM Option B (scattered mask, warm-starts from MRL)
#   6.  find_best     find best epoch for BAM-A and BAM-B
#   7.  eval          full evaluation table (BAM-A vs BAM-B vs MRL)
#   8.  ablations     ablation study incl. fixed-budget routing baseline
#   9.  analysis      routing ambiguity, failures, classifier robustness,
#                     mask specialization, bloom dim allocation
#   10. efficiency    sub-index latency benchmark
#   11. beir          cross-domain evaluation (SciFact, NFCorpus, FiQA, ArguAna)
#
# Prerequisites:
#   pip install -r requirements.txt
#   GPU recommended (CUDA). CPU runs will be slow for training.
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these paths before running
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR="./data/real"               # where JSONL files are written
MRL_CKPT_DIR="/tmp/mrl-ckpts"        # MRL baseline checkpoints
BAM_CKPT_DIR="/tmp/bam-ckpts"        # BAM Option A (prefix) checkpoints
BAM_V4_CKPT_DIR="/tmp/bam-v4-ckpts"  # BAM Option B (scattered) checkpoints
RESULTS_DIR="./results"              # all output goes here

MRL_CONFIG="configs/neurips.yaml"
BAM_CONFIG="configs/bam.yaml"
BAM_V4_CONFIG="configs/bam_v4.yaml"

# Hard negative mining curriculum stage (0.0–1.0; 0.7 = mine at 70% training)
CURRICULUM_STAGE="0.7"
NUM_NEG=3          # hard negatives per training query
NUM_BEIR_DATASETS="scifact nfcorpus fiqa arguana"
# ─────────────────────────────────────────────────────────────────────────────

# ── Argument parsing ──────────────────────────────────────────────────────────
STEP="all"
FROM_STEP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step) STEP="$2"; shift 2 ;;
        --from) FROM_STEP="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ALL_STEPS=(install data train_mrl find_mrl train_bam train_v4 find_best eval ablations analysis efficiency beir)

# Resolve --from: set STEP=all but skip earlier steps
if [[ -n "$FROM_STEP" ]]; then
    STEP="all"
    found=0
    SKIP_STEPS=()
    for s in "${ALL_STEPS[@]}"; do
        if [[ "$s" == "$FROM_STEP" ]]; then found=1; fi
        if [[ $found -eq 0 ]]; then SKIP_STEPS+=("$s"); fi
    done
    if [[ $found -eq 0 ]]; then
        echo "Unknown step for --from: $FROM_STEP"
        echo "Valid steps: ${ALL_STEPS[*]}"
        exit 1
    fi
fi

should_run() {
    local s="$1"
    if [[ "$STEP" != "all" ]]; then
        [[ "$STEP" == "$s" ]] && return 0 || return 1
    fi
    for skip in "${SKIP_STEPS[@]:-}"; do
        [[ "$skip" == "$s" ]] && return 1
    done
    return 0
}

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { echo ""; echo "══════════════════════════════════════════════════════"; \
         echo "  [$( date '+%H:%M:%S' )]  $1"; \
         echo "══════════════════════════════════════════════════════"; }
die()  { echo "ERROR: $1" >&2; exit 1; }

# Read the best checkpoint path written by find_best_epoch.py
# Falls back to $2 if the file doesn't exist yet.
best_ckpt() {
    local results_dir="$1" fallback="$2"
    local f="$results_dir/best_checkpoint_path.txt"
    if [[ -f "$f" ]]; then cat "$f"; else echo "$fallback"; fi
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — INSTALL DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA
# ─────────────────────────────────────────────────────────────────────────────
run_data() {
    log "STEP 1/11 — DATA PREPARATION"

    python3 data/build_real_data.py \
        || die "build_real_data.py failed"

    python3 data/annotate_bloom_pretrained.py \
        --data_dir "$DATA_DIR" \
        --method pretrained \
        || die "annotate_bloom_pretrained.py failed"

    # Bloom prediction caches (.bloom_cache.json) are generated automatically
    # by EducationalRetrievalDataset on first training run — no extra step needed.

    python3 data/curriculum_negatives.py \
        --pairs   "$DATA_DIR/train.jsonl" \
        --corpus  "$DATA_DIR/corpus.jsonl" \
        --output  "$DATA_DIR/train_curriculum.jsonl" \
        --stage   "$CURRICULUM_STAGE" \
        --num_neg "$NUM_NEG" \
        || die "curriculum_negatives.py failed"

    echo "Data ready at $DATA_DIR/"
    echo "  train.jsonl  →  $(wc -l < "$DATA_DIR/train.jsonl") pairs"
    echo "  val.jsonl    →  $(wc -l < "$DATA_DIR/val.jsonl") pairs"
    echo "  test.jsonl   →  $(wc -l < "$DATA_DIR/test.jsonl") pairs"
    echo "  corpus.jsonl →  $(wc -l < "$DATA_DIR/corpus.jsonl") passages"
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — TRAIN MRL BASELINE
# ─────────────────────────────────────────────────────────────────────────────
run_train_mrl() {
    log "STEP 2/11 — TRAINING: MRL Baseline (15 epochs)"

    [[ -f "$DATA_DIR/train.jsonl" ]] \
        || die "train.jsonl not found. Run: ./pipeline.sh --step data first."

    python3 scripts/train_baseline_mrl.py \
        --config "$MRL_CONFIG" \
        || die "train_baseline_mrl.py failed"

    echo "MRL checkpoints at $MRL_CKPT_DIR/"
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — FIND BEST MRL EPOCH
# ─────────────────────────────────────────────────────────────────────────────
run_find_mrl() {
    log "STEP 3/11 — FINDING BEST MRL EPOCH (corpus-level recall@10)"

    [[ -d "$MRL_CKPT_DIR/epoch_0" ]] \
        || die "No MRL epoch checkpoints found at $MRL_CKPT_DIR. Run train_mrl first."

    mkdir -p "$RESULTS_DIR/best_epochs/mrl"

    python3 scripts/find_best_epoch.py \
        --checkpoint_dir "$MRL_CKPT_DIR" \
        --config         "$MRL_CONFIG" \
        --model_type     mrl \
        --metric         recall@10 \
        --output_dir     "$RESULTS_DIR/best_epochs/mrl/" \
        || die "find_best_epoch (MRL) failed"

    echo "MRL best checkpoint → $MRL_CKPT_DIR/best/"
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — TRAIN BAM OPTION A (prefix routing)
# ─────────────────────────────────────────────────────────────────────────────
run_train_bam() {
    log "STEP 4/11 — TRAINING: BAM v4-A (prefix routing, warm-start from MRL)"

    MRL_BEST=$(best_ckpt "$RESULTS_DIR/best_epochs/mrl" "$MRL_CKPT_DIR/best")
    [[ -f "$MRL_BEST/checkpoint.pt" ]] \
        || die "MRL best checkpoint not found at $MRL_BEST. Run find_mrl first."

    python3 scripts/train_bam.py \
        --config       "$BAM_CONFIG" \
        --init_encoder "$MRL_BEST" \
        || die "train_bam.py (Option A) failed"

    echo "BAM Option A checkpoints at $BAM_CKPT_DIR/"
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — TRAIN BAM OPTION B (scattered mask)
# ─────────────────────────────────────────────────────────────────────────────
run_train_v4() {
    log "STEP 5/11 — TRAINING: BAM v4 Option B (scattered mask, warm-start from MRL)"

    MRL_BEST=$(best_ckpt "$RESULTS_DIR/best_epochs/mrl" "$MRL_CKPT_DIR/best")
    [[ -f "$MRL_BEST/checkpoint.pt" ]] \
        || die "MRL best checkpoint not found at $MRL_BEST. Run find_mrl first."

    python3 scripts/train_bam.py \
        --config       "$BAM_V4_CONFIG" \
        --init_encoder "$MRL_BEST" \
        || die "train_bam.py (Option B) failed"

    echo "BAM Option B checkpoints at $BAM_V4_CKPT_DIR/"
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — FIND BEST EPOCH (BAM-A and BAM-B)
# ─────────────────────────────────────────────────────────────────────────────
run_find_best() {
    log "STEP 6/11 — FINDING BEST EPOCH CHECKPOINTS"
    mkdir -p "$RESULTS_DIR/best_epochs/bam" "$RESULTS_DIR/best_epochs/bam_v4"

    echo "  BAM Option A..."
    [[ -d "$BAM_CKPT_DIR/epoch_0" ]] \
        || die "No BAM epoch checkpoints at $BAM_CKPT_DIR. Run train_bam first."

    python3 scripts/find_best_epoch.py \
        --checkpoint_dir "$BAM_CKPT_DIR" \
        --config         "$BAM_CONFIG" \
        --model_type     bam \
        --metric         recall@10 \
        --output_dir     "$RESULTS_DIR/best_epochs/bam/" \
        || die "find_best_epoch (BAM-A) failed"

    echo "  BAM Option B..."
    [[ -d "$BAM_V4_CKPT_DIR/epoch_0" ]] \
        || die "No BAM v4 epoch checkpoints at $BAM_V4_CKPT_DIR. Run train_v4 first."

    python3 scripts/find_best_epoch.py \
        --checkpoint_dir  "$BAM_V4_CKPT_DIR" \
        --config          "$BAM_V4_CONFIG" \
        --model_type      bam \
        --metric          recall@10 \
        --skip_warmup_epochs 0 \
        --output_dir      "$RESULTS_DIR/best_epochs/bam_v4/" \
        || die "find_best_epoch (BAM-B) failed"

    echo "  BAM-A best  → $BAM_CKPT_DIR/best/"
    echo "  BAM-B best  → $BAM_V4_CKPT_DIR/best/"
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — FULL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
run_eval() {
    log "STEP 7/11 — FULL EVALUATION"
    mkdir -p "$RESULTS_DIR/bam_eval"

    BAM_BEST=$(best_ckpt    "$RESULTS_DIR/best_epochs/bam"    "$BAM_CKPT_DIR/best")
    MRL_BEST=$(best_ckpt    "$RESULTS_DIR/best_epochs/mrl"    "$MRL_CKPT_DIR/best")
    BAM_V4_BEST=$(best_ckpt "$RESULTS_DIR/best_epochs/bam_v4" "$BAM_V4_CKPT_DIR/best")

    [[ -f "$BAM_BEST/checkpoint.pt"    ]] || die "BAM-A checkpoint missing: $BAM_BEST"
    [[ -f "$MRL_BEST/checkpoint.pt"    ]] || die "MRL checkpoint missing: $MRL_BEST"
    [[ -f "$BAM_V4_BEST/checkpoint.pt" ]] || die "BAM-B checkpoint missing: $BAM_V4_BEST"

    python3 scripts/eval_bam.py \
        --config          "$BAM_CONFIG" \
        --checkpoint      "$BAM_BEST" \
        --baseline        "$MRL_BEST" \
        --checkpoint_v4   "$BAM_V4_BEST" \
        --config_v4       "$BAM_V4_CONFIG" \
        --output_dir      "$RESULTS_DIR/bam_eval/" \
        || die "eval_bam.py failed"

    echo "Evaluation results → $RESULTS_DIR/bam_eval/results.json"
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — ABLATIONS
# ─────────────────────────────────────────────────────────────────────────────
run_ablations() {
    log "STEP 8/11 — ABLATION STUDY"
    mkdir -p "$RESULTS_DIR/ablations"

    BAM_BEST=$(best_ckpt    "$RESULTS_DIR/best_epochs/bam"    "$BAM_CKPT_DIR/best")
    MRL_BEST=$(best_ckpt    "$RESULTS_DIR/best_epochs/mrl"    "$MRL_CKPT_DIR/best")
    BAM_V4_BEST=$(best_ckpt "$RESULTS_DIR/best_epochs/bam_v4" "$BAM_V4_CKPT_DIR/best")

    [[ -f "$BAM_BEST/checkpoint.pt" ]] || die "BAM-A checkpoint missing: $BAM_BEST"

    python3 scripts/run_ablations.py \
        --config          "$BAM_CONFIG" \
        --checkpoint      "$BAM_BEST" \
        --baseline        "$MRL_BEST" \
        --checkpoint_v4   "$BAM_V4_BEST" \
        --config_v4       "$BAM_V4_CONFIG" \
        --output_dir      "$RESULTS_DIR/ablations/" \
        || die "run_ablations.py failed"

    echo "Ablation results → $RESULTS_DIR/ablations/ablation_results.json"
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
run_analysis() {
    log "STEP 9/11 — ANALYSIS SCRIPTS"
    mkdir -p "$RESULTS_DIR/analysis"

    BAM_BEST=$(best_ckpt    "$RESULTS_DIR/best_epochs/bam"    "$BAM_CKPT_DIR/best")
    BAM_V4_BEST=$(best_ckpt "$RESULTS_DIR/best_epochs/bam_v4" "$BAM_V4_CKPT_DIR/best")

    [[ -f "$BAM_BEST/checkpoint.pt"    ]] || die "BAM-A checkpoint missing: $BAM_BEST"
    [[ -f "$BAM_V4_BEST/checkpoint.pt" ]] || die "BAM-B checkpoint missing: $BAM_V4_BEST"

    echo "  [1/5] Routing ambiguity..."
    python3 scripts/analyze_routing_ambiguity.py \
        --config "$BAM_CONFIG" \
        --checkpoint "$BAM_BEST" \
        --output_dir "$RESULTS_DIR/analysis/" \
        || echo "  WARNING: analyze_routing_ambiguity.py failed (non-fatal)"

    echo "  [2/5] Evaluate-level failure analysis..."
    python3 scripts/analyze_evaluate_failures.py \
        --config "$BAM_CONFIG" \
        --checkpoint "$BAM_BEST" \
        --output_dir "$RESULTS_DIR/analysis/" \
        || echo "  WARNING: analyze_evaluate_failures.py failed (non-fatal)"

    echo "  [3/5] Classifier robustness + accuracy (reviewer C)..."
    python3 scripts/analyze_classifier_robustness.py \
        --config "$BAM_CONFIG" \
        --checkpoint "$BAM_BEST" \
        --output_dir "$RESULTS_DIR/analysis/" \
        || die "analyze_classifier_robustness.py failed"

    echo "  [4/5] Mask specialization (Option B)..."
    python3 scripts/analyze_mask_specialization.py \
        --config "$BAM_V4_CONFIG" \
        --checkpoint "$BAM_V4_BEST" \
        --output_dir "$RESULTS_DIR/analysis/" \
        || echo "  WARNING: analyze_mask_specialization.py failed (non-fatal)"

    echo "  [5/5] Bloom dim allocation — cognitive load hypothesis test (reviewer B)..."
    python3 scripts/analyze_bloom_dim_allocation.py \
        --config "$BAM_V4_CONFIG" \
        --checkpoint "$BAM_V4_BEST" \
        --output_dir "$RESULTS_DIR/analysis/" \
        --n_samples 2000 \
        || die "analyze_bloom_dim_allocation.py failed"

    echo "Analysis results → $RESULTS_DIR/analysis/"
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — EFFICIENCY BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
run_efficiency() {
    log "STEP 10/11 — EFFICIENCY BENCHMARK"
    mkdir -p "$RESULTS_DIR/efficiency"

    BAM_BEST=$(best_ckpt "$RESULTS_DIR/best_epochs/bam" "$BAM_CKPT_DIR/best")
    [[ -f "$BAM_BEST/checkpoint.pt" ]] || die "BAM-A checkpoint missing: $BAM_BEST"

    python3 -m evaluation.bloom_subindex \
        --config "$BAM_CONFIG" \
        --checkpoint "$BAM_BEST" \
        --output_dir "$RESULTS_DIR/efficiency/" \
        || die "bloom_subindex failed"

    echo "Efficiency results → $RESULTS_DIR/efficiency/"
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — BEIR CROSS-DOMAIN EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
run_beir() {
    log "STEP 11/11 — BEIR CROSS-DOMAIN EVALUATION"
    mkdir -p "$RESULTS_DIR/beir"

    BAM_BEST=$(best_ckpt "$RESULTS_DIR/best_epochs/bam" "$BAM_CKPT_DIR/best")
    MRL_BEST=$(best_ckpt "$RESULTS_DIR/best_epochs/mrl" "$MRL_CKPT_DIR/best")

    [[ -f "$BAM_BEST/checkpoint.pt" ]] || die "BAM-A checkpoint missing: $BAM_BEST"
    [[ -f "$MRL_BEST/checkpoint.pt" ]] || die "MRL checkpoint missing: $MRL_BEST"

    python3 scripts/eval_beir.py \
        --config     "$MRL_CONFIG" \
        --checkpoint "$BAM_BEST" \
        --baseline   "$MRL_BEST" \
        --datasets   $NUM_BEIR_DATASETS \
        --output_dir "$RESULTS_DIR/beir/" \
        || die "eval_beir.py failed"

    echo "BEIR results → $RESULTS_DIR/beir/"
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN DISPATCH
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$STEP" == "all" ]]; then
    should_run install    && run_install
    should_run data       && run_data
    should_run train_mrl  && run_train_mrl
    should_run find_mrl   && run_find_mrl
    should_run train_bam  && run_train_bam
    should_run train_v4   && run_train_v4
    should_run find_best  && run_find_best
    should_run eval       && run_eval
    should_run ablations  && run_ablations
    should_run analysis   && run_analysis
    should_run efficiency && run_efficiency
    should_run beir       && run_beir
    log "PIPELINE COMPLETE"
    echo ""
    echo "All results saved to $RESULTS_DIR/"
    echo ""
    echo "  Key outputs:"
    echo "    $RESULTS_DIR/bam_eval/results.json          ← main eval table"
    echo "    $RESULTS_DIR/ablations/ablation_results.json ← ablations"
    echo "    $RESULTS_DIR/analysis/bloom_dim_allocation.json ← hypothesis test"
    echo "    $RESULTS_DIR/analysis/classifier_accuracy.json  ← classifier ceiling"
    echo "    $RESULTS_DIR/beir/                          ← cross-domain"
else
    case "$STEP" in
        data)       run_data ;;
        train_mrl)  run_train_mrl ;;
        find_mrl)   run_find_mrl ;;
        train_bam)  run_train_bam ;;
        train_v4)   run_train_v4 ;;
        find_best)  run_find_best ;;
        eval)       run_eval ;;
        ablations)  run_ablations ;;
        analysis)   run_analysis ;;
        efficiency) run_efficiency ;;
        beir)       run_beir ;;
        *)
            echo "Unknown step: $STEP"
            echo "Valid steps: ${ALL_STEPS[*]}"
            exit 1
            ;;
    esac
fi
