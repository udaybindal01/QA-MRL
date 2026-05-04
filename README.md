# BAM-PQ: Bloom-Aligned Matryoshka Embeddings with Per-Query Dimension Routing

> **EMNLP 2026 Submission** — Query-adaptive dimension routing for educational information retrieval using Bloom's cognitive taxonomy.

---

## Overview

Standard dense retrieval treats all queries identically at any given dimension budget. A student asking *"What is osmosis?"* and one asking *"Propose an experimental design to test whether temperature affects enzyme activity"* impose fundamentally different representational demands — yet Matryoshka Representation Learning (MRL) forces both through the same leading prefix of dimensions.

**BAM-PQ** (Bloom-Aligned Matryoshka with Per-Query routing) addresses this by learning a per-query binary dimension mask conditioned on Bloom's cognitive taxonomy level. A taxonomy-aware prior anchors each query's mask to a cognitively appropriate subspace; a lightweight per-query residual MLP refines it based on query content. The result is a retrieval model that allocates representational capacity according to cognitive demand — without any per-level dimension targets.

**Key results (e5-large-v2 backbone, educational benchmark):**
- **+6.3 Recall@10** over MRL across all 6 Bloom levels using only **62% of dimensions**
- Gains consistent across the full cognitive spectrum: factual recall (+0.068), procedural reasoning (+0.084), creative synthesis (+0.058)
- Controlled fair comparison confirms gains reflect *which* dimensions are selected, not how many

---

## Table of Contents

1. [Architecture](#architecture)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Data Preparation](#data-preparation)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Pipeline Scripts](#pipeline-scripts)
8. [Configuration Reference](#configuration-reference)
9. [Model Variants](#model-variants)
10. [Supported Backbones](#supported-backbones)
11. [Results](#results)
12. [Development Notes](#development-notes)

---

## Architecture

### Data Flow

```
Query text
    │
    ▼
Bloom Council (4-model ensemble: DeBERTa-large, RoBERTa-large, BERT-base, SVC)
    │  b ∈ {0,1,2,3,4,5}  (85–88% accuracy)
    ▼
Transformer Encoder (e.g. intfloat/e5-large-v2)
    │  z ∈ ℝ^768  (CLS embedding)
    ▼
BloomQueryMaskHead
    ├── Bloom prior  E[b]          (learned per-level embedding, Gaussian-quantile init)
    │       +
    └── Query residual Δ(z)        (2-layer MLP, zero-initialized output)
            │  ℓ = E[b] + α · Δ(z)
            ▼
    Gumbel-STE → binary mask m ∈ {0,1}^768
            │
            ▼
    q̃ = normalize(z ⊙ m)          (sparse masked query)
            │
            ▼
    score(q, p) = q̃ᵀ d            (FAISS inner product, full-dim corpus index)

Passage text → Encoder → d ∈ ℝ^768 (no masking, stored once in FAISS)
```

### Model Progression

The codebase implements four models in increasing routing expressiveness:

| Model | Description | Routing |
|-------|-------------|---------|
| **Standard FT** | InfoNCE fine-tuning, full dimensions | None |
| **MRL** | Matryoshka at `{64,128,256,512,768}` | None (same prefix per query) |
| **BAM-B** | Per-Bloom-level scattered binary mask | Per cognitive level |
| **BAM-PQ** | BAM-B + per-query residual MLP | Per individual query |

### Loss Stack (BAM-PQ, 10 components)

| Loss | Weight | Purpose |
|------|--------|---------|
| `BloomMaskedContrastiveLoss` | 1.0 | InfoNCE in masked subspace, class-weighted by 1/√freq |
| `MaskSparsityLoss` | 1.0 | Per-Bloom active dims vs. cognitive-load targets |
| `BloomMaskDiversityLoss` | 0.5 | Pairwise repulsion between per-level masks |
| `BloomMaskVarianceLoss` | 0.3 | Spread information across all dim positions |
| `BloomMaskDistillationLoss` | 0.3 | Preserve full-dim similarity ordering in masked subspace |
| `MRLAnchorRegularizationLoss` | 0.3 | InfoNCE at MRL anchor dims (prevents forgetting) |
| `QueryRoutingContrastiveLoss` | 0.1 | Per-query routing diversity within a Bloom level |
| `BloomTwoFactorEfficiencyLoss` | 0.0→1.0 | Cognitive-complexity-weighted compression penalty |
| `DimVarianceRedistributionLoss` | 0.05 | Counter MRL prefix bias in encoder |
| `DropMask` (augmentation) | rate=0.1 | Randomly flip mask bits; structural, not a loss term |

### Training Schedule (BAM-PQ)

```
Epochs 0–7:   Encoder FROZEN, mask head trains at lr=2e-4
              → mask converges against stable representation target
              → efficiency loss inactive (warmup)

Epoch 8:      Encoder UNFROZEN at lr=1e-6
              → gradient checkpointing enabled to prevent OOM
              → encoder adapts gently to converged mask

Epochs 8–19:  Full co-adaptation, all loss components active
              → temperature anneals 0.1 → 0.02

Checkpoint selection: BSR metric (Bloom-Stratified Recall, α=0.5)
              → balances per-level R@10 against cognitive-weighted compression
              → warmup epochs excluded from selection
```

---

## Project Structure

```
QA-MRL/
├── configs/                        # 90+ YAML configuration files
│   ├── bam.yaml                    # BAM-B (Option A: contiguous prefix mask)
│   ├── bam_v4.yaml                 # BAM-B (Option B: scattered mask) — default
│   ├── bam_pq_*.yaml               # BAM-PQ per backbone (13 backbones × edu + msmarco)
│   ├── mrl_*.yaml                  # MRL baselines per backbone
│   ├── standard_ft_*.yaml          # Standard fine-tuning baselines
│   └── neurips.yaml, default.yaml  # Legacy configs
│
├── models/
│   ├── encoder.py                  # MRLEncoder — unified backbone wrapper
│   ├── bam.py                      # BloomAlignedMRL, BloomMaskHead, BloomQueryMaskHead
│   ├── bam_losses.py               # All loss functions (v11)
│   ├── qa_mrl.py                   # QAMRL — soft/group routing variant
│   ├── router.py                   # SoftRouter, GroupRouter
│   ├── pooling.py                  # CLS, mean, last_token, max pooling
│   └── losses.py                   # MRL contrastive losses
│
├── training/
│   ├── bam_trainer.py              # BAMTrainer — main training loop
│   ├── mrl_trainer.py              # MRL training loop
│   └── qa_mrl_trainer.py           # QA-MRL training loop
│
├── data/
│   ├── dataset.py                  # EducationalRetrievalDataset, dataloaders
│   ├── build_real_data.py          # Build corpus from SciQ/ARC/OBQA/QASC
│   ├── build_msmarco.py            # Build MS MARCO training data
│   ├── build_beir_training_data.py # Build BEIR in-domain training data
│   ├── annotate_bloom_pretrained.py# Bloom annotation via council
│   ├── annotate_bloom_batched.py   # Batched Gemini annotation
│   └── curriculum_negatives.py     # BM25 curriculum hard negative mining
│
├── evaluation/
│   ├── evaluator.py                # FullEvaluator — corpus-level FAISS retrieval
│   ├── bloom_stratified.py         # Per-Bloom-level metrics and BSR
│   ├── retrieval_metrics.py        # R@k, NDCG@k, MRR
│   ├── efficient_retrieval.py      # Latency/throughput benchmarking
│   └── pareto_analysis.py          # Recall vs. dims Pareto curves
│
├── scripts/
│   ├── train_bam.py                # Train BAM-B or BAM-PQ
│   ├── train_baseline_mrl.py       # Train MRL baseline
│   ├── eval_bam.py                 # Evaluate BAM vs. MRL with fair comparison
│   ├── find_best_epoch.py          # Post-hoc best checkpoint selection (BSR)
│   ├── eval_beir.py                # BEIR in-domain evaluation
│   ├── eval_fair_comparison.py     # Controlled scatter vs. prefix comparison
│   ├── run_ablations.py            # Full ablation suite
│   ├── run_diagnostics.py          # Routing analysis and diagnostics
│   ├── run_probing.py              # Linear probing of embedding dimensions
│   ├── run_efficiency.py           # Efficiency curves across dim budgets
│   └── generate_figures.py         # Paper figure generation
│
├── utils/
│   ├── misc.py                     # load_config, set_seed, count_parameters
│   └── logging_utils.py            # WandbLogger wrapper
│
├── bam_council_pipeline.sh         # Master pipeline (1220 lines)
├── pipeline_backbones_1.sh         # Group 1: e5-large, bge-large
├── pipeline_backbones_2.sh         # Group 2: qwen06b, qwen4b
├── pipeline_backbones_3.sh         # Group 3: arctic, roberta, phi3mini
├── pipeline_backbones_4.sh         # Group 4: qwen8b, llama1b, llama3b
├── pipeline_shared.sh              # Shared data prep and e5/bge MRL
└── requirements.txt
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/udaybindal01/QA-MRL.git
cd QA-MRL

# Create and activate environment
conda create -n qa-mrl python=3.10
conda activate qa-mrl

# Install core dependencies
pip install -r requirements.txt

# Optional: LLM2Vec backbones (Mistral-7B, LLaMA-3-8B)
pip install llm2vec

# Optional: GritLM-7B backbone
pip install gritlm
```

### Requirements

```
torch>=2.0.0
transformers>=4.40.0
sentence-transformers>=2.3.0
datasets>=2.16.0
faiss-gpu>=1.7.4
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
wandb>=0.16.0
pyyaml>=6.0
tqdm>=4.65.0
accelerate>=0.25.0
beir>=2.0.0
```

### Environment Variables

```bash
export HF_HOME=/path/to/model/cache          # HuggingFace model cache
export WANDB_PROJECT=bam-emnlp               # W&B project name
export WANDB_ENTITY=your-entity              # W&B entity (optional)
```

---

## Data Preparation

Run these scripts in order before training:

### Step 1 — Build educational corpus

Combines SciQ, ARC-Easy, ARC-Challenge, OpenBookQA, and QASC into a unified retrieval benchmark with BM25 hard negatives.

```bash
python data/build_real_data.py
```

**Output:**
```
./data/real/
├── train_curriculum.jsonl   # 26,361 training queries with curriculum negatives
├── val.jsonl                # Validation queries
├── test.jsonl               # 3,296 test queries
└── corpus.jsonl             # 40,640 passages
```

### Step 2 — Annotate with Bloom levels

Runs the four-model Bloom classification council over all queries. Predictions are cached as `.bloom_cache.json` files alongside each JSONL.

```bash
python data/annotate_bloom_pretrained.py
```

### Step 3 — Mine curriculum hard negatives

Mines BM25 hard negatives at curriculum stage 0.7 (30th-percentile-hardest candidates).

```bash
python data/curriculum_negatives.py
```

### Optional — MS MARCO data

```bash
python data/build_msmarco.py
# Data written to /tmp/data/msmarco/
```

### Optional — BEIR in-domain data

```bash
python data/build_beir_training_data.py --dataset scifact
python data/build_beir_training_data.py --dataset nfcorpus
python data/build_beir_training_data.py --dataset fiqa
```

---

## Training

### Train MRL baseline (required warm-start for BAM-PQ)

```bash
python scripts/train_baseline_mrl.py --config configs/mrl_e5large.yaml
```

### Train BAM-B (per-level scattered mask)

```bash
python scripts/train_bam.py \
    --config configs/bam_v4.yaml \
    --init_encoder /tmp/bam-mrl-ckpts/best
```

### Train BAM-PQ (per-query routing — main model)

```bash
python scripts/train_bam.py \
    --config configs/bam_pq.yaml \
    --init_encoder /tmp/bam-mrl-ckpts/best
```

### Train a specific backbone end-to-end

```bash
# Step 1: MRL warm-start
python scripts/train_baseline_mrl.py --config configs/mrl_bge_large.yaml

# Step 2: Select best MRL checkpoint
python scripts/find_best_epoch.py \
    --checkpoint_dir /tmp/bam-mrl-bge-ckpts/ \
    --config configs/mrl_bge_large.yaml

# Step 3: Train BAM-PQ
python scripts/train_bam.py \
    --config configs/bam_pq_bge_large.yaml \
    --init_encoder /tmp/bam-mrl-bge-ckpts/best
```

### Key training flags

| Flag | Description |
|------|-------------|
| `--config` | Path to YAML config file |
| `--init_encoder` | Path to warm-start checkpoint directory |
| `--freeze_encoder` | Keep encoder frozen throughout training |

---

## Evaluation

### Select best checkpoint (post-hoc, corpus-level)

> **Important:** In-batch validation NDCG during training is an approximation only. Always use `find_best_epoch.py` for true corpus-level checkpoint selection.

```bash
python scripts/find_best_epoch.py \
    --checkpoint_dir /tmp/bam-pq-ckpts/ \
    --config configs/bam_pq.yaml
```

Scans all `epoch_N/` checkpoints, evaluates each against the full corpus, selects by BSR, and symlinks the best to `checkpoint_dir/best`.

### Full evaluation (BAM-PQ vs. MRL)

```bash
python scripts/eval_bam.py \
    --config configs/bam_pq.yaml \
    --checkpoint /tmp/bam-pq-ckpts/best \
    --baseline /tmp/bam-mrl-ckpts/best
```

**Output metrics:** R@1, R@5, R@10, R@50, NDCG@10 — overall and stratified by Bloom level.

### Fair comparison (scatter vs. prefix-truncated MRL at same budget)

```bash
python scripts/eval_fair_comparison.py \
    --config configs/bam_pq.yaml \
    --bam_b_checkpoint /tmp/bam-b-ckpts/best \
    --mrl_checkpoint /tmp/bam-mrl-ckpts/best
```

### BEIR evaluation

```bash
python scripts/eval_beir.py \
    --config configs/bam_pq.yaml \
    --checkpoint /tmp/bam-pq-ckpts/best \
    --dataset scifact
```

### Analysis scripts

```bash
# Dimension specialization per Bloom level
python scripts/run_diagnostics.py --config configs/bam_pq.yaml

# Linear probing of embedding dimensions
python scripts/run_probing.py --config configs/bam_pq.yaml

# Efficiency curves (Recall@10 vs. active dims)
python scripts/run_efficiency.py --config configs/bam_pq.yaml

# Full ablation suite
python scripts/run_ablations.py --config configs/bam_pq.yaml

# Generate paper figures
python scripts/generate_figures.py
```

---

## Pipeline Scripts

For full multi-backbone experiments, use the pipeline scripts rather than individual commands.

### Master pipeline (`bam_council_pipeline.sh`)

Orchestrates the full training and evaluation pipeline across all 13 backbones and 5 datasets.

**Pipeline steps (in order):**

| Step | Description |
|------|-------------|
| `train_classifier` | Train Bloom council classifiers |
| `build` | Build datasets and hard negatives |
| `annotate` | Bloom-annotate all queries |
| `train_mrl` | Train shared MRL (e5large, bge) |
| `find_mrl` | Best epoch for shared MRL |
| `train_bam_b` | Train BAM-B (e5large only) |
| `find_bam_b` | Best epoch for BAM-B |
| `eval_pretrained` | Zero-shot pretrained evaluation |
| `train_standard_ft` | Train standard fine-tuning baselines |
| `find_standard_ft` | Best epoch for standard FT |
| `eval_standard_ft` | Evaluate standard FT |
| `train_mrl_bk` | Train per-backbone MRL (BAM-PQ warm-start) |
| `find_mrl_bk` | Best epoch for per-backbone MRL |
| `train_bam_pq` | Train BAM-PQ for all backbones |
| `find_bam_pq` | Best epoch for BAM-PQ |
| `eval` / `eval_bam_pq` | Full corpus evaluation |
| `fair_cmp` / `fair_cmp_pq` | Fair comparison at matched budgets |
| `eff_curves` | Efficiency curves |

**Usage:**

```bash
# Run full pipeline
./bam_council_pipeline.sh

# Start from a specific step (skip completed steps)
./bam_council_pipeline.sh --from train_bam_pq

# Run on specific datasets only
./bam_council_pipeline.sh --datasets "educational msmarco"

# Run for specific backbones only
./bam_council_pipeline.sh --backbone "e5large bge"

# Force retrain from scratch
./bam_council_pipeline.sh --force
```

### Backbone group scripts

Split into four groups to run in parallel on separate GPUs:

| Script | Backbones | Start Step | Notes |
|--------|-----------|------------|-------|
| `pipeline_backbones_1.sh` | e5-large, bge-large | `train_mrl_bk` | Retrieval-purpose encoders |
| `pipeline_backbones_2.sh` | qwen06b, qwen4b | `train_mrl_bk` | Decoder models |
| `pipeline_backbones_3.sh` | arctic, roberta, phi3mini | `eval_pretrained` | Mixed encoders |
| `pipeline_backbones_4.sh` | qwen8b, llama1b, llama3b | `train_mrl_bk` | Large models; llama1b/3b reverse two-stage |

```bash
# Run all four groups in parallel (requires 4 GPU nodes)
./pipeline_backbones_1.sh &
./pipeline_backbones_2.sh &
./pipeline_backbones_3.sh &
./pipeline_backbones_4.sh &
wait

# Resume a single group from a specific step
./pipeline_backbones_4.sh --from train_bam_pq
```

### Key environment variables

```bash
BACKBONES_TO_RUN="e5large bge"        # Override which backbones run
REUSE_TRAINED_MODELS=1                # Skip training if checkpoints exist
BEIR_DATA_ROOT=/path/to/beir          # Custom BEIR data location
```

---

## Configuration Reference

### Model section

```yaml
model:
  backbone: "intfloat/e5-large-v2"    # HuggingFace model ID
  backbone_type: "standard"           # standard | qwen | llm2vec | gritlm
  embedding_dim: 768                  # Output dimension (after projection)
  pooling: "cls"                      # cls | mean | last_token | max
  normalize_embeddings: true
  mrl_dims: [64, 128, 256, 512, 768]  # MRL checkpoint dimensions
  use_mask_routing: true              # Enable BAM routing head
  use_query_delta: true               # Enable per-query residual (BAM-PQ)
  query_delta_hidden: 256             # Hidden dim of residual MLP
  torch_dtype: "bfloat16"
```

### Training section (key fields)

```yaml
training:
  num_epochs: 20
  batch_size: 16
  gradient_accumulation_steps: 8     # Effective batch size = 128
  bf16: true
  optimizer:
    lr: 2.0e-4                       # Routing head LR
    encoder_lr: 2.0e-5               # Encoder LR (when unfrozen)
    encoder_finetune_lr: 1.0e-6      # Low LR after reverse two-stage unfreeze
  encoder_unfreeze_after_epochs: 8   # Reverse two-stage: unfreeze at this epoch
  checkpoint_dir: "/tmp/bam-pq-ckpts/"
  eval_every_n_steps: 500
  save_every_n_steps: 1000
```

### backbone_type guide

| `backbone_type` | Use for | Pooling | Notes |
|-----------------|---------|---------|-------|
| `standard` | BERT, RoBERTa, E5, BGE, Arctic | `cls` or `mean` | Standard AutoModel loading |
| `qwen` | Qwen, LLaMA, Phi-3-mini | `last_token` | Sets `pad_token=eos_token` automatically |
| `llm2vec` | LLM2Vec-Mistral-7B, LLM2Vec-LLaMA-8B | `mean` | Bidirectional via PEFT; requires `pip install llm2vec` |
| `gritlm` | GritLM-7B | `mean` | Requires `pip install gritlm` |

---

## Model Variants

### BAM-B vs. BAM-PQ

**BAM-B** (`use_query_delta: false`): One binary mask per Bloom level. All queries at the same cognitive level share the same mask. Simpler, faster inference, still substantially outperforms MRL.

**BAM-PQ** (`use_query_delta: true`): Per-query binary mask. The BAM-B mask is the starting point; a zero-initialized 2-layer MLP refines it based on the query's CLS embedding. The learned scalar `α` (initialized ≈ 0.05) controls how much the per-query residual contributes beyond the Bloom prior.

### Initialization requirement

BAM-PQ **must** be warm-started from a backbone-matched MRL checkpoint:

```
Pretrained weights → MRL fine-tuning → best MRL checkpoint
                                              ↓
                                   BAM-PQ initialization
```

Do **not** initialize from raw pretrained weights. The MRL warm-start provides a structured embedding space for the mask head to select from.

### Inference

1. Bloom council predicts cognitive level `b` from query text
2. Encoder computes CLS embedding `z`
3. `BloomQueryMaskHead` produces binary mask `m`
4. Masked query `q̃ = normalize(z ⊙ m)` is scored against full-dim FAISS index
5. Zero-masked dimensions contribute nothing to the dot product — savings are on the **query side only**; the corpus index does not need to be rebuilt

---

## Supported Backbones

| Key | Model | Native Dim | Output Dim | Type | Notes |
|-----|-------|-----------|------------|------|-------|
| `e5large` | intfloat/e5-large-v2 | 1024 | 768 | standard | Primary backbone |
| `bge` | BAAI/bge-large-en-v1.5 | 1024 | 768 | standard | |
| `arctic` | Snowflake/snowflake-arctic-embed-l | 1024 | 768 | standard | |
| `roberta` | FacebookAI/roberta-large | 1024 | 768 | standard | General-purpose encoder |
| `qwen06b` | Qwen/Qwen2.5-0.5B | 896 | 768 | qwen | Smallest decoder |
| `qwen4b` | Qwen/Qwen2.5-3B | 2048 | 768 | qwen | |
| `qwen8b` | Qwen/Qwen2.5-7B | 3584 | 768 | qwen | Frozen throughout; batch=1 |
| `llama1b` | meta-llama/Llama-3.2-1B | 2048 | 768 | qwen | Reverse two-stage |
| `llama3b` | meta-llama/Llama-3.2-3B | 3072 | 768 | qwen | Reverse two-stage |
| `llm2vec` | McGill-NLP/LLM2Vec-Mistral-7B | 4096 | 768 | llm2vec | |
| `llama8b` | McGill-NLP/LLM2Vec-Meta-Llama-3-8B | 4096 | 768 | llm2vec | |
| `gritlm` | GritLM/GritLM-7B | 4096 | 768 | gritlm | |
| `phi3mini` | microsoft/phi-3-mini-4k-instruct | 3072 | 768 | qwen | |

### Memory and batch size guide

| Backbone | GPU Memory | Recommended Batch | Grad Accumulation |
|----------|------------|-------------------|-------------------|
| e5large, bge, arctic, roberta | ~3 GB | 32 | 4 |
| phi3mini, llama3b | ~6 GB | 8 | 16 |
| qwen4b, llama1b | ~8 GB | 16 | 8 |
| qwen8b, llm2vec, llama8b, gritlm | ~16 GB | 1 | 128 |

All configurations maintain an effective batch size of 128.

---

## Results

### Educational benchmark (3,296 test queries, 40,640 passages, e5-large-v2)

| Model | R@10 | Dims | Savings |
|-------|------|------|---------|
| Standard FT | 0.441 | 768 | 0% |
| MRL | 0.467 | 768 | 0% |
| BAM-B | 0.523 | 446 | 42% |
| **BAM-PQ** | **0.530** | **473** | **38%** |

### Per-Bloom-level breakdown (e5-large-v2)

| Level | N | MRL | BAM-PQ | Dims | Gain |
|-------|---|-----|--------|------|------|
| L1 Remember | 1,091 | 0.567 | **0.635** | 468 | +0.068 |
| L2 Understand | 622 | 0.505 | **0.547** | 447 | +0.042 |
| L3 Apply | 640 | 0.319 | **0.403** | 474 | +0.084 |
| L4 Analyze | 384 | 0.487 | **0.537** | 486 | +0.050 |
| L5 Evaluate | 247 | 0.324 | **0.385** | 495 | +0.061 |
| L6 Create | 312 | 0.436 | **0.494** | 510 | +0.058 |
| **Overall** | **3,296** | **0.467** | **0.530** | **473** | **+0.063** |

---

## Development Notes

### Critical design decisions

**Do not zero-initialize the router weight matrix.** `dim_head[2].bias = 0` for a midpoint start. Zeroing `dim_head[2].weight` blocks `∂logit/∂hidden = weight = 0`, starving the Bloom embeddings of gradients entirely and locking all levels at ~448 dims forever.

**Gradient checkpointing at encoder unfreeze.** When the encoder unfreezes at epoch 8, backward pass must store activations for all encoder layers — for backbones ≥4B params this causes OOM on 40GB GPUs. `bam_trainer.py` automatically calls `gradient_checkpointing_enable()` at the unfreeze step.

**pad_token for decoder backbones.** All causal LM backbones (Qwen, LLaMA, Phi-3-mini) have no pad token by default. Always load tokenizers via `MRLEncoder._load_tokenizer()` rather than `AutoTokenizer.from_pretrained()` directly — the former sets `pad_token = eos_token` automatically.

**Bloom labels are 0-indexed internally.** Labels are 1-indexed in config YAML files and raw data (1=Remember, 6=Create) but converted to 0-indexed (0–5) at load time. All model and loss code uses 0-indexed labels.

**Val metrics during training are approximations.** In-batch validation NDCG only evaluates against the batch's own passages. Always run `find_best_epoch.py` against the full corpus for true checkpoint selection.

**Large model find_best shortcut.** Loading 11 × 16GB checkpoints sequentially causes OOM for qwen8b, qwen4b, llama8b. The pipeline automatically symlinks the `final` checkpoint as best for these models.

### Adding a new backbone

1. Create 6 configs: `mrl_{name}.yaml`, `standard_ft_{name}.yaml`, `bam_pq_{name}.yaml` and their `*_msmarco.yaml` variants
2. Register in `bam_council_pipeline.sh`: add to all six registry maps (`BACKBONE_MRL_EDU_CFG`, `BACKBONE_MRL_MSMARCO_CFG`, `BACKBONE_STANDARD_FT_EDU_CFG`, `BACKBONE_STANDARD_FT_MSMARCO_CFG`, `BACKBONE_EDU_CFG`, `BACKBONE_MSMARCO_CFG`) plus `BACKBONE_USE_MRL_INIT` and `BACKBONES_TO_RUN`
3. Add to the appropriate `pipeline_backbones_N.sh`
4. For decoder models use `backbone_type: "qwen"` and `pooling: "last_token"`; for standard encoders use `backbone_type: "standard"` and `pooling: "mean"` or `"cls"`

### Bloom cache management

Bloom predictions are cached alongside each JSONL as `{data_file}.bloom_cache.json`. If you re-train the council or change annotation, delete all caches:

```bash
find ./data -name "*.bloom_cache.json" -delete
```

### Experiment tracking

All runs log to Weights & Biases under project `bam-emnlp`. To disable:

```yaml
logging:
  use_wandb: false
```

---

## Citation

```bibtex
@inproceedings{bindal2026bampq,
  title     = {{BAM-PQ}: Bloom-Aligned Matryoshka Embeddings with Per-Query Dimension Routing},
  author    = {Bindal, Uday},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods
               in Natural Language Processing (EMNLP)},
  year      = {2026}
}
```

---

## License

This project is released for research purposes.
