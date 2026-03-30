# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QA-MRL is a research project on query-adaptive Matryoshka Representation Learning for educational information retrieval. It learns per-query dimension routing — selecting which embedding dimensions to use based on query characteristics (specifically Bloom's cognitive taxonomy level).

**Three model variants:**
- **Baseline MRL** — standard Matryoshka encoder with multi-resolution loss
- **QA-MRL** — adds a query-adaptive router that learns soft/hard dimension masks
- **BAM (v3)** — Bloom-Aligned MRL: classifies query Bloom level → maps to optimal truncation dimension via a global lookup table

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Data preparation (run in order):**
```bash
python data/build_real_data.py
python data/annotate_bloom_pretrained.py
python data/curriculum_negatives.py
```

**Training:**
```bash
python scripts/train_bam.py --config configs/bam.yaml
python scripts/train_qa_mrl.py --config configs/default.yaml
python scripts/train_baseline_mrl.py --config configs/neurips.yaml
```

**Evaluation:**
```bash
python scripts/eval_bam.py --config configs/bam.yaml --checkpoint /path/to/ckpt --baseline /path/to/baseline
python scripts/run_evaluation.py --config configs/neurips.yaml
python scripts/eval_beir.py --config configs/neurips.yaml
```

**Analysis:**
```bash
python scripts/run_diagnostics.py --config configs/default.yaml
python scripts/run_ablations.py --config configs/default.yaml
python scripts/run_efficiency.py
python scripts/generate_figures.py
```

## Architecture

### Data flow
```
Query → Transformer Encoder (BAAI/bge-base-en-v1.5, 768-dim)
      → BAM: Bloom classifier → global dim mapping → truncated embedding
      → QA-MRL: soft/group router → dimension mask → masked embedding
      → FAISS index → retrieved documents
```

### Key design decisions (BAM v3)
- **Bloom is query-only**: documents have no Bloom labels (prior approach of labeling documents caused contrastive loss misalignment)
- **Global dim mapping**: a Bloom-level → truncation-dimension lookup table replaces the per-query MLP that collapsed to fixed 384
- **No early stopping**: trains all 15 epochs, saves every checkpoint for post-hoc selection
- **Curriculum**: temperature annealing (warm→hard) rather than Bloom-distance negative weighting

### Loss stack (BAM)
- `BloomContrastiveLoss` — InfoNCE at full dimensions
- `BloomAlignedMRLLoss` — multi-resolution loss + per-truncation Bloom classifiers
- `TruncationPolicyLoss` — entropy bonus to prevent policy collapse
- `BAMCombinedLoss` — weighted aggregation

### Core modules
| File | Role |
|------|------|
| `models/encoder.py` | `MRLEncoder` — HuggingFace transformer wrapper, multi-resolution embeddings |
| `models/bam.py` | `BloomAlignedMRL` v3 — query Bloom classifier + dim mapping |
| `models/qa_mrl.py` | `QAMRL` — full model with query/document routers |
| `models/router.py` | `SoftRouter`, `GroupRouter` — dimension selection mechanisms |
| `models/bam_losses.py` | BAM-specific loss functions |
| `training/bam_trainer.py` | BAM training loop (no early stopping) |
| `evaluation/evaluator.py` | FAISS retrieval + bootstrap confidence intervals |
| `evaluation/bloom_stratified.py` | Metrics stratified by Bloom cognitive level |
| `data/dataset.py` | `EducationalRetrievalDataset` with hard negative mining |

## Configuration

Configs are YAML files in `configs/`. Key variants:
- `default.yaml` — standard QA-MRL
- `bam.yaml` — Bloom-aligned routing
- `neurips.yaml` — multi-backbone + BEIR evaluation
- `real_data.yaml` — educational corpora (SciQ, QASC, OpenBookQA)

Key hyperparameters:
- Backbone: `BAAI/bge-base-en-v1.5` (768-dim)
- MRL truncation dims: `[64, 128, 256, 384, 512, 768]`
- Training: 15 epochs, batch 16, gradient accumulation 8, AdamW + cosine LR + 10% warmup
- Loss weights: contrastive=1.0, MRL=0.3–0.5, specialization=0.05, Bloom=0.05–0.3

Data paths (configurable in YAML):
- Train/Val/Test: `/tmp/data/real/{train,val,test}.jsonl`
- Corpus: `/tmp/data/real/corpus.jsonl`
- Checkpoints: `/tmp/bam-ckpts/`

## GitHub Actions

- `.github/workflows/claude.yml` — Claude Code integration triggered by `@claude` mentions in issues/PRs
- `.github/workflows/claude-code-review.yml` — Automated code review
