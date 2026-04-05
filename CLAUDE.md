# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QA-MRL is a research project on query-adaptive Matryoshka Representation Learning for educational information retrieval. It learns per-query dimension routing — selecting which embedding dimensions to use based on query characteristics (specifically Bloom's cognitive taxonomy level).

**Three model variants:**
- **Baseline MRL** — standard Matryoshka encoder with multi-resolution loss
- **QA-MRL** — adds a query-adaptive router that learns soft/hard dimension masks
- **BAM (v4)** — Bloom-Aligned MRL: `BloomDimRouter` (learned MLP per Bloom level) → prefix mask over 768 dims via straight-through estimator

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

**Post-hoc best checkpoint selection** (run after training, uses corpus-level metrics not in-batch):
```bash
python scripts/find_best_epoch.py --checkpoint_dir /tmp/bam-ckpts/ --config configs/bam.yaml
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
      → BAM: BloomDimRouter (learned MLP) → continuous dim → prefix binary mask (STE)
             masked_emb = normalize(full_emb * mask)   # zero dims ignored at retrieval
      → QA-MRL: soft/group router → dimension mask → masked embedding
      → FAISS index → retrieved documents

Documents: always encoded at full 768 dims; dot-product naturally ignores zero query dims.
```

### BAM v4 key design decisions
- **Bloom is query-only**: documents have no Bloom labels (labeling documents causes contrastive loss misalignment)
- **`BloomDimRouter`**: 6 independent learned embeddings → 2-layer MLP → sigmoid → continuous dim ∈ [128, 768]. STE: hard prefix mask in forward, soft sigmoid in backward. NOT a lookup table.
- **Zero-init trap**: `dim_head[2].bias=0` for midpoint start (~448 dims); `dim_head[2].weight` uses Kaiming default (do NOT zero — zeroing blocks ∂logit/∂hidden = weight = 0, starving bloom_emb of gradients entirely and locking all levels at 448 forever). `dim_head[0]` also uses Kaiming default.
- **No early stopping**: trains all 15 epochs, saves every epoch for post-hoc selection via `find_best_epoch.py`. In-batch val NDCG is an approximation only.
- **Temperature annealing**: cosine 0.1→0.02 for unfrozen encoder; fixed at `temp_end` when encoder is frozen (annealing with frozen encoder parks the router at equilibrium).
- **Efficiency loss gate**: `encoder_warmup_epochs` (default 5) — efficiency loss is zero while encoder builds quality, then activates.

### Loss stack (BAM, `bam_losses.py` v7)
- `BloomMaskedContrastiveLoss` — class-weighted (1/√freq) InfoNCE in masked subspace + difficulty weighting
- `BloomTwoFactorEfficiencyLoss` — per-class averaged (not per-sample) efficiency penalty with cognitive weights (1 − b/6)
- `RouterDiversityLoss` — maximizes mean pairwise distance between all 6 Bloom levels' dims (pairwise distance, not variance, to avoid zero-gradient at init)
- `MRLAnchorRegularizationLoss` — InfoNCE at MRL anchor dims weighted D/√d to keep encoder sharp
- `BAMCombinedLoss` — orchestrates all four; call `set_epoch()` at start of each epoch

### Core modules
| File | Role |
|------|------|
| `models/encoder.py` | `MRLEncoder` — HuggingFace transformer wrapper, multi-resolution embeddings |
| `models/bam.py` | `BloomAlignedMRL` v4 + `BloomDimRouter` — MLP routing with STE prefix mask |
| `models/bam_losses.py` | BAM-specific losses (see loss stack above) |
| `models/qa_mrl.py` | `QAMRL` — full model with query/document routers |
| `models/router.py` | `SoftRouter`, `GroupRouter` — dimension selection mechanisms |
| `training/bam_trainer.py` | BAM training loop; supports resume, two-stage via `freeze_encoder` config |
| `evaluation/evaluator.py` | FAISS retrieval + bootstrap confidence intervals |
| `evaluation/bloom_stratified.py` | Metrics stratified by Bloom cognitive level |
| `data/dataset.py` | `EducationalRetrievalDataset` with hard negative mining |

## Configuration

Configs are YAML files in `configs/`. Key variants:
- `default.yaml` — standard QA-MRL
- `bam.yaml` — Bloom-aligned routing (BAM v4)
- `neurips.yaml` — multi-backbone + BEIR evaluation
- `real_data.yaml` — educational corpora (SciQ, QASC, OpenBookQA)

Key hyperparameters:
- Backbone: `BAAI/bge-base-en-v1.5` (768-dim)
- MRL truncation dims: `[64, 128, 256, 384, 512, 768]`
- Training: 15 epochs, batch 16, gradient accumulation 8, AdamW + cosine LR + 10% warmup
- Router LR is 10× encoder LR (`router_lr: 2e-4` vs `encoder_lr: 2e-5`)
- `bloom_frequencies` injected at runtime by `train_bam.py` — not hardcoded in config

Data paths (configurable in YAML):
- Train/Val/Test: `/tmp/data/real/{train,val,test}.jsonl`
- Corpus: `/tmp/data/real/corpus.jsonl`
- Checkpoints: `/tmp/bam-ckpts/`

## GitHub Actions

- `.github/workflows/claude.yml` — Claude Code integration triggered by `@claude` mentions in issues/PRs
- `.github/workflows/claude-code-review.yml` — Automated code review

## Development notes

- Only read the files relevant to the change you're making — do not load the entire codebase as context.
- Val metrics during training (`inbatch_best` checkpoint) are in-batch pairwise NDCG, not corpus-level retrieval. Always use `find_best_epoch.py` for the true best checkpoint.
- Bloom labels are 0-indexed internally (0=Remember … 5=Create) but 1-indexed in config YAML and data files.
- Push all the changes to git
