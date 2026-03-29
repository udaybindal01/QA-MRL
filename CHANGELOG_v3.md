# BAM v3 Changelog: Query-Only Bloom

## Core Conceptual Fix

**Bloom level is a property of the query, not the document.**

A document is just content. Its cognitive demand is determined by *what the query asks you to do with it*. The same passage can serve Remember, Understand, Apply, Analyze, Evaluate, and Create queries.

v2 assigned Bloom labels to corpus documents based on source dataset (SciQ → Bloom 2, QASC facts → Bloom 1, QASC combined → Bloom 4). This was:
1. Arbitrary and non-portable to other domains
2. Conceptually wrong (docs don't have cognitive levels)
3. Causing the contrastive loss to penalize correct retrievals when doc "Bloom" ≠ query Bloom

v3 removes ALL document Bloom labels. Bloom enters the model ONLY through:
- Per-truncation query Bloom classifiers in the MRL loss
- Query Bloom labels in training pairs (unchanged)

---

## Files Changed (7 modified, 2 new)

### `models/bam_losses.py` — REWRITTEN
- `BloomConditionedContrastiveLoss` → `ContrastiveLoss`
  - Removed `w_i = 1 + λ * |bloom(query) - bloom(negative_i)|`
  - Standard InfoNCE with no Bloom weighting on negatives
- `BloomAlignedMRLLoss`
  - Per-truncation Bloom classifiers now explicitly query-only
  - Added dropout (0.1) in classifier heads
  - Configurable `bloom_cls_weight` (default 0.3)
- `CurriculumScheduledLoss`
  - No longer ramps Bloom-distance weight (removed)
  - Now does temperature annealing: 0.1 (warm/easy) → 0.05 (target/hard)
- `TruncationPolicyLoss`
  - Removed `bloom_align = MSE(dim_norm, bloom_norm)` (used doc Bloom)
  - Added entropy bonus (`entropy_weight=0.05`) to prevent policy collapse
- `BAMCombinedLoss`
  - Updated to v3 components
  - `**kwargs` throughout to gracefully ignore legacy fields

### `training/bam_trainer.py` — REWRITTEN
- Removed `negative_blooms` from `train_step`
- **Removed early stopping entirely**
  - Val metric (in-batch NDCG) was misleading — caused stopping at epoch 7
    when corpus-level FAISS metrics were still improving
  - Now trains all `num_epochs` (15) and saves every epoch
  - Post-hoc evaluation finds the true best checkpoint
- Added policy entropy to progress bar
- Tracks `curriculum_temperature` instead of `bloom_weight`

### `data/build_real_data.py` — REWRITTEN
- **Documents no longer have `bloom_level` field**
  - Removed `"bloom_level": 2` from SciQ passages
  - Removed `"bloom_level": 1` from QASC/OBQA facts
  - Removed `"bloom_level": 4` from QASC combined facts
- **Fixed OpenBookQA loading**
  - Now tries `"additional"` config first (has `fact1`), falls back to `"main"`
  - Handles missing fields gracefully instead of silently producing 0 passages
- **Negative mining is topic-only** (no Bloom-distance tiers)
  - Tier 1: Same topic, different passage
  - Tier 2: Same subject, different topic
  - Tier 3: Random

### `data/curriculum_negatives.py` — REWRITTEN
- **Removed Bloom-based tier 1 and tier 2**
  - Old: "same topic, adjacent Bloom" / "same topic, distant Bloom"
  - These were nonsensical — docs don't have Bloom levels
- **New topic-based tiers:**
  - Tier 1 (hardest): Same topic, high keyword overlap with query
  - Tier 2 (hard): Same topic, any passage
  - Tier 3 (medium): Same subject, different topic
  - Tier 4 (easy): Random
- Returns `List[int]` (indices only), no `negative_blooms`

### `data/dataset.py` — MODIFIED
- `EducationalRetrievalDataset.__getitem__`: removed `negative_blooms` from batch
- `CorpusDataset.__getitem__`: `bloom_level` now uses `.get()` with default 0
  for backward compatibility with v2 data that still has the field

### `evaluation/evaluator.py` — REWRITTEN
- **Removed `bloom_aligned_recall@10`** metric
  - This checked if retrieved documents had matching Bloom labels to the query
  - Invalid in v3 (docs have no Bloom labels) and was conceptually wrong in v2
- **Added bootstrap confidence intervals** for all Bloom-stratified R@10 metrics
  - `bloom_{name}_recall@10_ci_lo`, `bloom_{name}_recall@10_ci_hi`
  - Reports sample size `bloom_{name}_n` for each level
  - Critical for levels with small n (Analyze n=9, Create n=20)

### `configs/bam.yaml` — MODIFIED
- Added `bloom_cls_weight: 0.3`
- Added `entropy_weight: 0.05`
- Removed `bloom_negative_weight: 1.0`

### `scripts/eval_bam.py` — NEW
- Proper evaluation script for BAM (was missing from repo)
- Uses `BloomAlignedMRL` (not old `QAMRL`)
- Prints bootstrap CIs for Bloom-stratified results
- Supports `--output_dir` for per-epoch evaluation

### `scripts/run_bam_v3_pipeline.sh` — NEW
- End-to-end pipeline: rebuild data → mine negatives → train baseline →
  train BAM → evaluate all epochs
- Handles HF cache, /tmp paths, config updates

---

## Training Pipeline (v3)

```bash
# 1. Rebuild data (no doc Bloom)
python data/build_real_data.py --config configs/bam.yaml --output_dir /tmp/data/real

# 2. Mine curriculum negatives (topic-based)
python data/curriculum_negatives.py \
    --pairs /tmp/data/real/train.jsonl \
    --corpus /tmp/data/real/corpus.jsonl \
    --output /tmp/data/real/train_curriculum.jsonl \
    --stage 0.7 --num_neg 3

# 3. Update config to use curriculum data
sed -i 's|train.jsonl|train_curriculum.jsonl|' configs/bam.yaml

# 4. Train MRL baseline
python scripts/train_baseline_mrl.py --config configs/bam.yaml

# 5. Train BAM v3 (all 15 epochs, no early stopping)
python scripts/train_bam.py --config configs/bam.yaml \
    --init_encoder /tmp/bam-ckpts/mrl_baseline_best/

# 6. Evaluate each epoch to find true best
for e in $(seq 0 14); do
    python scripts/eval_bam.py --config configs/bam.yaml \
        --checkpoint /tmp/bam-ckpts/epoch_$e/ \
        --baseline /tmp/bam-ckpts/mrl_baseline_best/ \
        --output_dir results/bam_v3_eval/epoch_$e/
done
```

---

## Paper Framing (updated)

> "We observe that cognitive complexity is a property of the query's intent,
> not the document's content. A single educational passage can serve queries
> spanning all six Bloom levels — from factual recall to creative synthesis.
> BAM's per-truncation Bloom classifiers force the query encoder to organize
> embedding dimensions by cognitive demand, while documents remain in a
> shared semantic space. This design is domain-portable: any retrieval
> corpus can be used without cognitive-level annotation, requiring only
> query-side Bloom classification (achievable via lightweight pattern matching
> or an LLM classifier)."
