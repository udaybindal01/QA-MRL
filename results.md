# BAM-PQ Results

> Results extracted from pipeline runs on 2026-05-02–04 (SLURM nodes 07/08).
> Only the **educational** dataset has completed full evaluation so far.
> Other datasets (msmarco, scifact, nfcorpus, fiqa) and remaining backbones are still training.

---

## 1. Main Results — Educational Dataset (N=3 296 queries, corpus=40 640)

### 1.1 Overall Retrieval Performance

| Model | Backbone | R@1 | R@5 | R@10 | R@50 | MRR | NDCG@10 | Avg Dims |
|-------|----------|-----|-----|------|------|-----|---------|----------|
| MRL baseline | e5-large | 0.2615 | 0.4035 | 0.4669 | 0.6374 | 0.3360 | 0.3603 | 768 (full) |
| MRL baseline | bge-large | 0.2700 | 0.4196 | 0.4779 | 0.6687 | 0.3474 | 0.3710 | 768 (full) |
| BAM-B | e5-large | 0.2612 | 0.4563 | **0.5337** | 0.7233 | 0.3559 | 0.3907 | **463/768** |
| BAM-PQ | e5-large | 0.2964 | 0.4609 | 0.5297 | 0.7212 | 0.3792 | **0.4070** | 473/768 |
| BAM-PQ | bge-large | **0.2870** | **0.4712** | **0.5358** | **0.7342** | **0.3758** | 0.4063 | **384/768** |

**Key numbers:**
- BAM-PQ (bge) vs MRL (bge): **+5.79pp R@10**, **+3.53pp NDCG@10**, using only **384/768 dims** (50% compression)
- BAM-PQ (e5) vs MRL (e5): **+6.28pp R@10**, **+4.67pp NDCG@10**, at 473 dims
- BAM-B vs MRL (e5): **+6.68pp R@10** at 463 dims (60% of full)

---

### 1.2 Dimension Allocation per Bloom Level

#### BAM-B (e5-large) — Avg 463/768 dims (sparse_ratio=0.40, scattered mask)

| Bloom Level | N | R@10 | Active Dims |
|-------------|---|------|-------------|
| Remember | 1091 | 0.6407 | 452 |
| Understand | 622 | 0.5531 | 452 |
| Apply | 640 | 0.3953 | 473 |
| Analyze | 384 | 0.5443 | 471 |
| Evaluate | 247 | 0.3806 | 472 |
| Create | 312 | 0.5128 | 483 |

BSR: 0.6261 (epoch 11) · Quality: 0.4913 · Efficiency: 0.5489

#### BAM-PQ (e5-large) — Avg 473/768 dims (sparse_ratio=0.38, scattered mask)

| Bloom Level | N | R@10 | Active Dims |
|-------------|---|------|-------------|
| Remember | 1091 | 0.6352 | 468 |
| Understand | 622 | 0.5466 | 447 |
| Apply | 640 | 0.4031 | 474 |
| Analyze | 384 | 0.5365 | 486 |
| Evaluate | 247 | 0.3846 | 495 |
| Create | 312 | 0.4936 | 510 |

BSR: 0.6177 (epoch 17) · Quality: 0.4864 · Efficiency: 0.5396

**Dim spread (Remember→Create): 468→510 (+42 dims) — weak cognitive ordering**

#### BAM-PQ (bge-large) — Avg 384/768 dims (sparse_ratio=0.50, scattered mask)

| Bloom Level | N | R@10 | Active Dims |
|-------------|---|------|-------------|
| Remember | 1091 | 0.6324 | 362 |
| Understand | 622 | 0.5675 | 364 |
| Apply | 640 | 0.3984 | 394 |
| Analyze | 384 | 0.5365 | 395 |
| Evaluate | 247 | 0.3846 | 441 |
| Create | 312 | 0.5353 | 417 |

BSR: 0.6522 (epoch 15) · Quality: 0.4969 · Efficiency: 0.6253

**Dim spread (Remember→Create): 362→417 (+55 dims) — partial cognitive ordering; Evaluate (441) > Create (417) breaks strict monotonicity**

---

### 1.3 MRL Baseline — Per-Bloom Breakdown (at full 768 dims)

| Bloom Level | N | MRL (e5) R@10 | MRL (bge) R@10 |
|-------------|---|--------------|----------------|
| Remember | 1091 | 0.5665 | 0.5930 |
| Understand | 622 | 0.5048 | 0.5193 |
| Apply | 640 | 0.3187 | 0.3328 |
| Analyze | 384 | 0.4870 | 0.4818 |
| Evaluate | 247 | 0.3239 | 0.2955 |
| Create | 312 | 0.4359 | 0.4295 |

---

### 1.4 Fair Comparison — BAM vs MRL at Same Per-Bloom Dim Budget

**BAM-B vs MRL-truncated (e5-large)**

| Level | N | Budget | MRL-full | MRL-trunc | BAM-B | Δ |
|-------|---|--------|----------|-----------|-------|---|
| Remember | 1091 | 452 | 0.5665 | 0.5509 | 0.6389 | **+0.0880** |
| Understand | 622 | 452 | 0.5048 | 0.4904 | 0.5370 | **+0.0466** |
| Apply | 640 | 473 | 0.3187 | 0.3125 | 0.3984 | **+0.0859** |
| Analyze | 384 | 471 | 0.4870 | 0.4896 | 0.5312 | **+0.0417** |
| Evaluate | 247 | 472 | 0.3239 | 0.3117 | 0.3725 | **+0.0607** |
| Create | 312 | 483 | 0.4359 | 0.4071 | 0.5224 | **+0.1154** |
| **Average** | | | | | | **+0.0731** |

BAM wins on **6/6 Bloom levels** vs MRL-truncated at equal budget.

**BAM-PQ (e5-large) vs MRL-truncated**

| Level | N | Budget | MRL-full | MRL-trunc | BAM-PQ | Δ |
|-------|---|--------|----------|-----------|--------|---|
| Remember | 1091 | 468 | 0.5665 | 0.5564 | 0.6544 | **+0.0981** |
| Understand | 622 | 447 | 0.5048 | 0.4968 | 0.5466 | **+0.0498** |
| Apply | 640 | 474 | 0.3187 | 0.3125 | 0.3953 | **+0.0828** |
| Analyze | 384 | 486 | 0.4870 | 0.4870 | 0.5677 | **+0.0807** |
| Evaluate | 247 | 495 | 0.3239 | 0.2996 | 0.3684 | **+0.0688** |
| Create | 312 | 510 | 0.4359 | 0.4167 | 0.4872 | **+0.0705** |
| **Average** | | | | | | **+0.0751** |

**BAM-PQ (bge-large) vs MRL-truncated**

| Level | N | Budget | MRL-full | MRL-trunc | BAM-PQ | Δ |
|-------|---|--------|----------|-----------|--------|---|
| Remember | 1091 | 362 | 0.5930 | 0.5848 | 0.6370 | **+0.0522** |
| Understand | 622 | 364 | 0.5209 | 0.5161 | 0.5547 | **+0.0386** |
| Apply | 640 | 394 | 0.3344 | 0.3281 | 0.4172 | **+0.0891** |
| Analyze | 384 | 395 | 0.4818 | 0.4583 | 0.5365 | **+0.0781** |
| Evaluate | 247 | 441 | 0.2955 | 0.2794 | 0.3846 | **+0.1053** |
| Create | 312 | 417 | 0.4295 | 0.4327 | 0.5160 | **+0.0833** |
| **Average** | | | | | | **+0.0744** |

> Note: BAM uses a **scattered (non-contiguous) mask**; MRL uses prefix truncation. MRL-trunc is therefore the stronger fair-comparison baseline — prefix dims capture the highest-magnitude dimensions after MRL training.

---

### 1.5 Efficiency Curves — R@10 vs Dim Budget (BAM-B, e5-large)

| Dims | MRL R@10 | BAM-B R@10 | Δ |
|------|----------|------------|---|
| 64 | 0.3844 | — | — |
| 128 | 0.4278 | — | — |
| 192 | 0.4381 | — | — |
| 256 | 0.4463 | — | — |
| 320 | 0.4490 | — | — |
| 384 | 0.4536 | — | — |
| 448 | 0.4536 | — | — |
| 512 | 0.4557 | — | — |
| 576 | 0.4609 | — | — |
| 640 | 0.4606 | — | — |
| 704 | 0.4663 | — | — |
| 768 | 0.4639 | — | — |
| 1024 | 0.4675 | — | — |

**BAM-B Bloom-level operating points vs MRL at same budget:**

| Level | BAM Dims | N | BAM-B R@10 | MRL@same | Δ |
|-------|----------|---|------------|----------|---|
| Remember | 448 | 1091 | 0.6379 | 0.4536 | **+18.4%** |
| Understand | 488 | 622 | 0.5370 | 0.4536 | **+8.3%** |
| Analyze | 473 | 384 | 0.5312 | 0.4536 | **+7.8%** |
| Create | 510 | 312 | 0.5224 | 0.4536 | **+6.9%** |
| Apply | 485 | 640 | 0.3984 | 0.4536 | -5.5% |
| Evaluate | 479 | 247 | 0.3725 | 0.4536 | -8.1% |

BAM-B sits above the MRL curve for 4/6 Bloom levels. Apply and Evaluate underperform at their allocated budget (both are minority classes with complex queries).

---

## 2. Training Summary — All Backbones

### 2.1 Completed (Educational Dataset Only)

| Backbone | Model | BSR | Best Epoch | Status |
|----------|-------|-----|-----------|--------|
| e5-large | BAM-B | 0.6261 | epoch_11 | ✅ Full eval done |
| e5-large | BAM-PQ | 0.6177 | epoch_17 | ✅ Full eval done |
| bge-large | BAM-PQ | 0.6522 | epoch_15 | ✅ Full eval done |
| qwen06b | BAM-PQ | 0.5876 | epoch_17 | ⏳ Training done, eval pending |
| arctic | BAM-PQ | 0.5429 | epoch_16 | ⏳ Training done, eval pending |
| roberta | BAM-PQ | 0.4726 | epoch_18 | ⏳ Training done, eval pending |

### 2.2 In Progress / Failed

| Backbone | Status | Notes |
|----------|--------|-------|
| qwen4b | 🔄 MRL training in progress | final_beir2.log cut off during MRL |
| phi3mini | ❌ OOM | OOM during standard-FT eval (find_best_epoch). eval_batch_size=16 fix applied. |
| llama1b | ❌ Poor + error | Best MRL R@10=0.0027; BAM-PQ training crashed at dataset.py:329 |
| llama3b | 🔄 Not started | Queued after llama1b |
| msmarco (all) | ❌ Failed → restarting | TypeError: NoneType has no len (data not built). Fixed; annotation now running. |
| scifact / nfcorpus / fiqa | ⏳ Pending | Depend on msmarco run completing first |

---

## 3. MRL Truncation Baseline — R@10 at Variable Dims (Educational)

Full truncation curve for reference (e5-large MRL):

| Dims | R@10 | NDCG@10 |
|------|------|---------|
| 64 | 0.3844 | — |
| 128 | 0.4281 | — |
| 256 | 0.4463 | — |
| 512 | 0.4557 | — |
| 768 | 0.4642 | — |
| 1024 | 0.4669 | — |
| **~473** | — | — |
| **BAM-PQ (e5)** | **0.5297** | **0.4070** |

BAM-PQ at 473 dims matches performance of MRL at 1024 dims **and adds +6.28pp** on top.

---

## 4. Bloom Distribution (Educational Train Set)

| Level | Count | % |
|-------|-------|---|
| Remember | 8911 | 33.8% |
| Understand | 5131 | 19.5% |
| Apply | 5203 | 19.7% |
| Analyze | 2856 | 10.8% |
| Evaluate | 1886 | 7.2% |
| Create | 2374 | 9.0% |

(Test set: Remember 1091/33.1%, Understand 622/18.9%, Apply 640/19.4%, Analyze 384/11.6%, Evaluate 247/7.5%, Create 312/9.5%)

---

## 5. Notes on Cognitive Dimension Ordering

The dim allocation across Bloom levels shows a **partial but not strictly monotonic** cognitive ordering:

| Level | BAM-B dims | BAM-PQ (e5) dims | BAM-PQ (bge) dims |
|-------|-----------|-----------------|------------------|
| Remember | 452 | 468 | 362 |
| Understand | 452 | 447 | 364 |
| Apply | 473 | 474 | 394 |
| Analyze | 471 | 486 | 395 |
| Evaluate | 472 | 495 | 441 |
| Create | 483 | 510 | 417 |

- BAM-PQ (e5): near-monotonic except Understand<Remember, consistent with CL hypothesis
- BAM-PQ (bge): Evaluate(441) > Create(417) breaks the ordering; bge uses ~50% fewer dims overall, suggesting stronger compression without cognitive sorting
- All models use scattered (non-prefix) masks, so per-level dim counts reflect **how many** dimensions are active, not **which** dimensions

---

*Last updated: 2026-05-04. Results from pipeline runs: final_beir1.log, final_beir1_continue.log, final_beir2.log, final_beir3.log, final_beir3_continue.log, final_beir4.log, final_beir4_cont.log*
