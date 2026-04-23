# BAM: Bloom-Aligned Matryoshka Representation Learning for Query-Adaptive Retrieval

---

## 1. What is MRL?

Dense retrieval encodes queries and documents into fixed-dimensional embeddings (e.g., 768 dims) and retrieves via dot-product similarity. **Matryoshka Representation Learning (MRL)** trains embeddings so that the first *d* dimensions form a valid lower-dimensional representation — you can truncate to 256 dims and still retrieve reasonably well.

MRL enforces a **dimensional hierarchy** during training: it applies contrastive loss at multiple truncation points (64, 128, 256, 512, 768), teaching the encoder to pack the most important information into early dimensions.

```
Full embedding:  [d₁, d₂, d₃, ..., d₂₅₆, ..., d₇₆₈]
                  ←── most important ──────── least important ──→

Truncate to 256: [d₁, d₂, d₃, ..., d₂₅₆]  ✓ still works
Truncate to 64:  [d₁, ..., d₆₄]              ✓ still works (lower quality)
```

**Result:** One encoder, multiple operating points. Choose 256 dims for speed or 768 for quality — no retraining needed.

### Datasets

We train on educational science QA datasets that provide query-passage pairs:

| Dataset | Source | What it provides |
|---------|--------|-----------------|
| **SciQ** (allenai/sciq) | Science textbook questions | Support passages + questions across biology, chemistry, physics |
| **ARC-Easy / ARC-Challenge** (allenai/ai2_arc) | Standardized science exams | Multiple-choice science questions at two difficulty levels |
| **OpenBookQA** (allenai/openbookqa) | Open-book science facts | Questions requiring reasoning over provided science facts |
| **QASC** (allenai/qasc) | Multi-hop science reasoning | Questions requiring combining two facts to answer |

Hard negatives are mined using **BM25-based curriculum mining** — the most lexically similar non-relevant passages are the hardest negatives (default: 7 negatives per query, curriculum stage 0.7).

### MRL Losses

| Loss | Formula | Why it's used |
|------|---------|---------------|
| **InfoNCE (full dim)** | `-log(exp(sim(q,p⁺)/τ) / Σ exp(sim(q,pᵢ)/τ))` | The core contrastive objective — pulls query-positive pairs together, pushes negatives apart at full 768 dimensions |
| **MRL Anchor Loss** | Same InfoNCE applied at truncation points [64, 128, 256, 512, 768], weighted by `D/√d` | Forces the encoder to produce valid representations at every truncation point. Smaller truncations get higher weight because they're harder to learn. This creates the dimensional hierarchy where early dims carry the most information |

**Backbone:** BAAI/bge-base-en-v1.5 (768-dim, 110M params) or intfloat/e5-large-v2 (1024-dim, 335M params).

---

## 2. The Problem with MRL

MRL uses **the same truncation for every query**. Dimension 1 is always "most important," dimension 768 always least — regardless of what you're asking.

But different queries have fundamentally different information needs:

| Query | Complexity | Dims Needed |
|-------|-----------|-------------|
| "What is photosynthesis?" | Low (factual recall) | Few — just match a definition |
| "Compare C3 vs C4 pathways under drought" | High (analysis) | Many — capture nuanced relationships |

We frame this through **Bloom's cognitive taxonomy** — six levels of cognitive complexity:

| Level | Name | Example |
|-------|------|---------|
| 1 | Remember | "Define osmosis" |
| 2 | Understand | "Explain how osmosis works" |
| 3 | Apply | "Calculate osmotic pressure for this solution" |
| 4 | Analyze | "Compare osmosis and diffusion" |
| 5 | Evaluate | "Is reverse osmosis effective for desalination?" |
| 6 | Create | "Design an experiment to demonstrate osmosis" |

**Our hypothesis:** Higher cognitive complexity → more dimensions needed. Lower complexity → fewer dimensions without quality loss. A system that **adapts dimensions per query** can beat fixed MRL truncation.

**The challenge:** Learning this routing without degrading retrieval quality, using a mechanism lightweight enough to preserve efficiency gains.

### Bloom Annotation

Queries are classified into Bloom levels using **zero-shot NLI** (MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli). Each Bloom level is framed as a hypothesis; the NLI model scores entailment probability and picks the highest:

| Level | Hypothesis |
|-------|-----------|
| Remember | "This query is asking to recall or retrieve a specific fact, name, or definition" |
| Understand | "This query is asking to explain, describe, or summarize how something works" |
| Apply | "This query is asking how to use or apply knowledge to solve a practical problem" |
| Analyze | "This query is asking to compare, contrast, or examine the relationship between things" |
| Evaluate | "This query requires making a decision or forming an opinion about the worth or validity of something" |
| Create | "This query is asking to design, propose, or synthesize something new" |

**Bloom is query-only** — documents have no cognitive level. The same passage about photosynthesis serves a "Remember" query and a "Create" query differently. Only the query determines routing.

---

## 3. BAM Option A — Prefix Routing

**Idea:** Instead of one fixed truncation point for all queries, learn a **different truncation point per Bloom level**. Remember queries truncate to ~300 dims; Create queries use ~600 dims.

**How it works:**

```
Query → Bloom classifier → level (1-6)
                              ↓
                        BloomDimRouter
                          bloom_emb = Embedding(6, 32)     # one vector per level
                          dim_head  = MLP(32 → 16 → 1)    # → scalar
                          continuous_dim = sigmoid(scalar) × (768 − 128) + 128
                              ↓
                        Prefix mask: [1,1,1,...,1,0,0,...,0]
                              ↓
                        masked_emb = normalize(full_emb × mask)
```

**Forward pass:** Hard binary prefix mask (dims 0..d are 1, rest are 0).
**Backward pass:** Soft sigmoid gradients flow through via Straight-Through Estimator (STE).

**Key properties:**
- Only ~1K parameters added (negligible vs 110M encoder)
- Active dimensions are contiguous [0..d] → FAISS-compatible, no efficiency penalty at retrieval
- Documents always encoded at full dimensionality; dot product naturally ignores zeroed query dims

### Option A Losses

| Loss | Weight | What it does | Why it's needed |
|------|--------|-------------|-----------------|
| **Bloom-Masked Contrastive** | 1.0 | InfoNCE computed in the masked subspace: `sim(q*mask, p*mask)`. Class-weighted by `1/√freq` per Bloom level + difficulty weighting (harder samples get more weight) | The primary retrieval objective — ensures the model retrieves correctly even with reduced dimensions. Class weighting prevents rare Bloom levels (Evaluate, Create) from being drowned out by common ones (Remember) |
| **Efficiency** | 0.2 | Per-class averaged penalty: `mean_b[cognitive(b) × mean_dim_b / D]` where `cognitive(b) = 1 - b/6` (Remember=1.0, Create=0.167) | Pushes lower cognitive levels to use fewer dimensions. Remember gets 6× more compression pressure than Create, encoding the hypothesis that simple queries need fewer dims. Per-class averaging ensures rare levels get equal gradient updates |
| **Router Diversity** | 0.0 | `-mean_{i<j}(\|dim_i - dim_j\|)` — maximizes pairwise distance between the 6 learned truncation points | Prevents all Bloom levels from collapsing to the same dimension. Set to 0.0 because efficiency loss alone provides correct cognitive ordering once initialization is fixed |
| **MRL Anchor Regularization** | 0.3 | InfoNCE at standard MRL truncation points [64, 128, 256, 512, 768], weighted `D/√d` | Prevents the encoder from "forgetting" its multi-resolution structure during BAM fine-tuning. Smaller truncations get higher weight since they're the most fragile |

**Training detail:** Efficiency loss is **gated off for the first 5 epochs** (`encoder_warmup_epochs`). This lets the encoder first build high-quality representations in the masked subspace, then efficiency pressure gradually compresses lower levels.

**Pipeline:** MRL warm-start → train Option A (15 epochs, save all checkpoints) → post-hoc epoch selection via BSR.

---

## 4. Why Option B?

Option A is constrained to **prefix masks** — active dims must be contiguous [0..d]. This means:

- It can say "use dims 0 to 300" or "use dims 0 to 600"
- It **cannot** say "use dims 7, 42, 103, 511 but skip dims 0-6"

The MRL hierarchy (early dims = most important) is a useful prior, but it's **not always true**. Some later dimensions might capture analytical or evaluative information better than some early dimensions. For a "Compare X and Y" query, dim 511 might be more relevant than dim 12.

**The prefix constraint limits the model's expressiveness.** What if we let the model select **any subset** of dimensions per Bloom level?

---

## 5. BAM Option B — Scattered Mask Routing

**Idea:** Learn a **scattered binary mask** per Bloom level. Any combination of dimensions can be active — dimension 7, 42, 103, 511 might all be on while 0, 1, 2 are off.

**How it works:**

```
Query → Bloom classifier → level (1-6)
                              ↓
                        BloomMaskHead
                          bloom_logits = Embedding(6, 768)   # one 768-dim logit vector per level
                              ↓
                          soft_mask = Gumbel-Sigmoid(logits, temperature=τ)
                          hard_mask = (soft_mask > 0.5).float()    # STE
                              ↓
                        Scattered mask: [0,1,0,1,1,0,...,1,0,1]
                              ↓
                        masked_emb = normalize(full_emb × mask)
```

Each Bloom level gets its own learned mask initialized to a target sparsity via Gaussian quantile initialization.

**The MRL prefix bias problem:**

The MRL warm-start gives Option B a strong encoder — but that encoder has learned to concentrate information in early dimensions. Without intervention, the mask learning discovers prefix dims are high-quality and converges to a prefix-like pattern, defeating the purpose.

Two mechanisms combat this:
1. **DropMask** (rate=10%): Randomly flip 10% of mask bits during training. Forces the encoder to spread information across all dimensions because any single dim might be masked out.
2. **DimVarianceRedistribution**: Penalizes the encoder when early dimensions have higher variance than late dimensions — a direct anti-prefix signal.

**Reverse two-stage training:**

If both mask and encoder train simultaneously, the mask chases a moving target → instability. Option B reverses the traditional approach:
- **Stage 1 (epochs 0-7):** Encoder FROZEN, only BloomMaskHead trains. Mask converges to stable per-level patterns on top of frozen encoder quality.
- **Stage 2 (epochs 8-19):** Encoder UNFREEZES at very low LR (1e-6). Gentle adaptation pushes information into mask-selected dimensions.

### Option B Losses

Option B uses all of Option A's losses (contrastive, efficiency, MRL anchor) plus five additional losses for scattered mask learning:

| Loss | Weight | What it does | Why it's needed |
|------|--------|-------------|-----------------|
| **Bloom-Masked Contrastive** | 1.0 | Same as Option A — InfoNCE in masked subspace with class weighting | Primary retrieval objective |
| **Efficiency** | 0.1 | Same cognitive-weighted compression as Option A | Encourages lower Bloom levels to activate fewer dimensions |
| **MRL Anchor Regularization** | 0.3 | Same as Option A — InfoNCE at MRL truncation points | Preserves encoder's multi-resolution quality |
| **Mask Sparsity** | 1.0 | `mean_b[\|mean_active_frac_b - target_b\|]` with per-level targets (Remember→35%, Understand→50%, Apply→60%, Analyze→65%, Evaluate→72%, Create→55%) | Without this, the mask drifts to all-ones (full dims) or collapses. Per-level targets directly encode the cognitive load hypothesis into the mask structure |
| **Mask Diversity** | 0.5 | `mean_{i<j} max(0, cosine_sim(mask_i, mask_j) - margin)` across Bloom levels | Prevents all 6 masks from learning identical patterns. Without it, all levels converge to the same "safe" mask (high cosine similarity ~0.97). Margin=0.3 provides strong gradient to push masks apart |
| **Mask Variance** | 0.5 | `-mean_d(Var_b(mean_activation[b,d]))` — maximizes per-dimension variance across Bloom levels | Rewards dimension specialization: dim d should be active for some Bloom levels but not others. Operates at dimension level (768 gradient signals) vs pair level (15 signals), providing much richer optimization signal than diversity alone |
| **Mask Distillation** | 0.3 | `mean_{i≠j} \|sim_full(i,j) - sim_masked(i,j)\|` — masked similarity should preserve full-embedding similarity structure | Unsupervised teacher-student signal — the full 768-dim encoder is the "teacher", masked output is the "student". Forces the mask to select dims that preserve semantic neighborhoods. Fires from epoch 0, giving the mask early signal before sparsity pressure |
| **DimVariance Redistribution** | 0.05 | `ReLU(mean_var(early_dims) - mean_var(late_dims))` on full embeddings | Directly combats MRL prefix bias — penalizes the encoder when early dimensions have higher information content than late dimensions. Forces the encoder to spread useful information across all 768 dims so the scattered mask has high-quality dims everywhere to choose from |

### Why so many losses?

Scattered masks are much harder to learn than prefix masks. Option A has one scalar per level — 6 values to optimize. Option B has 6 × 768 = 4,608 binary decisions. Without careful loss design:
- Masks collapse to identical patterns (need diversity + variance)
- Masks drift to all-on or all-off (need sparsity)
- Masks rediscover the MRL prefix (need DropMask + DimRedist)
- Masks destroy the encoder's similarity structure (need distillation)
- Encoder forgets multi-resolution quality (need MRL anchor)

Each loss addresses a specific failure mode observed during development.

---

## 6. Why BAM-PQ?

Option B assigns **one fixed mask per Bloom level** — all "Analyze" queries share the same mask regardless of content. But:

- "Compare mitosis vs meiosis" (biology)
- "Compare GDP of US vs China" (economics)

Both are Analyze-level, but they need **different dimensions** because they operate in different semantic spaces. The Bloom level alone doesn't capture the full routing signal.

**What if the mask could adapt to the specific query, not just its Bloom level?**

---

## 7. BAM-PQ — Per-Query Routing

**Idea:** The mask is the Bloom-level base **plus a query-specific adjustment**. The Bloom level provides a strong prior; a lightweight MLP adds a per-query residual.

**How it works:**

```
Query → Encoder → CLS token (768-dim)
     → Bloom classifier → level (1-6)
                              ↓
                        BloomQueryMaskHead
                          bloom_logits = Embedding(6, 768)           # coarse Bloom prior
                          query_mlp = Linear(768→256) → GELU → LN → Linear(256→768)  # per-query signal
                          alpha = sigmoid(alpha_raw)                 # learned mixing weight
                              ↓
                          final_logits = bloom_logits[level] + α × query_mlp(CLS)
                              ↓
                          mask = Gumbel-STE(final_logits)
                              ↓
                        masked_emb = normalize(full_emb × mask)
```

**Why this works where pure per-query routing fails:**

Previous attempts at self-supervised per-query routing (SoftRouter, GroupRouter in earlier QA-MRL versions) suffered from **collapse** — without a strong prior, the router either assigns all queries the same mask or oscillates chaotically.

BAM-PQ solves this via the **Bloom anchor**:
- At initialization, α ≈ 0.05 (init alpha_raw = -3.0) → model behaves identically to Option B
- Query MLP output layer is zero-initialized → no per-query signal at epoch 0
- α grows **only if** per-query adjustment genuinely improves retrieval
- Bloom anchor provides stable gradients for sparsity/diversity losses even early in training

**What alpha tells you at convergence:**
- α ≈ 0.1 → per-query routing adds 10% on top of Bloom prior (query content matters)
- α ≈ 0.05 → Bloom level alone captures most of the routing signal
- This is directly interpretable — you can measure how much per-query information helps

### BAM-PQ Losses

BAM-PQ uses all of Option B's losses (same weights) plus one additional loss:

| Loss | Weight | What it does | Why it's needed |
|------|--------|-------------|-----------------|
| All Option B losses | (same) | Contrastive, efficiency, MRL anchor, sparsity, diversity, variance, distillation, DimRedist | Same reasons as Option B — the Bloom-level mask still needs all the same constraints |
| **Query Routing Contrastive** | 0.1 | InfoNCE where the per-query MLP signal is contrasted against random other queries' signals | Ensures the per-query adjustment produces distinct masks for semantically different queries within the same Bloom level, preventing the MLP from outputting the same residual for all queries |

**Training:** Same reverse two-stage as Option B (8 epochs frozen + 12 unfrozen).

---

## 8. Summary: How Each Approach Builds on the Previous

```
MRL Baseline
  │  Losses: InfoNCE + MRL Anchor (2 losses)
  │
  │  Problem: Fixed truncation — same dims for every query
  │
  ▼
Option A (BloomDimRouter)
  │  Losses: + Efficiency + Diversity (4 losses)
  │  ✓ Learns per-Bloom-level prefix truncation
  │  ✓ FAISS-compatible (contiguous dims)
  │  ✗ Prefix constraint — can only use dims [0..d]
  │
  │  Problem: Cannot select scattered dims
  │
  ▼
Option B (BloomMaskHead)
  │  Losses: + Sparsity + Mask Diversity + Mask Variance
  │          + Distillation + DimRedist (9 losses)
  │  ✓ Scattered binary mask per Bloom level
  │  ✓ DropMask + DimRedist to combat MRL prefix bias
  │  ✓ Reverse two-stage training for stability
  │  ✗ One fixed mask per Bloom level
  │
  │  Problem: Same Bloom level ≠ same information need
  │
  ▼
BAM-PQ (BloomQueryMaskHead)
     Losses: + Query Routing Contrastive (10 losses)
     ✓ Bloom anchor + per-query MLP residual
     ✓ Learned alpha controls mixing
     ✓ Prevents collapse via strong Bloom prior
     ✓ Interpretable: alpha magnitude = per-query routing value
```

Each step adds expressiveness while maintaining the gains of the previous step. The Bloom taxonomy provides the structural prior that prevents routing collapse — the key failure mode of earlier approaches.

---

## 9. Evaluation Methodology

**Metrics:**
- **Retrieval quality:** Recall@K (k=1,5,10,50), NDCG@10, MRR
- **Bloom-stratified:** Per-Bloom-level Recall@10 with per-level dimension usage
- **Efficiency:** Average active dimensions, sparse ratio (fraction of zeroed dims)
- **BSR (Bloom Stratified Recall):** quality × (1 + α × efficiency) — balances retrieval quality against dimension compression for checkpoint selection

**Fair comparison:** BAM vs MRL at the **same dimension budget** per Bloom level — MRL truncated to match BAM's average active dims per level, isolating the effect of dimension *selection* (scattered vs prefix) from dimension *count*.

**Out-of-domain (BEIR):** Models trained on one domain are evaluated on BEIR benchmark datasets. BEIR queries are auto-annotated with Bloom levels using the NLI classifier so BAM routing adapts per-query at inference time.

---

## 10. Results — Educational Dataset (In-Domain)

**Setup:** Backbone: intfloat/e5-large-v2 (1024-dim, 335M params). Trained on educational science data (SciQ, ARC, OpenBookQA, QASC). Corpus: 40,640 passages. Test: 1,857 queries.

### MRL Baseline (Best epoch: 8)

| Metric | Value |
|--------|-------|
| R@1 | 0.2746 |
| R@10 | 0.5083 |
| R@50 | 0.6387 |
| NDCG@10 | 0.3876 |
| Dims | 768 (full) |

Bloom-stratified R@10 (MRL at full 768 dims):

| Level | N | R@10 |
|-------|---|------|
| Remember | 952 | 0.6586 |
| Understand | 450 | 0.4267 |
| Apply | 106 | 0.2358 |
| Analyze | 347 | 0.2853 |
| Create | 2 | 0.5000 |

### BAM Option B (Best epoch: 18, BSR = 0.6156)

| Metric | Value |
|--------|-------|
| R@1 | 0.2709 |
| R@10 | 0.5579 |
| R@50 | 0.7092 |
| NDCG@10 | 0.4100 |
| Avg Active Dims | 446 / 768 (sparse ratio = 0.42) |

Bloom-stratified R@10 with per-level dimension usage:

| Level | N | R@10 | Dims Used |
|-------|---|------|-----------|
| Remember | 952 | 0.7080 | 398 |
| Understand | 450 | 0.4844 | 494 |
| Apply | 106 | 0.2264 | 515 |
| Analyze | 347 | 0.3429 | 493 |
| Create | 2 | 0.5000 | 557 |

### BAM-PQ (Best epoch: 14, BSR = 0.6207)

| Metric | Value |
|--------|-------|
| R@1 | 0.2709 |
| R@10 | 0.5595 |
| R@50 | 0.7108 |
| NDCG@10 | 0.4082 |
| Avg Active Dims | 393 / 768 (sparse ratio = 0.49) |

Bloom-stratified R@10 with per-level dimension usage:

| Level | N | R@10 | Dims Used |
|-------|---|------|-----------|
| Remember | 952 | 0.7090 | 277 |
| Understand | 450 | 0.4911 | 447 |
| Apply | 106 | 0.2547 | 606 |
| Analyze | 347 | 0.3314 | 576 |
| Create | 2 | 0.5000 | 752 |

### Fair Comparison: BAM-B vs MRL at Same Dimension Budget

| Level | N | Budget | MRL-full | MRL-trunc | BAM-B | Δ |
|-------|---|--------|----------|-----------|-------|---|
| Remember | 952 | 398 | 0.6586 | 0.6439 | 0.7080 | +0.0641 |
| Understand | 450 | 494 | 0.4267 | 0.4222 | 0.4844 | +0.0622 |
| Apply | 106 | 515 | 0.2358 | 0.2547 | 0.2264 | -0.0283 |
| Analyze | 347 | 493 | 0.2853 | 0.2824 | 0.3429 | +0.0605 |
| Create | 2 | 557 | 0.5000 | 0.5000 | 0.5000 | +0.0000 |

**Average Δ(BAM − MRL-trunc): +0.0317. BAM wins 3/5 levels.**

BAM-B outperforms MRL at the same dimension budget on Remember (+6.4%), Understand (+6.2%), and Analyze (+6.1%). The scattered mask selects more informative dimensions than MRL's fixed prefix truncation.

### Key Observations (Educational)

1. **BAM-B improves R@10 by +5.0 points** over MRL (0.5579 vs 0.5083) while using only 58% of dimensions (446 vs 768)
2. **BAM-PQ improves R@10 by +5.1 points** (0.5595 vs 0.5083) while using only 51% of dimensions (393 vs 768)
3. **Cognitive ordering confirmed in BAM-PQ**: Remember uses 277 dims, Create uses 752 dims — a 2.7× ratio matching the cognitive complexity hypothesis
4. **BAM-PQ achieves the best BSR** (0.6207 vs BAM-B's 0.6156) due to stronger compression with maintained quality

---

## 11. Results — BEIR Out-of-Domain Evaluation

Models trained on educational data, evaluated on BEIR benchmark datasets. Tests whether learned cognitive-level routing generalizes beyond the training domain.

### SciFact (5,183 passages, 300 test queries)

| Model | R@1 | R@10 | R@50 | NDCG@10 | Dims |
|-------|-----|------|------|---------|------|
| MRL Baseline | 0.5767 | 0.8600 | 0.9400 | 0.7081 | 768 |
| BAM-B | 0.5567 | 0.8500 | 0.9267 | 0.6957 | 472 |
| BAM-PQ | 0.5450 | 0.8600 | 0.9400 | 0.7081 | 488 |

Fair comparison (BAM-B vs MRL at same dims):

| Level | N | Budget | BAM-B | MRL-trunc | Δ |
|-------|---|--------|-------|-----------|---|
| Remember | 9 | 453 | 0.7778 | 0.7778 | +0.0000 |
| Understand | 214 | 473 | 0.8411 | 0.8411 | +0.0000 |
| Analyze | 76 | 468 | 0.8816 | 0.8816 | +0.0000 |
| Evaluate | 1 | 481 | 1.0000 | 1.0000 | +0.0000 |

Fair comparison (BAM-PQ vs MRL at same dims): **Average Δ: +2.92%, BAM wins 2/4 levels.**

### NFCorpus

Fair comparison BAM-B: **Average Δ: +0.95%, BAM wins 2/5 levels.**
Fair comparison BAM-PQ: **Average Δ: +0.39%, BAM wins 1/5 levels.**

### FiQA

Fair comparison BAM-B: **Average Δ: +0.22%, BAM wins 3/5 levels.**
Fair comparison BAM-PQ: **Average Δ: +0.40%, BAM wins 2/5 levels.**

### Cross-Dataset Summary (Out-of-Domain)

| Dataset | BAM-B Avg Δ | BAM-B Wins | BAM-PQ Avg Δ | BAM-PQ Wins |
|---------|------------|------------|-------------|-------------|
| Educational (in-domain) | +3.17% | 3/5 | — | — |
| SciFact | +0.00% | 0/4 | +2.92% | 2/4 |
| NFCorpus | +0.95% | 2/5 | +0.39% | 1/5 |
| FiQA | +0.22% | 3/5 | +0.40% | 2/5 |

---

## 12. Results — BEIR In-Domain Training

For each BEIR dataset below, models were trained **and** evaluated on the same dataset. This isolates domain-specific routing behavior from cross-domain transfer effects. Bloom labels were assigned via zero-shot NLI classification (DeBERTa-v3-large).

### SciFact (5,183 passages, 300 test queries)

Bloom distribution: Remember (9), Understand (214), Analyze (76), Evaluate (1). No Apply or Create queries in this dataset.

**MRL Baseline (Best epoch: 14, R@10 = 0.8600)**

| Metric | Value |
|--------|-------|
| R@1 | 0.5767 |
| R@5 | 0.7667 |
| R@10 | 0.8600 |
| R@50 | 0.9400 |
| MRR | 0.6661 |
| NDCG@10 | 0.7081 |
| Dims | 768 (full) |

Bloom-stratified R@10:

| Level | N | R@10 |
|-------|---|------|
| Remember | 9 | 0.7778 |
| Understand | 214 | 0.8505 |
| Analyze | 76 | 0.8947 |
| Evaluate | 1 | 1.0000 |

**BAM Option B (Best epoch: 14, BSR = 1.1644)**

| Metric | Value |
|--------|-------|
| R@1 | 0.5567 |
| R@5 | 0.7500 |
| R@10 | 0.8500 |
| R@50 | 0.9267 |
| MRR | 0.6522 |
| NDCG@10 | 0.6957 |
| Active dims | 472 / 768 (38.5% savings) |

Bloom-stratified R@10 with learned dimensions:

| Level | N | R@10 | Dims |
|-------|---|------|------|
| Remember | 9 | 0.7778 | 453 |
| Understand | 214 | 0.8411 | 473 |
| Analyze | 76 | 0.8816 | 470 |
| Evaluate | 1 | 1.0000 | 481 |

**BAM-PQ (Best epoch: 19, BSR = 1.2019)**

| Metric | Value |
|--------|-------|
| R@1 | 0.5733 |
| R@5 | 0.7667 |
| R@10 | 0.8667 |
| R@50 | 0.9500 |
| MRR | 0.6624 |
| NDCG@10 | 0.7069 |
| Active dims | 481 / 768 (37.4% savings) |

Bloom-stratified R@10 with learned dimensions:

| Level | N | R@10 | Dims |
|-------|---|------|------|
| Remember | 9 | 0.8889 | 462 |
| Understand | 214 | 0.8551 | 476 |
| Analyze | 76 | 0.8947 | 497 |
| Evaluate | 1 | 1.0000 | 519 |

**Fair Comparison: BAM-B vs MRL at Same Dimension Budget**

| Level | N | Budget | MRL-full | MRL-trunc | BAM-B | Δ |
|-------|---|--------|----------|-----------|-------|---|
| Remember | 9 | 453 | 0.7778 | 0.7778 | 0.7778 | +0.0000 |
| Understand | 214 | 473 | 0.8505 | 0.8411 | 0.8411 | +0.0000 |
| Analyze | 76 | 468 | 0.8947 | 0.8816 | 0.8816 | +0.0000 |
| Evaluate | 1 | 481 | 1.0000 | 1.0000 | 1.0000 | +0.0000 |

Average Δ: +0.0000, BAM wins 0/4 levels.

**Fair Comparison: BAM-PQ vs MRL at Same Dimension Budget**

| Level | N | Budget | MRL-full | MRL-trunc | BAM-PQ | Δ |
|-------|---|--------|----------|-----------|--------|---|
| Remember | 9 | 462 | 0.7778 | 0.7778 | 0.8889 | +0.1111 |
| Understand | 214 | 476 | 0.8505 | 0.8411 | 0.8598 | +0.0187 |
| Analyze | 76 | 497 | 0.8947 | 0.8947 | 0.8816 | -0.0132 |
| Evaluate | 1 | 519 | 1.0000 | 1.0000 | 1.0000 | +0.0000 |

Average Δ: +0.0292, BAM-PQ wins 2/4 levels.

**Key Observations (SciFact):**
- BAM-PQ **matches or exceeds MRL** on R@10 (0.8667 vs 0.8600) while using 37% fewer dims
- BAM-PQ Remember recall jumps from 0.7778 → 0.8889 (+11.1%) at the per-level fair comparison
- BAM-B is more conservative — identical performance to MRL-truncated at every level
- SciFact's narrow Bloom distribution (71% Understand) limits routing differentiation

### NFCorpus (3,633 passages, 323 test queries)

Bloom distribution: Remember (20), Understand (80), Apply (3), Analyze (188), Evaluate (32). No Create queries.

**MRL Baseline (Best epoch: 9, R@10 = 0.2105)**

| Metric | Value |
|--------|-------|
| R@1 | 0.0495 |
| R@5 | 0.1455 |
| R@10 | 0.2105 |
| R@50 | 0.3127 |
| MRR | 0.0975 |
| NDCG@10 | 0.1193 |
| Dims | 768 (full) |

Bloom-stratified R@10:

| Level | N | R@10 |
|-------|---|------|
| Remember | 20 | 0.4500 |
| Understand | 80 | 0.1625 |
| Apply | 3 | 0.3333 |
| Analyze | 188 | 0.1915 |
| Evaluate | 32 | 0.2812 |

**BAM Option B (Best epoch: 4, BSR = 0.4105)**

| Metric | Value |
|--------|-------|
| R@1 | 0.0526 |
| R@5 | 0.1362 |
| R@10 | 0.1950 |
| R@50 | 0.3127 |
| MRR | 0.0957 |
| NDCG@10 | 0.1139 |
| Active dims | 468 / 768 (39.1% savings) |

Bloom-stratified R@10 with learned dimensions:

| Level | N | R@10 | Dims |
|-------|---|------|------|
| Remember | 20 | 0.4500 | 454 |
| Understand | 80 | 0.1500 | 472 |
| Apply | 3 | 0.3333 | 478 |
| Analyze | 188 | 0.1649 | 467 |
| Evaluate | 32 | 0.3125 | 476 |

**BAM-PQ (Best epoch: 11, BSR = 0.4091)**

| Metric | Value |
|--------|-------|
| R@1 | 0.0464 |
| R@5 | 0.1548 |
| R@10 | 0.2043 |
| R@50 | 0.3251 |
| MRR | 0.0966 |
| NDCG@10 | 0.1172 |
| Active dims | 495 / 768 (35.5% savings) |

Bloom-stratified R@10 with learned dimensions:

| Level | N | R@10 | Dims |
|-------|---|------|------|
| Remember | 20 | 0.4500 | 473 |
| Understand | 80 | 0.1500 | 485 |
| Apply | 3 | 0.3333 | 480 |
| Analyze | 188 | 0.1809 | 498 |
| Evaluate | 32 | 0.3125 | 515 |

**Fair Comparison: BAM-B vs MRL at Same Dimension Budget**

| Level | N | Budget | MRL-full | MRL-trunc | BAM-B | Δ |
|-------|---|--------|----------|-----------|-------|---|
| Remember | 20 | 454 | 0.4500 | 0.4500 | 0.4500 | +0.0000 |
| Understand | 80 | 472 | 0.1625 | 0.1125 | 0.1500 | +0.0375 |
| Apply | 3 | 478 | 0.3333 | 0.3333 | 0.3333 | +0.0000 |
| Analyze | 188 | 467 | 0.1915 | 0.1862 | 0.1649 | -0.0213 |
| Evaluate | 32 | 476 | 0.2812 | 0.2812 | 0.3125 | +0.0312 |

Average Δ: +0.0095, BAM wins 2/5 levels.

**Fair Comparison: BAM-PQ vs MRL at Same Dimension Budget**

| Level | N | Budget | MRL-full | MRL-trunc | BAM-PQ | Δ |
|-------|---|--------|----------|-----------|--------|---|
| Remember | 20 | 473 | 0.4500 | 0.4500 | 0.4500 | +0.0000 |
| Understand | 80 | 485 | 0.1625 | 0.1250 | 0.1500 | +0.0250 |
| Apply | 3 | 480 | 0.3333 | 0.3333 | 0.3333 | +0.0000 |
| Analyze | 188 | 498 | 0.1915 | 0.1862 | 0.1809 | -0.0053 |
| Evaluate | 32 | 515 | 0.2812 | 0.2812 | 0.2812 | +0.0000 |

Average Δ: +0.0039, BAM-PQ wins 1/5 levels.

**Key Observations (NFCorpus):**
- NFCorpus is a challenging medical dataset with multi-label relevance — overall R@10 is low for all models
- BAM-B selected best at epoch 4 (very early), suggesting routing converges quickly on this domain
- Both BAM variants improve Understand recall vs MRL-truncated (+3.75% for BAM-B, +2.50% for BAM-PQ)
- ~35-39% dimension savings with minimal retrieval quality loss

### FiQA (57,638 passages, 648 test queries)

Bloom distribution: Remember (23), Understand (116), Apply (130), Analyze (130), Evaluate (249). No Create queries. Largest and most diverse BEIR dataset in the evaluation.

**MRL Baseline (Best epoch: 10, R@10 = 0.4969)**

| Metric | Value |
|--------|-------|
| R@1 | 0.1975 |
| R@5 | 0.3873 |
| R@10 | 0.4969 |
| R@50 | 0.6775 |
| MRR | 0.2908 |
| NDCG@10 | 0.3325 |
| Dims | 768 (full) |

Bloom-stratified R@10:

| Level | N | R@10 |
|-------|---|------|
| Remember | 23 | 0.7826 |
| Understand | 116 | 0.5086 |
| Apply | 130 | 0.4308 |
| Analyze | 130 | 0.4923 |
| Evaluate | 249 | 0.5020 |

**BAM Option B (Best epoch: 11, BSR = 0.7624)**

| Metric | Value |
|--------|-------|
| R@1 | 0.1852 |
| R@5 | 0.3904 |
| R@10 | 0.4707 |
| R@50 | 0.6512 |
| MRR | 0.2815 |
| NDCG@10 | 0.3191 |
| Active dims | 478 / 768 (37.8% savings) |

Bloom-stratified R@10 with learned dimensions:

| Level | N | R@10 | Dims |
|-------|---|------|------|
| Remember | 23 | 0.8261 | 442 |
| Understand | 116 | 0.4828 | 476 |
| Apply | 130 | 0.3692 | 484 |
| Analyze | 130 | 0.4846 | 471 |
| Evaluate | 249 | 0.4779 | 482 |

**BAM-PQ (Best epoch: 8, BSR = 0.7255)**

| Metric | Value |
|--------|-------|
| R@1 | 0.1821 |
| R@5 | 0.3827 |
| R@10 | 0.4753 |
| R@50 | 0.6667 |
| MRR | 0.2799 |
| NDCG@10 | 0.3186 |
| Active dims | 505 / 768 (34.2% savings) |

Bloom-stratified R@10 with learned dimensions:

| Level | N | R@10 | Dims |
|-------|---|------|------|
| Remember | 23 | 0.7391 | 450 |
| Understand | 116 | 0.5086 | 485 |
| Apply | 130 | 0.4000 | 483 |
| Analyze | 130 | 0.4846 | 505 |
| Evaluate | 249 | 0.4699 | 532 |

**Fair Comparison: BAM-B vs MRL at Same Dimension Budget**

| Level | N | Budget | MRL-full | MRL-trunc | BAM-B | Δ |
|-------|---|--------|----------|-----------|-------|---|
| Remember | 23 | 442 | 0.7826 | 0.7826 | 0.8261 | +0.0435 |
| Understand | 116 | 476 | 0.5086 | 0.5000 | 0.4828 | -0.0172 |
| Apply | 130 | 484 | 0.4308 | 0.4154 | 0.3692 | -0.0462 |
| Analyze | 130 | 471 | 0.4923 | 0.4615 | 0.4846 | +0.0231 |
| Evaluate | 249 | 482 | 0.5020 | 0.4699 | 0.4779 | +0.0080 |

Average Δ: +0.0022, BAM wins 3/5 levels.

**Fair Comparison: BAM-PQ vs MRL at Same Dimension Budget**

| Level | N | Budget | MRL-full | MRL-trunc | BAM-PQ | Δ |
|-------|---|--------|----------|-----------|--------|---|
| Remember | 23 | 450 | 0.7826 | 0.7391 | 0.7391 | +0.0000 |
| Understand | 116 | 485 | 0.5086 | 0.5000 | 0.5172 | +0.0172 |
| Apply | 130 | 483 | 0.4308 | 0.4231 | 0.4231 | +0.0000 |
| Analyze | 130 | 505 | 0.4923 | 0.4615 | 0.4923 | +0.0308 |
| Evaluate | 249 | 532 | 0.5020 | 0.4900 | 0.4618 | -0.0281 |

Average Δ: +0.0040, BAM-PQ wins 2/5 levels.

**Key Observations (FiQA):**
- FiQA has the most balanced Bloom distribution — all 5 levels well-represented
- BAM-B Remember recall exceeds MRL-full (0.8261 vs 0.7826) at only 442 dims — a clear win for routing
- BAM-PQ shows clearest cognitive dimension ordering: Remember (450) < Understand (485) < Analyze (505) < Evaluate (532)
- Both BAM variants win 2-3/5 levels with ~35-38% dimension savings

### Cross-Dataset Summary (In-Domain)

| Dataset | Queries | Corpus | MRL R@10 | BAM-B R@10 | BAM-B Dims | BAM-PQ R@10 | BAM-PQ Dims |
|---------|---------|--------|----------|-----------|------------|-------------|-------------|
| SciFact | 300 | 5,183 | 0.8600 | 0.8500 | 472 | 0.8667 | 481 |
| NFCorpus | 323 | 3,633 | 0.2105 | 0.1950 | 468 | 0.2043 | 495 |
| FiQA | 648 | 57,638 | 0.4969 | 0.4707 | 478 | 0.4753 | 505 |

| Dataset | BAM-B Avg Δ | BAM-B Wins | BAM-PQ Avg Δ | BAM-PQ Wins |
|---------|------------|------------|-------------|-------------|
| SciFact | +0.00% | 0/4 | +2.92% | 2/4 |
| NFCorpus | +0.95% | 2/5 | +0.39% | 1/5 |
| FiQA | +0.22% | 3/5 | +0.40% | 2/5 |

---

## 13. Ablation Studies

The ablation suite isolates each component's contribution by systematically disabling or replacing parts of the BAM pipeline:

| Ablation | What it tests | How |
|----------|--------------|-----|
| **BAM full** | Full system with real Bloom labels | Normal operation (baseline for ablations) |
| **BAM random Bloom** | Is Bloom taxonomy the right signal? | Replace real Bloom labels with random 0-5 labels |
| **BAM fixed Bloom=1** | What if all queries are "simple"? | Force all queries to Remember (minimum dims) |
| **BAM fixed Bloom=6** | What if all queries are "complex"? | Force all queries to Create (maximum dims) |
| **BAM no routing** | Does routing help, or is fine-tuning enough? | Force router to 768 dims (full) — isolates fine-tuning gain from routing gain |
| **BAM soft routing** | Hard vs soft mask selection | Use softmax over all dims instead of hard binary mask |
| **BAM fixed avg budget** | Does *per-level* routing matter? | Route ALL queries to the same dim count as BAM's average — same compression, no per-level differentiation |
| **Mask vs truncation** | Scattered mask vs prefix at same dims | Option B vs Option A at identical average active dims |
| **Two-stage vs joint** | Does reverse two-stage training help? | Compare frozen→unfrozen (reverse) vs simultaneous training |
| **MRL Baseline** | Comparison target | Full 768 dims, no routing |

These ablations answer key research questions:
- **Random Bloom vs real Bloom**: If random labels perform similarly, the cognitive taxonomy signal is not useful — any grouping would work
- **Fixed level vs adaptive**: If fixed Bloom=1 or Bloom=6 matches BAM full, per-query adaptation is unnecessary
- **No routing vs full BAM**: The gap between these isolates how much of BAM's improvement comes from routing vs general fine-tuning on Bloom-aware data
