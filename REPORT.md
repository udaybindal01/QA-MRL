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

## 9. Evaluation

**Metrics:**
- **Retrieval quality:** Recall@K (k=1,5,10,20,50,100), NDCG@10, MAP
- **Bloom-stratified:** Per-Bloom-level Recall@10 with 95% bootstrap confidence intervals
- **Efficiency:** Average active dimensions, sparse ratio (fraction of zeroed dims)
- **BSR (Bloom Stratified Recall):** quality × (1 + α × efficiency) — balances retrieval quality against dimension compression for checkpoint selection

**Out-of-domain (BEIR):** Trained on educational science data → evaluated on HotpotQA, SciFact, NFCorpus, TREC-COVID, SciDocs, Climate-FEVER. BEIR queries are auto-annotated with Bloom levels using the NLI classifier so BAM routing adapts per-query.

**Fair comparison:** BAM vs MRL at the **same dimension budget** per Bloom level — MRL truncated to match BAM's average active dims, isolating the effect of dimension *selection* (scattered vs prefix) from dimension *count*.
