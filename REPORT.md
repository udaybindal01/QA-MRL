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
- Bloom is query-only — documents always encoded at full dimensionality; dot product naturally ignores zeroed query dims

**Training losses:**
- **Contrastive** — InfoNCE in masked subspace (class-weighted by 1/√freq for Bloom imbalance)
- **Efficiency** — pushes lower-complexity levels to fewer dims (Remember weight=1.0, Create weight=0.167)
- **Diversity** — spreads the 6 learned truncation points apart
- **MRL anchor** — keeps encoder sharp at standard MRL truncation points

**Pipeline:** MRL warm-start → train Option A with efficiency gated off for first 5 epochs (encoder builds quality before compression pressure begins).

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

Each Bloom level gets its own learned mask initialized to a target sparsity:
- Remember → 30% active (~230 dims)
- Understand → 56% active (~430 dims)
- Apply → 66% active (~507 dims)
- Analyze → 76% active (~584 dims)
- Evaluate → 86% active (~660 dims)
- Create → 27% active (~207 dims)

**The MRL prefix bias problem:**

The MRL warm-start gives Option B a strong encoder — but that encoder has learned to concentrate information in early dimensions. Without intervention, the mask learning discovers prefix dims are high-quality and converges to a prefix-like pattern, defeating the purpose.

Two mechanisms combat this:
1. **DropMask** (rate=10%): Randomly flip 10% of mask bits during training. Forces the encoder to spread information across all dimensions because any single dim might be masked out.
2. **DimVarianceRedistribution**: Penalizes the encoder when early dimensions have higher variance than late dimensions — a direct anti-prefix signal.

**Reverse two-stage training:**

If both mask and encoder train simultaneously, the mask chases a moving target → instability. Option B reverses the traditional approach:
- **Stage 1 (epochs 0-7):** Encoder FROZEN, only BloomMaskHead trains. Mask converges to stable per-level patterns on top of frozen encoder quality.
- **Stage 2 (epochs 8-19):** Encoder UNFREEZES at very low LR (1e-6). Gentle adaptation pushes information into mask-selected dimensions.

**Additional losses:**
- **Mask sparsity** — enforces per-level target active dims
- **Mask diversity** — penalizes different Bloom levels from learning identical masks
- **Mask variance** — each dim should be useful to some levels but not all
- **Mask distillation** — similarity structure in masked subspace should preserve full-embedding similarity structure

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

**Training:** Same reverse two-stage as Option B (8 epochs frozen + 12 unfrozen).

---

## 8. Summary: How Each Approach Builds on the Previous

```
MRL Baseline
  │
  │  Problem: Fixed truncation — same dims for every query
  │
  ▼
Option A (BloomDimRouter)
  │  ✓ Learns per-Bloom-level prefix truncation
  │  ✓ FAISS-compatible (contiguous dims)
  │  ✗ Prefix constraint — can only use dims [0..d]
  │
  │  Problem: Cannot select scattered dims
  │
  ▼
Option B (BloomMaskHead)
  │  ✓ Scattered binary mask per Bloom level
  │  ✓ DropMask + DimRedist to combat MRL prefix bias
  │  ✓ Reverse two-stage training for stability
  │  ✗ One fixed mask per Bloom level
  │
  │  Problem: Same Bloom level ≠ same information need
  │
  ▼
BAM-PQ (BloomQueryMaskHead)
     ✓ Bloom anchor + per-query MLP residual
     ✓ Learned alpha controls mixing
     ✓ Prevents collapse via strong Bloom prior
     ✓ Interpretable: alpha magnitude = per-query routing value
```

Each step adds expressiveness while maintaining the gains of the previous step. The Bloom taxonomy provides the structural prior that prevents routing collapse — the key failure mode of earlier approaches.
