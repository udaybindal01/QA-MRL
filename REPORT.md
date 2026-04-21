# BAM: Bloom-Aligned Matryoshka Representation Learning for Query-Adaptive Retrieval

## 1. Problem Statement

Standard dense retrieval encodes queries and documents into fixed-dimensional embeddings (e.g., 768 or 1024 dims) and retrieves via dot-product similarity. **Matryoshka Representation Learning (MRL)** improves efficiency by training embeddings such that the first *d* dimensions form a valid lower-dimensional representation — you can truncate to 256 dims and still retrieve reasonably well.

However, MRL enforces a **rigid dimensional hierarchy**: dimension 1 is always "most important," dimension 768 always least, regardless of the query. This is suboptimal because different queries have fundamentally different information needs:

- **"What is photosynthesis?"** (factual recall) — needs few dimensions to match a definition
- **"Compare the efficiency of C3 vs C4 photosynthetic pathways under drought conditions"** (analysis) — needs many dimensions to capture nuanced comparative relationships

We frame this insight through **Bloom's cognitive taxonomy**, which classifies questions into six levels of cognitive complexity:

| Level | Name | Example | Complexity |
|-------|------|---------|------------|
| 1 | Remember | "Define osmosis" | Lowest |
| 2 | Understand | "Explain how osmosis works" | |
| 3 | Apply | "Calculate osmotic pressure for this solution" | |
| 4 | Analyze | "Compare osmosis and diffusion" | |
| 5 | Evaluate | "Is reverse osmosis effective for desalination?" | |
| 6 | Create | "Design an experiment to demonstrate osmosis" | Highest |

**Our hypothesis:** Higher cognitive complexity queries need more embedding dimensions to capture the richer semantic relationships involved. Lower complexity queries can be represented with fewer dimensions without quality loss. A system that **adapts the number and selection of active dimensions per query** can achieve better quality-efficiency tradeoffs than fixed MRL truncation.

**The challenge:** Learning this routing without degrading the encoder's retrieval quality, and doing so with a lightweight mechanism that doesn't negate the efficiency gains.

---

## 2. Approach Overview

We develop four progressively more expressive approaches, each building on the previous:

```
MRL Baseline → Option A (prefix routing) → Option B (scattered masks) → BAM-PQ (per-query masks)
```

All approaches share:
- **Bloom is query-only**: Documents have no cognitive level — the same passage about photosynthesis serves a "Remember" query and a "Create" query differently. Only the query determines how many/which dimensions to use.
- **Documents encoded at full dimensionality**: The query mask zeros out dimensions; the dot product naturally ignores them.
- **MRL warm-start**: Each BAM variant initializes its encoder from a trained MRL baseline, giving it multi-resolution structure before routing is introduced.

---

## 3. MRL Baseline

**Architecture:** Standard transformer encoder (BAAI/bge-base-en-v1.5 at 768 dims, or intfloat/e5-large-v2 at 1024 dims) with Matryoshka training.

**Training:** InfoNCE contrastive loss at full dimensionality, plus MRL anchor losses at truncation points [64, 128, 256, 512, 768]. The anchor losses teach the encoder to place important information in early dimensions.

**Role in the pipeline:** The MRL baseline serves two purposes:
1. It is the comparison target — BAM must beat MRL at equivalent dimensionality to justify the routing overhead.
2. It provides the warm-start checkpoint for all BAM variants. Since e5-large has no built-in MRL structure (unlike BGE models which include MRL in pretraining), we must train this structure ourselves.

**Limitation:** Fixed truncation. To use 256 dims, you truncate to dims [0..255] for *every* query. A factual recall query and an analysis query both get the same 256-dim prefix.

---

## 4. Option A — Prefix Routing via BloomDimRouter

**Key idea:** Learn a different truncation point per Bloom level. Remember queries truncate to ~300 dims; Create queries use ~600 dims.

**Architecture:**
```
BloomDimRouter:
  bloom_emb = Embedding(6, 32)        # one learned vector per Bloom level
  dim_head  = Linear(32,16) → ReLU → Linear(16,1)  # MLP → scalar
  
  continuous_dim = sigmoid(dim_head(bloom_emb[level])) * (D_max - D_min) + D_min
  mask = prefix_binary_mask(continuous_dim)    # [1,1,1,...,1,0,0,...,0]
  
  Forward: hard prefix mask via STE
  Backward: soft sigmoid gradients
```

**Total parameters added:** ~1K (negligible vs 110M encoder)

**Why prefix:** Active dimensions are contiguous [0..d], enabling FAISS sub-index slicing — no efficiency penalty at retrieval time. Queries with different Bloom levels just use different prefix lengths.

**Training losses:**
- **Contrastive loss** — InfoNCE in the masked subspace (class-weighted by 1/√freq to handle Bloom imbalance)
- **Efficiency loss** — pushes lower-complexity levels to fewer dims via cognitive weights: Remember gets weight 1.0 (strong compression), Create gets weight 0.167 (mild compression)
- **Diversity loss** — maximizes pairwise distance between the 6 learned dimensions to prevent all levels collapsing to the same truncation point
- **MRL anchor regularization** — keeps the encoder sharp at standard MRL truncation points

**Key design lessons learned:**
- **Zero-init trap**: Initializing the output layer weights to zero blocks gradient flow to `bloom_emb`, locking all levels at ~448 dims forever. Kaiming initialization is required.
- **Diversity vs efficiency balance**: Diversity loss is undirected — it spreads dimensions but doesn't know which level should be high/low. Once Kaiming init is fixed, efficiency loss alone provides the correct cognitive ordering.
- **Encoder warmup**: Efficiency loss is gated off for the first 5 epochs so the encoder builds quality representations before compression pressure begins.

**Pipeline:** `optionA-working_pipeline.sh`
```
build data → Bloom annotation → hard negative mining → train MRL → find best MRL epoch → train BAM Option A
```

---

## 5. Option B — Scattered Mask Routing via BloomMaskHead

**Key idea:** Instead of constraining active dims to a prefix, learn a **scattered binary mask** per Bloom level. Dimension 7, 42, 103, 511 might all be active while 0, 1, 2 are not. This breaks MRL's nesting assumption entirely, allowing the model to select the most informative dimensions per cognitive level regardless of position.

**Architecture:**
```
BloomMaskHead:
  bloom_logits = Embedding(6, D)     # one D-dimensional logit vector per level
  
  # Initialize via Gaussian quantile to hit per-level target sparsity:
  #   Remember → 30% active (~230 dims)
  #   Understand → 56% active (~430 dims)
  #   etc.
  
  soft_mask  = Gumbel-Sigmoid(bloom_logits[level], temperature=τ)
  hard_mask  = (soft_mask > 0.5).float()   # STE: hard forward, soft backward
  
  # DropMask augmentation (training only):
  #   Randomly flip 10% of mask bits to force encoder to spread information
  #   across all dims, combating prefix bias inherited from MRL warm-start
```

**Why scattered:** A prefix mask can only express "use dims 0..d". A scattered mask can express "these specific 276 dims are most informative for Remember-level queries." The model learns which dimensions capture factual vs. analytical vs. evaluative information.

**Additional losses (beyond Option A's):**
- **Mask sparsity loss** — enforces per-level target active dims (e.g., Remember → 30%, Create → 27%)
- **Mask diversity loss** — penalizes different Bloom levels from learning identical masks (margin-based)
- **Mask variance loss** — maximizes per-dimension activation variance across levels (each dim should be useful to some levels but not all)
- **Mask distillation loss** — the similarity structure in the masked subspace should preserve the similarity structure of the full embedding

**DropMask and DimVarianceRedistribution:**
The MRL warm-start gives Option B a strong encoder, but that encoder has learned to concentrate information in prefix dimensions. Without intervention, the mask learning discovers that prefix dims are high-quality and converges to a prefix-like pattern, defeating the purpose of scattered masking. Two mechanisms combat this:
1. **DropMask** (rate=0.1): During training, randomly flip 10% of mask bits. This forces the encoder to spread information across all dimensions because any single dimension might be masked out.
2. **DimVarianceRedistributionLoss**: Penalizes the encoder when early dimensions have higher variance than late dimensions — a direct anti-prefix signal.

**Reverse two-stage training:**
Traditional approach: train encoder fully, then freeze and train router. Option B reverses this:
- **Stage 1 (epochs 0-7):** Encoder FROZEN, only BloomMaskHead trains at router_lr. The mask converges to stable per-level patterns on top of the frozen encoder quality.
- **Stage 2 (epochs 8-19):** Encoder UNFREEZES at very low LR (1e-6). Gentle adaptation pushes information into mask-selected dimensions. Stable mask = coherent gradients for the encoder.

**Rationale:** If both mask and encoder train simultaneously, the mask chases a moving target (encoder representation changes), leading to instability. Freezing the encoder first lets the mask find good dimension patterns, then the encoder fine-tunes to support those patterns.

**Pipeline:** `optionB-pipeline.sh` (requires Option A's MRL baseline as prerequisite)
```
re-mine negatives → train BAM Option B (MRL warm-start, reverse two-stage) → BSR selection → eval
```

---

## 6. BAM-PQ — Per-Query Routing via BloomQueryMaskHead

**Key idea:** Option B assigns one fixed mask per Bloom level — all "Analyze" queries get the same mask regardless of content. BAM-PQ adds a lightweight per-query residual: the mask is the Bloom-level base plus a query-specific adjustment.

**Architecture:**
```
BloomQueryMaskHead:
  bloom_logits = Embedding(6, D)                    # coarse Bloom prior (same as Option B)
  query_mlp    = Linear(D,256) → GELU → LN → Linear(256,D)   # fine per-query signal
  alpha_raw    = Parameter(scalar, init=-3.0)       # mixing weight
  
  alpha        = sigmoid(alpha_raw)                 # ≈0.047 at init → pure Bloom
  final_logits = bloom_logits[level] + alpha * query_mlp(normalize(cls_token))
  mask         = Gumbel-STE(final_logits)
```

**Why this works where pure per-query routing fails:**
Previous attempts at self-supervised per-query routing (the original QA-MRL with SoftRouter/GroupRouter) suffered from collapse — without a strong prior, the router either assigns all queries the same mask or oscillates chaotically. BAM-PQ solves this via the **Bloom anchor**:
- At initialization, alpha ≈ 0.05, so the model behaves identically to Option B
- The query MLP output layer is zero-initialized — no per-query signal at epoch 0
- As training progresses, alpha grows *only if* per-query adjustment genuinely improves retrieval
- The Bloom anchor provides stable gradients for the sparsity/diversity losses even early in training

**What alpha tells you at convergence:** If alpha converges to ~0.1, per-query routing adds 10% on top of the Bloom prior. If alpha stays near 0.05, the Bloom level alone captures most of the routing signal. This is directly interpretable.

**Training:** Same reverse two-stage as Option B (8 epochs frozen + 12 unfrozen).

---

## 7. Backbone Selection

| Backbone | Dims | Params | MRL Pretrained? | Notes |
|----------|------|--------|-----------------|-------|
| BAAI/bge-base-en-v1.5 | 768 | 110M | Yes | Initial experiments; built-in MRL structure |
| intfloat/e5-large-v2 | 1024 | 335M | No | Scaling experiments; contrastive-trained only, requires MRL warm-start from scratch |

E5-large was chosen specifically because it is **not** MRL pre-trained. This means the MRL warm-start step is essential — the encoder must learn multi-resolution structure entirely from our training. This is a stricter test of the approach: if BAM works on an encoder with no MRL prior, the routing mechanism is genuinely learning useful dimension organization.

---

## 8. Data and Bloom Annotation

**Training data sources:**
- **SciQ** (allenai/sciq) — science support passages + questions
- **ARC-Easy / ARC-Challenge** (allenai/ai2_arc) — standardized science questions
- **OpenBookQA** (allenai/openbookqa) — open-book science facts
- **QASC** (allenai/qasc) — multi-hop science reasoning facts

**Bloom annotation pipeline:**
1. **cip29/bert-blooms-taxonomy-classifier** — BERT model fine-tuned on Bloom-labeled educational data. Fast but collapses to ~82% "Remember" on non-educational queries.
2. **MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli** — Zero-shot NLI classifier. Each Bloom level is framed as a natural-language hypothesis; the NLI model scores entailment probability. Better distribution across all 6 levels, especially for Evaluate and Create.

**Bloom distribution challenge:** Educational QA datasets are heavily skewed toward Remember and Understand. Evaluate and Create queries are extremely rare (<1% combined). The `analyze_bloom_datasets.py` script surveys 29 datasets to identify which can supplement the underrepresented levels.

**Hard negative mining:** BM25-based curriculum negatives — the most lexically similar non-relevant passages are the hardest negatives. Configurable num_neg (default 7) and curriculum stage (default 0.7).

---

## 9. Evaluation Methodology

**Epoch selection:** In-batch validation NDCG is misleading because it only compares within the batch (16 candidates). All epoch checkpoints are saved; true best is selected via:
- `find_best_epoch.py` — Corpus-level FAISS retrieval metrics
- `find_best_epoch_bsr.py` — **Bloom Stratified Recall (BSR)**: quality × (1 + α × efficiency). Balances retrieval quality against dimension compression.

**Metrics:**
- **Retrieval quality:** Recall@K (k=1,5,10,20,50,100), NDCG@10, MAP
- **Bloom-stratified:** Per-Bloom-level Recall@10 with 95% bootstrap confidence intervals
- **Efficiency:** Average active dimensions, sparse ratio (fraction of zeroed dims)
- **Statistical significance:** Wilcoxon signed-rank test + bootstrap CIs for model comparisons

**Out-of-domain evaluation (BEIR):**
- Trained on educational science data → evaluated on HotpotQA (multi-hop reasoning), SciFact, NFCorpus, TREC-COVID, SciDocs, Climate-FEVER
- BEIR queries are auto-annotated with Bloom levels using the NLI classifier so BAM routing is per-query, not defaulting to level 5
- Tests whether learned cognitive-level routing generalizes beyond the training domain

**Fair comparison protocol:** `eval_fair_comparison.py` compares BAM-B vs MRL at the **same dimension budget** per Bloom level. For each level, MRL is truncated to the same average active dims as BAM-B uses, ensuring any quality difference is due to dimension *selection* (scattered vs prefix), not dimension *count*.

---

## 10. How Each Approach Builds on the Previous

```
MRL Baseline
  │
  │  "Fixed truncation is suboptimal — different queries need different dims"
  │
  ▼
Option A (BloomDimRouter)
  │  + Learns per-Bloom-level prefix truncation point
  │  + Prefix-contiguous → FAISS-compatible
  │  - Still constrained to prefix ordering
  │  - Cannot select dims 42 and 511 while skipping dims 0-41
  │
  │  "Prefix constraint limits flexibility — scattered selection could be better"
  │
  ▼
Option B (BloomMaskHead)
  │  + Scattered binary mask per Bloom level
  │  + DropMask + DimRedist to combat MRL prefix bias
  │  + Reverse two-stage training for mask stability
  │  - One fixed mask per Bloom level — all "Analyze" queries identical
  │  - "Compare mitosis vs meiosis" and "Compare GDP of US vs China"
  │    get the same mask despite different information needs
  │
  │  "Same Bloom level ≠ same information need — per-query adjustment could help"
  │
  ▼
BAM-PQ (BloomQueryMaskHead)
     + Bloom anchor + per-query MLP residual
     + Alpha controls mixing: starts as pure Option B, learns per-query signal
     + Interpretable: alpha magnitude quantifies per-query routing value
     + Zero-init query MLP: only deviates from Option B if genuinely helpful
```

Each step adds expressiveness while maintaining the gains of the previous step. The Bloom taxonomy provides the structural prior that prevents routing collapse — the key failure mode that earlier QA-MRL approaches (SoftRouter, GroupRouter) suffered from.

---

## 11. Training and Evaluation Pipelines

| Pipeline | Data | Steps | Purpose |
|----------|------|-------|---------|
| `optionA-working_pipeline.sh` | Educational (BGE-base) | 7 | MRL baseline + Option A |
| `optionB-pipeline.sh` | Educational (BGE-base) | 5 | Option B (requires Option A's MRL) |
| `e5large-pipeline.sh` | Educational (E5-large) | 13 | All three variants + BEIR eval |
| `msmarco-pipeline.sh` | MS MARCO (E5-large) | 13 | General IR claim (not just educational) |
| `mixed-pipeline.sh` | Educational + BEIR mix | 13 | Multi-domain training |
| `emnlp-pipeline.sh` | Educational (E5-large) | Full | Publication-ready pipeline |

All pipelines support `--from <step>` for resumption and follow the same pattern:
```
data preparation → MRL warm-start → BAM training → BSR epoch selection → in-domain eval → BEIR eval
```
