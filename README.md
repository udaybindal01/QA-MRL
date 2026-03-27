# QA-MRL: Query-Adaptive Matryoshka Retrieval for Educational Information Retrieval

## Paper Title
**"Query-Adaptive Matryoshka Retrieval: Learning Per-Query Dimension Routing for Educational Information Retrieval"**

## Abstract
Matryoshka Representation Learning (MRL) produces embeddings where the first *d* dimensions 
form a valid lower-dimensional representation. However, MRL enforces a rigid dimensional 
hierarchy—dimension 1 is always "most important"—regardless of the query. We demonstrate 
empirically that different query types (factual recall, conceptual, procedural, multi-hop) 
activate different dimensions, making this fixed ordering suboptimal. We propose **QA-MRL**, 
which learns a lightweight per-query dimension router that selects query-specific dimension 
subsets, breaking MRL's nesting assumption while retaining efficiency. Applied to educational 
retrieval, QA-MRL achieves significant gains on Bloom's-level stratified metrics, with 
especially large improvements on complex reasoning queries where standard MRL systematically 
fails.

## Project Structure
```
qa-mrl/
├── configs/
│   └── default_config.yaml         # All hyperparameters and experiment configs
├── data/
│   ├── __init__.py
│   ├── dataset.py                   # Dataset classes for educational retrieval
│   ├── data_loader.py               # DataLoader with hard negative mining
│   ├── bloom_annotator.py           # Bloom's taxonomy annotation pipeline
│   └── preprocessing.py             # Text preprocessing and corpus building
├── models/
│   ├── __init__.py
│   ├── encoder.py                   # Base bi-encoder with MRL support
│   ├── mrl_encoder.py              # Standard Matryoshka encoder (baseline)
│   ├── qa_mrl_encoder.py           # QA-MRL with dimension routing (ours)
│   ├── routing.py                   # Soft gating & hard group routing modules
│   └── pooling.py                   # Pooling strategies
├── training/
│   ├── __init__.py
│   ├── trainer.py                   # Main training loop
│   ├── losses.py                    # Contrastive + routing losses
│   ├── hard_negatives.py            # Hard negative mining strategies
│   └── scheduler.py                 # LR scheduling with warmup
├── evaluation/
│   ├── __init__.py
│   ├── retrieval_metrics.py         # Standard retrieval metrics
│   ├── bloom_stratified.py          # Bloom's-level stratified evaluation
│   ├── tail_topic_eval.py           # Tail/rare concept evaluation
│   ├── evaluator.py                 # Main evaluation orchestrator
│   └── latency_benchmark.py         # Efficiency benchmarking
├── analysis/
│   ├── __init__.py
│   ├── dimension_probing.py         # Probe which dims matter per query type
│   ├── gradient_attribution.py      # Gradient-based dimension importance
│   ├── visualization.py             # t-SNE, heatmaps, routing patterns
│   └── group_specialization.py      # Analyze what each dim group captures
├── scripts/
│   ├── run_diagnostic.py            # Step 1: Diagnostic analysis
│   ├── train_baseline_mrl.py        # Step 2: Train standard MRL baseline
│   ├── train_qa_mrl.py              # Step 3: Train QA-MRL (ours)
│   ├── evaluate_all.py              # Step 4: Full evaluation suite
│   ├── run_ablations.py             # Step 5: Ablation studies
│   └── generate_paper_figures.py    # Step 6: Generate all paper figures
├── utils/
│   ├── __init__.py
│   ├── logging_utils.py             # Logging and W&B integration
│   ├── misc.py                      # Seeds, device setup, etc.
│   └── faiss_index.py               # FAISS indexing utilities
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Run Diagnostic Analysis (Go/No-Go Gate)
```bash
python scripts/run_diagnostic.py --config configs/default_config.yaml
```
This trains a standard MRL model, then probes which dimensions matter for which 
query types. If different query types need different dimensions, proceed.

### 3. Train QA-MRL
```bash
python scripts/train_qa_mrl.py --config configs/default_config.yaml
```

### 4. Evaluate
```bash
python scripts/evaluate_all.py --config configs/default_config.yaml
```

### 5. Ablations
```bash
python scripts/run_ablations.py --config configs/default_config.yaml
```

## Key Contributions
1. **Diagnostic Analysis**: First systematic study showing MRL dimensions are query-type dependent
2. **Query-Adaptive Routing**: Lightweight per-query dimension router (soft gating + hard group routing)
3. **Asymmetric Query-Document Dimensions**: Queries and documents use different dimension subsets
4. **Bloom's-Stratified Evaluation**: Novel evaluation protocol for educational retrieval
5. **Specialization Loss**: Auxiliary loss encouraging dimension groups to capture distinct aspects
