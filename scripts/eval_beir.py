"""
BEIR Benchmark Evaluation for QA-MRL.

Evaluates on standard BEIR datasets to prove generality:
  - SciFact, NFCorpus, FiQA, ArguAna, TREC-COVID, etc.

Uses the beir library for standardized loading and evaluation.

Usage:
    python scripts/eval_beir.py --config configs/neurips.yaml \
        --checkpoint /tmp/qa-mrl-ckpts/best/ \
        --datasets scifact nfcorpus fiqa arguana \
        --output_dir results/beir/
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from models.qa_mrl import QAMRL
from models.encoder import MRLEncoder
from transformers import AutoTokenizer

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("Warning: faiss not available, using torch for search")


# ─────────────────────── BEIR Dataset Loading ───────────────────────

BEIR_DATASETS = [
    "scifact", "nfcorpus", "fiqa", "arguana", "trec-covid",
    "webis-touche2020", "quora", "scidocs", "nq", "hotpotqa",
    "fever", "climate-fever", "dbpedia-entity", "msmarco",
    "signal1m", "trec-news", "robust04", "bioasq",
]

# Smaller subset for quick evaluation
BEIR_QUICK = ["scifact", "nfcorpus", "fiqa", "arguana", "scidocs"]


def load_beir_dataset(dataset_name: str, split: str = "test"):
    """
    Load a BEIR dataset. Returns corpus, queries, qrels.

    Uses beir library if available, otherwise downloads manually.
    """
    try:
        from beir import util as beir_util
        from beir.datasets.data_loader import GenericDataLoader

        data_path = os.path.join("data/beir", dataset_name)
        if not os.path.exists(data_path):
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
            print(f"  Downloading {dataset_name}...")
            beir_util.download_and_unzip(url, "data/beir")

        corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)
        return corpus, queries, qrels

    except ImportError:
        # Fallback: use HuggingFace datasets
        from datasets import load_dataset

        print(f"  Loading {dataset_name} from HuggingFace...")
        try:
            ds = load_dataset(f"BeIR/{dataset_name}", "corpus", split="corpus")
            corpus = {}
            for row in ds:
                corpus[row["_id"]] = {
                    "title": row.get("title", ""),
                    "text": row.get("text", ""),
                }

            ds_q = load_dataset(f"BeIR/{dataset_name}", "queries", split="queries")
            queries = {}
            for row in ds_q:
                queries[row["_id"]] = row.get("text", "")

            # Load qrels
            ds_qrels = load_dataset(f"BeIR/{dataset_name}-qrels", split=split)
            qrels = defaultdict(dict)
            for row in ds_qrels:
                qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])

            return corpus, queries, dict(qrels)

        except Exception as e:
            print(f"  Failed to load {dataset_name}: {e}")
            return None, None, None


# ─────────────────────── Encoding ───────────────────────

@torch.no_grad()
def encode_texts(model, texts: List[str], tokenizer, device,
                  is_query: bool = False, batch_size: int = 128) -> np.ndarray:
    """Encode a list of texts into numpy embeddings."""
    model.eval()
    all_embs = []

    for i in tqdm(range(0, len(texts), batch_size), desc="  encoding", leave=False):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                       max_length=256 if not is_query else 128,
                       return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        if is_query and hasattr(model, "encode_queries"):
            out = model.encode_queries(enc["input_ids"], enc["attention_mask"])
            emb = out["masked_embedding"]
        elif not is_query and hasattr(model, "encode_documents"):
            out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
            emb = out["masked_embedding"]
        else:
            out = model(enc["input_ids"], enc["attention_mask"])
            emb = out["full"]

        all_embs.append(emb.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


# ─────────────────────── Retrieval ───────────────────────

def retrieve_faiss(query_embs: np.ndarray, corpus_embs: np.ndarray,
                    k: int = 100) -> Dict[int, List[Tuple[int, float]]]:
    """Retrieve using FAISS IndexFlatIP."""
    dim = corpus_embs.shape[1]
    query_embs = np.ascontiguousarray(query_embs.astype(np.float32))
    corpus_embs = np.ascontiguousarray(corpus_embs.astype(np.float32))

    if HAS_FAISS:
        index = faiss.IndexFlatIP(dim)
        index.add(corpus_embs)
        scores, indices = index.search(query_embs, k)
    else:
        # Fallback: torch
        q = torch.from_numpy(query_embs)
        c = torch.from_numpy(corpus_embs)
        results_scores, results_indices = [], []
        for i in range(0, len(q), 256):
            sim = torch.mm(q[i:i+256], c.t())
            s, idx = sim.topk(k, dim=-1)
            results_scores.append(s.numpy())
            results_indices.append(idx.numpy())
        scores = np.concatenate(results_scores)
        indices = np.concatenate(results_indices)

    return scores, indices


def retrieve_faiss_sparse(query_embs: np.ndarray, corpus_embs: np.ndarray,
                           query_masks: np.ndarray, k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Efficient sparse retrieval: only compute dot products over active dimensions.

    This is the TRUE efficiency implementation for QA-MRL.
    For each query, we extract only the non-zero dimensions and search
    over the corresponding corpus dimensions.

    For production: you'd precompute group-specific sub-indices.
    For evaluation: we do per-query sparse dot product.
    """
    N_q = query_embs.shape[0]
    N_c = corpus_embs.shape[0]

    all_scores = np.zeros((N_q, k), dtype=np.float32)
    all_indices = np.zeros((N_q, k), dtype=np.int64)

    # Group queries by their active mask pattern for batched processing
    mask_patterns = defaultdict(list)
    for i in range(N_q):
        active = tuple(np.where(query_masks[i] > 0.5)[0])
        mask_patterns[active].append(i)

    for active_dims, query_indices in tqdm(mask_patterns.items(),
                                            desc="  sparse search", leave=False):
        if len(active_dims) == 0:
            continue

        active_dims = list(active_dims)

        # Extract only active dimensions
        q_sparse = query_embs[query_indices][:, active_dims]  # [batch, d_active]
        c_sparse = corpus_embs[:, active_dims]                 # [N_c, d_active]

        # Normalize
        q_norm = q_sparse / (np.linalg.norm(q_sparse, axis=1, keepdims=True) + 1e-9)
        c_norm = c_sparse / (np.linalg.norm(c_sparse, axis=1, keepdims=True) + 1e-9)

        if HAS_FAISS and len(active_dims) > 1:
            d_active = len(active_dims)
            index = faiss.IndexFlatIP(d_active)
            index.add(np.ascontiguousarray(c_norm.astype(np.float32)))
            scores, indices = index.search(
                np.ascontiguousarray(q_norm.astype(np.float32)), k
            )
        else:
            q_t = torch.from_numpy(q_norm)
            c_t = torch.from_numpy(c_norm)
            sim = torch.mm(q_t, c_t.t())
            scores_t, indices_t = sim.topk(min(k, N_c), dim=-1)
            scores = scores_t.numpy()
            indices = indices_t.numpy()

        for j, qi in enumerate(query_indices):
            all_scores[qi] = scores[j]
            all_indices[qi] = indices[j]

    return all_scores, all_indices


# ─────────────────────── BEIR Metrics ───────────────────────

def compute_beir_metrics(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    ks: List[int] = [1, 5, 10, 100],
) -> Dict[str, float]:
    """Compute standard BEIR metrics: NDCG@10, Recall@K, MAP."""
    metrics = {}

    # NDCG@K
    for k in ks:
        ndcgs = []
        for qid in qrels:
            if qid not in results:
                ndcgs.append(0.0)
                continue
            qrel = qrels[qid]
            res = results[qid]

            # Sort by score
            sorted_docs = sorted(res.items(), key=lambda x: x[1], reverse=True)[:k]

            # DCG
            dcg = 0.0
            for i, (did, score) in enumerate(sorted_docs):
                rel = qrel.get(did, 0)
                dcg += (2**rel - 1) / np.log2(i + 2)

            # IDCG
            ideal_rels = sorted(qrel.values(), reverse=True)[:k]
            idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal_rels))

            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        metrics[f"ndcg@{k}"] = float(np.mean(ndcgs))

    # Recall@K
    for k in ks:
        recalls = []
        for qid in qrels:
            if qid not in results:
                recalls.append(0.0)
                continue
            qrel = {did: rel for did, rel in qrels[qid].items() if rel > 0}
            if not qrel:
                continue
            sorted_docs = sorted(results.get(qid, {}).items(),
                                key=lambda x: x[1], reverse=True)[:k]
            retrieved_relevant = sum(1 for did, _ in sorted_docs if did in qrel)
            recalls.append(retrieved_relevant / len(qrel))
        metrics[f"recall@{k}"] = float(np.mean(recalls))

    # MAP
    aps = []
    for qid in qrels:
        if qid not in results:
            aps.append(0.0)
            continue
        qrel = {did: rel for did, rel in qrels[qid].items() if rel > 0}
        if not qrel:
            continue
        sorted_docs = sorted(results.get(qid, {}).items(),
                            key=lambda x: x[1], reverse=True)
        hits = 0
        sum_prec = 0.0
        for i, (did, _) in enumerate(sorted_docs):
            if did in qrel:
                hits += 1
                sum_prec += hits / (i + 1)
        aps.append(sum_prec / len(qrel) if qrel else 0.0)
    metrics["map"] = float(np.mean(aps))

    return metrics


# ─────────────────────── Bootstrap CI ───────────────────────

def bootstrap_confidence_interval(
    qrels: Dict, results: Dict, metric_fn, n_bootstrap: int = 1000, ci: float = 0.95
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric."""
    qids = list(qrels.keys())
    n = len(qids)
    bootstrap_values = []

    for _ in range(n_bootstrap):
        sample_ids = np.random.choice(qids, size=n, replace=True)
        sample_qrels = {qid: qrels[qid] for qid in sample_ids if qid in qrels}
        sample_results = {qid: results.get(qid, {}) for qid in sample_ids}
        m = metric_fn(sample_qrels, sample_results)
        bootstrap_values.append(m)

    bootstrap_values = sorted(bootstrap_values)
    lower = bootstrap_values[int((1 - ci) / 2 * n_bootstrap)]
    upper = bootstrap_values[int((1 + ci) / 2 * n_bootstrap)]
    mean = np.mean(bootstrap_values)
    return mean, lower, upper


# ─────────────────────── Main Evaluation ───────────────────────

def evaluate_on_beir(
    model,
    tokenizer,
    device,
    dataset_name: str,
    model_name: str = "QA-MRL",
    use_sparse: bool = False,
    mrl_truncation_dims: List[int] = None,
) -> Dict[str, float]:
    """Evaluate a model on a single BEIR dataset."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on {dataset_name}")
    print(f"{'='*60}")

    # Load dataset
    corpus, queries, qrels = load_beir_dataset(dataset_name)
    if corpus is None:
        print(f"  Skipping {dataset_name} (failed to load)")
        return {}

    print(f"  Corpus: {len(corpus)}, Queries: {len(queries)}, Qrels: {len(qrels)}")

    # Prepare texts
    corpus_ids = list(corpus.keys())
    corpus_texts = [
        (corpus[cid].get("title", "") + " " + corpus[cid].get("text", "")).strip()
        for cid in corpus_ids
    ]
    corpus_id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}

    query_ids = [qid for qid in queries if qid in qrels]
    query_texts = [queries[qid] for qid in query_ids]

    if not query_texts:
        print(f"  No valid queries for {dataset_name}")
        return {}

    print(f"  Valid queries: {len(query_texts)}")

    # Encode
    print("  Encoding corpus...")
    t0 = time.time()
    corpus_embs = encode_texts(model, corpus_texts, tokenizer, device,
                                is_query=False, batch_size=128)
    encode_corpus_time = time.time() - t0

    print("  Encoding queries...")
    t0 = time.time()
    query_embs = encode_texts(model, query_texts, tokenizer, device,
                               is_query=True, batch_size=64)
    encode_query_time = time.time() - t0

    # Get query masks for sparse retrieval
    query_masks = None
    if use_sparse and hasattr(model, "encode_queries"):
        print("  Getting query masks for sparse retrieval...")
        all_masks = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(query_texts), 64):
                batch = query_texts[i:i+64]
                enc = tokenizer(batch, padding=True, truncation=True,
                               max_length=128, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                out = model.encode_queries(enc["input_ids"], enc["attention_mask"])
                all_masks.append(out["mask"].cpu().numpy())
        query_masks = np.concatenate(all_masks)

    # Retrieve
    print("  Retrieving...")
    t0 = time.time()
    if use_sparse and query_masks is not None:
        scores, indices = retrieve_faiss_sparse(query_embs, corpus_embs, query_masks, k=100)
        avg_active = (query_masks > 0.5).sum(axis=1).mean()
        print(f"  Sparse retrieval: avg {avg_active:.0f} active dims")
    else:
        scores, indices = retrieve_faiss(query_embs, corpus_embs, k=100)
    search_time = time.time() - t0

    # Build results dict
    results = {}
    for i, qid in enumerate(query_ids):
        results[qid] = {}
        for j in range(indices.shape[1]):
            corpus_idx = int(indices[i, j])
            if corpus_idx < len(corpus_ids):
                cid = corpus_ids[corpus_idx]
                results[qid][cid] = float(scores[i, j])

    # Compute metrics
    metrics = compute_beir_metrics(qrels, results)

    # Timing
    metrics["encode_corpus_s"] = encode_corpus_time
    metrics["encode_query_s"] = encode_query_time
    metrics["search_s"] = search_time
    metrics["ms_per_query"] = search_time / len(query_texts) * 1000
    if query_masks is not None:
        metrics["avg_active_dims"] = float((query_masks > 0.5).sum(axis=1).mean())

    # MRL truncation comparison
    if mrl_truncation_dims and not hasattr(model, "query_router"):
        print("  MRL truncation comparisons...")
        for d in mrl_truncation_dims:
            q_trunc = query_embs[:, :d].copy()
            c_trunc = corpus_embs[:, :d].copy()
            # Re-normalize
            q_trunc /= (np.linalg.norm(q_trunc, axis=1, keepdims=True) + 1e-9)
            c_trunc /= (np.linalg.norm(c_trunc, axis=1, keepdims=True) + 1e-9)

            t_scores, t_indices = retrieve_faiss(q_trunc, c_trunc, k=100)
            t_results = {}
            for i, qid in enumerate(query_ids):
                t_results[qid] = {}
                for j in range(t_indices.shape[1]):
                    cidx = int(t_indices[i, j])
                    if cidx < len(corpus_ids):
                        t_results[qid][corpus_ids[cidx]] = float(t_scores[i, j])

            t_metrics = compute_beir_metrics(qrels, t_results)
            for mk, mv in t_metrics.items():
                metrics[f"mrl_d{d}_{mk}"] = mv

    # Print
    print(f"\n  Results on {dataset_name}:")
    print(f"    NDCG@10: {metrics.get('ndcg@10', 0):.4f}")
    print(f"    R@10:    {metrics.get('recall@10', 0):.4f}")
    print(f"    R@100:   {metrics.get('recall@100', 0):.4f}")
    print(f"    MAP:     {metrics.get('map', 0):.4f}")
    if "avg_active_dims" in metrics:
        print(f"    Active dims: {metrics['avg_active_dims']:.0f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="BEIR evaluation")
    parser.add_argument("--config", default="configs/neurips.yaml")
    parser.add_argument("--checkpoint", required=True, help="QA-MRL checkpoint")
    parser.add_argument("--baseline", default=None, help="MRL baseline checkpoint")
    parser.add_argument("--datasets", nargs="+", default=BEIR_QUICK,
                        help="BEIR datasets to evaluate on")
    parser.add_argument("--sparse", action="store_true",
                        help="Use sparse retrieval (true efficiency)")
    parser.add_argument("--output_dir", default="results/beir/")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])

    all_results = {}

    # Evaluate QA-MRL
    print("\n" + "=" * 70)
    print("QA-MRL EVALUATION")
    print("=" * 70)
    qa_model = QAMRL(config)
    ckpt = os.path.join(args.checkpoint, "checkpoint.pt")
    if os.path.exists(ckpt):
        qa_model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
    qa_model.to(device).eval()

    qa_results = {}
    for ds_name in args.datasets:
        metrics = evaluate_on_beir(
            qa_model, tokenizer, device, ds_name,
            model_name="QA-MRL", use_sparse=args.sparse,
        )
        qa_results[ds_name] = metrics
    all_results["QA-MRL"] = qa_results

    # Evaluate baseline
    if args.baseline:
        print("\n" + "=" * 70)
        print("MRL BASELINE EVALUATION")
        print("=" * 70)
        mc = config["model"]
        bl_model = MRLEncoder(model_name=mc["backbone"], embedding_dim=mc["embedding_dim"],
                              mrl_dims=mc["mrl_dims"])
        ckpt = os.path.join(args.baseline, "checkpoint.pt")
        if os.path.exists(ckpt):
            bl_model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
        bl_model.to(device).eval()

        bl_results = {}
        for ds_name in args.datasets:
            metrics = evaluate_on_beir(
                bl_model, tokenizer, device, ds_name,
                model_name="MRL Baseline",
                mrl_truncation_dims=[64, 128, 256, 384, 512],
            )
            bl_results[ds_name] = metrics
        all_results["MRL Baseline"] = bl_results

    # Save
    with open(os.path.join(args.output_dir, "beir_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    # Print comparison table
    print("\n" + "=" * 80)
    print("BEIR COMPARISON TABLE")
    print("=" * 80)

    models = list(all_results.keys())
    header = f"{'Dataset':20s}" + "".join(f"{m:>20s}" for m in models)
    print(header)
    print("-" * len(header))

    for ds_name in args.datasets:
        row = f"{ds_name:20s}"
        for model_name in models:
            ndcg = all_results[model_name].get(ds_name, {}).get("ndcg@10", 0)
            row += f"{ndcg:>20.4f}"
        print(row)

    # Average
    print("-" * len(header))
    row = f"{'Average':20s}"
    for model_name in models:
        ndcgs = [all_results[model_name].get(ds, {}).get("ndcg@10", 0) for ds in args.datasets]
        row += f"{np.mean(ndcgs):>20.4f}"
    print(row)

    print(f"\nResults saved to {args.output_dir}/beir_results.json")


if __name__ == "__main__":
    main()