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
from models.bam import BloomAlignedMRL
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


def load_local_jsonl_beir(data_dir: str, split: str = "test"):
    """
    Load BEIR dataset from local JSONL files (pipeline pair format).

    corpus.jsonl  — one doc per line: {"_id": ..., "title": ..., "text": ...}
    {split}.jsonl — one pair per line (build_beir_training_data.py output):
        {"query": ..., "positive_id": ..., "negative_ids": [...],
         "bloom_level": ..., "subject": ...}

    Returns corpus, queries, qrels dicts suitable for retrieval eval.
    Each pair line becomes one query; qrel = {positive_id: 1}.
    Queries sharing the same text are merged (all their positives pooled).
    """
    import json as _json
    corpus_path = os.path.join(data_dir, "corpus.jsonl")
    split_path  = os.path.join(data_dir, f"{split}.jsonl")
    if not os.path.exists(corpus_path) or not os.path.exists(split_path):
        return None, None, None

    corpus = {}
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = _json.loads(line)
            cid = str(row.get("_id") or row.get("doc_id") or row.get("id", ""))
            if cid:
                corpus[cid] = {"title": row.get("title", ""), "text": row.get("text", "")}

    # Merge duplicate queries (same text → same qid, pool positives)
    text_to_qid: dict = {}
    queries: dict = {}
    qrels: dict = defaultdict(dict)
    bloom_map: dict = {}   # qid → 0-indexed bloom level

    with open(split_path) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = _json.loads(line)

            # Support pipeline pair format {"query":..., "positive_id":...}
            # and standard BEIR query format {"_id":..., "text":...}
            qtxt = row.get("query") or row.get("text") or ""
            pos_id = str(row.get("positive_id") or "")

            # Use query text as dedup key; fall back to index-based ID
            if qtxt in text_to_qid:
                qid = text_to_qid[qtxt]
            else:
                qid = str(row.get("_id") or row.get("query_id") or f"q{idx}")
                text_to_qid[qtxt] = qid
                queries[qid] = qtxt
                # bloom_level in pair records is 1-indexed; convert to 0-indexed
                bl = row.get("bloom_level") or row.get("predicted_bloom_level")
                if bl is not None:
                    bloom_map[qid] = max(0, min(int(bl) - 1, 5))

            if pos_id and pos_id in corpus:
                qrels[qid][pos_id] = 1

            # Also handle explicit relevant_docs list
            for rel in row.get("relevant_docs", []):
                cid = str(rel.get("doc_id") or rel.get("_id") or rel) \
                      if isinstance(rel, dict) else str(rel)
                if cid in corpus:
                    qrels[qid][cid] = int(rel.get("score", 1)) \
                                      if isinstance(rel, dict) else 1

    print(f"  Loaded from local JSONL: corpus={len(corpus):,}  "
          f"queries={len(queries):,}  qrels={len(qrels):,}  "
          f"bloom_labels={len(bloom_map):,}")
    return corpus, queries, dict(qrels), bloom_map


def load_beir_dataset(dataset_name: str, split: str = "test",
                      beir_data_root: str = None):
    """
    Load a BEIR dataset. Checks local path first, then downloads.
    beir_data_root: if set, look for {beir_data_root}/{dataset_name}/
    """
    # ── 1. Try local JSONL (pipeline pre-built format) ────────────────────────
    search_dirs = []
    if beir_data_root:
        search_dirs.append(os.path.join(beir_data_root, dataset_name))
    search_dirs += [
        os.path.join("data/beir", dataset_name),
        os.path.join("/tmp/data/beir", dataset_name),
    ]
    for local_dir in search_dirs:
        if os.path.isdir(local_dir):
            result = load_local_jsonl_beir(local_dir, split)
            if result[0] is not None:
                corpus, queries, qrels, bloom_map = result
                print(f"  Using local data at {local_dir}")
                return corpus, queries, qrels, bloom_map
            # Also try beir GenericDataLoader for standard BEIR format
            try:
                from beir.datasets.data_loader import GenericDataLoader
                corpus, queries, qrels = GenericDataLoader(local_dir).load(split=split)
                print(f"  Using local BEIR data at {local_dir}")
                return corpus, queries, qrels, {}
            except Exception:
                pass

    # ── 2. Download via beir library ──────────────────────────────────────────
    try:
        from beir import util as beir_util
        from beir.datasets.data_loader import GenericDataLoader

        data_path = os.path.join("data/beir", dataset_name)
        if not os.path.exists(data_path):
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
            print(f"  Downloading {dataset_name}...")
            beir_util.download_and_unzip(url, "data/beir")

        corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)
        return corpus, queries, qrels, {}

    except ImportError:
        pass
    except Exception as e:
        print(f"  beir library failed ({e}), trying HuggingFace ...")

    # ── 3. HuggingFace fallback ───────────────────────────────────────────────
    try:
        from datasets import load_dataset
        print(f"  Loading {dataset_name} from HuggingFace...")
        ds = load_dataset(f"BeIR/{dataset_name}", "corpus", split="corpus")
        corpus = {row["_id"]: {"title": row.get("title", ""), "text": row.get("text", "")}
                  for row in ds}
        ds_q = load_dataset(f"BeIR/{dataset_name}", "queries", split="queries")
        queries = {row["_id"]: row.get("text", "") for row in ds_q}
        ds_qrels = load_dataset(f"BeIR/{dataset_name}-qrels", split=split)
        qrels = defaultdict(dict)
        for row in ds_qrels:
            qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])
        return corpus, queries, dict(qrels), {}
    except Exception as e:
        print(f"  Failed to load {dataset_name}: {e}")
        return None, None, None, {}


# ─────────────────────── Encoding ───────────────────────

def annotate_bloom_labels(query_texts: List[str], device,
                          cache_path: str = None,
                          query_ids: List[str] = None) -> List[int]:
    """
    Returns 0-indexed Bloom labels (0=Remember … 5=Create).
    Loads from cache_path if available, otherwise runs the classifier.
    cache_path: path to {split}.jsonl.bloom_cache.json produced by the pipeline.
    """
    import json as _json
    from collections import Counter

    bloom_names = {0: "Remember", 1: "Understand", 2: "Apply",
                   3: "Analyze", 4: "Evaluate", 5: "Create"}

    # ── Try cache first ───────────────────────────────────────────────────────
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading Bloom labels from cache: {cache_path}")
        cache = _json.load(open(cache_path))
        # cache may be keyed by query_id or be a list
        if isinstance(cache, dict) and query_ids is not None:
            labels = []
            for qid in query_ids:
                entry = cache.get(str(qid), cache.get(qid))
                if entry is None:
                    labels.append(0)
                elif isinstance(entry, dict):
                    lv = entry.get("bloom_level", entry.get("label", 1))
                    labels.append(int(lv) - 1 if int(lv) >= 1 else int(lv))
                else:
                    lv = int(entry)
                    labels.append(lv - 1 if lv >= 1 else lv)
        elif isinstance(cache, list):
            labels = []
            for entry in cache:
                if isinstance(entry, dict):
                    lv = entry.get("bloom_level", entry.get("label", 1))
                    labels.append(int(lv) - 1 if int(lv) >= 1 else int(lv))
                else:
                    labels.append(int(entry) - 1)
            labels = labels[:len(query_texts)]
        else:
            # flat dict keyed by text or index
            labels = [0] * len(query_texts)
            for i, qt in enumerate(query_texts):
                if qt in cache:
                    lv = int(cache[qt]) if not isinstance(cache[qt], dict) \
                         else int(cache[qt].get("bloom_level", 1))
                    labels[i] = lv - 1 if lv >= 1 else lv

        dist = Counter(labels)
        print("  Bloom distribution (from cache):")
        for b in sorted(dist):
            print(f"    {bloom_names.get(b, b)}: {dist[b]} ({dist[b]/len(labels):.1%})")
        return labels

    # ── Run classifier ────────────────────────────────────────────────────────
    print("  Annotating queries with Bloom classifier...")
    try:
        from data.annotate_bloom_pretrained import load_pretrained_classifier, predict_bloom
        bloom_model, bloom_tok, id2label = load_pretrained_classifier(device=device)
        labels_1idx = predict_bloom(query_texts, bloom_model, bloom_tok,
                                    device=device, id2label=id2label)
        labels_0idx = [l - 1 for l in labels_1idx]
        del bloom_model
        if str(device) != "cpu":
            torch.cuda.empty_cache()
    except ImportError:
        # NLI zero-shot fallback
        print("  annotate_bloom_pretrained not found — using NLI zero-shot classifier.")
        from transformers import pipeline as hf_pipeline
        HYPOTHESES = [
            "This query is asking to recall or retrieve a specific fact, name, or definition.",
            "This query is asking to explain, describe, or summarize how something works.",
            "This query is asking how to use or apply knowledge to solve a practical problem.",
            "This query is asking to compare, contrast, or examine the relationship between things.",
            "This query is asking to evaluate evidence, assess effectiveness, or judge quality.",
            "This query is asking to design, propose, or synthesize something new.",
        ]
        clf = hf_pipeline("zero-shot-classification",
                          model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                          device=0 if str(device) != "cpu" else -1,
                          hypothesis_template="{}")
        labels_0idx = []
        for i in range(0, len(query_texts), 32):
            batch = query_texts[i:i+32]
            res = clf(batch, HYPOTHESES, multi_label=False)
            if isinstance(res, dict): res = [res]
            for r in res:
                labels_0idx.append(HYPOTHESES.index(r["labels"][0]))
        del clf
        if str(device) != "cpu":
            torch.cuda.empty_cache()

    dist = Counter(labels_0idx)
    print("  Bloom distribution:")
    for b in sorted(dist):
        print(f"    {bloom_names.get(b, b)}: {dist[b]} ({dist[b]/len(labels_0idx):.1%})")
    return labels_0idx


@torch.no_grad()
def encode_texts(model, texts: List[str], tokenizer, device,
                  is_query: bool = False, batch_size: int = 128,
                  bloom_labels: Optional[List[int]] = None) -> np.ndarray:
    """Encode a list of texts into numpy embeddings.

    Args:
        bloom_labels: 0-indexed Bloom labels per query (0=Remember...5=Create).
                      Only used when is_query=True and model has encode_queries.
    """
    model.eval()
    _query_instr = None
    if is_query:
        if hasattr(model, "encoder"):
            _query_instr = getattr(model.encoder, "query_instruction", None)
        elif hasattr(model, "query_instruction"):
            _query_instr = model.query_instruction

    all_embs = []

    for i in tqdm(range(0, len(texts), batch_size), desc="  encoding", leave=False):
        batch = texts[i:i+batch_size]
        if _query_instr:
            batch = [_query_instr + t for t in batch]
        enc = tokenizer(batch, padding=True, truncation=True,
                       max_length=256 if not is_query else 128,
                       return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        if is_query and hasattr(model, "encode_queries"):
            kwargs = {"input_ids": enc["input_ids"],
                      "attention_mask": enc["attention_mask"]}
            if bloom_labels is not None:
                batch_labels = bloom_labels[i:i+batch_size]
                kwargs["bloom_labels"] = torch.tensor(
                    batch_labels, dtype=torch.long, device=device
                )
            out = model.encode_queries(**kwargs)
            emb = out["masked_embedding"]
        elif not is_query and hasattr(model, "encode_documents"):
            out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
            emb = out["masked_embedding"]
        else:
            out = model(enc["input_ids"], enc["attention_mask"])
            emb = out["full"]

        # Use float16 for corpus embeddings to halve memory (5M-doc corpora like HotpotQA)
        arr = emb.cpu().half().numpy() if not is_query else emb.cpu().numpy()
        all_embs.append(arr)

    result = np.concatenate(all_embs, axis=0)
    # Retrieval needs float32; cast back only at search time (done in retrieve_faiss)
    return result


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
    BAM-PQ retrieval: masked query embeddings vs single full-corpus FAISS index.

    query_embs are already normalize(q_full * mask) from encode_queries; corpus_embs
    are normalize(c_full). Zero query dims are naturally ignored in the dot product,
    so this matches the deployment scenario (one FAISS index, masked queries).

    query_masks is kept in the signature for compatibility / avg_active_dims logging
    but is not used for retrieval — masking lives entirely on the query side.
    """
    return retrieve_faiss(query_embs, corpus_embs, k=k)


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
    split: str = "test",
    beir_data_root: str = None,
) -> Dict[str, float]:
    """Evaluate a model on a single BEIR dataset."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on {dataset_name}")
    print(f"{'='*60}")

    # Load dataset
    corpus, queries, qrels, local_bloom_map = load_beir_dataset(
        dataset_name, split=split, beir_data_root=beir_data_root)
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

    # Annotate queries with Bloom levels for BAM routing
    bloom_labels = None
    if hasattr(model, "encode_queries"):
        if local_bloom_map and all(qid in local_bloom_map for qid in query_ids):
            # Use bloom levels read directly from the JSONL pair records
            bloom_labels = [local_bloom_map[qid] for qid in query_ids]
            from collections import Counter
            dist = Counter(bloom_labels)
            bnames = {0:"Remember",1:"Understand",2:"Apply",3:"Analyze",4:"Evaluate",5:"Create"}
            print("  Bloom distribution (from pair records):")
            for b in sorted(dist):
                print(f"    {bnames.get(b,b)}: {dist[b]} ({dist[b]/len(bloom_labels):.1%})")
        else:
            # Fall back: check cache file then run classifier
            cache_path = None
            if beir_data_root:
                candidate = os.path.join(beir_data_root, dataset_name,
                                         f"{split}.jsonl.bloom_cache.json")
                if os.path.exists(candidate):
                    cache_path = candidate
            bloom_labels = annotate_bloom_labels(
                query_texts, device,
                cache_path=cache_path,
                query_ids=query_ids,
            )

    # Encode
    print("  Encoding corpus...")
    t0 = time.time()
    corpus_embs = encode_texts(model, corpus_texts, tokenizer, device,
                                is_query=False, batch_size=128)
    encode_corpus_time = time.time() - t0

    print("  Encoding queries...")
    t0 = time.time()
    query_embs = encode_texts(model, query_texts, tokenizer, device,
                               is_query=True, batch_size=64,
                               bloom_labels=bloom_labels)
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
                kwargs = {"input_ids": enc["input_ids"],
                          "attention_mask": enc["attention_mask"]}
                if bloom_labels is not None:
                    batch_labels = bloom_labels[i:i+64]
                    kwargs["bloom_labels"] = torch.tensor(
                        batch_labels, dtype=torch.long, device=device
                    )
                out = model.encode_queries(**kwargs)
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
    parser.add_argument("--model_type", choices=["qamrl", "bam", "mrl"], default="qamrl",
                        help="Model architecture: qamrl (default), bam (BloomAlignedMRL), mrl (MRLEncoder)")
    parser.add_argument("--sparse", action="store_true",
                        help="Use sparse retrieval (true efficiency)")
    parser.add_argument("--split", default="test",
                        help="Dataset split (default: test; use 'dev' for msmarco)")
    parser.add_argument("--output_dir", default="results/beir/")
    parser.add_argument("--beir_data_root", default=None,
                        help="Root dir containing local BEIR JSONL data "
                             "(e.g. /scratch/user/bampq-data/beir). "
                             "Checked before downloading.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])

    all_results = {}

    # Load primary model
    model_type = args.model_type
    print("\n" + "=" * 70)
    if model_type == "bam":
        model_label = "BAM"
        print(f"BAM EVALUATION (use_mask_routing={config['model'].get('use_mask_routing', False)})")
        print("=" * 70)
        primary_model = BloomAlignedMRL(config)
    elif model_type == "mrl":
        mc = config["model"]
        model_label = "MRL"
        print("MRL EVALUATION")
        print("=" * 70)
        primary_model = MRLEncoder(
            model_name=mc["backbone"], embedding_dim=mc["embedding_dim"],
            mrl_dims=mc["mrl_dims"], pooling=mc.get("pooling", "cls"),
            normalize=mc.get("normalize_embeddings", True),
        )
    else:
        model_label = "QA-MRL"
        print("QA-MRL EVALUATION")
        print("=" * 70)
        from models.qa_mrl import QAMRL  # deferred: requires einops
        primary_model = QAMRL(config)

    ckpt = os.path.join(args.checkpoint, "checkpoint.pt")
    if os.path.exists(ckpt):
        primary_model.load_state_dict(
            torch.load(ckpt, map_location=device)["model_state_dict"], strict=False
        )
    primary_model.to(device).eval()

    primary_results = {}
    mrl_trunc = config["model"].get("mrl_dims", None) if model_type == "mrl" else None
    for ds_name in args.datasets:
        metrics = evaluate_on_beir(
            primary_model, tokenizer, device, ds_name,
            model_name=model_label, use_sparse=args.sparse,
            mrl_truncation_dims=mrl_trunc,
            split=args.split,
            beir_data_root=args.beir_data_root,
        )
        primary_results[ds_name] = metrics
    all_results[model_label] = primary_results

    # Evaluate baseline
    if args.baseline:
        print("\n" + "=" * 70)
        print("MRL BASELINE EVALUATION")
        print("=" * 70)
        mc = config["model"]
        bl_model = MRLEncoder(model_name=mc["backbone"], embedding_dim=mc["embedding_dim"],
                              mrl_dims=mc["mrl_dims"],
                              pooling=mc.get("pooling", "cls"),
                              backbone_type=mc.get("backbone_type", "standard"))
        ckpt = os.path.join(args.baseline, "checkpoint.pt")
        if os.path.exists(ckpt):
            bl_model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"],
                                     strict=False)
        bl_model.to(device).eval()

        bl_results = {}
        for ds_name in args.datasets:
            metrics = evaluate_on_beir(
                bl_model, tokenizer, device, ds_name,
                model_name="MRL Baseline",
                mrl_truncation_dims=[64, 128, 256, 384, 512],
                split=args.split,
                beir_data_root=args.beir_data_root,
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