"""
Zero-shot evaluation of educational models on TREC-COVID, SciDocs, Climate-FEVER.

Models are trained ONLY on educational data — this measures domain transfer.
Queries are Bloom-annotated with the NLI classifier so per-level breakdowns work.

Models evaluated:
  MRL    — MRL encoder at 64/128/256/512/768/1024 and full dims
  BAM-B  — Bloom scattered mask (Option B)
  BAM-A  — Bloom prefix mask (Option A, optional — skipped if checkpoint absent)

Metrics: R@1, R@5, R@10, NDCG@10 — overall and stratified by Bloom level.

Usage:
    python scripts/eval_zero_shot.py \
        --mrl_checkpoint /tmp/multi-domain/educational/mrl/best/ \
        --mrl_config     results/multi_domain/educational/configs/mrl.yaml \
        --bam_b_checkpoint /tmp/multi-domain/educational/bam_b/best_bsr/ \
        --bam_b_config     results/multi_domain/educational/configs/bam_b.yaml \
        --datasets trec-covid scidocs climate-fever \
        --output_dir results/zero_shot/

    # Optional BAM-A
        --bam_a_checkpoint /tmp/multi-domain/educational/bam_a/best_bsr/ \
        --bam_a_config     results/multi_domain/educational/configs/bam_a.yaml \
"""

import argparse
import json
import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter, defaultdict
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

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

BLOOM_NAMES = {0: "Remember", 1: "Understand", 2: "Apply",
               3: "Analyze",  4: "Evaluate",   5: "Create"}

# Standard BEIR datasets for zero-shot transfer
ZERO_SHOT_DATASETS = ["trec-covid", "scidocs", "climate-fever"]


# ─────────────────────── Model loading ───────────────────────────────────────

def load_mrl_model(config, ckpt_path, device):
    mc = config["model"]
    model = MRLEncoder(
        model_name=mc["backbone"],
        embedding_dim=mc.get("embedding_dim", 768),
        mrl_dims=mc.get("mrl_dims", [64, 128, 256, 512, 768]),
    )
    ckpt_file = os.path.join(ckpt_path, "checkpoint.pt")
    if os.path.exists(ckpt_file):
        state = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(state["model_state_dict"], strict=False)
        print(f"  Loaded MRL from {ckpt_file}")
    else:
        print(f"  WARNING: no checkpoint.pt at {ckpt_path} — using random weights")
    return model.to(device).eval()


def load_bam_model(config, ckpt_path, device):
    config["training"]["loss"].setdefault("bloom_frequencies", [1 / 6] * 6)
    model = BloomAlignedMRL(config)
    ckpt_file = os.path.join(ckpt_path, "checkpoint.pt")
    if os.path.exists(ckpt_file):
        state = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(state["model_state_dict"], strict=False)
        print(f"  Loaded BAM from {ckpt_file}")
    else:
        print(f"  WARNING: no checkpoint.pt at {ckpt_path} — using random weights")
    return model.to(device).eval()


# ─────────────────────── BEIR dataset loading ────────────────────────────────

def load_beir_dataset(dataset_name: str, split: str = "test"):
    """Returns (corpus, queries, qrels) dicts in BEIR format."""
    try:
        from beir import util as beir_util
        from beir.datasets.data_loader import GenericDataLoader

        data_path = os.path.join("data/beir", dataset_name)
        if not os.path.exists(data_path):
            url = (f"https://public.ukp.informatik.tu-darmstadt.de/"
                   f"thakur/BEIR/datasets/{dataset_name}.zip")
            print(f"  Downloading {dataset_name} ...")
            beir_util.download_and_unzip(url, "data/beir")

        corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)
        return corpus, queries, qrels

    except ImportError:
        from datasets import load_dataset
        print(f"  Loading {dataset_name} from HuggingFace datasets ...")
        try:
            corpus_ds = load_dataset(f"BeIR/{dataset_name}", "corpus", split="corpus")
            corpus = {r["_id"]: {"title": r.get("title", ""), "text": r.get("text", "")}
                      for r in corpus_ds}

            queries_ds = load_dataset(f"BeIR/{dataset_name}", "queries", split="queries")
            queries = {r["_id"]: r.get("text", "") for r in queries_ds}

            qrels_ds = load_dataset(f"BeIR/{dataset_name}-qrels", split=split)
            qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
            for r in qrels_ds:
                qrels[str(r["query-id"])][str(r["corpus-id"])] = int(r["score"])
            return corpus, queries, dict(qrels)

        except Exception as exc:
            print(f"  FAILED to load {dataset_name}: {exc}")
            return None, None, None


# ─────────────────────── Bloom annotation ────────────────────────────────────

def annotate_bloom_nli(query_texts: List[str], device,
                       model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                       batch_size: int = 32) -> List[int]:
    """
    Classify queries into Bloom levels (0-indexed, 0=Remember…5=Create)
    using zero-shot NLI entailment.
    """
    from transformers import pipeline

    HYPOTHESES = [
        "This query is asking to recall or retrieve a specific fact, name, or definition.",
        "This query is asking to explain, describe, or summarize how something works.",
        "This query is asking how to use or apply knowledge to solve a practical problem.",
        "This query is asking to compare, contrast, or examine the relationship between things.",
        "This query is asking to evaluate evidence, assess effectiveness, or judge quality.",
        "This query is asking to design, propose, or synthesize something new.",
    ]

    print(f"  Annotating {len(query_texts)} queries with NLI Bloom classifier "
          f"({model_name}) ...")
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=0 if device.type == "cuda" else -1,
        hypothesis_template="{}",
    )

    labels: List[int] = []
    for i in tqdm(range(0, len(query_texts), batch_size),
                  desc="  Bloom-NLI", leave=False):
        batch = query_texts[i: i + batch_size]
        results = classifier(batch, HYPOTHESES, multi_label=False)
        if isinstance(results, dict):
            results = [results]
        for r in results:
            best_hyp = r["labels"][0]
            idx = HYPOTHESES.index(best_hyp)
            labels.append(idx)

    dist = Counter(labels)
    print("  Bloom distribution:")
    for b in sorted(dist):
        print(f"    {BLOOM_NAMES[b]}: {dist[b]} ({dist[b]/len(labels):.1%})")

    del classifier
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return labels


# ─────────────────────── Encoding ────────────────────────────────────────────

@torch.no_grad()
def encode_corpus(model, corpus_texts: List[str], tokenizer, device,
                  batch_size: int = 128) -> np.ndarray:
    all_embs = []
    for i in tqdm(range(0, len(corpus_texts), batch_size),
                  desc="  corpus", leave=False):
        batch = corpus_texts[i: i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        if hasattr(model, "encode_documents"):
            out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
            emb = out["masked_embedding"]
        else:
            out = model(enc["input_ids"], enc["attention_mask"])
            emb = out["full"]

        all_embs.append(emb.cpu().float().numpy())
    return np.concatenate(all_embs, axis=0)


@torch.no_grad()
def encode_queries_mrl(model, query_texts: List[str], tokenizer, device,
                        batch_size: int = 64) -> np.ndarray:
    all_embs = []
    for i in tqdm(range(0, len(query_texts), batch_size),
                  desc="  queries-mrl", leave=False):
        batch = query_texts[i: i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(enc["input_ids"], enc["attention_mask"])
        all_embs.append(out["full"].cpu().float().numpy())
    return np.concatenate(all_embs, axis=0)


@torch.no_grad()
def encode_queries_bam(model, query_texts: List[str], bloom_labels_0idx: List[int],
                        tokenizer, device, batch_size: int = 64):
    """
    Returns (full_embs, masks, active_dims, is_prefix).

    is_prefix=True  → Option A (discrete dim), use prefix truncation at retrieval.
    is_prefix=False → Option B (scattered binary mask), apply mask at retrieval.
    """
    all_full, all_masks, all_active, all_dims = [], [], [], []
    for i in tqdm(range(0, len(query_texts), batch_size),
                  desc="  queries-bam", leave=False):
        batch_texts = query_texts[i: i + batch_size]
        batch_bloom = bloom_labels_0idx[i: i + batch_size]

        enc = tokenizer(batch_texts, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        bloom_t = torch.tensor(batch_bloom, dtype=torch.long, device=device)

        out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                   bloom_labels=bloom_t)

        full = out.get("full_embedding", out.get("masked_embedding"))
        all_full.append(full.cpu().float().numpy())

        if "mask" in out:
            hard = (out["mask"] > 0.5).float()
            all_masks.append(hard.cpu().float().numpy())
            all_active.append(hard.sum(dim=-1).cpu().float().numpy())
        if "discrete_dim" in out:
            all_dims.append(out["discrete_dim"].cpu().float().numpy())

    full_embs = np.concatenate(all_full, axis=0)
    masks = np.concatenate(all_masks, axis=0) if all_masks else None
    active = np.concatenate(all_active, axis=0) if all_active else None
    dims = np.concatenate(all_dims, axis=0) if all_dims else None
    is_prefix = (dims is not None and masks is None)
    return full_embs, masks, active, dims, is_prefix


# ─────────────────────── Retrieval ───────────────────────────────────────────

def _build_index(corpus_embs: np.ndarray):
    embs = np.ascontiguousarray(corpus_embs.astype(np.float32))
    if HAS_FAISS:
        idx = faiss.IndexFlatIP(embs.shape[1])
        idx.add(embs)
        return ("faiss", idx)
    return ("numpy", embs)


def _search(index_tuple, query_embs: np.ndarray, k: int) -> np.ndarray:
    kind, idx = index_tuple
    q = np.ascontiguousarray(query_embs.astype(np.float32))
    if kind == "faiss":
        _, I = idx.search(q, k)
        return I
    # numpy fallback
    q_t = torch.from_numpy(q)
    c_t = torch.from_numpy(idx)
    all_idx = []
    for i in range(0, len(q_t), 256):
        sim = q_t[i: i + 256] @ c_t.T
        all_idx.append(sim.topk(k, dim=-1).indices.numpy())
    return np.concatenate(all_idx, axis=0)


def retrieve_mrl(q_embs_full: np.ndarray, c_embs_full: np.ndarray,
                 dim: int, k: int = 100) -> np.ndarray:
    """Retrieve using prefix-truncated MRL embeddings."""
    d = min(dim, q_embs_full.shape[1])
    q = q_embs_full[:, :d]
    c = c_embs_full[:, :d]
    # L2-normalize
    q_n = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    c_n = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)
    idx = _build_index(c_n)
    return _search(idx, q_n, k)


def retrieve_bam_scattered(q_full: np.ndarray, masks: np.ndarray,
                            c_full: np.ndarray, k: int = 100) -> np.ndarray:
    """
    Option B: each query gets its own scattered mask applied to both query and corpus.
    Because the mask varies per query we do per-query retrieval (no global index).
    """
    N = len(q_full)
    C = len(c_full)
    all_topk = np.zeros((N, k), dtype=np.int64)

    c_t = torch.from_numpy(c_full.astype(np.float32))
    for i in tqdm(range(N), desc="  BAM-B retrieval", leave=False):
        m = torch.from_numpy(masks[i].astype(np.float32))  # [D]
        q_v = torch.from_numpy(q_full[i].astype(np.float32)) * m  # [D]
        q_v = F.normalize(q_v.unsqueeze(0), p=2, dim=-1)
        c_v = F.normalize(c_t * m.unsqueeze(0), p=2, dim=-1)
        sim = (q_v @ c_v.T).squeeze(0)
        all_topk[i] = sim.topk(min(k, C), dim=-1).indices.numpy()
    return all_topk


def retrieve_bam_prefix(q_full: np.ndarray, dims: np.ndarray,
                         c_full: np.ndarray, k: int = 100) -> np.ndarray:
    """Option A: per-query prefix truncation."""
    N = len(q_full)
    C = len(c_full)
    all_topk = np.zeros((N, k), dtype=np.int64)
    c_t = torch.from_numpy(c_full.astype(np.float32))
    for i in tqdm(range(N), desc="  BAM-A retrieval", leave=False):
        d = max(1, int(dims[i]))
        q_v = F.normalize(
            torch.from_numpy(q_full[i, :d].astype(np.float32)).unsqueeze(0), p=2, dim=-1)
        c_v = F.normalize(c_t[:, :d], p=2, dim=-1)
        sim = (q_v @ c_v.T).squeeze(0)
        all_topk[i] = sim.topk(min(k, C), dim=-1).indices.numpy()
    return all_topk


# ─────────────────────── Metrics ─────────────────────────────────────────────

def compute_ndcg(topk_indices: np.ndarray, qrel_vector: np.ndarray,
                 k: int = 10) -> float:
    """Compute NDCG@k for a single query.
    qrel_vector: relevance scores indexed by corpus position (0 = not relevant).
    """
    gains = qrel_vector[topk_indices[:k]]
    dcg = sum(g / math.log2(r + 2) for r, g in enumerate(gains))
    ideal = sorted(qrel_vector, reverse=True)[:k]
    idcg = sum(g / math.log2(r + 2) for r, g in enumerate(ideal) if g > 0)
    return dcg / idcg if idcg > 0 else 0.0


def compute_recall(topk_indices: np.ndarray, qrel_vector: np.ndarray,
                   k: int = 10) -> float:
    return float((qrel_vector[topk_indices[:k]] > 0).any())


def evaluate_retrieval(topk_indices: np.ndarray,
                       qrels_array: np.ndarray,
                       bloom_labels: np.ndarray,
                       ks: Tuple[int, ...] = (1, 5, 10)) -> Dict:
    """
    topk_indices: [N, max_k]
    qrels_array:  [N, C] sparse — relevance of corpus doc for each query
    bloom_labels: [N] 0-indexed Bloom level
    """
    N = len(topk_indices)
    results = {}

    for k in ks:
        hits = np.array([
            compute_recall(topk_indices[i], qrels_array[i], k)
            for i in range(N)
        ])
        results[f"recall@{k}"] = float(hits.mean())

        ndcg_scores = np.array([
            compute_ndcg(topk_indices[i], qrels_array[i], k)
            for i in range(N)
        ])
        results[f"ndcg@{k}"] = float(ndcg_scores.mean())

    # Per-Bloom breakdown (ndcg@10 + recall@10)
    for b in range(6):
        mask = bloom_labels == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        b_name = BLOOM_NAMES[b]
        b_hits = np.array([
            compute_recall(topk_indices[i], qrels_array[i], 10)
            for i in np.where(mask)[0]
        ])
        b_ndcg = np.array([
            compute_ndcg(topk_indices[i], qrels_array[i], 10)
            for i in np.where(mask)[0]
        ])
        results[f"bloom_{b_name}_n"] = n_b
        results[f"bloom_{b_name}_recall@10"] = float(b_hits.mean())
        results[f"bloom_{b_name}_ndcg@10"] = float(b_ndcg.mean())

    return results


# ─────────────────────── Main evaluation per dataset ─────────────────────────

def run_dataset(dataset_name: str, args, device) -> Dict:
    print(f"\n{'='*70}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*70}")

    # 1. Load BEIR data
    corpus, queries, qrels = load_beir_dataset(dataset_name)
    if corpus is None:
        print(f"  SKIPPING {dataset_name} — could not load.")
        return {}

    # Filter queries that have at least one relevant doc
    query_ids = [qid for qid in queries if qid in qrels and len(qrels[qid]) > 0]
    print(f"  Corpus: {len(corpus):,}  Queries: {len(query_ids):,}")

    corpus_ids = list(corpus.keys())
    corpus_id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}

    corpus_texts = []
    for cid in corpus_ids:
        title = corpus[cid].get("title", "").strip()
        text  = corpus[cid].get("text", "").strip()
        corpus_texts.append((title + " " + text).strip() if title else text)

    query_texts = [queries[qid] for qid in query_ids]

    # Build qrels dense array [N_queries, N_corpus] — sparse matrix in numpy
    # For large corpora we store as a list of dicts and compute per-query
    qrels_list: List[np.ndarray] = []
    for qid in query_ids:
        rel_dict = qrels.get(qid, {})
        qrel_vec = np.zeros(len(corpus_ids), dtype=np.float32)
        for doc_id, score in rel_dict.items():
            if doc_id in corpus_id_to_idx:
                qrel_vec[corpus_id_to_idx[doc_id]] = float(score)
        qrels_list.append(qrel_vec)

    # 2. Bloom annotation
    bloom_labels_0idx = annotate_bloom_nli(
        query_texts, device,
        model_name=args.nli_model,
        batch_size=args.nli_batch_size,
    )
    bloom_arr = np.array(bloom_labels_0idx, dtype=np.int64)

    results_dict: Dict[str, Dict] = {}
    MAX_K = 100

    # ── MRL evaluation ───────────────────────────────────────────────────────
    mrl_config = load_config(args.mrl_config)
    tokenizer_mrl = AutoTokenizer.from_pretrained(mrl_config["model"]["backbone"])

    print("\n[MRL] Loading model and encoding ...")
    mrl_model = load_mrl_model(mrl_config, args.mrl_checkpoint, device)

    mrl_corpus_embs = encode_corpus(mrl_model, corpus_texts, tokenizer_mrl, device)
    mrl_query_embs  = encode_queries_mrl(mrl_model, query_texts, tokenizer_mrl, device)

    del mrl_model
    if device.type == "cuda": torch.cuda.empty_cache()

    full_dim = mrl_query_embs.shape[1]
    mrl_eval_dims = [d for d in mrl_config["model"].get("mrl_dims", [64,128,256,512,768,1024])
                     if d <= full_dim] + [full_dim]
    mrl_eval_dims = sorted(set(mrl_eval_dims))

    print(f"  MRL retrieval at dims: {mrl_eval_dims}")
    for dim in mrl_eval_dims:
        print(f"  MRL-{dim} ...", end=" ", flush=True)
        topk = retrieve_mrl(mrl_query_embs, mrl_corpus_embs, dim, MAX_K)
        metrics = evaluate_retrieval(topk, qrels_list, bloom_arr)
        results_dict[f"MRL-{dim}"] = metrics
        print(f"  R@10={metrics['recall@10']:.4f}  NDCG@10={metrics['ndcg@10']:.4f}")

    del mrl_corpus_embs, mrl_query_embs

    # ── BAM-B evaluation ─────────────────────────────────────────────────────
    bam_b_config = load_config(args.bam_b_config)
    tokenizer_bam_b = AutoTokenizer.from_pretrained(bam_b_config["model"]["backbone"])

    print("\n[BAM-B] Loading model and encoding ...")
    bam_b_model = load_bam_model(bam_b_config, args.bam_b_checkpoint, device)

    bam_b_corpus_embs = encode_corpus(bam_b_model, corpus_texts, tokenizer_bam_b, device)
    bam_b_q_full, bam_b_masks, bam_b_active, bam_b_dims, bam_b_prefix = \
        encode_queries_bam(bam_b_model, query_texts, bloom_labels_0idx,
                           tokenizer_bam_b, device)

    del bam_b_model
    if device.type == "cuda": torch.cuda.empty_cache()

    # Average active dims per level for reporting
    avg_dims_per_level = {}
    for b in range(6):
        mask = bloom_arr == b
        if mask.sum() == 0:
            continue
        if bam_b_active is not None:
            avg_dims_per_level[BLOOM_NAMES[b]] = float(bam_b_active[mask].mean())
        elif bam_b_dims is not None:
            avg_dims_per_level[BLOOM_NAMES[b]] = float(bam_b_dims[mask].mean())

    if bam_b_prefix:
        print("  BAM-B mode: prefix (Option A)")
        topk_bam_b = retrieve_bam_prefix(bam_b_q_full, bam_b_dims,
                                          bam_b_corpus_embs, MAX_K)
    else:
        print("  BAM-B mode: scattered mask (Option B)")
        topk_bam_b = retrieve_bam_scattered(bam_b_q_full, bam_b_masks,
                                             bam_b_corpus_embs, MAX_K)

    metrics_bam_b = evaluate_retrieval(topk_bam_b, qrels_list, bloom_arr)
    metrics_bam_b["avg_active_dims_per_level"] = avg_dims_per_level
    results_dict["BAM-B"] = metrics_bam_b
    print(f"  BAM-B  R@10={metrics_bam_b['recall@10']:.4f}  "
          f"NDCG@10={metrics_bam_b['ndcg@10']:.4f}  "
          f"avg_dims={avg_dims_per_level}")

    del bam_b_corpus_embs, bam_b_q_full, bam_b_masks

    # ── BAM-A evaluation (optional) ──────────────────────────────────────────
    if args.bam_a_checkpoint and args.bam_a_config:
        bam_a_ckpt_file = os.path.join(args.bam_a_checkpoint, "checkpoint.pt")
        if os.path.exists(bam_a_ckpt_file):
            bam_a_config = load_config(args.bam_a_config)
            tokenizer_bam_a = AutoTokenizer.from_pretrained(
                bam_a_config["model"]["backbone"])

            print("\n[BAM-A] Loading model and encoding ...")
            bam_a_model = load_bam_model(bam_a_config, args.bam_a_checkpoint, device)

            bam_a_corpus_embs = encode_corpus(
                bam_a_model, corpus_texts, tokenizer_bam_a, device)
            bam_a_q_full, bam_a_masks, bam_a_active, bam_a_dims, bam_a_prefix = \
                encode_queries_bam(bam_a_model, query_texts, bloom_labels_0idx,
                                   tokenizer_bam_a, device)

            del bam_a_model
            if device.type == "cuda": torch.cuda.empty_cache()

            if bam_a_prefix:
                topk_bam_a = retrieve_bam_prefix(bam_a_q_full, bam_a_dims,
                                                  bam_a_corpus_embs, MAX_K)
            else:
                topk_bam_a = retrieve_bam_scattered(bam_a_q_full, bam_a_masks,
                                                     bam_a_corpus_embs, MAX_K)

            metrics_bam_a = evaluate_retrieval(topk_bam_a, qrels_list, bloom_arr)
            results_dict["BAM-A"] = metrics_bam_a
            print(f"  BAM-A  R@10={metrics_bam_a['recall@10']:.4f}  "
                  f"NDCG@10={metrics_bam_a['ndcg@10']:.4f}")
        else:
            print(f"\n[BAM-A] checkpoint not found at {bam_a_ckpt_file} — skipping.")

    # ── Print summary table ──────────────────────────────────────────────────
    print(f"\n  {'Model':<16}  {'R@1':>6}  {'R@5':>6}  {'R@10':>6}  {'NDCG@10':>8}")
    print("  " + "-" * 48)
    for name, m in results_dict.items():
        r1   = m.get("recall@1",  float("nan"))
        r5   = m.get("recall@5",  float("nan"))
        r10  = m.get("recall@10", float("nan"))
        n10  = m.get("ndcg@10",   float("nan"))
        print(f"  {name:<16}  {r1:6.4f}  {r5:6.4f}  {r10:6.4f}  {n10:8.4f}")

    # ── Print Bloom breakdown for BAM-B ─────────────────────────────────────
    if "BAM-B" in results_dict:
        print(f"\n  BAM-B per-Bloom breakdown (NDCG@10 | R@10):")
        for b in range(6):
            bname = BLOOM_NAMES[b]
            n_key = f"bloom_{bname}_n"
            n_key_r = f"bloom_{bname}_recall@10"
            n_key_d = f"bloom_{bname}_ndcg@10"
            m = results_dict["BAM-B"]
            if n_key in m:
                print(f"    {bname:<12} n={m[n_key]:5d}  "
                      f"NDCG@10={m[n_key_d]:.4f}  R@10={m[n_key_r]:.4f}")

    return results_dict


# ─────────────────────── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation on TREC-COVID, SciDocs, Climate-FEVER")

    # Checkpoints
    parser.add_argument("--mrl_checkpoint", required=True,
                        help="Path to MRL checkpoint dir (contains checkpoint.pt)")
    parser.add_argument("--mrl_config", required=True,
                        help="MRL config YAML (for backbone + mrl_dims)")
    parser.add_argument("--bam_b_checkpoint", required=True,
                        help="Path to BAM-B checkpoint dir")
    parser.add_argument("--bam_b_config", required=True,
                        help="BAM-B config YAML")
    parser.add_argument("--bam_a_checkpoint", default=None,
                        help="(Optional) BAM-A checkpoint dir")
    parser.add_argument("--bam_a_config", default=None,
                        help="(Optional) BAM-A config YAML")

    # Datasets + output
    parser.add_argument("--datasets", nargs="+",
                        default=ZERO_SHOT_DATASETS,
                        help="BEIR dataset names to evaluate")
    parser.add_argument("--output_dir", default="results/zero_shot/")

    # NLI annotator
    parser.add_argument("--nli_model",
                        default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                        help="HuggingFace model for zero-shot Bloom annotation")
    parser.add_argument("--nli_batch_size", type=int, default=32)

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_results = {}
    for ds in args.datasets:
        ds_results = run_dataset(ds, args, device)
        all_results[ds] = ds_results

        # Save per-dataset JSON
        ds_out = os.path.join(args.output_dir, f"{ds}.json")
        with open(ds_out, "w") as f:
            json.dump(ds_results, f, indent=2)
        print(f"\n  Results saved to {ds_out}")

    # Save combined JSON
    combined_out = os.path.join(args.output_dir, "zero_shot_results.json")
    with open(combined_out, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print final summary across all datasets
    print(f"\n\n{'='*70}")
    print("  ZERO-SHOT SUMMARY (NDCG@10 / R@10)")
    print(f"{'='*70}")
    print(f"  {'Model':<16}" + "".join(f"  {ds[:12]:>14}" for ds in args.datasets))
    print("  " + "-" * (16 + 16 * len(args.datasets)))

    all_models = []
    for ds in args.datasets:
        all_models += list(all_results.get(ds, {}).keys())
    all_models = list(dict.fromkeys(all_models))

    for model_name in all_models:
        row = f"  {model_name:<16}"
        for ds in args.datasets:
            m = all_results.get(ds, {}).get(model_name, {})
            r10  = m.get("recall@10",  float("nan"))
            n10  = m.get("ndcg@10",    float("nan"))
            row += f"  {n10:6.4f}/{r10:6.4f}"
        print(row)

    print(f"\nAll results saved to {combined_out}")


if __name__ == "__main__":
    main()
