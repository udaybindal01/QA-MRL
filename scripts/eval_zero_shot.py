"""
Zero-shot evaluation of educational models on TREC-COVID, SciDocs, Climate-FEVER.

Models are trained ONLY on educational data — this measures domain transfer.
Queries are Bloom-annotated with the NLI classifier so per-level breakdowns work.

Models evaluated:
  MRL    — MRL encoder at 64/128/256/512/768/1024 and full dims
  BAM-B  — Bloom scattered mask (Option B), per-level FAISS
  BAM-PQ — Bloom + per-query residual mask, single FAISS against full corpus
  BAM-A  — Bloom prefix mask (Option A, optional — skipped if checkpoint absent)

Metrics: R@1, R@5, R@10, NDCG@10 — overall and stratified by Bloom level.

Memory handling:
  - qrels stored as sparse dicts (not dense arrays) — essential for large corpora
  - corpus embeddings stored as float16 (half memory)
  - retrieval done in chunks so full corpus never needs to be in GPU VRAM
  - BAM-B uses per-level FAISS indices (6 unique masks × 1 FAISS build each)
  - BAM-PQ uses one FAISS index (full corpus) + per-query masked queries
  - --max_corpus_size caps very large corpora (default 500k; climate-fever has 5.4M)

Usage:
    python scripts/eval_zero_shot.py \
        --mrl_checkpoint /tmp/multi-domain/educational/mrl/best/ \
        --mrl_config     results/multi_domain/educational/configs/mrl.yaml \
        --bam_b_checkpoint /tmp/multi-domain/educational/bam_b/best_bsr/ \
        --bam_b_config     results/multi_domain/educational/configs/bam_b.yaml \
        --bam_pq_checkpoint /tmp/multi-domain/educational/bam_pq/best_bsr/ \
        --bam_pq_config     results/multi_domain/educational/configs/bam_pq.yaml \
        --datasets trec-covid scidocs climate-fever \
        --output_dir results/zero_shot/
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

def load_beir_dataset(dataset_name: str, split: str = "test",
                      max_corpus_size: Optional[int] = None):
    """Returns (corpus, queries, qrels, truncated_flag) in BEIR format."""
    corpus, queries, qrels = None, None, None

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
    except ImportError:
        pass
    except Exception as exc:
        print(f"  beir library failed ({exc}), falling back to HuggingFace ...")

    if corpus is None:
        for attempt in range(3):
            try:
                from datasets import load_dataset
                print(f"  Loading {dataset_name} from HuggingFace (attempt {attempt+1}) ...")
                corpus_ds = load_dataset(
                    f"BeIR/{dataset_name}", "corpus", split="corpus",
                    trust_remote_code=True)
                corpus = {r["_id"]: {"title": r.get("title", ""),
                                      "text":  r.get("text",  "")}
                          for r in corpus_ds}
                queries_ds = load_dataset(
                    f"BeIR/{dataset_name}", "queries", split="queries",
                    trust_remote_code=True)
                queries = {r["_id"]: r.get("text", "") for r in queries_ds}

                qrels_ds = load_dataset(
                    f"BeIR/{dataset_name}-qrels", split=split,
                    trust_remote_code=True)
                qrels_raw: Dict[str, Dict[str, int]] = defaultdict(dict)
                for r in qrels_ds:
                    qrels_raw[str(r["query-id"])][str(r["corpus-id"])] = int(r["score"])
                qrels = dict(qrels_raw)
                break
            except Exception as exc2:
                print(f"  HuggingFace attempt {attempt+1} failed: {exc2}")
                if attempt == 2:
                    return None, None, None, False

    if corpus is None or queries is None or qrels is None:
        return None, None, None, False

    truncated = False
    if max_corpus_size and len(corpus) > max_corpus_size:
        print(f"  Corpus has {len(corpus):,} docs — truncating to {max_corpus_size:,} "
              f"(set --max_corpus_size 0 to disable)")
        # Keep all docs that appear in qrels first, then fill with random
        assert qrels is not None
        relevant_ids = set()
        for rel_dict in qrels.values():
            relevant_ids.update(rel_dict.keys())
        kept = list(relevant_ids & set(corpus.keys()))
        all_ids = list(corpus.keys())
        remaining = [cid for cid in all_ids if cid not in relevant_ids]
        np.random.shuffle(remaining)
        kept += remaining[:max(0, max_corpus_size - len(kept))]
        corpus = {cid: corpus[cid] for cid in kept[:max_corpus_size]}
        truncated = True
        print(f"  Truncated corpus: {len(corpus):,} docs "
              f"(includes all {len(relevant_ids):,} relevant docs)")

    return corpus, queries, qrels, truncated


# ─────────────────────── Bloom annotation ────────────────────────────────────

def annotate_bloom_nli(query_texts: List[str], device,
                       model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                       batch_size: int = 32) -> List[int]:
    from transformers import pipeline

    HYPOTHESES = [
        "This query is asking to recall or retrieve a specific fact, name, or definition.",
        "This query is asking to explain, describe, or summarize how something works.",
        "This query is asking how to use or apply knowledge to solve a practical problem.",
        "This query is asking to compare, contrast, or examine the relationship between things.",
        "This query is asking to evaluate evidence, assess effectiveness, or judge quality.",
        "This query is asking to design, propose, or synthesize something new.",
    ]

    print(f"  Annotating {len(query_texts)} queries with NLI Bloom classifier ...")
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
            labels.append(HYPOTHESES.index(best_hyp))

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
def encode_texts_to_array(model, texts: List[str], tokenizer, device,
                           is_query: bool = False,
                           bloom_labels: Optional[List[int]] = None,
                           batch_size: int = 128) -> np.ndarray:
    """Encode texts → float16 numpy array. Keeps full embedding dim."""
    _query_instr = None
    if is_query:
        if hasattr(model, "encoder"):
            _query_instr = getattr(model.encoder, "query_instruction", None)
        elif hasattr(model, "query_instruction"):
            _query_instr = model.query_instruction

    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size),
                  desc="  queries" if is_query else "  corpus", leave=False):
        batch = texts[i: i + batch_size]
        if _query_instr:
            batch = [_query_instr + t for t in batch]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=128 if is_query else 256,
                        return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        if is_query and hasattr(model, "encode_queries"):
            kwargs = dict(input_ids=enc["input_ids"],
                          attention_mask=enc["attention_mask"])
            if bloom_labels is not None:
                bl = bloom_labels[i: i + batch_size]
                kwargs["bloom_labels"] = torch.tensor(bl, dtype=torch.long, device=device)
            out = model.encode_queries(**kwargs)
            emb = out.get("full_embedding", out.get("masked_embedding"))
        elif not is_query and hasattr(model, "encode_documents"):
            out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
            emb = out["masked_embedding"]
        else:
            out = model(enc["input_ids"], enc["attention_mask"])
            emb = out["full"]

        all_embs.append(emb.cpu().half().numpy())   # float16 — halves memory

    return np.concatenate(all_embs, axis=0)


# ─────────────────────── FAISS index helpers ─────────────────────────────────

def _build_faiss(embs_f16: np.ndarray) -> object:
    """Build FAISS IndexFlatIP from float16 array (cast to float32 internally)."""
    embs_f32 = np.ascontiguousarray(embs_f16.astype(np.float32))
    if HAS_FAISS:
        idx = faiss.IndexFlatIP(embs_f32.shape[1])
        idx.add(embs_f32)
        return ("faiss", idx)
    return ("numpy", embs_f32)


def _search_index(index_tuple, q_f32: np.ndarray, k: int) -> np.ndarray:
    kind, idx = index_tuple
    q = np.ascontiguousarray(q_f32.astype(np.float32))
    if kind == "faiss":
        _, I = idx.search(q, k)
        return I
    # numpy fallback (chunked to avoid RAM spike)
    c = torch.from_numpy(idx)
    q_t = torch.from_numpy(q)
    out = []
    for i in range(0, len(q_t), 256):
        sim = q_t[i: i + 256] @ c.T
        out.append(sim.topk(k, dim=-1).indices.numpy())
    return np.concatenate(out, axis=0)


def _norm(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x.astype(np.float32), axis=1, keepdims=True)
    return (x.astype(np.float32) / (norms + 1e-9))


# ─────────────────────── MRL retrieval ───────────────────────────────────────

def retrieve_mrl_dims(q_embs_f16: np.ndarray, c_embs_f16: np.ndarray,
                      dims: List[int], k: int = 100) -> Dict[int, np.ndarray]:
    """Build one FAISS per dim and return {dim: topk_indices}."""
    results = {}
    for d in dims:
        d = min(d, q_embs_f16.shape[1])
        q_n = _norm(q_embs_f16[:, :d])
        c_n = _norm(c_embs_f16[:, :d])
        idx = _build_faiss(c_n)
        results[d] = _search_index(idx, q_n, k)
        del idx
    return results


# ─────────────────────── BAM retrieval ───────────────────────────────────────

@torch.no_grad()
def extract_level_masks(bam_model, embedding_dim: int, device) -> Dict[int, np.ndarray]:
    """
    Extract the hard binary mask for each Bloom level (0-5).
    BAM-B has 6 unique masks — one per level. We get them by passing a dummy query
    with each bloom_label; the mask depends only on the label, not the query text.
    """
    masks = {}
    dummy_ids  = torch.zeros(1, 4, dtype=torch.long, device=device)
    dummy_mask = torch.ones(1, 4, dtype=torch.long, device=device)
    for level in range(6):
        bloom_t = torch.tensor([level], dtype=torch.long, device=device)
        out = bam_model.encode_queries(dummy_ids, dummy_mask, bloom_labels=bloom_t)
        if "mask" in out:
            hard = (out["mask"] > 0.5).float().squeeze(0).cpu().numpy()
            masks[level] = hard
        elif "discrete_dim" in out:
            # Option A — prefix mask: active dims = first d dims
            d = int(out["discrete_dim"].item())
            hard = np.zeros(embedding_dim, dtype=np.float32)
            hard[:d] = 1.0
            masks[level] = hard
    return masks


def retrieve_bam_scattered(q_full_f16: np.ndarray,
                            bam_q_bloom: np.ndarray,
                            level_masks: Dict[int, np.ndarray],
                            c_full_f16: np.ndarray,
                            k: int = 100) -> np.ndarray:
    """
    Per-level FAISS retrieval for BAM-B scattered mask.
    Builds 6 indexed corpus variants (one per level mask) and routes each query.
    """
    N = len(q_full_f16)
    all_topk = np.zeros((N, k), dtype=np.int64)

    # Build one FAISS index per level (only for levels that appear in queries)
    present_levels = set(bam_q_bloom.tolist())
    level_indices = {}
    for lev in present_levels:
        if lev not in level_masks:
            continue
        m = level_masks[lev].astype(np.float32)              # [D]
        c_masked = c_full_f16.astype(np.float32) * m[None, :]
        c_norm   = _norm(c_masked)
        level_indices[lev] = _build_index_obj(c_norm)
        print(f"    Level {BLOOM_NAMES.get(lev, lev)}: "
              f"active_dims={int(m.sum())}  corpus indexed")

    # Query routing
    for lev in present_levels:
        if lev not in level_indices:
            continue
        qmask = bam_q_bloom == lev
        q_idx = np.where(qmask)[0]
        m = level_masks[lev].astype(np.float32)
        q_masked = q_full_f16[q_idx].astype(np.float32) * m[None, :]
        q_norm   = _norm(q_masked)
        topk = _search_index(level_indices[lev], q_norm, k)
        all_topk[q_idx] = topk

    return all_topk


def _build_index_obj(c_norm: np.ndarray):
    """Alias for _build_faiss that accepts pre-normalised float32 array."""
    c_f32 = np.ascontiguousarray(c_norm.astype(np.float32))
    if HAS_FAISS:
        idx = faiss.IndexFlatIP(c_f32.shape[1])
        idx.add(c_f32)
        return ("faiss", idx)
    return ("numpy", c_f32)


def retrieve_bam_prefix(q_full_f16: np.ndarray,
                         bam_q_bloom: np.ndarray,
                         level_masks: Dict[int, np.ndarray],
                         c_full_f16: np.ndarray,
                         k: int = 100) -> np.ndarray:
    """Option A: prefix truncation using per-level discrete dim."""
    N = len(q_full_f16)
    all_topk = np.zeros((N, k), dtype=np.int64)

    present_levels = set(bam_q_bloom.tolist())
    level_indices = {}
    for lev in present_levels:
        if lev not in level_masks:
            continue
        d = int(level_masks[lev].sum())            # number of active dims (prefix len)
        c_norm = _norm(c_full_f16[:, :d])
        level_indices[lev] = (_build_index_obj(c_norm), d)

    for lev in present_levels:
        if lev not in level_indices:
            continue
        faiss_idx, d = level_indices[lev]
        qmask = bam_q_bloom == lev
        q_idx = np.where(qmask)[0]
        q_norm = _norm(q_full_f16[q_idx, :d])
        all_topk[q_idx] = _search_index(faiss_idx, q_norm, k)

    return all_topk


# ─────────────────────── BAM-PQ retrieval ────────────────────────────────────

@torch.no_grad()
def encode_bam_pq_queries(model, query_texts: List[str], bloom_labels: List[int],
                           tokenizer, device, batch_size: int = 64
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode queries with BAM-PQ, returning masked embeddings + per-query active dim counts.
    BAM-PQ masks differ per query (bloom anchor + query residual), so we capture
    masked_embedding directly rather than applying level masks post-hoc.
    """
    _query_instr = None
    if hasattr(model, "encoder"):
        _query_instr = getattr(model.encoder, "query_instruction", None)
    elif hasattr(model, "query_instruction"):
        _query_instr = model.query_instruction

    all_embs: List[np.ndarray] = []
    all_dims: List[int] = []
    for i in tqdm(range(0, len(query_texts), batch_size),
                  desc="  BAM-PQ queries", leave=False):
        batch_texts = query_texts[i: i + batch_size]
        if _query_instr:
            batch_texts = [_query_instr + t for t in batch_texts]
        batch_bloom = bloom_labels[i: i + batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        bloom_t = torch.tensor(batch_bloom, dtype=torch.long, device=device)
        out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                   bloom_labels=bloom_t)
        masked = out.get("masked_embedding", out.get("full_embedding"))
        all_embs.append(masked.cpu().half().numpy())
        if "mask" in out:
            dims = (out["mask"] > 0.5).float().sum(dim=-1).long().cpu().tolist()
        elif "discrete_dim" in out:
            dims = out["discrete_dim"].long().cpu().tolist()
        else:
            dims = [masked.shape[-1]] * len(batch_texts)
        all_dims.extend(dims if isinstance(dims, list) else [dims] * len(batch_texts))

    return np.concatenate(all_embs, axis=0), np.array(all_dims, dtype=np.int64)


def retrieve_bam_pq(pq_q_embs_f16: np.ndarray,
                    c_full_f16: np.ndarray,
                    k: int = 100) -> np.ndarray:
    """
    BAM-PQ retrieval: masked query embeddings vs full corpus embeddings.
    Since query zeros out irrelevant dims, a single FAISS flat index works.
    Queries are already masked — score = q_masked · d_full ignores zero dims.
    """
    c_norm = _norm(c_full_f16)
    index = _build_index_obj(c_norm)
    q_norm = _norm(pq_q_embs_f16)
    return _search_index(index, q_norm, k)


# ─────────────────────── Metrics (sparse qrels) ──────────────────────────────

def compute_ndcg_sparse(topk_indices: np.ndarray,
                         qrel_dict: Dict[int, float], k: int) -> float:
    gains = [qrel_dict.get(int(idx), 0.0) for idx in topk_indices[:k]]
    dcg  = sum(g / math.log2(r + 2) for r, g in enumerate(gains))
    idcg = sum(g / math.log2(r + 2)
               for r, g in enumerate(sorted(qrel_dict.values(), reverse=True)[:k])
               if g > 0)
    return dcg / idcg if idcg > 0 else 0.0


def compute_recall_sparse(topk_indices: np.ndarray,
                           qrel_dict: Dict[int, float], k: int) -> float:
    return float(any(qrel_dict.get(int(idx), 0) > 0 for idx in topk_indices[:k]))


def evaluate_retrieval(topk_indices: np.ndarray,
                       qrels_sparse: List[Dict[int, float]],
                       bloom_labels: np.ndarray,
                       ks: Tuple[int, ...] = (1, 5, 10)) -> Dict:
    N = len(topk_indices)
    results = {}

    for k in ks:
        r_arr    = np.array([compute_recall_sparse(topk_indices[i], qrels_sparse[i], k)
                             for i in range(N)])
        ndcg_arr = np.array([compute_ndcg_sparse(topk_indices[i], qrels_sparse[i], k)
                             for i in range(N)])
        results[f"recall@{k}"]  = float(r_arr.mean())
        results[f"ndcg@{k}"]    = float(ndcg_arr.mean())

    for b in range(6):
        mask = bloom_labels == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        idx_b = np.where(mask)[0]
        bname = BLOOM_NAMES[b]
        results[f"bloom_{bname}_n"] = n_b
        results[f"bloom_{bname}_recall@10"] = float(np.mean([
            compute_recall_sparse(topk_indices[i], qrels_sparse[i], 10) for i in idx_b]))
        results[f"bloom_{bname}_ndcg@10"]   = float(np.mean([
            compute_ndcg_sparse(topk_indices[i], qrels_sparse[i], 10)   for i in idx_b]))

    return results


# ─────────────────────── Per-dataset runner ──────────────────────────────────

def run_dataset(dataset_name: str, args, device) -> Dict:
    print(f"\n{'='*70}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*70}")

    max_corp = args.max_corpus_size if args.max_corpus_size > 0 else None
    corpus, queries, qrels, truncated = load_beir_dataset(
        dataset_name, max_corpus_size=max_corp)
    if corpus is None:
        print(f"  SKIPPING {dataset_name} — could not load.")
        return {}

    query_ids = [qid for qid in queries if qid in qrels and len(qrels[qid]) > 0]
    print(f"  Corpus: {len(corpus):,}  Queries (with qrels): {len(query_ids):,}"
          + ("  [TRUNCATED]" if truncated else ""))

    corpus_ids = list(corpus.keys())
    corpus_id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}

    corpus_texts = []
    for cid in corpus_ids:
        title = corpus[cid].get("title", "").strip()
        text  = corpus[cid].get("text",  "").strip()
        corpus_texts.append((title + " " + text).strip() if title else text)

    query_texts = [queries[qid] for qid in query_ids]

    # Sparse qrels (not dense arrays — essential for large corpora)
    qrels_sparse: List[Dict[int, float]] = []
    for qid in query_ids:
        sparse = {corpus_id_to_idx[did]: float(sc)
                  for did, sc in qrels.get(qid, {}).items()
                  if did in corpus_id_to_idx}
        qrels_sparse.append(sparse)

    # Bloom annotation
    bloom_labels = np.array(annotate_bloom_nli(
        query_texts, device,
        model_name=args.nli_model,
        batch_size=args.nli_batch_size,
    ), dtype=np.int64)

    MAX_K = 100
    results_dict: Dict[str, Dict] = {}
    meta = {"dataset": dataset_name, "n_corpus": len(corpus_ids),
            "n_queries": len(query_ids), "corpus_truncated": truncated}

    # ── MRL ─────────────────────────────────────────────────────────────────
    mrl_config = load_config(args.mrl_config)
    tok_mrl    = AutoTokenizer.from_pretrained(mrl_config["model"]["backbone"])
    print("\n[MRL] Loading and encoding ...")
    mrl_model  = load_mrl_model(mrl_config, args.mrl_checkpoint, device)

    mrl_c_embs = encode_texts_to_array(mrl_model, corpus_texts, tok_mrl, device,
                                        batch_size=128)
    mrl_q_embs = encode_texts_to_array(mrl_model, query_texts, tok_mrl, device,
                                        is_query=True, batch_size=64)
    del mrl_model
    if device.type == "cuda": torch.cuda.empty_cache()

    full_dim = mrl_q_embs.shape[1]
    eval_dims = sorted(set(
        [d for d in mrl_config["model"].get("mrl_dims", [64,128,256,512,768,1024])
         if d <= full_dim] + [full_dim]))

    print(f"  Evaluating MRL at dims: {eval_dims}")
    dim_topks = retrieve_mrl_dims(mrl_q_embs, mrl_c_embs, eval_dims, MAX_K)
    for d, topk in dim_topks.items():
        m = evaluate_retrieval(topk, qrels_sparse, bloom_labels)
        results_dict[f"MRL-{d}"] = m
        print(f"  MRL-{d:4d}  R@10={m['recall@10']:.4f}  NDCG@10={m['ndcg@10']:.4f}")
    del mrl_c_embs, mrl_q_embs, dim_topks

    # ── BAM-B ────────────────────────────────────────────────────────────────
    bam_b_config = load_config(args.bam_b_config)
    tok_bam_b    = AutoTokenizer.from_pretrained(bam_b_config["model"]["backbone"])
    print("\n[BAM-B] Loading and encoding ...")
    bam_b_model  = load_bam_model(bam_b_config, args.bam_b_checkpoint, device)

    emb_dim = bam_b_config["model"].get("embedding_dim", 1024)
    level_masks_b = extract_level_masks(bam_b_model, emb_dim, device)
    is_prefix_b = all((m.sum() == m[:int(m.sum())].sum()) for m in level_masks_b.values())

    bam_b_c_embs = encode_texts_to_array(bam_b_model, corpus_texts, tok_bam_b, device,
                                          batch_size=128)
    bam_b_q_embs = encode_texts_to_array(bam_b_model, query_texts, tok_bam_b, device,
                                          is_query=True,
                                          bloom_labels=bloom_labels.tolist(),
                                          batch_size=64)
    del bam_b_model
    if device.type == "cuda": torch.cuda.empty_cache()

    print(f"  BAM-B mode: {'prefix' if is_prefix_b else 'scattered mask'}")
    if is_prefix_b:
        topk_b = retrieve_bam_prefix(bam_b_q_embs, bloom_labels, level_masks_b,
                                      bam_b_c_embs, MAX_K)
    else:
        topk_b = retrieve_bam_scattered(bam_b_q_embs, bloom_labels, level_masks_b,
                                         bam_b_c_embs, MAX_K)

    m_b = evaluate_retrieval(topk_b, qrels_sparse, bloom_labels)
    avg_dims = {BLOOM_NAMES[lev]: int(level_masks_b[lev].sum())
                for lev in level_masks_b}
    m_b["avg_active_dims_per_level"] = avg_dims
    results_dict["BAM-B"] = m_b
    print(f"  BAM-B  R@10={m_b['recall@10']:.4f}  NDCG@10={m_b['ndcg@10']:.4f}  "
          f"dims={avg_dims}")
    del bam_b_c_embs, bam_b_q_embs, topk_b

    # ── BAM-PQ (optional) ────────────────────────────────────────────────────
    if args.bam_pq_checkpoint and args.bam_pq_config:
        ckpt_f = os.path.join(args.bam_pq_checkpoint, "checkpoint.pt")
        if os.path.exists(ckpt_f):
            bam_pq_config = load_config(args.bam_pq_config)
            tok_bam_pq    = AutoTokenizer.from_pretrained(
                bam_pq_config["model"]["backbone"])
            print("\n[BAM-PQ] Loading and encoding ...")
            bam_pq_model  = load_bam_model(bam_pq_config, args.bam_pq_checkpoint, device)

            # Corpus: full unmasked embeddings (masking happens on query side)
            bam_pq_c_embs = encode_texts_to_array(
                bam_pq_model, corpus_texts, tok_bam_pq, device, batch_size=128)

            # Queries: masked embeddings (bloom anchor + query residual)
            bam_pq_q_embs, pq_per_query_dims = encode_bam_pq_queries(
                bam_pq_model, query_texts, bloom_labels.tolist(),
                tok_bam_pq, device)
            del bam_pq_model
            if device.type == "cuda": torch.cuda.empty_cache()

            topk_pq = retrieve_bam_pq(bam_pq_q_embs, bam_pq_c_embs, MAX_K)
            m_pq = evaluate_retrieval(topk_pq, qrels_sparse, bloom_labels)

            # Per-level active dim stats
            pq_level_dims = {}
            for lev in range(6):
                mask_lev = bloom_labels == lev
                if mask_lev.any():
                    pq_level_dims[BLOOM_NAMES[lev]] = float(
                        pq_per_query_dims[mask_lev].mean())
            m_pq["avg_active_dims_per_level"] = pq_level_dims
            m_pq["avg_active_dims_overall"] = float(pq_per_query_dims.mean())
            results_dict["BAM-PQ"] = m_pq
            print(f"  BAM-PQ R@10={m_pq['recall@10']:.4f}  NDCG@10={m_pq['ndcg@10']:.4f}  "
                  f"avg_dims={m_pq['avg_active_dims_overall']:.1f}  per_level={pq_level_dims}")
            del bam_pq_c_embs, bam_pq_q_embs, topk_pq
        else:
            print(f"\n[BAM-PQ] checkpoint not found at {ckpt_f} — skipping.")

    # ── BAM-A (optional) ─────────────────────────────────────────────────────
    if args.bam_a_checkpoint and args.bam_a_config:
        ckpt_f = os.path.join(args.bam_a_checkpoint, "checkpoint.pt")
        if os.path.exists(ckpt_f):
            bam_a_config = load_config(args.bam_a_config)
            tok_bam_a    = AutoTokenizer.from_pretrained(
                bam_a_config["model"]["backbone"])
            print("\n[BAM-A] Loading and encoding ...")
            bam_a_model  = load_bam_model(bam_a_config, args.bam_a_checkpoint, device)

            emb_dim_a = bam_a_config["model"].get("embedding_dim", 1024)
            level_masks_a = extract_level_masks(bam_a_model, emb_dim_a, device)

            bam_a_c_embs = encode_texts_to_array(
                bam_a_model, corpus_texts, tok_bam_a, device, batch_size=128)
            bam_a_q_embs = encode_texts_to_array(
                bam_a_model, query_texts, tok_bam_a, device,
                is_query=True, bloom_labels=bloom_labels.tolist(), batch_size=64)
            del bam_a_model
            if device.type == "cuda": torch.cuda.empty_cache()

            is_prefix_a = all((m.sum() == m[:int(m.sum())].sum())
                               for m in level_masks_a.values())
            if is_prefix_a:
                topk_a = retrieve_bam_prefix(bam_a_q_embs, bloom_labels, level_masks_a,
                                              bam_a_c_embs, MAX_K)
            else:
                topk_a = retrieve_bam_scattered(bam_a_q_embs, bloom_labels, level_masks_a,
                                                 bam_a_c_embs, MAX_K)
            m_a = evaluate_retrieval(topk_a, qrels_sparse, bloom_labels)
            results_dict["BAM-A"] = m_a
            print(f"  BAM-A  R@10={m_a['recall@10']:.4f}  NDCG@10={m_a['ndcg@10']:.4f}")
            del bam_a_c_embs, bam_a_q_embs, topk_a
        else:
            print(f"\n[BAM-A] checkpoint not found at {ckpt_f} — skipping.")

    # ── Print summary table ──────────────────────────────────────────────────
    print(f"\n  {'Model':<16}  {'R@1':>6}  {'R@5':>6}  {'R@10':>6}  {'NDCG@10':>8}")
    print("  " + "-" * 48)
    for name, m in results_dict.items():
        r1  = m.get("recall@1",  float("nan"))
        r5  = m.get("recall@5",  float("nan"))
        r10 = m.get("recall@10", float("nan"))
        n10 = m.get("ndcg@10",   float("nan"))
        print(f"  {name:<16}  {r1:6.4f}  {r5:6.4f}  {r10:6.4f}  {n10:8.4f}")

    # Per-model Bloom breakdown
    for model_key in ("BAM-B", "BAM-PQ"):
        if model_key in results_dict:
            print(f"\n  {model_key} per-Bloom breakdown (NDCG@10 | R@10 | avg_dims):")
            m = results_dict[model_key]
            level_dims = m.get("avg_active_dims_per_level", {})
            for b in range(6):
                bname = BLOOM_NAMES[b]
                if f"bloom_{bname}_n" in m:
                    dims_str = (f"  dims={level_dims[bname]:.0f}"
                                if bname in level_dims else "")
                    print(f"    {bname:<12} n={m[f'bloom_{bname}_n']:5d}  "
                          f"NDCG@10={m[f'bloom_{bname}_ndcg@10']:.4f}  "
                          f"R@10={m[f'bloom_{bname}_recall@10']:.4f}{dims_str}")

    results_dict["_meta"] = meta
    return results_dict


# ─────────────────────── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mrl_checkpoint",    required=True)
    parser.add_argument("--mrl_config",        required=True)
    parser.add_argument("--bam_b_checkpoint",  required=True)
    parser.add_argument("--bam_b_config",      required=True)
    parser.add_argument("--bam_pq_checkpoint", default=None)
    parser.add_argument("--bam_pq_config",     default=None)
    parser.add_argument("--bam_a_checkpoint",  default=None)
    parser.add_argument("--bam_a_config",      default=None)
    parser.add_argument("--datasets", nargs="+", default=ZERO_SHOT_DATASETS)
    parser.add_argument("--output_dir",        default="results/zero_shot/")
    parser.add_argument("--nli_model",
        default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    parser.add_argument("--nli_batch_size",    type=int, default=32)
    parser.add_argument("--max_corpus_size",   type=int, default=500000,
                        help="Cap corpus size for very large datasets. "
                             "0 = no limit. climate-fever has 5.4M docs.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.max_corpus_size > 0:
        print(f"Max corpus size: {args.max_corpus_size:,} "
              f"(climate-fever will be subsampled; all relevant docs kept)")

    all_results = {}
    for ds in args.datasets:
        ds_results = run_dataset(ds, args, device)
        all_results[ds] = ds_results
        ds_out = os.path.join(args.output_dir, f"{ds}.json")
        with open(ds_out, "w") as f:
            json.dump(ds_results, f, indent=2)
        print(f"\n  Saved → {ds_out}")

    combined_out = os.path.join(args.output_dir, "zero_shot_results.json")
    with open(combined_out, "w") as f:
        json.dump(all_results, f, indent=2)

    # Final summary
    print(f"\n\n{'='*70}")
    print("  ZERO-SHOT SUMMARY  (NDCG@10 / R@10)")
    print(f"{'='*70}")
    all_models = list(dict.fromkeys(
        m for ds in args.datasets for m in all_results.get(ds, {}) if not m.startswith("_")))
    header = f"  {'Model':<16}" + "".join(f"  {ds[:14]:>16}" for ds in args.datasets)
    print(header)
    print("  " + "-" * len(header))
    for model_name in all_models:
        row = f"  {model_name:<16}"
        for ds in args.datasets:
            m = all_results.get(ds, {}).get(model_name, {})
            n10 = m.get("ndcg@10",   float("nan"))
            r10 = m.get("recall@10", float("nan"))
            row += f"  {n10:6.4f}/{r10:6.4f}  "
        print(row)

    print(f"\nAll results saved to {combined_out}")


if __name__ == "__main__":
    main()
