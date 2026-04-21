"""
Statistical significance testing: MRL vs BAM-B vs BAM-PQ.

For each dataset, encodes corpus+queries with all three models, computes
per-query R@10 and NDCG@10, then runs:
  - Paired Wilcoxon signed-rank test (appropriate for bounded IR metrics)
  - Bootstrap 95% CI on mean difference
  - Bonferroni-corrected p-values across all comparisons

Comparisons tested:
  BAM-B  vs MRL    (primary claim)
  BAM-PQ vs MRL    (extended claim)
  BAM-PQ vs BAM-B  (novelty: does per-query delta help?)

Output: per-dataset significance table + combined JSON.

Usage:
    python scripts/run_significance_tests.py \
        --datasets educational scifact nfcorpus fiqa \
        --ckpt_root /tmp/multi-domain \
        --cfg_root   results/multi_domain \
        --output_dir results/significance/
"""

import argparse
import json
import math
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config
from models.bam import BloomAlignedMRL
from models.encoder import MRLEncoder
from transformers import AutoTokenizer

_wilcoxon = None
try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon
    _wilcoxon = _scipy_wilcoxon
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found — using permutation test instead of Wilcoxon.")

BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply",
               4: "Analyze",  5: "Evaluate",   6: "Create"}


# ─────────────────────── Model loading ───────────────────────────────────────

def load_bam_model(config, ckpt_path, device):
    config["training"]["loss"].setdefault("bloom_frequencies", [1/6]*6)
    model = BloomAlignedMRL(config)
    f = os.path.join(ckpt_path, "checkpoint.pt")
    if os.path.exists(f):
        ckpt = torch.load(f, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"    Loaded {f}")
    return model.to(device).eval()


def load_mrl_model(config, ckpt_path, device):
    mc = config["model"]
    model = MRLEncoder(model_name=mc["backbone"],
                       embedding_dim=mc["embedding_dim"],
                       mrl_dims=mc["mrl_dims"])
    f = os.path.join(ckpt_path, "checkpoint.pt")
    if os.path.exists(f):
        ckpt = torch.load(f, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"    Loaded {f}")
    return model.to(device).eval()


# ─────────────────────── Encoding ────────────────────────────────────────────

@torch.no_grad()
def encode_corpus(model, texts, tokenizer, device, batch_size=128):
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="    corpus", leave=False):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        if hasattr(model, "encode_documents"):
            out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
            emb = out["masked_embedding"]
        else:
            out = model(enc["input_ids"], enc["attention_mask"])
            emb = out["full"]
        all_embs.append(emb.cpu().float())
    return torch.cat(all_embs)


@torch.no_grad()
def encode_queries_bam(model, samples, tokenizer, device, batch_size=64):
    """Returns (full_embs, masks, active_dims, is_prefix)."""
    all_full, all_masks, all_active, all_dims = [], [], [], []
    for i in tqdm(range(0, len(samples), batch_size), desc="    queries", leave=False):
        batch = samples[i:i+batch_size]
        texts  = [s["query"] for s in batch]
        blooms = [s["bloom_level"] - 1 for s in batch]
        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        bloom_t = torch.tensor(blooms, dtype=torch.long, device=device)
        out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                   bloom_labels=bloom_t)
        full = out.get("full_embedding", out.get("masked_embedding"))
        all_full.append(full.cpu().float())
        if "mask" in out:
            hard = (out["mask"] > 0.5).float()
            all_masks.append(hard.cpu())
            all_active.append(hard.sum(dim=-1).cpu())
        if "discrete_dim" in out:
            all_dims.append(out["discrete_dim"].cpu())
    full_embs  = torch.cat(all_full)
    masks      = torch.cat(all_masks)  if all_masks  else None
    active     = torch.cat(all_active) if all_active else None
    dims       = torch.cat(all_dims)   if all_dims   else None
    is_prefix  = (dims is not None and masks is None)
    return full_embs, masks, active, dims, is_prefix


@torch.no_grad()
def encode_queries_mrl(model, samples, tokenizer, device, batch_size=64):
    all_embs = []
    for i in tqdm(range(0, len(samples), batch_size), desc="    queries", leave=False):
        batch = samples[i:i+batch_size]
        texts = [s["query"] for s in batch]
        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(enc["input_ids"], enc["attention_mask"])
        all_embs.append(out["full"].cpu().float())
    return torch.cat(all_embs)


# ─────────────────────── Per-query retrieval ─────────────────────────────────

def per_query_recall(q_embs: torch.Tensor, c_embs: torch.Tensor,
                     gt_indices: np.ndarray, k: int = 10,
                     masks: Optional[torch.Tensor] = None,
                     dims: Optional[torch.Tensor] = None,
                     is_prefix: bool = False,
                     device=None, chunk: int = 256) -> np.ndarray:
    """
    Returns binary hit array [N] — 1 if relevant doc in top-k, else 0.
    Handles MRL (no mask), Option A (prefix), Option B (scattered mask).
    """
    N = len(q_embs)
    hits = np.zeros(N, dtype=np.float32)
    C = c_embs.shape[0]

    if masks is None and dims is None:
        # MRL — full-dim normalized dot product in chunks
        c_norm = F.normalize(c_embs.to(device), p=2, dim=-1)
        for i in range(0, N, chunk):
            q = F.normalize(q_embs[i:i+chunk].to(device), p=2, dim=-1)
            sim = torch.mm(q, c_norm.T)
            topk = sim.topk(k, dim=-1).indices.cpu().numpy()
            for j, row in enumerate(topk):
                hits[i+j] = int(gt_indices[i+j] in row)
    elif is_prefix:
        assert dims is not None
        c_t = c_embs.to(device)
        for i in range(N):
            d = max(1, int(dims[i].item()))
            q_v = F.normalize(q_embs[i:i+1, :d].to(device), p=2, dim=-1)
            c_v = F.normalize(c_t[:, :d], p=2, dim=-1)
            sim = torch.mm(q_v, c_v.T)
            topk = sim.topk(k, dim=-1).indices.cpu().numpy()[0]
            hits[i] = int(gt_indices[i] in topk)
    else:
        assert masks is not None
        c_t = c_embs.to(device)
        for i in range(N):
            m = masks[i].to(device)
            q_v = F.normalize(q_embs[i:i+1].to(device) * m, p=2, dim=-1)
            c_v = F.normalize(c_t * m, p=2, dim=-1)
            sim = torch.mm(q_v, c_v.T)
            topk = sim.topk(k, dim=-1).indices.cpu().numpy()[0]
            hits[i] = int(gt_indices[i] in topk)
    return hits


def per_query_ndcg(q_embs: torch.Tensor, c_embs: torch.Tensor,
                   gt_indices: np.ndarray, k: int = 10,
                   masks: Optional[torch.Tensor] = None,
                   dims: Optional[torch.Tensor] = None,
                   is_prefix: bool = False,
                   device=None, chunk: int = 256) -> np.ndarray:
    """Returns NDCG@k array [N] (binary relevance)."""
    N = len(q_embs)
    ndcg_scores = np.zeros(N, dtype=np.float32)
    C = c_embs.shape[0]

    def _ndcg_from_topk(topk_row, gt_idx):
        for rank, idx in enumerate(topk_row):
            if idx == gt_idx:
                return 1.0 / math.log2(rank + 2)
        return 0.0

    ideal = 1.0  # binary: one relevant doc → IDCG = 1/log2(2) = 1.0

    if masks is None and dims is None:
        c_norm = F.normalize(c_embs.to(device), p=2, dim=-1)
        for i in range(0, N, chunk):
            q = F.normalize(q_embs[i:i+chunk].to(device), p=2, dim=-1)
            sim = torch.mm(q, c_norm.T)
            topk = sim.topk(k, dim=-1).indices.cpu().numpy()
            for j, row in enumerate(topk):
                ndcg_scores[i+j] = _ndcg_from_topk(row, gt_indices[i+j]) / ideal
    elif is_prefix:
        assert dims is not None
        c_t = c_embs.to(device)
        for i in range(N):
            d = max(1, int(dims[i].item()))
            q_v = F.normalize(q_embs[i:i+1, :d].to(device), p=2, dim=-1)
            c_v = F.normalize(c_t[:, :d], p=2, dim=-1)
            sim = torch.mm(q_v, c_v.T)
            topk = sim.topk(k, dim=-1).indices.cpu().numpy()[0]
            ndcg_scores[i] = _ndcg_from_topk(topk, gt_indices[i]) / ideal
    else:
        assert masks is not None
        c_t = c_embs.to(device)
        for i in range(N):
            m = masks[i].to(device)
            q_v = F.normalize(q_embs[i:i+1].to(device) * m, p=2, dim=-1)
            c_v = F.normalize(c_t * m, p=2, dim=-1)
            sim = torch.mm(q_v, c_v.T)
            topk = sim.topk(k, dim=-1).indices.cpu().numpy()[0]
            ndcg_scores[i] = _ndcg_from_topk(topk, gt_indices[i]) / ideal
    return ndcg_scores


# ─────────────────────── Statistics ──────────────────────────────────────────

def wilcoxon_test(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Paired Wilcoxon signed-rank test. Returns p-value (two-tailed)."""
    diff = scores_a - scores_b
    if np.all(diff == 0):
        return 1.0
    if len(diff) < 10:
        return 1.0
    if HAS_SCIPY and _wilcoxon is not None:
        try:
            _, p = _wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
            return float(p)
        except Exception:
            pass
    # Fallback: permutation test
    return permutation_test(scores_a, scores_b)


def permutation_test(scores_a: np.ndarray, scores_b: np.ndarray,
                     n_perm: int = 5000, seed: int = 42) -> float:
    obs = abs(scores_a.mean() - scores_b.mean())
    rng = np.random.RandomState(seed)
    combined = np.stack([scores_a, scores_b], axis=1)
    count = 0
    for _ in range(n_perm):
        flip = rng.randint(0, 2, size=len(scores_a)).astype(bool)
        perm_a = np.where(flip, combined[:, 1], combined[:, 0])
        perm_b = np.where(flip, combined[:, 0], combined[:, 1])
        if abs(perm_a.mean() - perm_b.mean()) >= obs:
            count += 1
    return count / n_perm


def bootstrap_mean_diff(scores_a: np.ndarray, scores_b: np.ndarray,
                         n_boot: int = 2000, ci: float = 0.95,
                         seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap CI on mean(a) - mean(b). Returns (mean_diff, lo, hi)."""
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        diffs.append(scores_a[idx].mean() - scores_b[idx].mean())
    diffs = np.sort(diffs)
    alpha = (1 - ci) / 2
    lo = diffs[int(alpha * n_boot)]
    hi = diffs[int((1 - alpha) * n_boot)]
    return float(scores_a.mean() - scores_b.mean()), float(lo), float(hi)


def sig_stars(p: float, bonferroni_n: int = 1) -> str:
    p_adj = min(p * bonferroni_n, 1.0)
    if p_adj < 0.001: return "***"
    if p_adj < 0.01:  return "**"
    if p_adj < 0.05:  return "*"
    return "ns"


def format_p(p: float, bonferroni_n: int = 1) -> str:
    p_adj = min(p * bonferroni_n, 1.0)
    if p_adj < 0.001: return "<0.001"
    return f"{p_adj:.3f}"


# ─────────────────────── Per-dataset runner ──────────────────────────────────

def run_dataset(ds: str, args, device) -> Dict:
    print(f"\n{'='*70}")
    print(f"  Dataset: {ds}")
    print(f"{'='*70}")

    cfg_dir  = os.path.join(args.cfg_root, ds, "configs")
    mrl_cfg_path   = os.path.join(cfg_dir, "mrl.yaml")
    bam_b_cfg_path = os.path.join(cfg_dir, "bam_b.yaml")
    bam_pq_cfg_path= os.path.join(cfg_dir, "bam_pq.yaml")

    mrl_ckpt   = os.path.join(args.ckpt_root, ds, "mrl",   "best")
    bam_b_ckpt = os.path.join(args.ckpt_root, ds, "bam_b", "best_bsr")
    bam_pq_ckpt= os.path.join(args.ckpt_root, ds, "bam_pq","best_bsr")

    # Fallback to base configs
    if not os.path.exists(mrl_cfg_path):
        mrl_cfg_path = "configs/mrl_e5large.yaml"
    if not os.path.exists(bam_b_cfg_path):
        bam_b_cfg_path = "configs/bam_optionb_e5large.yaml"
    if not os.path.exists(bam_pq_cfg_path):
        bam_pq_cfg_path = "configs/bam_pq.yaml"

    mrl_config   = load_config(mrl_cfg_path)
    bam_b_config = load_config(bam_b_cfg_path)

    # Determine data paths
    if ds == "educational":
        test_path   = mrl_config["data"]["test_path"]
        corpus_path = mrl_config["data"]["corpus_path"]
    else:
        test_path   = os.path.join(args.beir_root, ds, "test.jsonl")
        corpus_path = os.path.join(args.beir_root, ds, "corpus.jsonl")

    # Load data
    print(f"  Loading data from {test_path} ...")
    corpus  = [json.loads(l) for l in open(corpus_path)]
    samples = [json.loads(l) for l in open(test_path)]
    corpus_id_to_idx = {p["id"]: i for i, p in enumerate(corpus)}
    valid = [s for s in samples if s.get("positive_id","") in corpus_id_to_idx]
    print(f"  Corpus: {len(corpus):,}  Valid queries: {len(valid):,}")

    corpus_texts = [p["text"] for p in corpus]
    gt_indices   = np.array([corpus_id_to_idx[s["positive_id"]] for s in valid])
    query_blooms = np.array([s["bloom_level"] for s in valid])

    results = {}
    has_pq = os.path.exists(os.path.join(bam_pq_ckpt, "checkpoint.pt"))

    # ── MRL ──────────────────────────────────────────────────────────────────
    print("\n  [MRL] encoding ...")
    tok_mrl = AutoTokenizer.from_pretrained(mrl_config["model"]["backbone"])
    mrl_model = load_mrl_model(mrl_config, mrl_ckpt, device)
    mrl_c = encode_corpus(mrl_model, corpus_texts, tok_mrl, device)
    mrl_q = encode_queries_mrl(mrl_model, valid, tok_mrl, device)
    del mrl_model
    if device.type == "cuda": torch.cuda.empty_cache()

    mrl_r10   = per_query_recall(mrl_q, mrl_c, gt_indices, k=10, device=device)
    mrl_ndcg  = per_query_ndcg(mrl_q, mrl_c, gt_indices, k=10, device=device)
    results["MRL"] = {"r10": mrl_r10, "ndcg10": mrl_ndcg}
    print(f"    MRL  R@10={mrl_r10.mean():.4f}  NDCG@10={mrl_ndcg.mean():.4f}")

    # ── BAM-B ────────────────────────────────────────────────────────────────
    print("\n  [BAM-B] encoding ...")
    tok_b = AutoTokenizer.from_pretrained(bam_b_config["model"]["backbone"])
    bam_b_model = load_bam_model(bam_b_config, bam_b_ckpt, device)
    bam_b_c = encode_corpus(bam_b_model, corpus_texts, tok_b, device)
    bam_b_q, bam_b_masks, _, bam_b_dims, bam_b_prefix = \
        encode_queries_bam(bam_b_model, valid, tok_b, device)
    del bam_b_model
    if device.type == "cuda": torch.cuda.empty_cache()

    bam_b_r10  = per_query_recall(bam_b_q, bam_b_c, gt_indices, k=10,
                                   masks=bam_b_masks, dims=bam_b_dims,
                                   is_prefix=bam_b_prefix, device=device)
    bam_b_ndcg = per_query_ndcg(bam_b_q, bam_b_c, gt_indices, k=10,
                                  masks=bam_b_masks, dims=bam_b_dims,
                                  is_prefix=bam_b_prefix, device=device)
    results["BAM-B"] = {"r10": bam_b_r10, "ndcg10": bam_b_ndcg}
    print(f"    BAM-B  R@10={bam_b_r10.mean():.4f}  NDCG@10={bam_b_ndcg.mean():.4f}")

    # ── BAM-PQ ───────────────────────────────────────────────────────────────
    if has_pq:
        print("\n  [BAM-PQ] encoding ...")
        bam_pq_config = load_config(bam_pq_cfg_path)
        bam_pq_config["training"]["loss"].setdefault("bloom_frequencies", [1/6]*6)
        tok_pq = AutoTokenizer.from_pretrained(bam_pq_config["model"]["backbone"])
        bam_pq_model = load_bam_model(bam_pq_config, bam_pq_ckpt, device)
        bam_pq_c = encode_corpus(bam_pq_model, corpus_texts, tok_pq, device)
        bam_pq_q, bam_pq_masks, _, bam_pq_dims, bam_pq_prefix = \
            encode_queries_bam(bam_pq_model, valid, tok_pq, device)
        del bam_pq_model
        if device.type == "cuda": torch.cuda.empty_cache()

        bam_pq_r10  = per_query_recall(bam_pq_q, bam_pq_c, gt_indices, k=10,
                                        masks=bam_pq_masks, dims=bam_pq_dims,
                                        is_prefix=bam_pq_prefix, device=device)
        bam_pq_ndcg = per_query_ndcg(bam_pq_q, bam_pq_c, gt_indices, k=10,
                                      masks=bam_pq_masks, dims=bam_pq_dims,
                                      is_prefix=bam_pq_prefix, device=device)
        results["BAM-PQ"] = {"r10": bam_pq_r10, "ndcg10": bam_pq_ndcg}
        print(f"    BAM-PQ R@10={bam_pq_r10.mean():.4f}  NDCG@10={bam_pq_ndcg.mean():.4f}")
    else:
        print(f"\n  [BAM-PQ] checkpoint not found at {bam_pq_ckpt} — skipping.")

    # ── Statistical tests ────────────────────────────────────────────────────
    # Bonferroni correction: 2 metrics × up to 3 pairs = 6 tests per dataset
    n_comparisons = 2 * (3 if has_pq else 1)

    comparisons = [("BAM-B", "MRL")]
    if has_pq:
        comparisons += [("BAM-PQ", "MRL"), ("BAM-PQ", "BAM-B")]

    print(f"\n  {'─'*72}")
    print(f"  Significance tests  (Bonferroni n={n_comparisons}, Wilcoxon signed-rank)")
    print(f"  {'─'*72}")
    print(f"  {'Comparison':<20}  {'Metric':<8}  {'Δ mean':>8}  {'95% CI':>20}  "
          f"{'p (adj)':>8}  {'sig':>4}")
    print(f"  {'─'*72}")

    stat_results = {}
    for (model_a, model_b) in comparisons:
        for metric in ("r10", "ndcg10"):
            sa = results[model_a][metric]
            sb = results[model_b][metric]
            p_raw = wilcoxon_test(sa, sb)
            mean_diff, lo, hi = bootstrap_mean_diff(sa, sb)
            p_adj = min(p_raw * n_comparisons, 1.0)
            stars = sig_stars(p_raw, n_comparisons)
            label = f"{model_a} vs {model_b}"
            metric_label = "R@10" if metric == "r10" else "NDCG@10"
            ci_str = f"[{lo:+.4f}, {hi:+.4f}]"
            print(f"  {label:<20}  {metric_label:<8}  {mean_diff:+8.4f}  "
                  f"{ci_str:>20}  {format_p(p_raw, n_comparisons):>8}  {stars:>4}")
            key = f"{model_a}_vs_{model_b}_{metric}"
            stat_results[key] = {
                "mean_diff": mean_diff, "ci_lo": lo, "ci_hi": hi,
                "p_raw": p_raw, "p_bonferroni": p_adj, "significant": stars != "ns",
                "stars": stars,
            }

    # ── Per-Bloom breakdown ───────────────────────────────────────────────────
    print(f"\n  Per-Bloom R@10 breakdown")
    print(f"  {'Level':<12}  {'n':>5}  {'MRL':>7}  {'BAM-B':>7}"
          + (f"  {'BAM-PQ':>8}" if has_pq else "")
          + f"  {'Δ(B-M)':>8}  {'p_B':>8}"
          + (f"  {'Δ(PQ-M)':>9}  {'p_PQ':>8}" if has_pq else ""))
    print(f"  {'─'*80}")

    bloom_results = {}
    for level in range(1, 7):
        mask = query_blooms == level
        n = int(mask.sum())
        if n < 5:
            continue
        bname = BLOOM_NAMES[level]
        mrl_b   = results["MRL"]["r10"][mask]
        bam_b_b = results["BAM-B"]["r10"][mask]

        p_b = wilcoxon_test(bam_b_b, mrl_b)
        stars_b = sig_stars(p_b)

        row = (f"  {bname:<12}  {n:5d}  {mrl_b.mean():7.4f}  "
               f"{bam_b_b.mean():7.4f}")

        bloom_entry = {
            "n": n,
            "mrl_r10": float(mrl_b.mean()),
            "bam_b_r10": float(bam_b_b.mean()),
            "bam_b_vs_mrl_p": p_b,
            "bam_b_vs_mrl_stars": stars_b,
        }

        if has_pq:
            bam_pq_b = results["BAM-PQ"]["r10"][mask]
            p_pq = wilcoxon_test(bam_pq_b, mrl_b)
            stars_pq = sig_stars(p_pq)
            delta_b  = bam_b_b.mean()  - mrl_b.mean()
            delta_pq = bam_pq_b.mean() - mrl_b.mean()
            row += (f"  {bam_pq_b.mean():8.4f}  {delta_b:+8.4f}  "
                    f"{format_p(p_b):>8}  {delta_pq:+9.4f}  {format_p(p_pq):>8}")
            bloom_entry.update({
                "bam_pq_r10": float(bam_pq_b.mean()),
                "bam_pq_vs_mrl_p": p_pq,
                "bam_pq_vs_mrl_stars": stars_pq,
            })
        else:
            delta_b = bam_b_b.mean() - mrl_b.mean()
            row += f"  {delta_b:+8.4f}  {format_p(p_b):>8}"

        print(row)
        bloom_results[bname] = bloom_entry

    return {
        "dataset": ds,
        "n_queries": len(valid),
        "means": {m: {"r10": float(v["r10"].mean()), "ndcg10": float(v["ndcg10"].mean())}
                  for m, v in results.items()},
        "significance": stat_results,
        "per_bloom": bloom_results,
        "n_comparisons_bonferroni": n_comparisons,
    }


# ─────────────────────── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",  nargs="+",
                        default=["educational", "scifact", "nfcorpus", "fiqa"])
    parser.add_argument("--ckpt_root", default="/tmp/multi-domain")
    parser.add_argument("--cfg_root",  default="results/multi_domain")
    parser.add_argument("--beir_root", default="/tmp/data/beir")
    parser.add_argument("--output_dir",default="results/significance/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if not HAS_SCIPY:
        print("Install scipy for Wilcoxon test: pip install scipy")

    all_results = {}
    for ds in args.datasets:
        try:
            all_results[ds] = run_dataset(ds, args, device)
        except Exception as e:
            print(f"\n  ERROR on {ds}: {e}")
            import traceback; traceback.print_exc()
            all_results[ds] = {"error": str(e)}

    # ── Combined summary ──────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  SIGNIFICANCE SUMMARY — BAM-B vs MRL  (Bonferroni-corrected)")
    print(f"{'='*70}")
    print(f"  {'Dataset':<14}  {'N':>5}  {'ΔMRL R@10':>10}  {'CI':>20}  "
          f"{'p (adj)':>8}  {'sig':>4}  {'ΔNDCG@10':>10}  {'sig':>4}")
    print(f"  {'─'*82}")
    for ds, r in all_results.items():
        if "error" in r: continue
        n = r["n_queries"]
        sig = r["significance"]
        r10 = sig.get("BAM-B_vs_MRL_r10", {})
        nd  = sig.get("BAM-B_vs_MRL_ndcg10", {})
        ci  = f"[{r10.get('ci_lo',0):+.4f},{r10.get('ci_hi',0):+.4f}]"
        print(f"  {ds:<14}  {n:5d}  {r10.get('mean_diff',0):+10.4f}  "
              f"{ci:>20}  {r10.get('p_bonferroni',1):>8.3f}  "
              f"{r10.get('stars','ns'):>4}  "
              f"{nd.get('mean_diff',0):+10.4f}  {nd.get('stars','ns'):>4}")

    if any("BAM-PQ_vs_MRL_r10" in r.get("significance", {}) for r in all_results.values()):
        print(f"\n  SIGNIFICANCE SUMMARY — BAM-PQ vs MRL  (Bonferroni-corrected)")
        print(f"  {'─'*82}")
        for ds, r in all_results.items():
            if "error" in r: continue
            sig = r["significance"]
            pq = sig.get("BAM-PQ_vs_MRL_r10", {})
            nd = sig.get("BAM-PQ_vs_MRL_ndcg10", {})
            if not pq: continue
            ci = f"[{pq.get('ci_lo',0):+.4f},{pq.get('ci_hi',0):+.4f}]"
            print(f"  {ds:<14}  {r['n_queries']:5d}  {pq.get('mean_diff',0):+10.4f}  "
                  f"{ci:>20}  {pq.get('p_bonferroni',1):>8.3f}  "
                  f"{pq.get('stars','ns'):>4}  "
                  f"{nd.get('mean_diff',0):+10.4f}  {nd.get('stars','ns'):>4}")

    # Save
    out_path = os.path.join(args.output_dir, "significance_results.json")
    # Convert numpy arrays to plain floats before saving
    def _clean(obj):
        if isinstance(obj, dict):  return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_clean(v) for v in obj]
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        return obj
    with open(out_path, "w") as f:
        json.dump(_clean(all_results), f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
