"""
Complete BAM-PQ analysis: correlation, dimension allocation, efficiency, significance.

Sections:
  1. Bloom–Dim Correlation  — Kendall's tau: does the model learn cognitive ordering?
  2. Dimension Allocation   — per-Bloom mean/std dims, pairwise mask cosine similarity
  3. Efficiency Curves      — R@10 vs dim budget for BAM-PQ, MRL, BAM-B (optional)
  4. Significance Tests     — Wilcoxon + bootstrap CI + Bonferroni (BAM-PQ vs MRL, BAM-B vs MRL, BAM-PQ vs BAM-B)

Usage (single dataset / backbone):
    python scripts/analyze_bampq_complete.py \
        --config          results/multi_domain/educational/configs/bam_pq_e5large.yaml \
        --mrl_config      results/multi_domain/educational/configs/mrl_e5large.yaml \
        --bampq_ckpt      /tmp/multi-domain/educational/bam_pq_e5large/best_bsr \
        --mrl_ckpt        /tmp/multi-domain/educational/mrl_e5large/best \
        --bam_b_ckpt      /tmp/multi-domain/educational/bam_b/best_bsr \
        --bam_b_config    results/multi_domain/educational/configs/bam_b.yaml \
        --output_dir      results/analysis/educational/e5large/

Usage (all backbones, all datasets):
    python scripts/analyze_bampq_complete.py \
        --all \
        --datasets educational msmarco scifact nfcorpus fiqa \
        --backbones e5large bge qwen06b llama1b \
        --ckpt_root /tmp/multi-domain \
        --cfg_root  results/multi_domain \
        --output_dir results/analysis/
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
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from models.bam import BloomAlignedMRL
from models.encoder import MRLEncoder
from transformers import AutoTokenizer

try:
    from scipy.stats import kendalltau as _kendalltau, wilcoxon as _scipy_wilcoxon
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found — kendalltau/wilcoxon will use fallbacks.")

BLOOM_NAMES = {0: "Remember", 1: "Understand", 2: "Apply",
               3: "Analyze",  4: "Evaluate",   5: "Create"}
BLOOM_NAMES_1IDX = {1: "Remember", 2: "Understand", 3: "Apply",
                    4: "Analyze",  5: "Evaluate",   6: "Create"}


# ─────────────────────────── Model loading ────────────────────────────────────

def load_bam(config, ckpt_path, device):
    config["training"]["loss"].setdefault("bloom_frequencies", [1/6]*6)
    model = BloomAlignedMRL(config)
    f = os.path.join(ckpt_path, "checkpoint.pt")
    if os.path.exists(f):
        ckpt = torch.load(f, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"    Loaded BAM: {f}")
    else:
        print(f"    WARNING: checkpoint not found at {f}")
    return model.to(device).eval()


def load_mrl(config, ckpt_path, device):
    mc = config["model"]
    model = MRLEncoder(model_name=mc["backbone"],
                       embedding_dim=mc["embedding_dim"],
                       mrl_dims=mc["mrl_dims"])
    f = os.path.join(ckpt_path, "checkpoint.pt")
    if os.path.exists(f):
        ckpt = torch.load(f, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"    Loaded MRL: {f}")
    else:
        print(f"    WARNING: checkpoint not found at {f}")
    return model.to(device).eval()


# ─────────────────────────── Encoding helpers ─────────────────────────────────

@torch.no_grad()
def encode_corpus(model, texts, tokenizer, device, batch_size=64):
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="    corpus", leave=False):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        if hasattr(model, "encode_documents"):
            out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
            emb = out.get("masked_embedding", out.get("full", out.get("embedding")))
        else:
            out = model(enc["input_ids"], enc["attention_mask"])
            emb = out.get("full", out.get("embedding"))
        all_embs.append(emb.cpu().float())
    return torch.cat(all_embs)


@torch.no_grad()
def encode_queries_bam(model, samples, tokenizer, device, batch_size=64):
    """Returns (full_embs [N,D], masks [N,D] or None, active_dims [N] or None,
                discrete_dims [N] or None, is_prefix bool)."""
    all_full, all_masks, all_active, all_dims = [], [], [], []
    for i in tqdm(range(0, len(samples), batch_size), desc="    queries", leave=False):
        batch = samples[i:i+batch_size]
        texts  = [s["query"] for s in batch]
        blooms = [s["bloom_level"] - 1 for s in batch]
        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        bl_t = torch.tensor(blooms, dtype=torch.long, device=device)
        out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                   bloom_labels=bl_t)
        full = out.get("full_embedding", out.get("masked_embedding"))
        all_full.append(full.cpu().float())
        if "mask" in out:
            hard = (out["mask"] > 0.5).float()
            all_masks.append(hard.cpu())
            all_active.append(hard.sum(dim=-1).cpu())
        if "discrete_dim" in out:
            all_dims.append(out["discrete_dim"].cpu())
        elif "active_dims" in out:
            all_dims.append(out["active_dims"].cpu())

    full_embs = torch.cat(all_full)
    masks     = torch.cat(all_masks)  if all_masks  else None
    active    = torch.cat(all_active) if all_active else None
    dims      = torch.cat(all_dims)   if all_dims   else None
    is_prefix = (dims is not None and masks is None)
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
        all_embs.append(out.get("full", out.get("embedding")).cpu().float())
    return torch.cat(all_embs)


# ─────────────────────────── Retrieval metrics ────────────────────────────────

def per_query_recall(q_embs, c_embs, gt_indices, k=10,
                     masks=None, dims=None, is_prefix=False,
                     device=None, chunk=256):
    N = len(q_embs)
    hits = np.zeros(N, dtype=np.float32)

    if masks is None and dims is None:
        c_norm = F.normalize(c_embs.to(device), p=2, dim=-1)
        for i in range(0, N, chunk):
            q = F.normalize(q_embs[i:i+chunk].to(device), p=2, dim=-1)
            sim = torch.mm(q, c_norm.T)
            topk = sim.topk(k, dim=-1).indices.cpu().numpy()
            for j, row in enumerate(topk):
                hits[i+j] = int(gt_indices[i+j] in row)
    elif is_prefix and dims is not None:
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


def per_query_ndcg(q_embs, c_embs, gt_indices, k=10,
                   masks=None, dims=None, is_prefix=False,
                   device=None, chunk=256):
    N = len(q_embs)
    scores = np.zeros(N, dtype=np.float32)

    def _ndcg(topk_row, gt):
        for rank, idx in enumerate(topk_row):
            if idx == gt:
                return 1.0 / math.log2(rank + 2)
        return 0.0

    if masks is None and dims is None:
        c_norm = F.normalize(c_embs.to(device), p=2, dim=-1)
        for i in range(0, N, chunk):
            q = F.normalize(q_embs[i:i+chunk].to(device), p=2, dim=-1)
            sim = torch.mm(q, c_norm.T)
            topk = sim.topk(k, dim=-1).indices.cpu().numpy()
            for j, row in enumerate(topk):
                scores[i+j] = _ndcg(row, gt_indices[i+j])
    elif is_prefix and dims is not None:
        c_t = c_embs.to(device)
        for i in range(N):
            d = max(1, int(dims[i].item()))
            q_v = F.normalize(q_embs[i:i+1, :d].to(device), p=2, dim=-1)
            c_v = F.normalize(c_t[:, :d], p=2, dim=-1)
            sim = torch.mm(q_v, c_v.T)
            topk = sim.topk(k, dim=-1).indices.cpu().numpy()[0]
            scores[i] = _ndcg(topk, gt_indices[i])
    else:
        assert masks is not None
        c_t = c_embs.to(device)
        for i in range(N):
            m = masks[i].to(device)
            q_v = F.normalize(q_embs[i:i+1].to(device) * m, p=2, dim=-1)
            c_v = F.normalize(c_t * m, p=2, dim=-1)
            sim = torch.mm(q_v, c_v.T)
            topk = sim.topk(k, dim=-1).indices.cpu().numpy()[0]
            scores[i] = _ndcg(topk, gt_indices[i])
    return scores


# ─────────────────────────── Statistical helpers ──────────────────────────────

def kendall_tau(levels, means):
    if HAS_SCIPY:
        tau, p = _kendalltau(levels, means)
        return float(tau), float(p)
    # Manual Kendall's tau
    n = len(levels)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i+1, n):
            sign_l = levels[j] - levels[i]
            sign_m = means[j]  - means[i]
            if sign_l * sign_m > 0:  concordant += 1
            elif sign_l * sign_m < 0: discordant += 1
    denom = n * (n-1) / 2
    tau = (concordant - discordant) / denom if denom > 0 else 0.0
    return tau, 1.0  # no p-value without scipy


def wilcoxon_test(a, b):
    diff = a - b
    if np.all(diff == 0) or len(diff) < 10:
        return 1.0
    if HAS_SCIPY:
        try:
            _, p = _scipy_wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
            return float(p)
        except Exception:
            pass
    # Permutation fallback
    rng = np.random.RandomState(42)
    obs = abs(a.mean() - b.mean())
    combined = np.stack([a, b], axis=1)
    count = sum(
        1 for _ in range(5000)
        if abs(np.where(rng.randint(0,2,len(a)).astype(bool),
                        combined[:,1], combined[:,0]).mean()
               - np.where(rng.randint(0,2,len(a)).astype(bool),
                          combined[:,0], combined[:,1]).mean()) >= obs
    )
    return count / 5000


def bootstrap_ci(a, b, n_boot=2000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    n = len(a)
    diffs = [a[rng.choice(n, n, replace=True)].mean()
             - b[rng.choice(n, n, replace=True)].mean()
             for _ in range(n_boot)]
    diffs = np.sort(diffs)
    alpha = (1 - ci) / 2
    return (float(a.mean() - b.mean()),
            float(diffs[int(alpha * n_boot)]),
            float(diffs[int((1-alpha) * n_boot)]))


def sig_stars(p, n=1):
    pa = min(p * n, 1.0)
    if pa < 0.001: return "***"
    if pa < 0.01:  return "**"
    if pa < 0.05:  return "*"
    return "ns"


# ─────────────────────────── Section 1+2: Dim analysis ───────────────────────

def analyze_dimensions(model, samples, tokenizer, device,
                        batch_size=64, n_samples=2000):
    """
    Returns:
        bloom_dims  dict{0..5: np.array of active dims per query}
        mean_masks  dict{0..5: tensor[D]}
        alpha       float or None  (BAM-PQ alpha value if present)
    """
    samples = samples[:n_samples]
    bloom_dims  = defaultdict(list)
    bloom_masks = defaultdict(list)
    alpha_vals  = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(samples), batch_size),
                      desc="    dim analysis", leave=False):
            batch = samples[i:i+batch_size]
            texts  = [s["query"] for s in batch]
            blooms = [s["bloom_level"] - 1 for s in batch]
            enc = tokenizer(texts, padding=True, truncation=True,
                            max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            bl_t = torch.tensor(blooms, dtype=torch.long, device=device)
            out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                       bloom_labels=bl_t)

            if "mask" in out:
                hard = (out["mask"] > 0.5).float().cpu()
                active = hard.sum(dim=-1).tolist()
                for j, bl in enumerate(blooms):
                    bloom_dims[bl].append(active[j])
                    bloom_masks[bl].append(hard[j])
            elif "discrete_dim" in out:
                dims_vals = out["discrete_dim"].cpu().tolist()
                for j, bl in enumerate(blooms):
                    bloom_dims[bl].append(dims_vals[j])

            if "alpha" in out:
                alpha_vals.extend(out["alpha"].cpu().flatten().tolist())

    mean_masks = {}
    for b in range(6):
        if bloom_masks[b]:
            mean_masks[b] = torch.stack(bloom_masks[b]).mean(dim=0)

    bloom_dims_np = {b: np.array(v) for b, v in bloom_dims.items() if v}

    alpha_mean = float(np.mean(alpha_vals)) if alpha_vals else None
    return bloom_dims_np, mean_masks, alpha_mean


def print_dim_analysis(bloom_dims_np, mean_masks, alpha_mean=None, max_dims=768):
    print("\n" + "="*60)
    print("  SECTION 1+2: Dimension Allocation Analysis")
    print("="*60)

    if alpha_mean is not None:
        print(f"\n  BAM-PQ alpha (query-specific weight): {alpha_mean:.4f}")
        if alpha_mean < 0.1:
            print("  → Per-query MLP barely contributes; BAM-PQ ≈ BAM-B")

    print(f"\n  Per-Bloom Mean Active Dims (max={max_dims})")
    print(f"  {'Level':>12s}  {'N':>6s}  {'Mean':>8s}  {'Std':>7s}  "
          f"{'Min':>6s}  {'Max':>6s}  {'Frac':>7s}")
    print(f"  {'-'*60}")

    levels_present, means = [], []
    for b in range(6):
        if b not in bloom_dims_np:
            continue
        d = bloom_dims_np[b]
        m = d.mean()
        means.append(m)
        levels_present.append(b)
        print(f"  {BLOOM_NAMES[b]:>12s}  {len(d):>6d}  {m:>8.1f}  {d.std():>7.1f}  "
              f"  {d.min():>4.0f}  {d.max():>4.0f}  {m/max_dims:>7.3f}")

    spread = max(means) - min(means) if means else 0.0
    print(f"\n  Spread (max-min across levels): {spread:.1f} dims")

    # Kendall's tau
    tau, p = 0.0, 1.0
    if len(levels_present) >= 3:
        tau, p = kendall_tau(levels_present, means)
        print(f"\n  Kendall's tau (Bloom level vs active dims): tau={tau:+.3f}, p={p:.4f}")
        if tau > 0 and p < 0.05:
            print("  → SUPPORTS cognitive load hypothesis (p<0.05)")
        elif tau > 0:
            print(f"  → Positive trend (tau={tau:+.3f}) but not significant (p={p:.3f})")
        else:
            print(f"  → Does NOT support cognitive load hypothesis")
    else:
        print("\n  Too few Bloom levels for Kendall's tau.")

    # Pairwise mask similarity
    if len(mean_masks) >= 2:
        present = sorted(mean_masks.keys())
        masks_t = torch.stack([mean_masks[b] for b in present])
        normed  = F.normalize(masks_t, p=2, dim=-1)
        sim     = torch.mm(normed, normed.t()).numpy()

        print(f"\n  Pairwise Mask Cosine Similarity (lower = more specialized)")
        header = f"  {'':12s}" + "".join(f"{BLOOM_NAMES[b]:>12s}" for b in present)
        print(header)
        for i, bi in enumerate(present):
            row = f"  {BLOOM_NAMES[bi]:12s}"
            for j in range(len(present)):
                row += f"{sim[i,j]:>12.3f}"
            print(row)
        off = sim[np.triu_indices(len(present), k=1)]
        print(f"\n  Mean off-diagonal similarity: {off.mean():.3f} (0=specialized, 1=identical)")
    else:
        sim = None

    return {"kendall_tau": tau, "kendall_p": p,
            "per_bloom_mean_dims": {BLOOM_NAMES[b]: float(bloom_dims_np[b].mean())
                                    for b in bloom_dims_np},
            "per_bloom_n": {BLOOM_NAMES[b]: int(len(bloom_dims_np[b]))
                            for b in bloom_dims_np},
            "spread_dims": float(spread),
            "supports_cognitive_hypothesis": bool(tau > 0 and p < 0.05),
            "alpha_mean": alpha_mean}


# ─────────────────────────── Section 3: Efficiency curves ─────────────────────

def efficiency_curves(mrl_q, mrl_c, bampq_q, bampq_c,
                      bampq_masks, bampq_dims, bampq_is_prefix,
                      gt_indices, query_blooms,
                      bam_b_q=None, bam_b_c=None,
                      bam_b_masks=None, bam_b_dims=None, bam_b_is_prefix=False,
                      max_dims=768, k=10, device=None):
    """Sweep fixed-dim budgets for MRL; compute BAM-PQ operating points per Bloom."""

    # Generate dim budgets in 64-dim steps up to max_dims (covers 768, 1024, etc.)
    dim_budgets = sorted(set(range(64, max_dims + 1, 64)) | {max_dims})
    dim_budgets = [d for d in dim_budgets if d <= max_dims]

    print("\n" + "="*60)
    print("  SECTION 3: Efficiency Curves")
    print("="*60)

    # MRL curve
    mrl_curve = {}
    print("\n  Computing MRL R@10 curve ...")
    for d in dim_budgets:
        q_d = F.normalize(mrl_q[:, :d], p=2, dim=-1)
        c_d = F.normalize(mrl_c[:, :d], p=2, dim=-1)
        hits = []
        for i in range(0, len(q_d), 512):
            q_chunk = q_d[i:i+512].to(device)
            sim = torch.mm(q_chunk, c_d.to(device).T)
            topk = sim.topk(k, dim=-1).indices.cpu().numpy()
            for j, row in enumerate(topk):
                hits.append(int(gt_indices[i+j] in row))
        mrl_curve[d] = float(np.mean(hits))

    # BAM-PQ operating points per Bloom
    bampq_per_bloom = {}
    for b in range(6):
        lmask = (query_blooms == (b+1))
        n = int(lmask.sum())
        if n == 0:
            continue
        idx = np.where(lmask)[0]
        gt_l = gt_indices[idx]

        if bampq_is_prefix and bampq_dims is not None:
            avg_d = int(bampq_dims[idx].float().mean().item())
            hits = []
            for j, qi in enumerate(idx):
                d = max(1, min(int(bampq_dims[qi].item()), bampq_c.shape[1]))
                q_v = F.normalize(bampq_q[qi:qi+1, :d].to(device), p=2, dim=-1)
                c_v = F.normalize(bampq_c[:, :d].to(device), p=2, dim=-1)
                sim = torch.mm(q_v, c_v.T)
                topk = sim.topk(k, dim=-1).indices.cpu().numpy()[0]
                hits.append(int(gt_l[j] in topk))
        elif bampq_masks is not None:
            avg_d = int((bampq_masks[idx] > 0.5).float().sum(dim=-1).mean().item())
            hits = []
            for j, qi in enumerate(idx):
                m = bampq_masks[qi].to(device)
                q_v = F.normalize(bampq_q[qi:qi+1].to(device) * m, p=2, dim=-1)
                c_v = F.normalize(bampq_c.to(device) * m, p=2, dim=-1)
                sim = torch.mm(q_v, c_v.T)
                topk = sim.topk(k, dim=-1).indices.cpu().numpy()[0]
                hits.append(int(gt_l[j] in topk))
        else:
            continue

        bampq_per_bloom[BLOOM_NAMES[b]] = {
            "avg_dims": avg_d,
            f"recall@{k}": float(np.mean(hits)),
            "n": n,
        }

    # BAM-B operating points (optional)
    bamb_per_bloom = {}
    if bam_b_q is not None:
        for b in range(6):
            lmask = (query_blooms == (b+1))
            n = int(lmask.sum())
            if n == 0:
                continue
            idx = np.where(lmask)[0]
            gt_l = gt_indices[idx]

            if bam_b_is_prefix and bam_b_dims is not None:
                avg_d = int(bam_b_dims[idx].float().mean().item())
                hits = []
                for j, qi in enumerate(idx):
                    d = max(1, min(int(bam_b_dims[qi].item()), bam_b_c.shape[1]))
                    q_v = F.normalize(bam_b_q[qi:qi+1, :d].to(device), p=2, dim=-1)
                    c_v = F.normalize(bam_b_c[:, :d].to(device), p=2, dim=-1)
                    sim = torch.mm(q_v, c_v.T)
                    topk = sim.topk(k, dim=-1).indices.cpu().numpy()[0]
                    hits.append(int(gt_l[j] in topk))
            elif bam_b_masks is not None:
                avg_d = int((bam_b_masks[idx] > 0.5).float().sum(dim=-1).mean().item())
                hits = []
                for j, qi in enumerate(idx):
                    m = bam_b_masks[qi].to(device)
                    q_v = F.normalize(bam_b_q[qi:qi+1].to(device) * m, p=2, dim=-1)
                    c_v = F.normalize(bam_b_c.to(device) * m, p=2, dim=-1)
                    sim = torch.mm(q_v, c_v.T)
                    topk = sim.topk(k, dim=-1).indices.cpu().numpy()[0]
                    hits.append(int(gt_l[j] in topk))
            else:
                continue

            bamb_per_bloom[BLOOM_NAMES[b]] = {
                "avg_dims": avg_d,
                f"recall@{k}": float(np.mean(hits)),
                "n": n,
            }

    # Print MRL curve
    print(f"\n  MRL R@{k} vs dimensions:")
    print(f"  {'Dims':>6}  {'MRL R@10':>10}")
    print(f"  {'-'*20}")
    for d in dim_budgets:
        print(f"  {d:>6}  {mrl_curve[d]:>10.4f}")

    # Print BAM-PQ per-Bloom operating points
    def _print_ops(per_bloom, label):
        print(f"\n  {label} per-Bloom operating points vs MRL@same budget:")
        print(f"  {'Level':12s}  {'Dims':>6}  {'N':>5}  {label:>8}  {'MRL@same':>10}  {'Δ':>8}")
        print(f"  {'-'*55}")
        for name, lv in sorted(per_bloom.items(), key=lambda x: x[1]["avg_dims"]):
            d = lv["avg_dims"]
            r = lv[f"recall@{k}"]
            n = lv["n"]
            lower_d = max((dd for dd in dim_budgets if dd <= d), default=dim_budgets[0])
            r_mrl = mrl_curve.get(lower_d, mrl_curve[dim_budgets[0]])
            delta = r - r_mrl
            sign  = "+" if delta >= 0 else ""
            print(f"  {name:12s}  {d:>6}  {n:>5}  {r:>8.4f}  {r_mrl:>10.4f}  {sign}{delta*100:.1f}%")

    _print_ops(bampq_per_bloom, "BAM-PQ")
    if bamb_per_bloom:
        _print_ops(bamb_per_bloom, "BAM-B")

    return {
        "mrl_curve": {str(d): v for d, v in mrl_curve.items()},
        "bampq_per_bloom": bampq_per_bloom,
        "bamb_per_bloom": bamb_per_bloom,
        "k": k,
    }


# ─────────────────────────── Section 4: Significance testing ──────────────────

def significance_tests(results, query_blooms):
    """
    results: dict with keys "MRL", "BAM-PQ", "BAM-B" each having r10/ndcg10 arrays.
    """
    print("\n" + "="*60)
    print("  SECTION 4: Significance Testing")
    print("="*60)

    has_pq = "BAM-PQ" in results
    has_b  = "BAM-B" in results

    comparisons = []
    if has_b:  comparisons.append(("BAM-B",  "MRL"))
    if has_pq: comparisons.append(("BAM-PQ", "MRL"))
    if has_pq and has_b: comparisons.append(("BAM-PQ", "BAM-B"))

    n_comparisons = 2 * len(comparisons)  # two metrics

    print(f"\n  Wilcoxon signed-rank + Bootstrap 95% CI + Bonferroni (n={n_comparisons})")
    print(f"  {'Comparison':<20}  {'Metric':<8}  {'Δ mean':>8}  {'95% CI':>22}  "
          f"{'p(adj)':>8}  {'sig':>4}")
    print(f"  {'-'*78}")

    stat_out = {}
    for (model_a, model_b) in comparisons:
        for metric in ("r10", "ndcg10"):
            sa = results[model_a][metric]
            sb = results[model_b][metric]
            p_raw = wilcoxon_test(sa, sb)
            mean_diff, lo, hi = bootstrap_ci(sa, sb)
            p_adj = min(p_raw * n_comparisons, 1.0)
            stars = sig_stars(p_raw, n_comparisons)
            label = f"{model_a} vs {model_b}"
            m_label = "R@10" if metric == "r10" else "NDCG@10"
            ci_str = f"[{lo:+.4f}, {hi:+.4f}]"
            print(f"  {label:<20}  {m_label:<8}  {mean_diff:+8.4f}  "
                  f"{ci_str:>22}  {p_adj:>8.3f}  {stars:>4}")
            key = f"{model_a}_vs_{model_b}_{metric}"
            stat_out[key] = {"mean_diff": mean_diff, "ci_lo": lo, "ci_hi": hi,
                             "p_raw": p_raw, "p_bonferroni": p_adj,
                             "significant": stars != "ns", "stars": stars}

    # Per-Bloom R@10 breakdown
    print(f"\n  Per-Bloom R@10")
    header = f"  {'Level':12s}  {'N':>5}  {'MRL':>7}"
    if has_b:  header += f"  {'BAM-B':>7}  {'Δ(B)':>7}  {'p(B)':>7}"
    if has_pq: header += f"  {'BAM-PQ':>8}  {'Δ(PQ)':>7}  {'p(PQ)':>8}"
    print(header)
    print(f"  {'-'*80}")

    bloom_out = {}
    for level in range(1, 7):
        mask = query_blooms == level
        n = int(mask.sum())
        if n < 5:
            continue
        bname = BLOOM_NAMES_1IDX[level]
        mrl_b = results["MRL"]["r10"][mask]
        row = f"  {bname:12s}  {n:5d}  {mrl_b.mean():7.4f}"
        entry = {"n": n, "mrl_r10": float(mrl_b.mean())}

        if has_b:
            bb = results["BAM-B"]["r10"][mask]
            pb = wilcoxon_test(bb, mrl_b)
            row += f"  {bb.mean():7.4f}  {bb.mean()-mrl_b.mean():+7.4f}  {format_p(pb,1):>7}"
            entry["bam_b_r10"] = float(bb.mean())
            entry["bam_b_p"] = pb

        if has_pq:
            pq = results["BAM-PQ"]["r10"][mask]
            pp = wilcoxon_test(pq, mrl_b)
            row += f"  {pq.mean():8.4f}  {pq.mean()-mrl_b.mean():+7.4f}  {format_p(pp,1):>8}"
            entry["bam_pq_r10"] = float(pq.mean())
            entry["bam_pq_p"] = pp

        print(row)
        bloom_out[bname] = entry

    return stat_out, bloom_out


def format_p(p, n=1):
    pa = min(p * n, 1.0)
    return "<0.001" if pa < 0.001 else f"{pa:.3f}"


# ─────────────────────────── Main per-dataset runner ──────────────────────────

def run_dataset(bampq_config, mrl_config, bampq_ckpt, mrl_ckpt,
                bam_b_config, bam_b_ckpt,
                output_dir, device, k=10,
                n_dim_samples=2000, corpus_batch=64):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(bampq_config["model"]["backbone"])
    max_dims   = bampq_config["model"]["embedding_dim"]

    # Resolve data paths — bampq config is authoritative
    test_path   = bampq_config["data"]["test_path"]
    corpus_path = bampq_config["data"]["corpus_path"]

    print(f"\n  Loading data ...")
    corpus  = [json.loads(l) for l in open(corpus_path)]
    samples = [json.loads(l) for l in open(test_path)]
    corpus_id_to_idx = {p["id"]: i for i, p in enumerate(corpus)}
    valid   = [s for s in samples if s.get("positive_id","") in corpus_id_to_idx]
    gt_indices   = np.array([corpus_id_to_idx[s["positive_id"]] for s in valid])
    query_blooms = np.array([s["bloom_level"] for s in valid])
    corpus_texts = [p["text"] for p in corpus]
    print(f"  Corpus: {len(corpus):,}  Valid queries: {len(valid):,}")

    results_agg = {}

    # ── BAM-PQ ────────────────────────────────────────────────────────────────
    print("\n  [BAM-PQ] loading ...")
    bampq_model = load_bam(bampq_config, bampq_ckpt, device)
    bampq_c = encode_corpus(bampq_model, corpus_texts, tokenizer, device, corpus_batch)
    bampq_q, bampq_masks, _, bampq_dims, bampq_is_prefix = \
        encode_queries_bam(bampq_model, valid, tokenizer, device)

    # Dim analysis on BAM-PQ
    bloom_dims_np, mean_masks, alpha_mean = \
        analyze_dimensions(bampq_model, valid, tokenizer, device,
                           n_samples=n_dim_samples)
    del bampq_model
    if device.type == "cuda": torch.cuda.empty_cache()

    dim_results = print_dim_analysis(bloom_dims_np, mean_masks, alpha_mean, max_dims)

    bampq_r10  = per_query_recall(bampq_q, bampq_c, gt_indices, k,
                                   masks=bampq_masks, dims=bampq_dims,
                                   is_prefix=bampq_is_prefix, device=device)
    bampq_ndcg = per_query_ndcg(bampq_q, bampq_c, gt_indices, k,
                                  masks=bampq_masks, dims=bampq_dims,
                                  is_prefix=bampq_is_prefix, device=device)
    results_agg["BAM-PQ"] = {"r10": bampq_r10, "ndcg10": bampq_ndcg}
    print(f"\n  BAM-PQ   R@{k}={bampq_r10.mean():.4f}  NDCG@{k}={bampq_ndcg.mean():.4f}")

    # ── MRL ───────────────────────────────────────────────────────────────────
    print("\n  [MRL] loading ...")
    mrl_tok = AutoTokenizer.from_pretrained(mrl_config["model"]["backbone"])
    mrl_model = load_mrl(mrl_config, mrl_ckpt, device)
    mrl_c = encode_corpus(mrl_model, corpus_texts, mrl_tok, device, corpus_batch)
    mrl_q = encode_queries_mrl(mrl_model, valid, mrl_tok, device)
    del mrl_model
    if device.type == "cuda": torch.cuda.empty_cache()

    mrl_r10  = per_query_recall(mrl_q, mrl_c, gt_indices, k, device=device)
    mrl_ndcg = per_query_ndcg(mrl_q, mrl_c, gt_indices, k, device=device)
    results_agg["MRL"] = {"r10": mrl_r10, "ndcg10": mrl_ndcg}
    print(f"  MRL      R@{k}={mrl_r10.mean():.4f}  NDCG@{k}={mrl_ndcg.mean():.4f}")

    # ── BAM-B (optional) ──────────────────────────────────────────────────────
    bam_b_q = bam_b_c = bam_b_masks = bam_b_dims = None
    bam_b_is_prefix = False
    if bam_b_config and bam_b_ckpt and os.path.exists(os.path.join(bam_b_ckpt, "checkpoint.pt")):
        print("\n  [BAM-B] loading ...")
        b_tok = AutoTokenizer.from_pretrained(bam_b_config["model"]["backbone"])
        bam_b_model = load_bam(bam_b_config, bam_b_ckpt, device)
        bam_b_c = encode_corpus(bam_b_model, corpus_texts, b_tok, device, corpus_batch)
        bam_b_q, bam_b_masks, _, bam_b_dims, bam_b_is_prefix = \
            encode_queries_bam(bam_b_model, valid, b_tok, device)
        del bam_b_model
        if device.type == "cuda": torch.cuda.empty_cache()

        bamb_r10  = per_query_recall(bam_b_q, bam_b_c, gt_indices, k,
                                      masks=bam_b_masks, dims=bam_b_dims,
                                      is_prefix=bam_b_is_prefix, device=device)
        bamb_ndcg = per_query_ndcg(bam_b_q, bam_b_c, gt_indices, k,
                                    masks=bam_b_masks, dims=bam_b_dims,
                                    is_prefix=bam_b_is_prefix, device=device)
        results_agg["BAM-B"] = {"r10": bamb_r10, "ndcg10": bamb_ndcg}
        print(f"  BAM-B    R@{k}={bamb_r10.mean():.4f}  NDCG@{k}={bamb_ndcg.mean():.4f}")

    # ── Efficiency curves ─────────────────────────────────────────────────────
    eff_results = efficiency_curves(
        mrl_q, mrl_c, bampq_q, bampq_c,
        bampq_masks, bampq_dims, bampq_is_prefix,
        gt_indices, query_blooms,
        bam_b_q=bam_b_q, bam_b_c=bam_b_c,
        bam_b_masks=bam_b_masks, bam_b_dims=bam_b_dims,
        bam_b_is_prefix=bam_b_is_prefix,
        max_dims=max_dims, k=k, device=device
    )

    # ── Significance tests ────────────────────────────────────────────────────
    stat_results, bloom_stats = significance_tests(results_agg, query_blooms)

    # ── Save all results ──────────────────────────────────────────────────────
    def _clean(obj):
        if isinstance(obj, dict):          return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):          return [_clean(v) for v in obj]
        if isinstance(obj, np.ndarray):    return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)):     return int(obj)
        return obj

    out = {
        "n_queries":  len(valid),
        "n_corpus":   len(corpus),
        "dim_analysis": _clean(dim_results),
        "efficiency": _clean(eff_results),
        "significance": _clean(stat_results),
        "per_bloom_stats": _clean(bloom_stats),
        "means": {m: {"r10": float(v["r10"].mean()), "ndcg10": float(v["ndcg10"].mean())}
                  for m, v in results_agg.items()},
    }
    path = os.path.join(output_dir, "bampq_analysis.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved → {path}")
    return out


# ─────────────────────────── CLI ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Single-run mode
    parser.add_argument("--config",        default=None, help="BAM-PQ config yaml")
    parser.add_argument("--mrl_config",    default=None, help="MRL config yaml")
    parser.add_argument("--bampq_ckpt",    default=None, help="BAM-PQ best_bsr dir")
    parser.add_argument("--mrl_ckpt",      default=None, help="MRL best dir")
    parser.add_argument("--bam_b_ckpt",    default=None, help="BAM-B best_bsr dir (optional)")
    parser.add_argument("--bam_b_config",  default=None, help="BAM-B config yaml (optional)")

    # Multi-run mode
    parser.add_argument("--all",           action="store_true",
                        help="Run all backbone×dataset combinations")
    parser.add_argument("--datasets",      nargs="+",
                        default=["educational", "msmarco", "scifact", "nfcorpus", "fiqa"])
    parser.add_argument("--backbones",     nargs="+",
                        default=["e5large", "bge", "qwen06b", "qwen4b",
                                 "llama1b", "llama3b", "phi3mini"])
    parser.add_argument("--ckpt_root",     default="/tmp/multi-domain")
    parser.add_argument("--cfg_root",      default="results/multi_domain")

    # Common
    parser.add_argument("--output_dir",    default="results/analysis/")
    parser.add_argument("--k",             type=int, default=10)
    parser.add_argument("--n_dim_samples", type=int, default=2000,
                        help="Max test queries for dim analysis")
    parser.add_argument("--corpus_batch",  type=int, default=64)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  scipy: {HAS_SCIPY}")

    if args.all:
        all_results = {}
        for ds in args.datasets:
            for bk in args.backbones:
                label = f"{ds}/{bk}"
                print(f"\n{'='*70}")
                print(f"  Running: {label}")
                print(f"{'='*70}")

                pq_ckpt = f"{args.ckpt_root}/{ds}/bam_pq_{bk}/best_bsr"
                mrl_ckpt = f"{args.ckpt_root}/{ds}/mrl_{bk}/best"
                b_ckpt  = f"{args.ckpt_root}/{ds}/bam_b/best_bsr"

                pq_cfg = f"{args.cfg_root}/{ds}/configs/bam_pq_{bk}.yaml"
                mrl_cfg = f"{args.cfg_root}/{ds}/configs/mrl_{bk}.yaml"
                b_cfg  = f"{args.cfg_root}/{ds}/configs/bam_b.yaml"

                # Skip if required checkpoints missing
                if not os.path.exists(os.path.join(pq_ckpt, "checkpoint.pt")):
                    print(f"  Skipping {label} — BAM-PQ checkpoint not found: {pq_ckpt}")
                    continue
                if not os.path.exists(os.path.join(mrl_ckpt, "checkpoint.pt")):
                    print(f"  Skipping {label} — MRL checkpoint not found: {mrl_ckpt}")
                    continue
                if not os.path.exists(pq_cfg):
                    print(f"  Skipping {label} — BAM-PQ config not found: {pq_cfg}")
                    continue
                if not os.path.exists(mrl_cfg):
                    print(f"  Skipping {label} — MRL config not found: {mrl_cfg}")
                    continue

                pq_config  = load_config(pq_cfg)
                mrl_config = load_config(mrl_cfg)
                b_config   = load_config(b_cfg) if os.path.exists(b_cfg) else None
                if not os.path.exists(os.path.join(b_ckpt, "checkpoint.pt")):
                    b_ckpt = None

                out_dir = os.path.join(args.output_dir, ds, bk)
                try:
                    r = run_dataset(pq_config, mrl_config, pq_ckpt, mrl_ckpt,
                                    b_config, b_ckpt, out_dir, device,
                                    k=args.k, n_dim_samples=args.n_dim_samples,
                                    corpus_batch=args.corpus_batch)
                    all_results[label] = r
                except Exception as e:
                    print(f"\n  ERROR on {label}: {e}")
                    import traceback; traceback.print_exc()
                    all_results[label] = {"error": str(e)}

        # Cross-run summary
        print(f"\n\n{'='*70}")
        print("  CROSS-BACKBONE / CROSS-DATASET SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Label':30s}  {'N':>5}  {'BAM-PQ R@10':>12}  {'MRL R@10':>10}  "
              f"{'Δ':>8}  {'tau':>6}  {'sig':>5}")
        print(f"  {'-'*80}")
        for label, r in all_results.items():
            if "error" in r: continue
            means = r.get("means", {})
            pq_r  = means.get("BAM-PQ", {}).get("r10", float("nan"))
            mrl_r = means.get("MRL",    {}).get("r10", float("nan"))
            delta = pq_r - mrl_r
            tau   = r.get("dim_analysis", {}).get("kendall_tau", float("nan"))
            stars = r.get("significance", {}).get(
                "BAM-PQ_vs_MRL_r10", {}).get("stars", "—")
            print(f"  {label:30s}  {r['n_queries']:>5d}  "
                  f"{pq_r:>12.4f}  {mrl_r:>10.4f}  {delta:+8.4f}  "
                  f"{tau:>+6.3f}  {stars:>5}")

        summary_path = os.path.join(args.output_dir, "summary_all.json")
        def _clean(obj):
            if isinstance(obj, dict):         return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):         return [_clean(v) for v in obj]
            if isinstance(obj, np.ndarray):   return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)): return float(obj)
            if isinstance(obj, (np.int32, np.int64)):     return int(obj)
            return obj
        with open(summary_path, "w") as f:
            json.dump(_clean(all_results), f, indent=2)
        print(f"\nFull summary saved → {summary_path}")

    else:
        # Single-run mode
        if not all([args.config, args.mrl_config, args.bampq_ckpt, args.mrl_ckpt]):
            parser.error("Provide --config, --mrl_config, --bampq_ckpt, --mrl_ckpt "
                         "or use --all")

        pq_config  = load_config(args.config)
        mrl_config = load_config(args.mrl_config)
        b_config   = load_config(args.bam_b_config) if args.bam_b_config else None
        b_ckpt     = args.bam_b_ckpt

        os.makedirs(args.output_dir, exist_ok=True)
        run_dataset(pq_config, mrl_config, args.bampq_ckpt, args.mrl_ckpt,
                    b_config, b_ckpt, args.output_dir, device,
                    k=args.k, n_dim_samples=args.n_dim_samples,
                    corpus_batch=args.corpus_batch)


if __name__ == "__main__":
    main()
