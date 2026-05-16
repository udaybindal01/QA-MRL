"""
Fair per-Bloom-level comparison: BAM-PQ vs MRL at the SAME dimension budget.

For each Bloom level, BAM-PQ uses N_b active dims (e.g. Remember≈396, Understand≈675).
This script evaluates MRL truncated to exactly N_b dims on that level's queries,
so the retrieval budgets match.

Usage:
    python scripts/eval_fair_comparison.py \
        --config configs/bam.yaml \
        --bam_checkpoint /tmp/bam-ckpts/best/ \
        --mrl_checkpoint /tmp/mrl-ckpts/best/ \
        --output_dir results/fair_comparison/

Optional: use pre-computed BAM results to skip re-encoding BAM queries:
    --bam_results results/bam_eval/results.json
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from models.bam import BloomAlignedMRL
from models.encoder import MRLEncoder
from evaluation.evaluator import BLOOM_NAMES, bootstrap_ci
from transformers import AutoTokenizer


# ── helpers ──────────────────────────────────────────────────────────────────

def load_bam(config, ckpt_path, device):
    config["training"]["loss"].setdefault("bloom_frequencies", [1 / 6] * 6)
    model = BloomAlignedMRL(config)
    f = os.path.join(ckpt_path, "checkpoint.pt")
    if os.path.exists(f):
        ckpt = torch.load(f, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  Loaded BAM from {f}")
    return model.to(device).eval()


def load_mrl(config, ckpt_path, device):
    mc = config["model"]
    model = MRLEncoder(
        model_name=mc["backbone"],
        embedding_dim=mc["embedding_dim"],
        mrl_dims=mc["mrl_dims"],
        pooling=mc.get("pooling", "cls"),
        backbone_type=mc.get("backbone_type", "standard"),
        query_instruction=mc.get("query_instruction", None),
        peft_model_name=mc.get("peft_model_name", None),
    )
    f = os.path.join(ckpt_path, "checkpoint.pt")
    if os.path.exists(f):
        ckpt = torch.load(f, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  Loaded MRL from {f}")
    return model.to(device).eval()


def encode_corpus(model, corpus_texts, tokenizer, device, batch_size=128):
    all_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(corpus_texts), batch_size),
                      desc="  corpus", leave=False):
            batch = corpus_texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=256, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            if hasattr(model, "encode_documents"):
                out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
                all_embs.append(out["masked_embedding"].cpu())
            else:
                out = model(enc["input_ids"], enc["attention_mask"])
                all_embs.append(out["full"].cpu())
    return torch.cat(all_embs)   # [C, D]


def encode_queries_bam(model, samples, tokenizer, device, batch_size=64):
    """
    Returns full (unmasked) embeddings, per-query active dims, and hard masks.
    Works for both Option A (prefix) and Option B (scattered).
    """
    all_full, all_dims, all_masks = [], [], []
    all_active_dims = []

    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            texts = [s["query"] for s in batch]
            bloom_0 = [s["bloom_level"] - 1 for s in batch]

            enc = tokenizer(texts, padding=True, truncation=True,
                            max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}

            bloom_labels = torch.tensor(bloom_0, dtype=torch.long, device=device)
            out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                       bloom_labels=bloom_labels)

            if "full_embedding" in out:
                all_full.append(out["full_embedding"].cpu())
            else:
                all_full.append(out["masked_embedding"].cpu())

            if "discrete_dim" in out:
                all_dims.append(out["discrete_dim"].cpu())
            if "mask" in out:
                hard = (out["mask"] > 0.5).float()
                all_masks.append(hard.cpu())
                all_active_dims.append(hard.sum(dim=-1).cpu())

    full_embs = torch.cat(all_full)
    dims = torch.cat(all_dims) if all_dims else None
    masks = torch.cat(all_masks) if all_masks else None
    active_dims = torch.cat(all_active_dims) if all_active_dims else None
    return full_embs, dims, masks, active_dims


def encode_queries_mrl(model, samples, tokenizer, device, batch_size=64):
    """Full-dim MRL embeddings (pre-normalization for later per-level truncation)."""
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            texts = [s["query"] for s in batch]
            enc = tokenizer(texts, padding=True, truncation=True,
                            max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(enc["input_ids"], enc["attention_mask"])
            all_embs.append(out["full"].cpu())
    return torch.cat(all_embs)   # [N, D]


def recall_at_k(q_embs, c_embs, gt_indices, k, device, chunk=256):
    """Compute R@k for given query/corpus embeddings (already normalized)."""
    N = len(q_embs)
    hits = []
    for i in range(0, N, chunk):
        q = q_embs[i:i + chunk].to(device)
        sim = torch.mm(q, c_embs.to(device).t())
        topk = sim.topk(k, dim=-1).indices.cpu().numpy()
        for j, row in enumerate(topk):
            hits.append(int(gt_indices[i + j] in row))
    return np.array(hits, dtype=float)


def bam_retrieval_per_level(
    level_idx, full_embs, dims, masks, corpus_embs, level_gt,
    device, is_prefix, k=10
):
    """
    Run BAM retrieval for queries at one Bloom level, returning hit array.

    level_idx: global query indices for this level (used to slice full_embs/dims/masks)
    level_gt:  gt_indices already subset to this level (len == len(level_idx))
    Option A (is_prefix=True): normalize(q[:d]) · normalize(c[:d])
    Option B (is_prefix=False): normalize(q*mask) · normalize(c*mask)
    """
    N = len(level_idx)
    hits = np.zeros(N)

    if is_prefix:
        for j, qi in enumerate(level_idx):
            d = int(dims[qi].item())
            d = max(1, min(d, corpus_embs.shape[1]))
            q_v = F.normalize(full_embs[qi:qi+1, :d], p=2, dim=-1).to(device)
            c_v = F.normalize(corpus_embs[:, :d], p=2, dim=-1).to(device)
            sim = torch.mm(q_v, c_v.t())
            topk = sim.topk(k, dim=-1).indices.cpu().numpy()[0]
            hits[j] = int(level_gt[j] in topk)
    else:
        for j, qi in enumerate(level_idx):
            m = masks[qi].to(device)
            q_v = F.normalize(full_embs[qi:qi+1].to(device) * m, p=2, dim=-1)
            c_v = F.normalize(corpus_embs.to(device) * m, p=2, dim=-1)
            sim = torch.mm(q_v, c_v.t())
            topk = sim.topk(k, dim=-1).indices.cpu().numpy()[0]
            hits[j] = int(level_gt[j] in topk)

    return hits


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bam.yaml")
    parser.add_argument("--mrl_config", default=None,
                        help="Config for the MRL model (backbone/pooling/etc). "
                             "Defaults to --config if not provided.")
    parser.add_argument("--bam_checkpoint", required=True)
    parser.add_argument("--mrl_checkpoint", required=True)
    parser.add_argument("--bam_results", default=None,
                        help="Path to pre-computed BAM results.json "
                             "(skips BAM encoding if provided)")
    parser.add_argument("--output_dir", default="results/fair_comparison/")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    mrl_config = load_config(args.mrl_config) if args.mrl_config else config
    set_seed(config["training"]["seed"])
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])

    test_path   = config["data"]["test_path"]
    corpus_path = config["data"]["corpus_path"]
    k           = args.k

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading corpus...")
    corpus = [json.loads(l) for l in open(corpus_path)]
    corpus_id_to_idx = {p["id"]: i for i, p in enumerate(corpus)}
    print(f"  {len(corpus)} passages")

    print("Loading test queries...")
    all_samples = [json.loads(l) for l in open(test_path)]
    valid = [s for s in all_samples if s.get("positive_id", "") in corpus_id_to_idx]
    print(f"  {len(valid)} valid queries (positive in corpus)")

    gt_indices = np.array([corpus_id_to_idx[s["positive_id"]] for s in valid])
    query_blooms = np.array([s["bloom_level"] for s in valid])   # 1-indexed

    corpus_texts = [p["text"] for p in corpus]

    # ── MRL: encode corpus + queries ─────────────────────────────────────────
    print("\nEncoding corpus + queries with MRL...")
    mrl_model = load_mrl(mrl_config, args.mrl_checkpoint, device)
    mrl_corpus_embs = encode_corpus(mrl_model, corpus_texts, tokenizer, device)
    print(f"  MRL corpus: {mrl_corpus_embs.shape}")
    mrl_q_embs = encode_queries_mrl(mrl_model, valid, tokenizer, device)
    print(f"  MRL queries: {mrl_q_embs.shape}")
    del mrl_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── BAM: encode corpus + queries ─────────────────────────────────────────
    # Corpus must be encoded with BAM's own encoder (different weights from MRL).
    # Comparing BAM queries against MRL corpus would be a cross-model dot product.
    bam_bloom_dims = {}   # level (1-indexed) → avg active dims

    print("\nEncoding corpus + queries with BAM...")
    bam_model = load_bam(config, args.bam_checkpoint, device)
    bam_corpus_embs = encode_corpus(bam_model, corpus_texts, tokenizer, device)
    print(f"  BAM corpus: {bam_corpus_embs.shape}")
    bam_full_embs, bam_dims, bam_masks, bam_active = encode_queries_bam(
        bam_model, valid, tokenizer, device
    )

    # Read per-level dims from saved results if available (avoids re-encoding)
    if args.bam_results and os.path.exists(args.bam_results):
        with open(args.bam_results) as f:
            bam_res = json.load(f)
        bam_key = next((k for k in bam_res if "BAM" in k and "Encoder" not in k
                        and "MRL" not in k), None)
        if bam_key:
            for level in range(1, 7):
                dim_key = f"bloom_{BLOOM_NAMES[level]}_avg_dim"
                if dim_key in bam_res[bam_key]:
                    bam_bloom_dims[level] = int(round(bam_res[bam_key][dim_key]))
            if bam_bloom_dims:
                print(f"  Per-level dims from saved results: {bam_bloom_dims}")

    # Compute per-level dims from the freshly encoded queries if not read from file
    if not bam_bloom_dims:
        for level in range(1, 7):
            lmask = query_blooms == level
            if lmask.sum() == 0:
                continue
            if bam_active is not None:
                bam_bloom_dims[level] = int(round(bam_active[lmask].float().mean().item()))
            elif bam_dims is not None:
                bam_bloom_dims[level] = int(round(bam_dims[lmask].float().mean().item()))
        print(f"  Per-level dims (from live encoding): {bam_bloom_dims}")

    del bam_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    is_prefix = bam_dims is not None  # True = Option A, False = Option B (scattered)
    print(f"\nBAM routing mode: {'Option A (prefix)' if is_prefix else 'Option B (scattered mask)'}")

    # ── Per-level fair evaluation ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Fair Comparison: BAM vs MRL @ same dim budget (R@{k})")
    print(f"{'='*70}")
    header = (f"  {'Level':14s}  {'N':>5}  {'Budget':>7}  "
              f"{'MRL-full':>9}  {'MRL-trunc':>10}  "
              f"{'BAM-PQ':>9}  {'Δ(BAM-MRL_trunc)':>17}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    results = {}
    for level in range(1, 7):
        mask = query_blooms == level
        n = int(mask.sum())
        if n == 0:
            continue
        name = BLOOM_NAMES[level]
        level_idx = np.where(mask)[0]
        # Subset gt_indices to this level — fixes off-by-index bug when recall_at_k
        # uses i+j as a local offset (not a global index into gt_indices).
        level_gt = gt_indices[level_idx]
        budget = bam_bloom_dims.get(level, None)

        # MRL at full dims (MRL corpus + MRL queries, same model on both sides)
        q_full_norm = F.normalize(mrl_q_embs[level_idx], p=2, dim=-1)
        c_full_norm = F.normalize(mrl_corpus_embs, p=2, dim=-1)
        hits_mrl_full = recall_at_k(q_full_norm, c_full_norm, level_gt, k, device)

        # MRL truncated to BAM's budget (MRL corpus prefix-truncated)
        hits_mrl_trunc = None
        if budget is not None:
            d = max(1, min(budget, mrl_q_embs.shape[1]))
            q_trunc = F.normalize(mrl_q_embs[level_idx, :d], p=2, dim=-1)
            c_trunc = F.normalize(mrl_corpus_embs[:, :d], p=2, dim=-1)
            hits_mrl_trunc = recall_at_k(q_trunc, c_trunc, level_gt, k, device)

        # BAM at its per-level budget (BAM corpus + BAM queries, same model on both sides)
        hits_bam = bam_retrieval_per_level(
            level_idx, bam_full_embs, bam_dims, bam_masks,
            bam_corpus_embs, level_gt, device, is_prefix=is_prefix, k=k
        )

        r_mrl_full  = float(hits_mrl_full.mean())
        r_mrl_trunc = float(hits_mrl_trunc.mean()) if hits_mrl_trunc is not None else float("nan")
        r_bam       = float(hits_bam.mean())
        delta       = r_bam - r_mrl_trunc if not np.isnan(r_mrl_trunc) else float("nan")

        # Bootstrap CIs
        _, lo_mrl_trunc, hi_mrl_trunc = bootstrap_ci(hits_mrl_trunc) if hits_mrl_trunc is not None else (0, 0, 0)
        _, lo_bam,       hi_bam       = bootstrap_ci(hits_bam)

        budget_str = f"{budget:4d}" if budget else "  N/A"
        delta_str  = f"{delta:+.4f}" if not np.isnan(delta) else "  N/A"
        print(f"  {name:14s}  {n:5d}  {budget_str}    "
              f"{r_mrl_full:9.4f}  {r_mrl_trunc:10.4f}  "
              f"{r_bam:8.4f}  {delta_str}")

        results[name] = {
            "n": n,
            "budget_dims": budget,
            f"mrl_full_recall@{k}": r_mrl_full,
            f"mrl_truncated_recall@{k}": r_mrl_trunc,
            f"bam_recall@{k}": r_bam,
            f"delta_bam_minus_mrl_trunc": delta,
            f"mrl_trunc_recall@{k}_ci": [lo_mrl_trunc, hi_mrl_trunc],
            f"bam_recall@{k}_ci": [lo_bam, hi_bam],
        }

    # ── Summary ──────────────────────────────────────────────────────────────
    valid_deltas = [v["delta_bam_minus_mrl_trunc"] for v in results.values()
                    if not np.isnan(v["delta_bam_minus_mrl_trunc"])]
    avg_delta = float(np.mean(valid_deltas)) if valid_deltas else float("nan")
    wins = sum(1 for d in valid_deltas if d > 0)

    print(f"\n  Average Δ(BAM − MRL-trunc): {avg_delta:+.4f}")
    print(f"  BAM wins (positive Δ): {wins}/{len(valid_deltas)} levels")

    note = ("Option A: prefix slicing — comparable FAISS sub-index efficiency."
            if is_prefix else
            "Option B: scattered mask — BAM uses non-contiguous dims; "
            "MRL uses prefix dims of the same count. "
            "Note: MRL prefix-truncation is a stronger competitor "
            "for the same computational budget.")
    print(f"\n  Note: {note}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "routing_mode": "prefix" if is_prefix else "scattered",
        "k": k,
        "per_level": results,
        "avg_delta_bam_minus_mrl_trunc": avg_delta,
        "bam_wins": wins,
        "total_levels": len(valid_deltas),
        "note": note,
    }
    path = os.path.join(args.output_dir, "fair_comparison.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
