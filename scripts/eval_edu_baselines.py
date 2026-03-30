"""
Evaluate retrieval baselines on the educational test set.

Baselines:
  1. BM25 (lexical, rank_bm25)
  2. BGE-base at full 768 dims, no MRL (backbone only)
  3. MRL Baseline (our backbone + MRL training, no Bloom routing)
  4. BAM (full model)

Also reports Bloom-stratified R@10 and avg dims used (for efficiency comparison).

Usage:
    python scripts/eval_edu_baselines.py \
        --config configs/bam.yaml \
        --bam_checkpoint /tmp/bam-ckpts/best/ \
        --mrl_checkpoint /tmp/mrl-ckpts/best/ \
        --output_dir results/baselines/
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from models.bam import BloomAlignedMRL
from models.encoder import MRLEncoder
from transformers import AutoModel, AutoTokenizer


BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply",
               4: "Analyze", 5: "Evaluate", 6: "Create"}


# ─────────────────────── Shared helpers ───────────────────────────────────

def load_corpus_and_queries(corpus_path, test_path):
    corpus = []
    with open(corpus_path) as f:
        for line in f:
            corpus.append(json.loads(line.strip()))
    corpus_id_to_idx = {p["id"]: i for i, p in enumerate(corpus)}

    samples = []
    with open(test_path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    valid = [s for s in samples if s.get("positive_id", "") in corpus_id_to_idx]

    return corpus, corpus_id_to_idx, valid


def compute_metrics(query_embs, corpus_embs, gt_indices, query_blooms, device):
    """Compute R@k, NDCG@10, MRR, Bloom-stratified R@10."""
    N = len(query_embs)
    if not isinstance(query_embs, torch.Tensor):
        query_embs = torch.tensor(query_embs, dtype=torch.float32)
    if not isinstance(corpus_embs, torch.Tensor):
        corpus_embs = torch.tensor(corpus_embs, dtype=torch.float32)

    rankings = []
    for i in range(0, N, 256):
        sim = torch.mm(query_embs[i:i + 256].to(device), corpus_embs.t().to(device))
        topk = sim.topk(100, dim=-1).indices.cpu().numpy()
        rankings.append(topk)
    rankings = np.concatenate(rankings)

    metrics = {}
    for k in [1, 5, 10, 50]:
        hits = np.array([gt_indices[i] in rankings[i, :k] for i in range(N)])
        metrics[f"recall@{k}"] = float(hits.mean())

    mrrs = [1.0 / (np.where(rankings[i] == gt_indices[i])[0][0] + 1)
            if len(np.where(rankings[i] == gt_indices[i])[0]) > 0 else 0.0
            for i in range(N)]
    metrics["mrr"] = float(np.mean(mrrs))

    ndcgs = []
    for i in range(N):
        for j, idx in enumerate(rankings[i, :10]):
            if idx == gt_indices[i]:
                ndcgs.append(1.0 / np.log2(j + 2))
                break
        else:
            ndcgs.append(0.0)
    metrics["ndcg@10"] = float(np.mean(ndcgs))

    for level in range(1, 7):
        mask = query_blooms == level
        if mask.sum() == 0:
            continue
        lr = rankings[mask]; lg = gt_indices[mask]; nl = int(mask.sum())
        hits = np.array([lg[i] in lr[i, :10] for i in range(nl)])
        metrics[f"bloom_{BLOOM_NAMES[level]}_recall@10"] = float(hits.mean())
        metrics[f"bloom_{BLOOM_NAMES[level]}_n"] = int(nl)

    return metrics


# ─────────────────────── BM25 ─────────────────────────────────────────────

def eval_bm25(corpus, valid, corpus_id_to_idx) -> Dict:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("  rank_bm25 not installed. Run: pip install rank_bm25")
        return {}

    print("  Building BM25 index...")
    tokenized = [doc["text"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)

    query_texts = [s["query"] for s in valid]
    gt_indices = np.array([corpus_id_to_idx[s["positive_id"]] for s in valid])
    query_blooms = np.array([s["bloom_level"] for s in valid])
    N = len(valid)

    print("  BM25 retrieval...")
    rankings = []
    for q in tqdm(query_texts, desc="  BM25", leave=False):
        scores = bm25.get_scores(q.lower().split())
        top = np.argsort(scores)[::-1][:100]
        rankings.append(top)
    rankings = np.array(rankings)

    metrics = {"avg_dims": 0.0}   # BM25 doesn't use dense dims
    for k in [1, 5, 10, 50]:
        hits = np.array([gt_indices[i] in rankings[i, :k] for i in range(N)])
        metrics[f"recall@{k}"] = float(hits.mean())
    mrrs = [1.0 / (np.where(rankings[i] == gt_indices[i])[0][0] + 1)
            if len(np.where(rankings[i] == gt_indices[i])[0]) > 0 else 0.0
            for i in range(N)]
    metrics["mrr"] = float(np.mean(mrrs))
    ndcgs = []
    for i in range(N):
        for j, idx in enumerate(rankings[i, :10]):
            if idx == gt_indices[i]:
                ndcgs.append(1.0 / np.log2(j + 2))
                break
        else:
            ndcgs.append(0.0)
    metrics["ndcg@10"] = float(np.mean(ndcgs))
    for level in range(1, 7):
        mask = query_blooms == level
        if mask.sum() == 0:
            continue
        lr = rankings[mask]; lg = gt_indices[mask]; nl = int(mask.sum())
        hits = np.array([lg[i] in lr[i, :10] for i in range(nl)])
        metrics[f"bloom_{BLOOM_NAMES[level]}_recall@10"] = float(hits.mean())

    return metrics


# ─────────────────────── BGE-base zero-shot ───────────────────────────────

@torch.no_grad()
def eval_bge_base(corpus, valid, corpus_id_to_idx, model_name, device) -> Dict:
    """Evaluate backbone model (no MRL training) at full 768 dims."""
    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    def encode_texts(texts, max_len=256, batch_size=128):
        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="  encoding", leave=False):
            enc = tokenizer(texts[i:i+batch_size], padding=True, truncation=True,
                            max_length=max_len, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            emb = F.normalize(out.last_hidden_state[:, 0], p=2, dim=-1)
            all_embs.append(emb.cpu())
        return torch.cat(all_embs)

    corpus_embs = encode_texts([c["text"] for c in corpus], max_len=256)
    query_embs = encode_texts([s["query"] for s in valid], max_len=128)

    del model
    torch.cuda.empty_cache()

    gt_indices = np.array([corpus_id_to_idx[s["positive_id"]] for s in valid])
    query_blooms = np.array([s["bloom_level"] for s in valid])
    metrics = compute_metrics(query_embs, corpus_embs, gt_indices, query_blooms, device)
    metrics["avg_dims"] = 768.0
    return metrics


# ─────────────────────── MRL Baseline ────────────────────────────────────

@torch.no_grad()
def eval_mrl_baseline(corpus, valid, corpus_id_to_idx, config, checkpoint_path, device) -> Dict:
    mc = config["model"]
    model = MRLEncoder(model_name=mc["backbone"], embedding_dim=mc["embedding_dim"],
                       mrl_dims=mc["mrl_dims"])
    ckpt = os.path.join(checkpoint_path, "checkpoint.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"],
                               strict=False)
        print(f"  Loaded MRL from {ckpt}")
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(mc["backbone"])

    def encode(texts, max_len):
        all_embs = []
        for i in tqdm(range(0, len(texts), 128), desc="  encoding", leave=False):
            enc = tokenizer(texts[i:i+128], padding=True, truncation=True,
                            max_length=max_len, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(enc["input_ids"], enc["attention_mask"])
            all_embs.append(out["full"].cpu())
        return torch.cat(all_embs)

    corpus_embs = encode([c["text"] for c in corpus], 256)
    query_embs = encode([s["query"] for s in valid], 128)

    gt_indices = np.array([corpus_id_to_idx[s["positive_id"]] for s in valid])
    query_blooms = np.array([s["bloom_level"] for s in valid])
    metrics = compute_metrics(query_embs, corpus_embs, gt_indices, query_blooms, device)
    metrics["avg_dims"] = 768.0
    return metrics


# ─────────────────────── BAM ─────────────────────────────────────────────

@torch.no_grad()
def eval_bam(corpus, valid, corpus_id_to_idx, config, checkpoint_path, device) -> Dict:
    model = BloomAlignedMRL(config)
    ckpt = os.path.join(checkpoint_path, "checkpoint.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"],
                               strict=False)
        print(f"  Loaded BAM from {ckpt}")
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])

    # Encode corpus at full dims
    corpus_embs = []
    for i in tqdm(range(0, len(corpus), 128), desc="  corpus", leave=False):
        batch = [c["text"] for c in corpus[i:i+128]]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
        corpus_embs.append(out["full_embedding"].cpu())
    corpus_embs = torch.cat(corpus_embs)

    # Encode queries with Bloom routing
    query_embs = []
    avg_dims_list = []
    for i in tqdm(range(0, len(valid), 64), desc="  queries", leave=False):
        batch_samples = valid[i:i+64]
        enc = tokenizer([s["query"] for s in batch_samples], padding=True,
                        truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        bloom_labels = torch.tensor(
            [s["bloom_level"] - 1 for s in batch_samples], dtype=torch.long, device=device
        )
        out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                   bloom_labels=bloom_labels)
        query_embs.append(out["masked_embedding"].cpu())
        avg_dims_list.append(out["policy_output"]["selected_dim"].cpu())

    query_embs = torch.cat(query_embs)
    avg_dims = torch.cat(avg_dims_list)

    gt_indices = np.array([corpus_id_to_idx[s["positive_id"]] for s in valid])
    query_blooms = np.array([s["bloom_level"] for s in valid])
    metrics = compute_metrics(query_embs, corpus_embs, gt_indices, query_blooms, device)
    metrics["avg_dims"] = float(avg_dims.mean().item())
    return metrics


# ─────────────────────── Main ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bam.yaml")
    parser.add_argument("--bam_checkpoint", default=None)
    parser.add_argument("--mrl_checkpoint", default=None)
    parser.add_argument("--output_dir", default="results/baselines/")
    parser.add_argument("--skip_bge", action="store_true",
                        help="Skip BGE-base zero-shot eval (saves time)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpus, corpus_id_to_idx, valid = load_corpus_and_queries(
        config["data"]["corpus_path"], config["data"]["test_path"]
    )
    print(f"  Corpus: {len(corpus)}, Valid queries: {len(valid)}")

    all_results = {}

    # BM25
    print("\n── BM25 ──────────────────────────────")
    all_results["BM25"] = eval_bm25(corpus, valid, corpus_id_to_idx)

    # BGE-base zero-shot (backbone, no training)
    if not args.skip_bge:
        print("\n── BGE-base (zero-shot, no MRL) ──────")
        all_results["BGE-base (zero-shot)"] = eval_bge_base(
            corpus, valid, corpus_id_to_idx,
            config["model"]["backbone"], device,
        )

    # MRL Baseline
    if args.mrl_checkpoint:
        print("\n── MRL Baseline ──────────────────────")
        all_results["MRL Baseline"] = eval_mrl_baseline(
            corpus, valid, corpus_id_to_idx, config, args.mrl_checkpoint, device
        )

    # BAM
    if args.bam_checkpoint:
        print("\n── BAM (ours) ────────────────────────")
        all_results["BAM"] = eval_bam(
            corpus, valid, corpus_id_to_idx, config, args.bam_checkpoint, device
        )

    # Save
    out_path = os.path.join(args.output_dir, "edu_baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    # ── Print table ───────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print("EDUCATIONAL TEST SET — BASELINE COMPARISON")
    print("=" * 95)
    hdr = f"{'Method':28s}{'R@1':>7s}{'R@10':>7s}{'R@50':>7s}{'NDCG@10':>9s}{'MRR':>7s}{'AvgDim':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for name, res in all_results.items():
        print(f"{name:28s}"
              f"{res.get('recall@1', 0):>7.4f}"
              f"{res.get('recall@10', 0):>7.4f}"
              f"{res.get('recall@50', 0):>7.4f}"
              f"{res.get('ndcg@10', 0):>9.4f}"
              f"{res.get('mrr', 0):>7.4f}"
              f"{res.get('avg_dims', 768):>8.0f}")

    print(f"\n{'Bloom-Stratified R@10':28s}")
    print("-" * 95)
    bloom_hdr = f"{'Method':28s}" + "".join(f"{BLOOM_NAMES[l]:>13s}" for l in range(1, 7))
    print(bloom_hdr)
    for name, res in all_results.items():
        row = f"{name:28s}"
        for l in range(1, 7):
            row += f"{res.get(f'bloom_{BLOOM_NAMES[l]}_recall@10', 0):>13.4f}"
        print(row)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
