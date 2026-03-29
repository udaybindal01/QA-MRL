"""
Ablation Study for BAM v3.

Tests the contribution of each component:
  1. BAM full (all components)
  2. BAM no routing (fixed 384 truncation — tests if routing matters)
  3. BAM random routing (random truncation dim per query)
  4. BAM fixed routing (same truncation for all queries)
  5. BAM no Bloom classifiers (MRL reweighting only, no query Bloom signal)
  6. BAM no D/sqrt(d) (uniform MRL weighting instead of reweighted)
  7. MRL Baseline (no BAM training at all)

Usage:
    python scripts/run_ablations.py --config configs/bam.yaml \
        --checkpoint /tmp/bam-ckpts/best/ \
        --baseline /tmp/bam-ckpts/mrl_baseline_best/
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from models.bam import BloomAlignedMRL
from models.encoder import MRLEncoder
from transformers import AutoTokenizer


BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply",
               4: "Analyze", 5: "Evaluate", 6: "Create"}


@torch.no_grad()
def evaluate_bam_ablation(
    model, test_path, corpus_path, tokenizer, device,
    truncation_strategy="normal", fixed_dim=384,
) -> Dict[str, float]:
    """
    Evaluate BAM with different truncation strategies.

    truncation_strategy:
        "normal"  - Use policy's learned truncation
        "none"    - No truncation (full 768 dims, ignoring policy)
        "random"  - Random truncation dim per query
        "fixed"   - Fixed truncation dim for all queries
    """
    model.eval()

    # Encode corpus (always full dims)
    corpus = []
    with open(corpus_path) as f:
        for line in f:
            corpus.append(json.loads(line.strip()))
    corpus_id_to_idx = {p["id"]: i for i, p in enumerate(corpus)}

    corpus_embs = []
    for i in tqdm(range(0, len(corpus), 128), desc="  corpus", leave=False):
        batch = [c["text"] for c in corpus[i:i+128]]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        if hasattr(model, "encode_documents"):
            out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
            corpus_embs.append(out["full_embedding"].cpu())
        else:
            out = model(enc["input_ids"], enc["attention_mask"])
            corpus_embs.append(out["full"].cpu())
    corpus_embs = torch.cat(corpus_embs)

    # Load test queries
    samples = []
    with open(test_path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    valid = [s for s in samples if s.get("positive_id", "") in corpus_id_to_idx]

    # Encode queries with ablated truncation
    query_embs = []
    avg_dims = []
    mrl_dims = [64, 128, 256, 384, 512, 768]

    for i in tqdm(range(0, len(valid), 64), desc="  queries", leave=False):
        batch_texts = [s["query"] for s in valid[i:i+64]]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        if hasattr(model, "encode_queries"):
            out = model.encode_queries(enc["input_ids"], enc["attention_mask"])
            full_emb = out["full_embedding"]

            if truncation_strategy == "normal":
                emb = out["masked_embedding"]
                avg_dims.append(out["policy_output"]["selected_dim"].cpu())
            elif truncation_strategy == "none":
                emb = full_emb
                avg_dims.append(torch.full((len(batch_texts),), 768.0))
            elif truncation_strategy == "random":
                # Random truncation dim per query
                dims = torch.tensor([mrl_dims[torch.randint(len(mrl_dims), (1,)).item()]
                                     for _ in range(len(batch_texts))])
                emb = torch.zeros_like(full_emb)
                for j, d in enumerate(dims):
                    emb[j, :d] = full_emb[j, :d]
                emb = F.normalize(emb, p=2, dim=-1)
                avg_dims.append(dims.float())
            elif truncation_strategy == "fixed":
                emb = torch.zeros_like(full_emb)
                emb[:, :fixed_dim] = full_emb[:, :fixed_dim]
                emb = F.normalize(emb, p=2, dim=-1)
                avg_dims.append(torch.full((len(batch_texts),), float(fixed_dim)))

            query_embs.append(emb.cpu())
        else:
            out = model(enc["input_ids"], enc["attention_mask"])
            query_embs.append(out["full"].cpu())
            avg_dims.append(torch.full((len(batch_texts),), 768.0))

    query_embs = torch.cat(query_embs)
    all_dims = torch.cat(avg_dims) if avg_dims else None

    # Retrieve and compute metrics
    gt_indices = np.array([corpus_id_to_idx[s["positive_id"]] for s in valid])
    query_blooms = np.array([s["bloom_level"] for s in valid])
    N = len(valid)

    rankings = []
    for i in range(0, N, 256):
        sim = torch.mm(query_embs[i:i+256], corpus_embs.t())
        topk = sim.topk(100, dim=-1).indices.numpy()
        rankings.append(topk)
    rankings = np.concatenate(rankings)

    metrics = {}
    for k in [1, 5, 10, 50]:
        hits = np.array([gt_indices[i] in rankings[i, :k] for i in range(N)])
        metrics[f"recall@{k}"] = float(hits.mean())

    mrrs = []
    for i in range(N):
        found = np.where(rankings[i] == gt_indices[i])[0]
        mrrs.append(1.0 / (found[0] + 1) if len(found) > 0 else 0.0)
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

    # Bloom-stratified
    for level in range(1, 7):
        mask = query_blooms == level
        if mask.sum() == 0:
            continue
        name = BLOOM_NAMES[level]
        lr = rankings[mask]
        lg = gt_indices[mask]
        nl = int(mask.sum())
        hits = np.array([lg[i] in lr[i, :10] for i in range(nl)])
        metrics[f"bloom_{name}_recall@10"] = float(hits.mean())

    if all_dims is not None:
        metrics["avg_dims"] = float(all_dims.mean().item())

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bam.yaml")
    parser.add_argument("--checkpoint", required=True, help="BAM checkpoint")
    parser.add_argument("--baseline", default=None, help="MRL baseline checkpoint")
    parser.add_argument("--output_dir", default="results/ablations/")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])

    test_path = config["data"]["test_path"]
    corpus_path = config["data"]["corpus_path"]

    # Load BAM
    bam_model = BloomAlignedMRL(config)
    ckpt = os.path.join(args.checkpoint, "checkpoint.pt")
    if os.path.exists(ckpt):
        bam_model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"], strict=False)
    bam_model.to(device).eval()

    all_results = {}

    # Ablation 1: Full BAM
    print("\n--- BAM full ---")
    all_results["BAM full"] = evaluate_bam_ablation(
        bam_model, test_path, corpus_path, tokenizer, device, "normal"
    )

    # Ablation 2: No routing (all 768 dims)
    print("\n--- BAM no routing (768 dims) ---")
    all_results["BAM no routing"] = evaluate_bam_ablation(
        bam_model, test_path, corpus_path, tokenizer, device, "none"
    )

    # Ablation 3: Random routing
    print("\n--- BAM random routing ---")
    all_results["BAM random routing"] = evaluate_bam_ablation(
        bam_model, test_path, corpus_path, tokenizer, device, "random"
    )

    # Ablation 4: Fixed routing (384 dims)
    print("\n--- BAM fixed 384 ---")
    all_results["BAM fixed 384"] = evaluate_bam_ablation(
        bam_model, test_path, corpus_path, tokenizer, device, "fixed", fixed_dim=384
    )

    # Ablation 5: Fixed routing (256 dims)
    print("\n--- BAM fixed 256 ---")
    all_results["BAM fixed 256"] = evaluate_bam_ablation(
        bam_model, test_path, corpus_path, tokenizer, device, "fixed", fixed_dim=256
    )

    # Ablation 6: Fixed routing (128 dims)
    print("\n--- BAM fixed 128 ---")
    all_results["BAM fixed 128"] = evaluate_bam_ablation(
        bam_model, test_path, corpus_path, tokenizer, device, "fixed", fixed_dim=128
    )

    # MRL Baseline
    if args.baseline:
        print("\n--- MRL Baseline ---")
        mc = config["model"]
        bl_model = MRLEncoder(model_name=mc["backbone"], embedding_dim=mc["embedding_dim"],
                              mrl_dims=mc["mrl_dims"])
        bl_ckpt = os.path.join(args.baseline, "checkpoint.pt")
        if os.path.exists(bl_ckpt):
            bl_model.load_state_dict(torch.load(bl_ckpt, map_location=device)["model_state_dict"], strict=False)
        bl_model.to(device).eval()

        all_results["MRL Baseline"] = evaluate_bam_ablation(
            bl_model, test_path, corpus_path, tokenizer, device, "none"
        )

    # Save
    with open(os.path.join(args.output_dir, "ablation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    # Print tables
    print("\n" + "=" * 100)
    print("ABLATION STUDY")
    print("=" * 100)

    header = f"{'Method':30s}{'R@10':>8s}{'R@50':>8s}{'NDCG':>8s}{'MRR':>8s}{'Dims':>8s}"
    print(header)
    print("-" * len(header))
    for name, res in all_results.items():
        print(f"{name:30s}{res.get('recall@10',0):>8.4f}{res.get('recall@50',0):>8.4f}"
              f"{res.get('ndcg@10',0):>8.4f}{res.get('mrr',0):>8.4f}"
              f"{res.get('avg_dims',768):>8.0f}")

    print(f"\n{'Bloom-Stratified R@10':30s}")
    print("-" * 100)
    header = f"{'Method':30s}" + "".join(f"{BLOOM_NAMES[l]:>12s}" for l in range(1, 7))
    print(header)
    for name, res in all_results.items():
        row = f"{name:30s}"
        for l in range(1, 7):
            key = f"bloom_{BLOOM_NAMES[l]}_recall@10"
            row += f"{res.get(key, 0):>12.4f}"
        print(row)

    print(f"\nSaved to {args.output_dir}/ablation_results.json")


if __name__ == "__main__":
    main()
