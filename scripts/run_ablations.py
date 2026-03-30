"""
Ablation Study for BAM.

Ablations:
  1. BAM full           — Bloom routing + FiLM conditioning (full model)
  2. BAM random Bloom   — Random Bloom labels passed to model (tests if Bloom routing matters)
  3. BAM no FiLM        — FiLM conditioning zeroed out (tests if Bloom conditioning helps)
  4. BAM fixed Bloom=1  — All queries forced to Bloom 1 (always minimum dims)
  5. BAM fixed Bloom=6  — All queries forced to Bloom 6 (always maximum dims)
  6. MRL Baseline       — No routing, full 768 dims

Key distinction: "random Bloom" passes RANDOM BLOOM LABELS through the model
(so the routing + FiLM conditioning runs on garbage labels) vs "no FiLM" which
disables the FiLM conditioning entirely. Together they separate routing from conditioning.

Usage:
    python scripts/run_ablations.py \
        --config configs/bam.yaml \
        --checkpoint /tmp/bam-ckpts/best/ \
        --baseline /tmp/mrl-ckpts/best/ \
        --output_dir results/ablations/
"""

import argparse
import contextlib
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from models.bam import BloomAlignedMRL
from models.encoder import MRLEncoder
from transformers import AutoTokenizer


BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply",
               4: "Analyze", 5: "Evaluate", 6: "Create"}


@contextlib.contextmanager
def zero_film(model):
    """Context manager that zeroes FiLM output — disables Bloom conditioning."""
    def _hook(module, input, output):
        return torch.zeros_like(output)
    handle = model.bloom_film_linear.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def evaluate_bam(
    model, test_path, corpus_path, tokenizer, device,
    ablation: str = "normal",
) -> Dict[str, float]:
    """
    Evaluate BAM under different ablation conditions.

    ablation:
        "normal"       — real Bloom labels from test data
        "random_bloom" — random Bloom labels (0-5) replacing real ones
        "no_film"      — FiLM output zeroed (bloom_film_linear → 0)
        "fixed_1"      — all queries set to Bloom level 0 (0-indexed)
        "fixed_6"      — all queries set to Bloom level 5 (0-indexed)
    """
    model.eval()

    # ── Encode corpus ────────────────────────────────────────────────
    corpus = []
    with open(corpus_path) as f:
        for line in f:
            corpus.append(json.loads(line.strip()))
    corpus_id_to_idx = {p["id"]: i for i, p in enumerate(corpus)}

    corpus_embs = []
    for i in tqdm(range(0, len(corpus), 128), desc="  corpus", leave=False):
        batch = [c["text"] for c in corpus[i:i + 128]]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
        corpus_embs.append(out["full_embedding"].cpu())
    corpus_embs = torch.cat(corpus_embs)  # [C, 768]

    # ── Load test queries ─────────────────────────────────────────────
    samples = []
    with open(test_path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    valid = [s for s in samples if s.get("positive_id", "") in corpus_id_to_idx]

    # ── Encode queries ────────────────────────────────────────────────
    query_embs = []
    avg_dims_list = []

    ctx = zero_film(model) if ablation == "no_film" else contextlib.nullcontext()

    with ctx:
        for i in tqdm(range(0, len(valid), 64), desc="  queries", leave=False):
            batch_samples = valid[i:i + 64]
            batch_texts = [s["query"] for s in batch_samples]
            enc = tokenizer(batch_texts, padding=True, truncation=True,
                            max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}

            # Build bloom_labels for this ablation
            B = len(batch_texts)
            if ablation == "random_bloom":
                # Random 0-indexed Bloom labels
                bloom_labels = torch.randint(0, 6, (B,), device=device)
            elif ablation == "fixed_1":
                bloom_labels = torch.zeros(B, dtype=torch.long, device=device)
            elif ablation == "fixed_6":
                bloom_labels = torch.full((B,), 5, dtype=torch.long, device=device)
            else:
                # "normal" or "no_film": use real labels (1-indexed in data → convert to 0-indexed)
                bloom_labels = torch.tensor(
                    [s["bloom_level"] - 1 for s in batch_samples],
                    dtype=torch.long, device=device
                )

            out = model.encode_queries(
                enc["input_ids"], enc["attention_mask"],
                bloom_labels=bloom_labels,
            )
            query_embs.append(out["masked_embedding"].cpu())
            avg_dims_list.append(out["policy_output"]["selected_dim"].cpu())

    query_embs = torch.cat(query_embs)        # [Q, 768]
    all_dims = torch.cat(avg_dims_list)       # [Q]

    # ── Retrieval ─────────────────────────────────────────────────────
    gt_indices = np.array([corpus_id_to_idx[s["positive_id"]] for s in valid])
    query_blooms = np.array([s["bloom_level"] for s in valid])
    N = len(valid)

    rankings = []
    for i in range(0, N, 256):
        sim = torch.mm(query_embs[i:i + 256], corpus_embs.t())
        topk = sim.topk(100, dim=-1).indices.numpy()
        rankings.append(topk)
    rankings = np.concatenate(rankings)   # [N, 100]

    # ── Metrics ───────────────────────────────────────────────────────
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

    metrics["avg_dims"] = float(all_dims.mean().item())

    # Bloom-stratified R@10
    for level in range(1, 7):
        mask = query_blooms == level
        if mask.sum() == 0:
            continue
        lr = rankings[mask]
        lg = gt_indices[mask]
        nl = int(mask.sum())
        hits = np.array([lg[i] in lr[i, :10] for i in range(nl)])
        metrics[f"bloom_{BLOOM_NAMES[level]}_recall@10"] = float(hits.mean())
        metrics[f"bloom_{BLOOM_NAMES[level]}_n"] = int(nl)

    return metrics


@torch.no_grad()
def evaluate_mrl_baseline(model, test_path, corpus_path, tokenizer, device) -> Dict[str, float]:
    """Evaluate MRLEncoder baseline at full 768 dims."""
    model.eval()

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            corpus.append(json.loads(line.strip()))
    corpus_id_to_idx = {p["id"]: i for i, p in enumerate(corpus)}

    corpus_embs = []
    for i in tqdm(range(0, len(corpus), 128), desc="  corpus", leave=False):
        batch = [c["text"] for c in corpus[i:i + 128]]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(enc["input_ids"], enc["attention_mask"])
        corpus_embs.append(out["full"].cpu())
    corpus_embs = torch.cat(corpus_embs)

    samples = []
    with open(test_path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    valid = [s for s in samples if s.get("positive_id", "") in corpus_id_to_idx]

    query_embs = []
    for i in tqdm(range(0, len(valid), 64), desc="  queries", leave=False):
        batch_texts = [s["query"] for s in valid[i:i + 64]]
        enc = tokenizer(batch_texts, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(enc["input_ids"], enc["attention_mask"])
        query_embs.append(out["full"].cpu())
    query_embs = torch.cat(query_embs)

    gt_indices = np.array([corpus_id_to_idx[s["positive_id"]] for s in valid])
    query_blooms = np.array([s["bloom_level"] for s in valid])
    N = len(valid)

    rankings = []
    for i in range(0, N, 256):
        sim = torch.mm(query_embs[i:i + 256], corpus_embs.t())
        topk = sim.topk(100, dim=-1).indices.numpy()
        rankings.append(topk)
    rankings = np.concatenate(rankings)

    metrics = {"avg_dims": 768.0}
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
    for level in range(1, 7):
        mask = query_blooms == level
        if mask.sum() == 0:
            continue
        lr = rankings[mask]; lg = gt_indices[mask]; nl = int(mask.sum())
        hits = np.array([lg[i] in lr[i, :10] for i in range(nl)])
        metrics[f"bloom_{BLOOM_NAMES[level]}_recall@10"] = float(hits.mean())

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bam.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--output_dir", default="results/ablations/")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])
    test_path = config["data"]["test_path"]
    corpus_path = config["data"]["corpus_path"]

    # Load BAM checkpoint
    bam_model = BloomAlignedMRL(config)
    ckpt_path = os.path.join(args.checkpoint, "checkpoint.pt")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        bam_model.load_state_dict(state["model_state_dict"], strict=False)
        print(f"  Loaded BAM from {ckpt_path}")
    bam_model.to(device).eval()

    all_results = {}

    ablations = [
        ("BAM full",          "normal"),
        ("BAM random Bloom",  "random_bloom"),
        ("BAM no FiLM",       "no_film"),
        ("BAM fixed Bloom=1", "fixed_1"),
        ("BAM fixed Bloom=6", "fixed_6"),
    ]

    for name, ablation in ablations:
        print(f"\n{'─'*60}")
        print(f"  {name}")
        print(f"{'─'*60}")
        all_results[name] = evaluate_bam(
            bam_model, test_path, corpus_path, tokenizer, device, ablation
        )

    if args.baseline:
        print(f"\n{'─'*60}")
        print("  MRL Baseline")
        print(f"{'─'*60}")
        mc = config["model"]
        bl_model = MRLEncoder(
            model_name=mc["backbone"],
            embedding_dim=mc["embedding_dim"],
            mrl_dims=mc["mrl_dims"],
        )
        bl_ckpt = os.path.join(args.baseline, "checkpoint.pt")
        if os.path.exists(bl_ckpt):
            bl_model.load_state_dict(
                torch.load(bl_ckpt, map_location=device)["model_state_dict"], strict=False
            )
        bl_model.to(device).eval()
        all_results["MRL Baseline"] = evaluate_mrl_baseline(
            bl_model, test_path, corpus_path, tokenizer, device
        )

    # Save
    out_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    # ── Print summary table ───────────────────────────────────────────
    print("\n" + "=" * 100)
    print("ABLATION STUDY RESULTS")
    print("=" * 100)
    hdr = f"{'Method':30s}{'R@1':>7s}{'R@10':>7s}{'R@50':>7s}{'NDCG@10':>9s}{'MRR':>7s}{'AvgDim':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for name, res in all_results.items():
        print(f"{name:30s}"
              f"{res.get('recall@1',0):>7.4f}"
              f"{res.get('recall@10',0):>7.4f}"
              f"{res.get('recall@50',0):>7.4f}"
              f"{res.get('ndcg@10',0):>9.4f}"
              f"{res.get('mrr',0):>7.4f}"
              f"{res.get('avg_dims',768):>8.0f}")

    print(f"\n{'Bloom-Stratified R@10':30s}")
    print("-" * 100)
    bloom_hdr = f"{'Method':30s}" + "".join(f"{BLOOM_NAMES[l]:>13s}" for l in range(1, 7))
    print(bloom_hdr)
    for name, res in all_results.items():
        row = f"{name:30s}"
        for l in range(1, 7):
            row += f"{res.get(f'bloom_{BLOOM_NAMES[l]}_recall@10', 0):>13.4f}"
        print(row)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
