"""
Ablation Study for BAM.

Original ablations (BAM v3/v4-A):
  1. BAM full          — Bloom routing with real labels
  2. BAM random Bloom  — Random Bloom labels (tests taxonomy vs random signal)
  3. BAM fixed Bloom=1 — All queries forced to Remember (minimum dims)
  4. BAM fixed Bloom=6 — All queries forced to Create (maximum dims)
  5. BAM no routing    — Router forced to 768 dims (isolates fine-tuning vs routing)
  6. MRL Baseline      — Full 768 dims, no routing

New ablations (v4+):
  7. mask_vs_truncation   — BAM v4 Option B vs BAM v4-A at same avg dims
  8. soft_vs_hard_routing — soft (softmax @ all_dims) vs hard (argmax) routing
  9. two_stage_vs_joint   — pass --checkpoint_stage1 (joint) and compare
 10. (pcgrad_vs_standard requires re-training; done via separate checkpoints)

Usage:
    python scripts/run_ablations.py \
        --config configs/bam.yaml \
        --checkpoint /tmp/bam-ckpts/best/ \
        --baseline /tmp/mrl-ckpts/best/ \
        --output_dir results/ablations/ \
        [--checkpoint_v4 /tmp/bam-v4-ckpts/best/] \
        [--config_v4 configs/bam_v4.yaml] \
        [--checkpoint_joint /tmp/bam-joint-ckpts/best/]
"""

import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from models.bam import BloomAlignedMRL
from models.encoder import MRLEncoder
from transformers import AutoTokenizer

BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply",
               4: "Analyze", 5: "Evaluate", 6: "Create"}


@torch.no_grad()
def evaluate_bam(model, test_path, corpus_path, tokenizer, device,
                 ablation: str = "normal"):
    """
    Evaluate BAM under ablation conditions.

    ablation:
      "normal"        — real Bloom labels from test data
      "random_bloom"  — random Bloom labels 0-5
      "fixed_1"       — all queries set to Bloom 0 (Remember = min dims)
      "fixed_6"       — all queries set to Bloom 5 (Create = max dims)
      "no_routing"    — router forced to 768 dims (isolates fine-tuning)
      "soft_routing"  — bloom_probs = softmax(uniform) @ all_dims (soft routing)
    """
    model.eval()

    # For no_routing: temporarily force all levels to max dims
    # Works for Option A (BloomDimRouter). For Option B, mask head can't be overridden
    # this way — use BAM Encoder (no routing) baseline instead.
    orig_bloom_emb = None
    if ablation == "no_routing" and not model.use_mask_routing:
        # Set all bloom embeddings to map to ~768 dims (sigmoid(10) ≈ 1.0)
        orig_bloom_emb = model.bloom_router.bloom_emb.weight.data.clone()
        orig_dim_head = [p.data.clone() for p in model.bloom_router.dim_head.parameters()]
        # Override: set output bias to large positive value → sigmoid ≈ 1 → max dim
        for p in model.bloom_router.dim_head.parameters():
            p.data.zero_()
        model.bloom_router.dim_head[-1].bias.data.fill_(10.0)

    try:
        # Encode corpus
        corpus = []
        with open(corpus_path) as f:
            for line in f:
                corpus.append(json.loads(line.strip()))
        corpus_id_to_idx = {p["id"]: i for i, p in enumerate(corpus)}

        corpus_full_embs = []
        for i in tqdm(range(0, len(corpus), 128), desc="  corpus", leave=False):
            batch = [c["text"] for c in corpus[i:i + 128]]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=256, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
            corpus_full_embs.append(out["full_embedding"].cpu())
        corpus_full_embs = torch.cat(corpus_full_embs)   # [C, 768]

        samples = []
        with open(test_path) as f:
            for line in f:
                samples.append(json.loads(line.strip()))
        valid = [s for s in samples if s.get("positive_id", "") in corpus_id_to_idx]

        query_embs_list, query_dims_list, query_active_list = [], [], []
        for i in tqdm(range(0, len(valid), 64), desc="  queries", leave=False):
            batch_samples = valid[i:i + 64]
            batch_texts = [s["query"] for s in batch_samples]
            enc = tokenizer(batch_texts, padding=True, truncation=True,
                            max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            B = len(batch_texts)

            if ablation == "random_bloom":
                bloom_labels = torch.randint(0, 6, (B,), device=device)
            elif ablation == "fixed_1":
                bloom_labels = torch.zeros(B, dtype=torch.long, device=device)
            elif ablation == "fixed_6":
                bloom_labels = torch.full((B,), 5, dtype=torch.long, device=device)
            else:  # normal, no_routing, soft_routing
                bloom_labels = torch.tensor(
                    [s["bloom_level"] - 1 for s in batch_samples],
                    dtype=torch.long, device=device,
                )

            bloom_probs = None
            if ablation == "soft_routing" and not model.use_mask_routing:
                # Use uniform softmax (equal weights) to demonstrate soft routing path
                bloom_probs = F.softmax(torch.ones(B, 6, device=device), dim=-1)

            out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                       bloom_labels=bloom_labels,
                                       bloom_probs=bloom_probs)
            query_embs_list.append(out["full_embedding"].cpu())
            if "discrete_dim" in out:
                query_dims_list.append(out["discrete_dim"].detach().cpu())
            if "active_dims" in out:
                query_active_list.append(out["active_dims"].detach().cpu())

        # For Option B, use masked_embedding directly
        use_prefix_slicing = bool(query_dims_list) and not model.use_mask_routing
        if use_prefix_slicing:
            query_full_embs = torch.cat(query_embs_list)
            query_dims = torch.cat(query_dims_list).round().long()
        else:
            # Use masked embedding for Option B
            query_masked = []
            for i in tqdm(range(0, len(valid), 64), desc="  re-encode masked", leave=False):
                batch_samples = valid[i:i + 64]
                batch_texts = [s["query"] for s in batch_samples]
                enc = tokenizer(batch_texts, padding=True, truncation=True,
                                max_length=128, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                B = len(batch_texts)
                bloom_labels = torch.tensor(
                    [s["bloom_level"] - 1 for s in batch_samples],
                    dtype=torch.long, device=device,
                )
                out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                           bloom_labels=bloom_labels)
                query_masked.append(out["masked_embedding"].cpu())
            query_full_embs = torch.cat(query_masked)
            query_dims = None

        N = len(valid)
        rankings = np.zeros((N, 100), dtype=np.int64)

        if use_prefix_slicing:
            for k_val in query_dims.unique():
                k = int(k_val.item())
                k = max(1, min(k, 768))
                group_idx = (query_dims == k_val).nonzero(as_tuple=True)[0]
                c_k = F.normalize(corpus_full_embs[:, :k], p=2, dim=-1).to(device)
                for i in range(0, len(group_idx), 256):
                    chunk = group_idx[i:i + 256]
                    q_k = F.normalize(query_full_embs[chunk, :k], p=2, dim=-1).to(device)
                    sim = torch.mm(q_k, c_k.t())
                    topk = sim.topk(100, dim=-1).indices.cpu().numpy()
                    rankings[chunk.numpy()] = topk
                del c_k
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        else:
            for i in range(0, N, 256):
                q_chunk = query_full_embs[i:i + 256].to(device)
                sim = torch.mm(q_chunk, corpus_full_embs.to(device).t())
                topk = sim.topk(100, dim=-1).indices.cpu().numpy()
                rankings[i:i + 256] = topk
                del sim
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        # Metrics
        gt_indices = np.array([corpus_id_to_idx[s["positive_id"]] for s in valid])
        query_blooms = np.array([s["bloom_level"] for s in valid])
        metrics = {}

        for k in [1, 5, 10, 50]:
            hits = np.array([gt_indices[i] in rankings[i, :k] for i in range(N)])
            metrics[f"recall@{k}"] = float(hits.mean())

        mrrs = [
            1.0 / (np.where(rankings[i] == gt_indices[i])[0][0] + 1)
            if len(np.where(rankings[i] == gt_indices[i])[0]) > 0 else 0.0
            for i in range(N)
        ]
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

        # Avg dims
        if query_dims_list:
            all_dims = torch.cat(query_dims_list)
            metrics["avg_dims"] = float(all_dims.float().mean().item())
        elif query_active_list:
            all_active = torch.cat(query_active_list)
            metrics["avg_dims"] = float(all_active.float().mean().item())
        else:
            metrics["avg_dims"] = 768.0

        for level in range(1, 7):
            mask = query_blooms == level
            if mask.sum() == 0:
                continue
            lr = rankings[mask]
            lg = gt_indices[mask]
            nl = int(mask.sum())
            hits = np.array([lg[i] in lr[i, :10] for i in range(nl)])
            metrics[f"bloom_{BLOOM_NAMES[level]}_recall@10"] = float(hits.mean())
            metrics[f"bloom_{BLOOM_NAMES[level]}_n"] = nl
            if query_dims_list:
                metrics[f"bloom_{BLOOM_NAMES[level]}_avg_dim"] = float(
                    torch.cat(query_dims_list)[mask].float().mean().item()
                )

    finally:
        # Restore original router params if overridden
        if orig_bloom_emb is not None:
            model.bloom_router.bloom_emb.weight.data = orig_bloom_emb
            for p, saved in zip(model.bloom_router.dim_head.parameters(), orig_dim_head):
                p.data = saved

    return metrics


@torch.no_grad()
def evaluate_mrl_baseline(model, test_path, corpus_path, tokenizer, device):
    """MRL baseline at full 768 dims."""
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
        rankings.append(sim.topk(100, dim=-1).indices.numpy())
    rankings = np.concatenate(rankings)

    metrics = {"avg_dims": 768.0}
    for k in [1, 5, 10, 50]:
        hits = np.array([gt_indices[i] in rankings[i, :k] for i in range(N)])
        metrics[f"recall@{k}"] = float(hits.mean())

    mrrs = [
        1.0 / (np.where(rankings[i] == gt_indices[i])[0][0] + 1)
        if len(np.where(rankings[i] == gt_indices[i])[0]) > 0 else 0.0
        for i in range(N)
    ]
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
    parser.add_argument("--checkpoint", required=True,
                        help="BAM v3/v4-A checkpoint dir")
    parser.add_argument("--baseline", default=None,
                        help="MRL checkpoint dir")
    parser.add_argument("--checkpoint_v4", default=None,
                        help="BAM v4 Option B checkpoint dir (for mask_vs_truncation ablation)")
    parser.add_argument("--config_v4", default="configs/bam_v4.yaml")
    parser.add_argument("--checkpoint_joint", default=None,
                        help="BAM trained jointly (no stage 2) for two_stage_vs_joint ablation")
    parser.add_argument("--output_dir", default="results/ablations/")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])
    test_path = config["data"]["test_path"]
    corpus_path = config["data"]["corpus_path"]

    config["training"]["loss"]["bloom_frequencies"] = [1/6] * 6
    bam_model = BloomAlignedMRL(config)
    ckpt_path = os.path.join(args.checkpoint, "checkpoint.pt")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        bam_model.load_state_dict(state["model_state_dict"], strict=False)
        print(f"Loaded BAM from {ckpt_path}")
    bam_model.to(device).eval()

    all_results = {}

    # --- Original ablations (Option A) ---
    original_ablations = [
        ("BAM full",           "normal"),
        ("BAM random Bloom",   "random_bloom"),
        ("BAM fixed Bloom=1",  "fixed_1"),
        ("BAM fixed Bloom=6",  "fixed_6"),
        ("BAM no routing",     "no_routing"),
        ("BAM soft routing",   "soft_routing"),
    ]
    for name, ablation in original_ablations:
        print(f"\n{'─' * 60}\n  {name}\n{'─' * 60}")
        all_results[name] = evaluate_bam(
            bam_model, test_path, corpus_path, tokenizer, device, ablation
        )

    # --- New ablation: mask_vs_truncation (Option B vs Option A at same avg dims) ---
    if args.checkpoint_v4:
        print(f"\n{'─' * 60}\n  BAM v4 Option B (mask_vs_truncation)\n{'─' * 60}")
        config_v4 = load_config(args.config_v4)
        config_v4["training"]["loss"]["bloom_frequencies"] = [1/6] * 6
        bam_v4 = BloomAlignedMRL(config_v4)
        f = os.path.join(args.checkpoint_v4, "checkpoint.pt")
        if os.path.exists(f):
            bam_v4.load_state_dict(
                torch.load(f, map_location=device)["model_state_dict"], strict=False
            )
        bam_v4.to(device).eval()
        all_results["BAM v4 Option B"] = evaluate_bam(
            bam_v4, test_path, corpus_path, tokenizer, device, "normal"
        )

    # --- New ablation: two_stage_vs_joint ---
    if args.checkpoint_joint:
        print(f"\n{'─' * 60}\n  BAM joint (two_stage_vs_joint)\n{'─' * 60}")
        bam_joint = BloomAlignedMRL(config)
        f = os.path.join(args.checkpoint_joint, "checkpoint.pt")
        if os.path.exists(f):
            bam_joint.load_state_dict(
                torch.load(f, map_location=device)["model_state_dict"], strict=False
            )
        bam_joint.to(device).eval()
        all_results["BAM joint (no stage 2)"] = evaluate_bam(
            bam_joint, test_path, corpus_path, tokenizer, device, "normal"
        )

    # --- MRL Baseline ---
    if args.baseline:
        print(f"\n{'─' * 60}\n  MRL Baseline (768 dims)\n{'─' * 60}")
        mc = config["model"]
        bl_model = MRLEncoder(model_name=mc["backbone"], embedding_dim=mc["embedding_dim"],
                              mrl_dims=mc["mrl_dims"])
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

    # Summary table
    print("\n" + "=" * 100)
    print("ABLATION STUDY RESULTS")
    print("=" * 100)
    hdr = f"{'Method':35s}{'R@1':>7s}{'R@10':>7s}{'R@50':>7s}{'NDCG@10':>9s}{'MRR':>7s}{'AvgDim':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for name, res in all_results.items():
        print(f"{name:35s}"
              f"{res.get('recall@1', 0):>7.4f}"
              f"{res.get('recall@10', 0):>7.4f}"
              f"{res.get('recall@50', 0):>7.4f}"
              f"{res.get('ndcg@10', 0):>9.4f}"
              f"{res.get('mrr', 0):>7.4f}"
              f"{res.get('avg_dims', 768):>8.0f}")

    print(f"\n{'Bloom-Stratified R@10':35s}")
    print("-" * 100)
    bloom_hdr = f"{'Method':35s}" + "".join(f"{BLOOM_NAMES[l]:>13s}" for l in range(1, 7))
    print(bloom_hdr)
    for name, res in all_results.items():
        row = f"{name:35s}"
        for l in range(1, 7):
            row += f"{res.get(f'bloom_{BLOOM_NAMES[l]}_recall@10', 0):>13.4f}"
        print(row)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
