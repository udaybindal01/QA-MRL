"""
Find best BAM epoch using Bloom Stratified Recall (BSR).

Plain recall@10 is dominated by Remember (64% of queries) and ignores
dimension efficiency. BSR is a Pareto tradeoff: quality × efficiency.

  quality    = Σ_b [ w_b * R@10_b ]
               class weights w_b = 1/sqrt(freq_b), normalised to sum=1
               → each Bloom level contributes equally regardless of frequency

  efficiency = Σ_b [ cognitive(b) * (1 − dim_b/768) ] / Σ_b cognitive(b)
               cognitive(b) = 1 − b/6  (Remember=1.0 → Create=0.167)
               → rewards compressing Remember; doesn't penalise Create staying large
               → normalised to [0, 1]

  BSR = quality × (1 + α × efficiency)       (default α = 0.5)

  Pareto interpretation:
    − If all dims = 768 (no routing): efficiency = 0, BSR = quality.
    − If routing is perfect (cognitive ordering + compression): BSR > quality.
    − Bad quality at rare Bloom levels (Understand, Evaluate) pulls quality down
      even if Remember is high, so BSR correctly penalises that.

Usage:
    python scripts/find_best_epoch_bsr.py \\
        --config       configs/bam.yaml \\
        --checkpoint_dir /tmp/bam-ckpts/ \\
        --output_dir   ./results/best_epochs/bam_bsr/

    # Tune efficiency weight:
    python scripts/find_best_epoch_bsr.py ... --alpha 0.3
"""

import argparse, json, os, sys, shutil, math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from models.bam import BloomAlignedMRL
from evaluation.evaluator import FullEvaluator
from transformers import AutoTokenizer

BLOOM_NAMES = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
EMBEDDING_DIM = 768.0  # overridden at runtime from config["model"]["embedding_dim"]


def compute_bloom_frequencies(train_path: str):
    """Count Bloom label distribution in training data. Returns list of 6 floats."""
    counts = [0] * 6
    total = 0
    with open(train_path) as f:
        for line in f:
            import json as _json
            item = _json.loads(line.strip())
            # bloom_level is 1-indexed in JSONL, 0-indexed internally
            bl = item.get("bloom_level", 1)
            idx = max(0, min(int(bl) - 1, 5))
            counts[idx] += 1
            total += 1
    if total == 0:
        return [1 / 6] * 6
    return [c / total for c in counts]


def class_weights(frequencies):
    """1/sqrt(freq), normalised to sum=1. [6] list."""
    w = [1.0 / math.sqrt(max(f, 1e-6)) for f in frequencies]
    s = sum(w)
    return [x / s for x in w]


def cognitive_weights():
    """1 - b/6 for b in 0..5. Remember=1.0, Create=0.167."""
    return [1.0 - b / 6.0 for b in range(6)]


def compute_bsr(metrics: dict, cw: list, alpha: float):
    """
    Compute BSR from evaluator metrics dict.

    Levels with 0 test queries are skipped (evaluator omits their keys).
    Weights are renormalised over present levels so BSR stays in the same range.

    Returns (bsr, quality, efficiency, per_bloom_r10, per_bloom_dim).
    per_bloom_r10/dim are full 6-element lists with None for absent levels.
    Returns None only if NO level metrics exist at all.
    """
    per_bloom_r10 = []
    per_bloom_dim = []
    present = []   # indices of levels that have test queries

    for b, name in enumerate(BLOOM_NAMES):
        r10_key = f"bloom_{name}_recall@10"
        dim_key = f"bloom_{name}_avg_dim"
        if r10_key in metrics and dim_key in metrics:
            per_bloom_r10.append(metrics[r10_key])
            per_bloom_dim.append(metrics[dim_key])
            present.append(b)
        else:
            per_bloom_r10.append(None)
            per_bloom_dim.append(None)

    if not present:
        return None

    cog = cognitive_weights()

    # Renormalise class weights and cognitive weights over present levels only
    cw_sum  = sum(cw[b]  for b in present)
    cog_sum = sum(cog[b] for b in present)

    quality = sum(
        (cw[b] / cw_sum) * per_bloom_r10[b] for b in present
    )
    efficiency = sum(
        cog[b] * (1.0 - per_bloom_dim[b] / EMBEDDING_DIM) for b in present
    ) / cog_sum

    bsr = quality * (1.0 + alpha * efficiency)

    return bsr, quality, efficiency, per_bloom_r10, per_bloom_dim


def load_bam(config, ckpt_dir, device):
    model = BloomAlignedMRL(config)
    f = os.path.join(ckpt_dir, "checkpoint.pt")
    if os.path.exists(f):
        ckpt = torch.load(f, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def evaluate_epoch(model, test_path, corpus_path, tokenizer, device):
    """
    Evaluate one epoch checkpoint.

    Corpus: full embeddings (all dims active) — single pre-built index.
    Queries: masked embeddings (bloom-routed) — same as training forward pass.
    Similarity: masked_query · full_corpus  (zero dims naturally ignored).

    This matches the training objective and the realistic deployment scenario.
    Returns a metrics dict with bloom_{name}_recall@10, bloom_{name}_avg_dim,
    recall@{1,5,10,50}, ndcg@10, mrr.
    """
    import json as _json

    # Load corpus
    corpus = []
    with open(corpus_path) as f:
        for line in f:
            corpus.append(_json.loads(line.strip()))
    corpus_id_to_idx = {p["id"]: i for i, p in enumerate(corpus)}

    # Encode corpus at full dims
    corpus_embs = []
    for i in tqdm(range(0, len(corpus), 128), desc="  corpus", leave=False):
        batch = [c["text"] for c in corpus[i:i+128]]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
        corpus_embs.append(out["full_embedding"].float().cpu())
    corpus_embs = torch.cat(corpus_embs)  # [C, D]

    # Load test queries
    samples = []
    with open(test_path) as f:
        for line in f:
            samples.append(_json.loads(line.strip()))
    valid = [s for s in samples if s.get("positive_id", "") in corpus_id_to_idx]

    # Encode queries with bloom-adaptive mask
    query_embs, query_blooms, query_active_dims = [], [], []
    for i in tqdm(range(0, len(valid), 64), desc="  queries", leave=False):
        batch = valid[i:i+64]
        enc = tokenizer([q["query"] for q in batch], padding=True,
                        truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        bloom_labels = torch.tensor(
            [q["bloom_level"] - 1 for q in batch], dtype=torch.long, device=device
        )
        out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                   bloom_labels=bloom_labels)
        query_embs.append(out["masked_embedding"].float().cpu())
        query_blooms.extend([q["bloom_level"] - 1 for q in batch])
        if "active_dims" in out:
            query_active_dims.append(out["active_dims"].float().cpu())

    query_embs = torch.cat(query_embs)       # [N, D]
    query_blooms = np.array(query_blooms)    # [N]
    gt_indices = np.array([corpus_id_to_idx[s["positive_id"]] for s in valid])

    # Retrieval: masked query · full corpus
    KS = [1, 5, 10, 50]
    max_k = max(KS)
    rankings = np.zeros((len(valid), max_k), dtype=np.int64)
    chunk = 256
    corpus_embs_dev = corpus_embs.to(device)
    for i in range(0, len(valid), chunk):
        q = query_embs[i:i+chunk].to(device)
        sim = torch.mm(q, corpus_embs_dev.t())
        topk = sim.topk(max_k, dim=-1).indices.cpu().numpy()
        rankings[i:i+chunk] = topk
    del corpus_embs_dev

    # Standard metrics
    metrics = {}
    for k in KS:
        hits = (rankings[:, :k] == gt_indices[:, None]).any(axis=1)
        metrics[f"recall@{k}"] = float(hits.mean())

    # NDCG@10
    ndcg_scores = []
    for i in range(len(valid)):
        rank_of_pos = np.where(rankings[i] == gt_indices[i])[0]
        if len(rank_of_pos) == 0 or rank_of_pos[0] >= 10:
            ndcg_scores.append(0.0)
        else:
            ndcg_scores.append(1.0 / math.log2(rank_of_pos[0] + 2))
    metrics["ndcg@10"] = float(np.mean(ndcg_scores))

    # MRR
    mrr_scores = []
    for i in range(len(valid)):
        rank_of_pos = np.where(rankings[i] == gt_indices[i])[0]
        mrr_scores.append(1.0 / (rank_of_pos[0] + 1) if len(rank_of_pos) > 0 else 0.0)
    metrics["mrr"] = float(np.mean(mrr_scores))

    # Per-bloom R@10 and avg_dim
    active_dims_tensor = torch.cat(query_active_dims) if query_active_dims else None
    for b, name in enumerate(BLOOM_NAMES):
        idx = np.where(query_blooms == b)[0]
        if len(idx) == 0:
            continue
        hits = (rankings[idx, :10] == gt_indices[idx, None]).any(axis=1)
        metrics[f"bloom_{name}_recall@10"] = float(hits.mean())
        if active_dims_tensor is not None:
            metrics[f"bloom_{name}_avg_dim"] = float(active_dims_tensor[idx].mean().item())
        else:
            # Fallback: count non-zero dims in the masked embeddings
            metrics[f"bloom_{name}_avg_dim"] = float(
                (query_embs[idx] != 0).float().sum(dim=-1).mean().item()
            )

    # Overall avg_dim
    if active_dims_tensor is not None:
        metrics["avg_active_dims"] = float(active_dims_tensor.mean().item())

    # Print summary
    sparse_ratio = 1.0 - metrics.get("avg_active_dims", EMBEDDING_DIM) / EMBEDDING_DIM
    print(f"\n  === Results (N={len(valid)}) ===")
    for k in KS:
        print(f"  R@{k}:    {metrics[f'recall@{k}']:.4f}")
    print(f"  MRR:    {metrics['mrr']:.4f}")
    print(f"  NDCG@10:{metrics['ndcg@10']:.4f}")
    if "avg_active_dims" in metrics:
        print(f"  Active dims: {metrics['avg_active_dims']:.0f} / {int(EMBEDDING_DIM)}"
              f"  (sparse_ratio={sparse_ratio:.2f})")
    print(f"\n  Bloom-Stratified R@10:")
    for b, name in enumerate(BLOOM_NAMES):
        r10 = metrics.get(f"bloom_{name}_recall@10")
        dim = metrics.get(f"bloom_{name}_avg_dim")
        n   = int((query_blooms == b).sum())
        if r10 is not None:
            print(f"    {name:12s} (n={n:4d}): R@10={r10:.4f}  dim={dim:.0f}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          required=True)
    parser.add_argument("--checkpoint_dir",  required=True)
    parser.add_argument("--output_dir",      default=None)
    parser.add_argument("--alpha",           type=float, default=0.5,
                        help="Efficiency weight in BSR = quality × (1 + α × efficiency). "
                             "0 = quality only (class-weighted R@10). "
                             "Higher = more reward for dimension compression.")
    parser.add_argument("--skip_warmup_epochs", type=int, default=None,
                        help="Skip first N epochs (efficiency=0 during warmup). "
                             "Defaults to encoder_warmup_epochs from config.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Override global EMBEDDING_DIM from config so efficiency formula scales correctly
    global EMBEDDING_DIM
    EMBEDDING_DIM = float(config["model"].get("embedding_dim", 768))
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])
    evaluator = FullEvaluator(config)

    test_path   = config["data"]["test_path"]
    corpus_path = config["data"]["corpus_path"]
    train_path  = config["data"]["train_path"]
    out_dir     = args.output_dir or args.checkpoint_dir
    os.makedirs(out_dir, exist_ok=True)

    # Bloom class weights from training data
    freqs = compute_bloom_frequencies(train_path)
    cw    = class_weights(freqs)
    cog   = cognitive_weights()

    print("Bloom class weights (1/√freq, normalised):")
    for b, name in enumerate(BLOOM_NAMES):
        print(f"  {name:12s}: freq={freqs[b]:.3f}  class_weight={cw[b]:.3f}  "
              f"cognitive={cog[b]:.3f}")
    print(f"\nBSR = quality × (1 + {args.alpha} × efficiency)")
    print(f"  quality    = Σ_b [ class_weight_b × R@10_b ]")
    print(f"  efficiency = Σ_b [ cognitive_b × (1 − dim_b/{int(EMBEDDING_DIM)}) ] / Σ cognitive_b\n")

    # Collect epoch checkpoints
    epoch_dirs = []
    for name in sorted(os.listdir(args.checkpoint_dir)):
        if name.startswith("epoch_"):
            d = os.path.join(args.checkpoint_dir, name)
            if os.path.exists(os.path.join(d, "checkpoint.pt")):
                try:
                    epoch_dirs.append((int(name.split("_")[1]), name, d))
                except ValueError:
                    pass
    final_dir = os.path.join(args.checkpoint_dir, "final")
    if os.path.exists(os.path.join(final_dir, "checkpoint.pt")):
        epoch_dirs.append((9999, "final", final_dir))
    epoch_dirs.sort(key=lambda x: x[0])

    if not epoch_dirs:
        print(f"ERROR: No epoch_N/ checkpoints in {args.checkpoint_dir}")
        sys.exit(1)

    warmup_cutoff = args.skip_warmup_epochs
    if warmup_cutoff is None:
        warmup_cutoff = config["training"]["loss"].get("encoder_warmup_epochs", 0)

    print(f"Found {len(epoch_dirs)} checkpoints. Warmup cutoff: {warmup_cutoff} epochs.")
    print("=" * 90)

    # Header
    hdr = (f"{'Epoch':10s} {'BSR':>7s} {'Quality':>8s} {'Efficiency':>11s} "
           + "  ".join(f"{n[:4]:>6s}" for n in BLOOM_NAMES)
           + f"  {'AvgDim':>7s}")
    print(hdr)
    print(f"{'':10s} {'':>7s} {'(cls-wtd)':>8s} {'(cog-wtd)':>11s} "
          + "  ".join(f"{'R@10':>6s}" for _ in BLOOM_NAMES)
          + f"  {'':>7s}")
    print("-" * len(hdr))

    results = {}
    best_bsr        = -1.0
    best_epoch_name = None
    best_epoch_dir  = None

    for epoch_num, name, ckpt_dir in epoch_dirs:
        model = load_bam(config, ckpt_dir, device)
        metrics = evaluator.evaluate_model(
            model, test_path, corpus_path, tokenizer, device,
            compute_bootstrap=False,
        )
        results[name] = metrics

        is_warmup = (epoch_num != 9999 and epoch_num < warmup_cutoff)
        bsr_out   = compute_bsr(metrics, cw, args.alpha)

        if bsr_out is None:
            print(f"{name:10s}  [missing per-bloom metrics — skipping]")
        else:
            bsr, quality, efficiency, r10s, dims = bsr_out
            present_dims = [d for d in dims if d is not None]
            avg_dim = sum(present_dims) / len(present_dims) if present_dims else 0
            warmup_tag = " [W]" if is_warmup else "    "
            r10_str = "  ".join(f"{r:>6.4f}" if r is not None else f"{'—':>6s}" for r in r10s)
            print(f"{name:10s} {bsr:>7.4f} {quality:>8.4f} {efficiency:>11.4f}  "
                  f"{r10_str}  {avg_dim:>7.0f}{warmup_tag}")

            results[name]["bsr"]        = bsr
            results[name]["bsr_quality"]     = quality
            results[name]["bsr_efficiency"]  = efficiency

            if not is_warmup and bsr > best_bsr:
                best_bsr        = bsr
                best_epoch_name = name
                best_epoch_dir  = ckpt_dir

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("-" * len(hdr))
    print("  [W] = warmup epoch, not eligible for best-epoch selection")

    if best_epoch_dir is None:
        print("\nERROR: No eligible post-warmup epochs found.")
        sys.exit(1)

    print(f"\nBest epoch: {best_epoch_name}  (BSR = {best_bsr:.4f})")

    # Print per-bloom breakdown for best epoch
    best_metrics = results[best_epoch_name]
    bsr_out = compute_bsr(best_metrics, cw, args.alpha)
    if bsr_out:
        _, quality, efficiency, r10s, dims = bsr_out
        print(f"  Quality (class-weighted R@10): {quality:.4f}")
        print(f"  Efficiency (cognitive-weighted compression): {efficiency:.4f}")
        print(f"  Per-Bloom breakdown:")
        for b, bloom_name in enumerate(BLOOM_NAMES):
            if r10s[b] is None:
                print(f"    {bloom_name:12s}: (no test queries — skipped)")
            else:
                print(f"    {bloom_name:12s}: R@10={r10s[b]:.4f}  dim={dims[b]:.0f}  "
                      f"class_w={cw[b]:.3f}  cog={cog[b]:.3f}")

    # Copy best checkpoint
    best_dest = os.path.join(args.checkpoint_dir, "best_bsr")
    if os.path.exists(best_dest):
        shutil.rmtree(best_dest)
    shutil.copytree(best_epoch_dir, best_dest)
    print(f"\nCopied best checkpoint → {best_dest}/")

    # Save results
    results_path = os.path.join(out_dir, "epoch_results_bsr.json")
    with open(results_path, "w") as f:
        json.dump({
            "metric": "bsr",
            "alpha": args.alpha,
            "bloom_frequencies": {BLOOM_NAMES[b]: freqs[b] for b in range(6)},
            "class_weights":     {BLOOM_NAMES[b]: cw[b]    for b in range(6)},
            "best_epoch": best_epoch_name,
            "best_bsr":   best_bsr,
            "epochs": {
                n: {k: float(v) for k, v in m.items() if isinstance(v, (int, float))}
                for n, m in results.items()
            }
        }, f, indent=2)
    print(f"Results saved → {results_path}")


if __name__ == "__main__":
    main()
