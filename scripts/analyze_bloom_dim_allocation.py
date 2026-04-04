"""
Post-hoc Bloom Dimension Allocation Analysis.

Tests whether BAM Option B spontaneously learns to allocate more dimensions
to higher Bloom cognitive levels, WITHOUT per-level sparsity targets baked
into the loss. If the cognitive load hypothesis holds, a model trained with
only a uniform global sparsity constraint should still produce monotonically
increasing avg active dims from Remember (b=0) to Create (b=5).

Reports:
  1. Per-Bloom mean active dims (from hard mask)
  2. Kendall's tau for correlation between Bloom level and active dims
     (tau=1.0 → perfect cognitive ordering, tau=-1.0 → reversed)
  3. p-value under the null hypothesis that ordering is random
  4. Pairwise mask cosine similarity matrix (specialization)

Usage:
    python scripts/analyze_bloom_dim_allocation.py \
        --config configs/bam_v4.yaml \
        --checkpoint /tmp/bam-v4-ckpts/best/ \
        --output_dir results/analysis/
"""

import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from models.bam import BloomAlignedMRL
from transformers import AutoTokenizer

BLOOM_NAMES = {0: "Remember", 1: "Understand", 2: "Apply",
               3: "Analyze",  4: "Evaluate",   5: "Create"}


@torch.no_grad()
def compute_per_bloom_dims(model, test_path, tokenizer, device,
                           n_samples: int = 2000):
    """
    Encode up to n_samples test queries and collect active dim counts per Bloom level.

    Uses predicted bloom labels from cache if available, else ground-truth.
    Returns:
        bloom_dims:  dict {0..5: List[float]} — active dim counts per sample per level
        bloom_masks: dict {0..5: Tensor[768]} — mean hard mask per level
    """
    samples = []
    with open(test_path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    samples = samples[:n_samples]

    # Prefer predicted bloom labels to match training (same source)
    cache_path = test_path + ".bloom_cache.json"
    bloom_cache = None
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        if len(cache) >= len(samples):
            bloom_cache = cache[:len(samples)]

    bloom_dims  = {b: [] for b in range(6)}
    bloom_masks = {b: [] for b in range(6)}

    model.eval()
    batch_size = 64
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        texts  = [s["query"] for s in batch]
        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        if bloom_cache is not None:
            bloom_labels = torch.tensor(
                bloom_cache[i:i + len(batch)], dtype=torch.long, device=device
            )
        else:
            bloom_labels = torch.tensor(
                [s["bloom_level"] - 1 for s in batch], dtype=torch.long, device=device
            )

        out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                   bloom_labels=bloom_labels)
        hard_mask = (out["mask"] > 0.5).float().cpu()  # [B, 768]
        active    = hard_mask.sum(dim=-1).tolist()      # [B]

        for j, bl in enumerate(bloom_labels.cpu().tolist()):
            bloom_dims[bl].append(active[j])
            bloom_masks[bl].append(hard_mask[j])

    # Aggregate per-level mean masks
    mean_masks = {}
    for b in range(6):
        if bloom_masks[b]:
            mean_masks[b] = torch.stack(bloom_masks[b]).mean(dim=0)

    return bloom_dims, mean_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/bam_v4.yaml")
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--output_dir",  default="results/analysis/")
    parser.add_argument("--n_samples",   type=int, default=2000)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])

    config["training"]["loss"].setdefault("bloom_frequencies", [1/6] * 6)
    model = BloomAlignedMRL(config)
    ckpt = os.path.join(args.checkpoint, "checkpoint.pt")
    if os.path.exists(ckpt):
        result = model.load_state_dict(
            torch.load(ckpt, map_location=device)["model_state_dict"], strict=False
        )
        if result.missing_keys:
            print(f"WARNING missing keys: {result.missing_keys[:5]}")
    model.to(device)

    test_path = config["data"]["test_path"]
    print(f"Analyzing dim allocation on {args.n_samples} samples from {test_path}...")
    bloom_dims, mean_masks = compute_per_bloom_dims(
        model, test_path, tokenizer, device, n_samples=args.n_samples
    )

    # ── 1. Per-Bloom mean active dims ──────────────────────────────────────
    print("\n=== Per-Bloom Mean Active Dims (768 total) ===")
    print(f"{'Level':>12s}  {'N':>6s}  {'Mean dims':>10s}  {'Std':>7s}  {'Fraction':>9s}")
    print("-" * 50)
    means = []
    levels_present = []
    for b in range(6):
        if not bloom_dims[b]:
            continue
        d = np.array(bloom_dims[b])
        m = d.mean()
        means.append(m)
        levels_present.append(b)
        print(f"  {BLOOM_NAMES[b]:10s}  {len(d):>6d}  {m:>10.1f}  "
              f"{d.std():>7.1f}  {m/768:>9.3f}")

    # ── 2. Kendall's tau — cognitive ordering test ─────────────────────────
    if len(means) >= 3:
        tau, p_value = kendalltau(levels_present, means)
        print(f"\nKendall's tau (level vs active dims): tau={tau:+.3f}, p={p_value:.4f}")
        if tau > 0 and p_value < 0.05:
            print("  → SUPPORTS cognitive load hypothesis: higher Bloom levels use "
                  "more dims (p<0.05, without per-level training targets).")
        elif tau > 0:
            print(f"  → Positive trend (tau={tau:+.3f}) but not significant (p={p_value:.3f}). "
                  "Collect more samples or interpret cautiously.")
        else:
            print(f"  → Does NOT support cognitive load hypothesis (tau={tau:+.3f}).")
    else:
        print("\nToo few Bloom levels present for Kendall's tau.")
        tau, p_value = 0.0, 1.0

    # ── 3. Pairwise mask cosine similarity ─────────────────────────────────
    print("\n=== Pairwise Mask Cosine Similarity (lower = more specialization) ===")
    present = sorted(mean_masks.keys())
    masks_t = torch.stack([mean_masks[b] for b in present])
    normed  = F.normalize(masks_t, p=2, dim=-1)
    sim     = torch.mm(normed, normed.t()).numpy()

    header = f"{'':12s}" + "".join(f"{BLOOM_NAMES[b]:>12s}" for b in present)
    print(header)
    for i, bi in enumerate(present):
        row = f"  {BLOOM_NAMES[bi]:10s}"
        for j in range(len(present)):
            row += f"{sim[i, j]:>12.3f}"
        print(row)

    off_diag = sim[np.triu_indices(len(present), k=1)]
    print(f"\n  Mean off-diagonal similarity: {off_diag.mean():.3f} "
          f"(0 = fully specialized, 1 = identical masks)")

    # ── 4. Save results ────────────────────────────────────────────────────
    results = {
        "per_bloom_mean_dims": {
            BLOOM_NAMES[b]: float(np.mean(bloom_dims[b])) if bloom_dims[b] else None
            for b in range(6)
        },
        "per_bloom_n": {BLOOM_NAMES[b]: len(bloom_dims[b]) for b in range(6)},
        "kendall_tau": float(tau),
        "kendall_p": float(p_value),
        "supports_cognitive_hypothesis": bool(tau > 0 and p_value < 0.05),
        "mean_mask_similarity": float(off_diag.mean()),
        "pairwise_similarity": {
            f"{BLOOM_NAMES[present[i]]}_vs_{BLOOM_NAMES[present[j]]}": float(sim[i, j])
            for i in range(len(present)) for j in range(i + 1, len(present))
        },
    }
    out_path = os.path.join(args.output_dir, "bloom_dim_allocation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
