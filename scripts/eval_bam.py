"""
Evaluate BAM models: v3/v4-A (prefix routing) and BAM v4 (Option B scattered mask).

Usage:
    # BAM v3/v4-A only:
    python scripts/eval_bam.py --config configs/bam.yaml \
        --checkpoint /tmp/bam-ckpts/best/ \
        --baseline /tmp/bam-ckpts/mrl_baseline_best/

    # Include BAM v4 (Option B):
    python scripts/eval_bam.py --config configs/bam.yaml \
        --checkpoint /tmp/bam-ckpts/best/ \
        --baseline /tmp/bam-ckpts/mrl_baseline_best/ \
        --checkpoint_v4 /tmp/bam-v4-ckpts/best/ \
        --config_v4 configs/bam_v4.yaml
"""
import argparse, json, sys, os
import torch
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from models.bam import BloomAlignedMRL
from models.encoder import MRLEncoder
from evaluation.evaluator import FullEvaluator, BLOOM_NAMES
from transformers import AutoTokenizer


def load_bam(config, ckpt_path, device):
    config["training"]["loss"].setdefault("bloom_frequencies", [1/6] * 6)
    model = BloomAlignedMRL(config)
    f = os.path.join(ckpt_path, "checkpoint.pt")
    if os.path.exists(f):
        ckpt = torch.load(f, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  Loaded BAM from {f}")
    model.to(device).eval()
    return model


def load_mrl(config, ckpt_path, device):
    mc = config["model"]
    model = MRLEncoder(model_name=mc["backbone"], embedding_dim=mc["embedding_dim"],
                       mrl_dims=mc["mrl_dims"])
    f = os.path.join(ckpt_path, "checkpoint.pt")
    if os.path.exists(f):
        ckpt = torch.load(f, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  Loaded MRL baseline from {f}")
    model.to(device).eval()
    return model


def compute_mask_specialization(model, test_path, tokenizer, device, n_samples=500):
    """
    Compute pairwise cosine similarity between per-Bloom average soft masks (Option B).
    Returns 6×6 similarity matrix and per-pair overlap percentages.
    """
    if not hasattr(model, "bloom_mask_head") or not model.use_mask_routing:
        return None, None

    model.eval()
    samples = []
    with open(test_path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    samples = samples[:n_samples]

    import torch.nn.functional as F
    bloom_masks = {b: [] for b in range(6)}

    with torch.no_grad():
        for i in range(0, len(samples), 64):
            batch = samples[i:i + 64]
            texts = [s["query"] for s in batch]
            enc = tokenizer(texts, padding=True, truncation=True,
                            max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            bloom_labels = torch.tensor(
                [s["bloom_level"] - 1 for s in batch], dtype=torch.long, device=device
            )
            out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                       bloom_labels=bloom_labels)
            hard_mask = (out["mask"] > 0.5).float().cpu()
            for j, bl in enumerate(bloom_labels.cpu().tolist()):
                bloom_masks[bl].append(hard_mask[j])

    # Per-Bloom average mask
    bloom_mean_masks = {}
    for b in range(6):
        if bloom_masks[b]:
            bloom_mean_masks[b] = torch.stack(bloom_masks[b]).mean(dim=0)

    if len(bloom_mean_masks) < 2:
        return None, None

    levels = sorted(bloom_mean_masks.keys())
    means = torch.stack([bloom_mean_masks[b] for b in levels])  # [L, 768]
    normed = F.normalize(means, p=2, dim=-1)
    sim_matrix = torch.mm(normed, normed.t()).numpy()

    # Overlap: % of dims shared between each pair
    overlap_matrix = np.zeros((len(levels), len(levels)))
    for i, bi in enumerate(levels):
        for j, bj in enumerate(levels):
            mi = (bloom_mean_masks[bi] > 0.5).float()
            mj = (bloom_mean_masks[bj] > 0.5).float()
            overlap = (mi * mj).sum().item()
            total = mi.sum().clamp(min=1).item()
            overlap_matrix[i, j] = overlap / total

    return sim_matrix, overlap_matrix, levels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bam.yaml")
    parser.add_argument("--checkpoint", required=True, help="BAM v3/v4-A checkpoint dir")
    parser.add_argument("--baseline", default=None, help="MRL baseline checkpoint dir")
    parser.add_argument("--mrl_continued", default=None,
                        help="MRL-continued checkpoint dir (same budget as BAM)")
    parser.add_argument("--checkpoint_v4", default=None,
                        help="BAM v4 Option B checkpoint dir")
    parser.add_argument("--config_v4", default="configs/bam_v4.yaml",
                        help="Config for BAM v4 Option B")
    parser.add_argument("--output_dir", default="results/bam_eval/")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])

    test_path = config["data"]["test_path"]
    corpus_path = config["data"]["corpus_path"]

    evaluator = FullEvaluator(config)
    all_results = {}

    # --- BAM v3/v4-A (prefix routing) ---
    print("=" * 60)
    print("BAM v3/v4-A (prefix routing)")
    print("=" * 60)
    bam_model = load_bam(config, args.checkpoint, device)
    bam_metrics = evaluator.evaluate_model(
        bam_model, test_path, corpus_path, tokenizer, device,
        compute_bootstrap=True,
    )
    all_results["BAM"] = bam_metrics

    # BAM encoder truncated (no routing) — isolates routing gain from encoder quality gain
    print("\n" + "=" * 60)
    print("BAM Encoder (no routing)")
    print("=" * 60)
    bam_encoder_model = load_mrl(config, args.checkpoint, device)
    bam_enc_metrics = evaluator.evaluate_model(
        bam_encoder_model, test_path, corpus_path, tokenizer, device,
        mrl_truncation_dims=config["model"]["mrl_dims"],
        compute_bootstrap=True,
    )
    all_results["BAM Encoder (no routing)"] = bam_enc_metrics

    # --- BAM v4 Option B (scattered mask) ---
    if args.checkpoint_v4:
        print("\n" + "=" * 60)
        print("BAM v4 Option B (scattered mask)")
        print("=" * 60)
        config_v4 = load_config(args.config_v4)
        config_v4["training"]["loss"].setdefault("bloom_frequencies", [1/6] * 6)
        bam_v4_model = load_bam(config_v4, args.checkpoint_v4, device)
        bam_v4_metrics = evaluator.evaluate_model(
            bam_v4_model, test_path, corpus_path, tokenizer, device,
            compute_bootstrap=True,
        )
        all_results["BAM v4 (Option B)"] = bam_v4_metrics

    # --- MRL Baseline ---
    if args.baseline:
        print("\n" + "=" * 60)
        print("MRL Baseline")
        print("=" * 60)
        bl_model = load_mrl(config, args.baseline, device)
        bl_metrics = evaluator.evaluate_model(
            bl_model, test_path, corpus_path, tokenizer, device,
            mrl_truncation_dims=config["model"]["mrl_dims"],
            compute_bootstrap=True,
        )
        all_results["MRL Baseline"] = bl_metrics

    if args.mrl_continued:
        print("\n" + "=" * 60)
        print("MRL-Continued Baseline")
        print("=" * 60)
        mc_model = load_mrl(config, args.mrl_continued, device)
        mc_metrics = evaluator.evaluate_model(
            mc_model, test_path, corpus_path, tokenizer, device,
            mrl_truncation_dims=config["model"]["mrl_dims"],
            compute_bootstrap=True,
        )
        all_results["MRL Continued"] = mc_metrics

    # --- Print comparison table ---
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    std_metrics = ["recall@1", "recall@5", "recall@10", "recall@50", "mrr", "ndcg@10"]
    header = f"{'Metric':45s}" + "".join(f"{n:>14s}" for n in all_results)
    print(header)
    print("-" * len(header))
    for m in std_metrics:
        row = f"{m:45s}"
        for name, res in all_results.items():
            row += f"{res.get(m, 0):>14.4f}"
        print(row)

    if "avg_active_dims" in bam_metrics:
        print(f"\n{'avg_active_dims':45s}{bam_metrics['avg_active_dims']:>14.0f}")

    # --- Bloom-stratified R@10 with routing entropy ---
    print(f"\n{'Bloom-Stratified R@10':45s}")
    print("-" * len(header))
    for level in range(1, 7):
        name = BLOOM_NAMES[level]
        key = f"bloom_{name}_recall@10"
        n_key = f"bloom_{name}_n"
        ci_lo_key = f"bloom_{name}_recall@10_ci_lo"
        ci_hi_key = f"bloom_{name}_recall@10_ci_hi"
        ent_key = f"bloom_{name}_routing_entropy"

        row = f"  {name:43s}"
        for mname, res in all_results.items():
            row += f"{res.get(key, 0):>14.4f}"
        print(row)

        if ci_lo_key in bam_metrics:
            n = bam_metrics.get(n_key, 0)
            lo = bam_metrics[ci_lo_key]
            hi = bam_metrics[ci_hi_key]
            print(f"    {'':41s}  (n={n}, 95% CI=[{lo:.3f}, {hi:.3f}])")

        # Routing entropy column (BAM only, Option A with soft routing)
        if ent_key in bam_metrics:
            print(f"    routing entropy (BAM): {bam_metrics[ent_key]:.4f}")

    # --- Per-Bloom dims ---
    print(f"\n{'Avg Active Dims per Bloom Level':45s}")
    for level in range(1, 7):
        name = BLOOM_NAMES[level]
        dim_key = f"bloom_{name}_avg_dim"
        row = f"  {name:43s}"
        for mname, res in all_results.items():
            val = res.get(dim_key, res.get("avg_active_dims", 0))
            row += f"{val:>14.0f}"
        print(row)

    # --- Truncation comparison ---
    dims_list = config["model"]["mrl_dims"]
    print(f"\n{'Truncation R@10 comparison':45s}")
    print("-" * 55)
    print(f"  {'Dims':>6}  {'MRL':>8}  {'MRL-cont':>10}  {'BAM-enc':>9}  {'BAM (routed)':>13}")
    for d in dims_list:
        key = f"mrl_d{d}_recall@10"
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else " " * 8
        print(f"  {d:>6}  "
              f"{fmt(all_results.get('MRL Baseline', {}).get(key, '-')):>8}  "
              f"{fmt(all_results.get('MRL Continued', {}).get(key, '-')):>10}  "
              f"{fmt(all_results.get('BAM Encoder (no routing)', {}).get(key, '-')):>9}")
    avg_dim = bam_metrics.get("avg_active_dims", 0)
    print(f"  {'~'+str(int(avg_dim)):>6}  {'':>8}  {'':>10}  {'':>9}  "
          f"{bam_metrics.get('recall@10', 0):>13.4f}  ← BAM routed")

    # --- Option B mask specialization ---
    if args.checkpoint_v4:
        print("\n" + "=" * 60)
        print("BAM v4 Mask Specialization")
        print("=" * 60)
        result = compute_mask_specialization(bam_v4_model, test_path, tokenizer, device)
        if result[0] is not None:
            sim_matrix, overlap_matrix, levels = result
            bloom_level_names = [BLOOM_NAMES[b + 1] for b in levels]
            print("  Pairwise cosine similarity between Bloom-level average masks:")
            print(f"  {'':14s}" + "".join(f"{n:>12s}" for n in bloom_level_names))
            for i, ni in enumerate(bloom_level_names):
                row = f"  {ni:14s}"
                for j in range(len(levels)):
                    row += f"{sim_matrix[i, j]:>12.3f}"
                print(row)

            # Save specialization matrix
            spec_path = os.path.join(args.output_dir, "mask_specialization_summary.json")
            with open(spec_path, "w") as f:
                json.dump({
                    "levels": [BLOOM_NAMES[b + 1] for b in levels],
                    "cosine_similarity": sim_matrix.tolist(),
                    "overlap_matrix": overlap_matrix.tolist(),
                }, f, indent=2)
            print(f"  Saved to {spec_path}")
        else:
            print("  (model does not use mask routing)")

    # Save all results
    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                k: {kk: (float(vv) if isinstance(vv, float) else vv)
                    for kk, vv in v.items()}
                for k, v in all_results.items()
            },
            f, indent=2,
        )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
