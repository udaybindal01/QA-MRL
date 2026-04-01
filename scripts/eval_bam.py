"""
Evaluate BAM (Bloom-Aligned Matryoshka) model v3.

Usage:
    python scripts/eval_bam.py --config configs/bam.yaml \
        --checkpoint /tmp/bam-ckpts/best/ \
        --baseline /tmp/bam-ckpts/mrl_baseline_best/

    # Evaluate a specific epoch:
    python scripts/eval_bam.py --config configs/bam.yaml \
        --checkpoint /tmp/bam-ckpts/epoch_5/ \
        --baseline /tmp/bam-ckpts/mrl_baseline_best/
"""
import argparse, json, sys, os
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from models.bam import BloomAlignedMRL
from models.encoder import MRLEncoder
from evaluation.evaluator import FullEvaluator, BLOOM_NAMES
from transformers import AutoTokenizer


def load_bam(config, ckpt_path, device):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bam.yaml")
    parser.add_argument("--checkpoint", required=True, help="BAM checkpoint dir")
    parser.add_argument("--baseline", default=None, help="MRL baseline checkpoint dir")
    parser.add_argument("--mrl_continued", default=None,
                        help="MRL-continued checkpoint dir (same training budget as BAM, fair comparison)")
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

    # Evaluate BAM
    print("=" * 60)
    print("BAM Evaluation")
    print("=" * 60)
    bam_model = load_bam(config, args.checkpoint, device)
    bam_metrics = evaluator.evaluate_model(
        bam_model, test_path, corpus_path, tokenizer, device,
        compute_bootstrap=True,
    )
    all_results["BAM"] = bam_metrics
    print(f"  {bam_model.__class__.__name__}: corpus={len(open(corpus_path).readlines())}, "
          f"queries from {test_path}")

    # BAM encoder truncated (no routing) — isolates routing gain from encoder quality gain.
    # Uses the same BAM encoder but ignores the Bloom router; truncates at fixed dims.
    # If BAM full > BAM encoder truncated at equal dims → routing genuinely helps.
    print("\n" + "=" * 60)
    print("BAM Encoder Truncated (no routing, fair encoder quality baseline)")
    print("=" * 60)
    bam_encoder_model = load_mrl(config, args.checkpoint, device)
    bam_enc_metrics = evaluator.evaluate_model(
        bam_encoder_model, test_path, corpus_path, tokenizer, device,
        mrl_truncation_dims=config["model"]["mrl_dims"],
        compute_bootstrap=True,
    )
    all_results["BAM Encoder (no routing)"] = bam_enc_metrics

    # Evaluate MRL Baseline
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

    # MRL-continued baseline (fair: same total training steps as BAM)
    if args.mrl_continued:
        print("\n" + "=" * 60)
        print("MRL-Continued Baseline (same total training as BAM)")
        print("=" * 60)
        mc_model = load_mrl(config, args.mrl_continued, device)
        mc_metrics = evaluator.evaluate_model(
            mc_model, test_path, corpus_path, tokenizer, device,
            mrl_truncation_dims=config["model"]["mrl_dims"],
            compute_bootstrap=True,
        )
        all_results["MRL Continued"] = mc_metrics

    # Print comparison
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    std_metrics = ["recall@1", "recall@5", "recall@10", "recall@50", "mrr", "ndcg@10"]
    print(f"{'Metric':45s}" + "".join(f"{n:>12s}" for n in all_results))
    print("-" * (45 + 12 * len(all_results)))
    for m in std_metrics:
        row = f"{m:45s}"
        for name, res in all_results.items():
            row += f"{res.get(m, 0):>12.4f}"
        print(row)

    # Policy stats
    if "avg_active_dims" in bam_metrics:
        print(f"\n{'avg_policy_dim':45s}{bam_metrics['avg_active_dims']:>12.0f}")

    # Bloom-stratified
    print(f"\n{'Bloom-Stratified R@10':45s}")
    print("-" * (45 + 12 * len(all_results)))
    for level in range(1, 7):
        name = BLOOM_NAMES[level]
        key = f"bloom_{name}_recall@10"
        n_key = f"bloom_{name}_n"
        ci_lo_key = f"bloom_{name}_recall@10_ci_lo"
        ci_hi_key = f"bloom_{name}_recall@10_ci_hi"

        row = f"  {name:43s}"
        for mname, res in all_results.items():
            val = res.get(key, 0)
            row += f"{val:>12.4f}"
        print(row)

        # Print CI for BAM
        if ci_lo_key in bam_metrics:
            n = bam_metrics.get(n_key, 0)
            lo = bam_metrics[ci_lo_key]
            hi = bam_metrics[ci_hi_key]
            print(f"    {'':41s}  (n={n}, 95% CI=[{lo:.3f}, {hi:.3f}])")

    # BAM per-Bloom dims (from BloomDimRouter)
    print(f"\n{'BAM Dims per Bloom Level':45s}")
    for level in range(1, 7):
        name = BLOOM_NAMES[level]
        key = f"bloom_{name}_avg_dim"
        if key in bam_metrics:
            print(f"  {name:43s}{bam_metrics[key]:>12.0f}")
        elif "avg_active_dims" in bam_metrics:
            print(f"  {name:43s}{bam_metrics['avg_active_dims']:>12.0f}")

    # Truncation comparison across all models
    dims_list = config["model"]["mrl_dims"]
    print(f"\n{'Truncation R@10 comparison':45s}")
    print("-" * 50)
    print(f"  {'Dims':>6}  {'MRL':>8}  {'MRL-cont':>10}  {'BAM-enc':>9}  {'BAM (routed)':>13}")
    for d in dims_list:
        key = f"mrl_d{d}_recall@10"
        mrl_r10 = all_results.get("MRL Baseline", {}).get(key, "-")
        mrl_c_r10 = all_results.get("MRL Continued", {}).get(key, "-")
        bam_enc_r10 = all_results.get("BAM Encoder (no routing)", {}).get(key, "-")
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else f"{'':>8}"
        print(f"  {d:>6}  {fmt(mrl_r10):>8}  {fmt(mrl_c_r10):>10}  {fmt(bam_enc_r10):>9}")
    avg_dim = bam_metrics.get("avg_active_dims", 0)
    print(f"  {'~'+str(int(avg_dim)):>6}  {'':>8}  {'':>10}  {'':>9}  {bam_metrics.get('recall@10', 0):>13.4f}  ← BAM routed")

    # Save
    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump({k: {kk: (float(vv) if isinstance(vv, (float,)) else vv)
                       for kk, vv in v.items()} for k, v in all_results.items()}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
