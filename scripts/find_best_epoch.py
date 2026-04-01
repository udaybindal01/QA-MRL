"""
Find the true best epoch checkpoint using full-corpus evaluation.

The in-training validator computes NDCG on val-set pairwise similarity
(query vs val positives only), NOT the real 40K-document corpus. This
inflates metrics for early epochs and picks the wrong "best" checkpoint.

This script evaluates every saved epoch checkpoint against the full corpus
and identifies the true best by R@10 or NDCG@10.

Usage:
    # For BAM:
    python scripts/find_best_epoch.py \\
        --config configs/bam.yaml \\
        --checkpoint_dir /tmp/bam-ckpts/ \\
        --model_type bam \\
        --metric recall@10

    # For MRL baseline:
    python scripts/find_best_epoch.py \\
        --config configs/neurips.yaml \\
        --checkpoint_dir /tmp/mrl-ckpts/ \\
        --model_type mrl \\
        --metric recall@10

After running, the best checkpoint is copied to {checkpoint_dir}/best/
"""

import argparse, json, os, sys, shutil
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from models.bam import BloomAlignedMRL
from models.encoder import MRLEncoder
from evaluation.evaluator import FullEvaluator
from transformers import AutoTokenizer


def load_bam(config, ckpt_dir, device):
    model = BloomAlignedMRL(config)
    f = os.path.join(ckpt_dir, "checkpoint.pt")
    if os.path.exists(f):
        ckpt = torch.load(f, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()
    return model


def load_mrl(config, ckpt_dir, device):
    mc = config["model"]
    model = MRLEncoder(
        model_name=mc["backbone"],
        embedding_dim=mc["embedding_dim"],
        mrl_dims=mc["mrl_dims"],
    )
    f = os.path.join(ckpt_dir, "checkpoint.pt")
    if os.path.exists(f):
        ckpt = torch.load(f, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Root dir containing epoch_0/, epoch_1/, ... subdirs")
    parser.add_argument("--model_type", choices=["bam", "mrl"], default="bam")
    parser.add_argument("--metric", default="recall@10",
                        choices=["recall@1", "recall@5", "recall@10", "recall@50",
                                 "mrr", "ndcg@10"],
                        help="Metric to use for best-epoch selection")
    parser.add_argument("--output_dir", default=None,
                        help="Where to save per-epoch results JSON (default: checkpoint_dir)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])
    evaluator = FullEvaluator(config)

    test_path = config["data"]["test_path"]
    corpus_path = config["data"]["corpus_path"]
    out_dir = args.output_dir or args.checkpoint_dir
    os.makedirs(out_dir, exist_ok=True)

    # Find all epoch checkpoints
    epoch_dirs = []
    for name in sorted(os.listdir(args.checkpoint_dir)):
        if name.startswith("epoch_"):
            d = os.path.join(args.checkpoint_dir, name)
            ckpt_file = os.path.join(d, "checkpoint.pt")
            if os.path.exists(ckpt_file):
                try:
                    epoch_num = int(name.split("_")[1])
                    epoch_dirs.append((epoch_num, name, d))
                except ValueError:
                    pass
    # Also include "final" if present
    final_dir = os.path.join(args.checkpoint_dir, "final")
    if os.path.exists(os.path.join(final_dir, "checkpoint.pt")):
        epoch_dirs.append((9999, "final", final_dir))

    epoch_dirs.sort(key=lambda x: x[0])

    if not epoch_dirs:
        print(f"ERROR: No epoch_N/ checkpoints found in {args.checkpoint_dir}")
        print("Make sure training completed and saved per-epoch checkpoints.")
        sys.exit(1)

    print(f"Found {len(epoch_dirs)} checkpoints to evaluate.")
    print(f"Selecting best by: {args.metric}")
    print(f"Model type: {args.model_type}")
    print("=" * 70)

    results = {}
    best_metric_val = -1.0
    best_epoch_name = None
    best_epoch_dir = None

    header = f"{'Epoch':10s} {'R@1':>8s} {'R@10':>8s} {'R@50':>8s} {'NDCG@10':>10s} {'AvgDims':>9s}"
    print(header)
    print("-" * len(header))

    for epoch_num, name, ckpt_dir in epoch_dirs:
        # Load model fresh for each epoch
        if args.model_type == "bam":
            model = load_bam(config, ckpt_dir, device)
        else:
            model = load_mrl(config, ckpt_dir, device)

        metrics = evaluator.evaluate_model(
            model, test_path, corpus_path, tokenizer, device,
            compute_bootstrap=False,
        )
        results[name] = metrics

        r1 = metrics.get("recall@1", 0)
        r10 = metrics.get("recall@10", 0)
        r50 = metrics.get("recall@50", 0)
        ndcg = metrics.get("ndcg@10", 0)
        avg_dims = metrics.get("avg_active_dims", 768)

        print(f"{name:10s} {r1:>8.4f} {r10:>8.4f} {r50:>8.4f} {ndcg:>10.4f} {avg_dims:>9.0f}")

        target = metrics.get(args.metric, 0)
        if target > best_metric_val:
            best_metric_val = target
            best_epoch_name = name
            best_epoch_dir = ckpt_dir

        # Free GPU memory between epochs
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    print("-" * len(header))
    print(f"\nBest epoch: {best_epoch_name}  ({args.metric} = {best_metric_val:.4f})")

    # Copy best to {checkpoint_dir}/best/
    best_dest = os.path.join(args.checkpoint_dir, "best")
    if os.path.exists(best_dest):
        shutil.rmtree(best_dest)
    shutil.copytree(best_epoch_dir, best_dest)
    print(f"Copied best checkpoint → {best_dest}/")

    # Save per-epoch results
    results_path = os.path.join(out_dir, "epoch_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "best_epoch": best_epoch_name,
            "best_metric": args.metric,
            "best_value": best_metric_val,
            "epochs": {
                name: {k: float(v) for k, v in m.items()
                       if isinstance(v, (int, float))}
                for name, m in results.items()
            }
        }, f, indent=2)
    print(f"Per-epoch results saved → {results_path}")


if __name__ == "__main__":
    main()
