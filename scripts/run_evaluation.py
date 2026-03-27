"""Run full evaluation suite with proper query→passage ID matching.
Usage: python scripts/run_evaluation.py --config configs/real_data.yaml \
    --checkpoint /tmp/qa-mrl-ckpts/best/ --baseline /tmp/qa-mrl-ckpts/mrl_baseline_best/
"""
import argparse, json, sys, os
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from models.qa_mrl import QAMRL
from models.encoder import MRLEncoder
from evaluation.evaluator import FullEvaluator
from transformers import AutoTokenizer


def load_qa_mrl(config, ckpt_path, device):
    model = QAMRL(config)
    f = os.path.join(ckpt_path, "checkpoint.pt")
    if os.path.exists(f):
        ckpt = torch.load(f, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded QA-MRL from {f}")
    model.to(device).eval()
    return model


def load_mrl(config, ckpt_path, device):
    mc = config["model"]
    model = MRLEncoder(model_name=mc["backbone"], embedding_dim=mc["embedding_dim"],
                       mrl_dims=mc["mrl_dims"])
    f = os.path.join(ckpt_path, "checkpoint.pt")
    if os.path.exists(f):
        ckpt = torch.load(f, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded MRL baseline from {f}")
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_data.yaml")
    parser.add_argument("--checkpoint", required=True, help="QA-MRL checkpoint dir")
    parser.add_argument("--baseline", default=None, help="MRL baseline checkpoint dir")
    parser.add_argument("--output_dir", default="results/evaluation/")
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

    # Evaluate QA-MRL
    print("=" * 60)
    print("QA-MRL Evaluation")
    print("=" * 60)
    qa_model = load_qa_mrl(config, args.checkpoint, device)
    all_results["QA-MRL"] = evaluator.evaluate_model(
        qa_model, test_path, corpus_path, tokenizer, device
    )

    # Evaluate MRL Baseline
    if args.baseline:
        print("\n" + "=" * 60)
        print("MRL Baseline Evaluation")
        print("=" * 60)
        bl_model = load_mrl(config, args.baseline, device)
        all_results["MRL Baseline"] = evaluator.evaluate_model(
            bl_model, test_path, corpus_path, tokenizer, device,
            mrl_truncation_dims=[64, 128, 256, 384, 512],
        )

    # Save results
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({k: {kk: (float(vv) if isinstance(vv, float) else vv)
                       for kk, vv in v.items()} for k, v in all_results.items()}, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    # Standard metrics
    std_metrics = ["recall@1", "recall@5", "recall@10", "recall@50", "mrr", "ndcg@10"]
    print(f"\n{'Metric':35s}" + "".join(f"{n:>15s}" for n in all_results))
    print("-" * (35 + 15 * len(all_results)))
    for m in std_metrics:
        row = f"{m:35s}"
        for name, res in all_results.items():
            row += f"{res.get(m, 0):>15.4f}"
        print(row)

    # Efficiency
    print(f"\n{'avg_active_dims':35s}", end="")
    for name, res in all_results.items():
        v = res.get("avg_active_dims", 768)
        print(f"{v:>15.0f}", end="")
    print()

    # Bloom stratified
    print(f"\n{'Bloom-Stratified R@10':35s}")
    print("-" * (35 + 15 * len(all_results)))
    for level in range(1, 7):
        from evaluation.evaluator import BLOOM_NAMES
        name = BLOOM_NAMES[level]
        key = f"bloom_{name}_recall@10"
        row = f"  {name:33s}"
        for mname, res in all_results.items():
            row += f"{res.get(key, 0):>15.4f}"
        print(row)

    # MRL truncation comparison (if baseline was evaluated)
    if args.baseline and "MRL Baseline" in all_results:
        bl = all_results["MRL Baseline"]
        qa = all_results["QA-MRL"]
        print(f"\n{'MRL Truncation vs QA-MRL':35s}")
        print("-" * 65)
        for d in [64, 128, 256, 384, 512]:
            key = f"mrl_d{d}_recall@10"
            if key in bl:
                print(f"  MRL d={d:3d}:  R@10={bl[key]:.4f}")
        print(f"  MRL d=768:  R@10={bl.get('recall@10', 0):.4f}")
        print(f"  QA-MRL (~{qa.get('avg_active_dims', 768):.0f} dims): "
              f"R@10={qa.get('recall@10', 0):.4f}")

    print(f"\nResults saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()