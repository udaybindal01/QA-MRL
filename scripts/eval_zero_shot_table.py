"""
Zero-shot BEIR evaluation table.

Trained on (e.g.) educational, evaluated zero-shot on scifact/nfcorpus/fiqa.
Reports R@10 and R@50 for:
  - MRL baseline at each prefix-truncation dimension (mrl_dims from config)
  - BAM-PQ routed (uses model's average active dimensions)

Uses FullEvaluator which:
  - Computes masked-query vs full-corpus dot product (deployment-realistic,
    matches the training objective; not the oracle-eval variant).
  - Casts embeddings to float32 (handles bf16 backbones like Qwen).
  - Loads predicted Bloom labels from per-dataset .bloom_cache.json.

Outputs:
  results.json           — raw metrics per (dataset, model)
  zero_shot_table.tex    — LaTeX table for the paper (R@10, R@50 only)
  Console table          — same numbers, human-readable

Usage:
  python3 scripts/eval_zero_shot_table.py \
      --mrl_config        configs/mrl_bge_base.yaml \
      --mrl_checkpoint    /scratch/.../educational/mrl_bge_base/best \
      --bam_pq_config     configs/bam_pq_bge_base.yaml \
      --bam_pq_checkpoint /scratch/.../educational/bam_pq_bge_base/best_bsr \
      --datasets scifact nfcorpus fiqa \
      --beir_data_root /scratch/.../bampq-data/beir \
      --output_dir results/zero_shot/bge_base/
"""
import argparse
import copy
import json
import os
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed                       # noqa: E402
from models.bam import BloomAlignedMRL                             # noqa: E402
from models.encoder import MRLEncoder                              # noqa: E402
from evaluation.evaluator import FullEvaluator                     # noqa: E402


def _load_state(model, ckpt_path, device, label):
    f = os.path.join(ckpt_path, "checkpoint.pt")
    if not os.path.exists(f):
        raise FileNotFoundError(f"{label} checkpoint not found: {f}")
    ckpt = torch.load(f, map_location=device)
    result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if result.missing_keys:
        print(f"  WARNING [{label}] missing keys: {result.missing_keys[:5]}"
              f"{' …' if len(result.missing_keys) > 5 else ''}")
    if result.unexpected_keys:
        print(f"  WARNING [{label}] unexpected keys: {result.unexpected_keys[:5]}"
              f"{' …' if len(result.unexpected_keys) > 5 else ''}")
    print(f"  Loaded {label} from {f}")


def load_bam(config, ckpt_path, device):
    config["training"]["loss"].setdefault("bloom_frequencies", [1 / 6] * 6)
    model = BloomAlignedMRL(config)
    _load_state(model, ckpt_path, device, "BAM-PQ")
    return model.to(device).eval()


def load_mrl(config, ckpt_path, device):
    mc = config["model"]
    model = MRLEncoder(
        model_name=mc["backbone"],
        embedding_dim=mc["embedding_dim"],
        mrl_dims=mc["mrl_dims"],
        pooling=mc.get("pooling", "cls"),
        backbone_type=mc.get("backbone_type", "standard"),
        query_instruction=mc.get("query_instruction", None),
        peft_model_name=mc.get("peft_model_name", None),
    )
    _load_state(model, ckpt_path, device, "MRL")
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mrl_config",        required=True)
    parser.add_argument("--mrl_checkpoint",    required=True)
    parser.add_argument("--bam_pq_config",     required=True)
    parser.add_argument("--bam_pq_checkpoint", required=True)
    parser.add_argument("--datasets",          nargs="+",
                        default=["scifact", "nfcorpus", "fiqa"])
    parser.add_argument("--beir_data_root",    default="/tmp/data/beir")
    parser.add_argument("--output_dir",        default="results/zero_shot/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    bam_config = load_config(args.bam_pq_config)
    mrl_config = load_config(args.mrl_config)
    set_seed(bam_config["training"]["seed"])
    tokenizer = AutoTokenizer.from_pretrained(bam_config["model"]["backbone"])
    mrl_dims = mrl_config["model"]["mrl_dims"]
    full_dim = mrl_config["model"]["embedding_dim"]

    print(f"Backbone: {bam_config['model']['backbone']}  (full dim = {full_dim})")
    print(f"MRL truncation dims: {mrl_dims}")
    print(f"Datasets: {args.datasets}\n")

    print("Loading models...")
    bam_model = load_bam(bam_config, args.bam_pq_checkpoint, device)
    mrl_model = load_mrl(mrl_config, args.mrl_checkpoint, device)

    all_results = {}
    for ds in args.datasets:
        test_path   = os.path.join(args.beir_data_root, ds, "test.jsonl")
        corpus_path = os.path.join(args.beir_data_root, ds, "corpus.jsonl")
        if not (os.path.exists(test_path) and os.path.exists(corpus_path)):
            print(f"  SKIP {ds}: data not found at {args.beir_data_root}/{ds}/")
            continue

        print(f"\n=== Zero-shot eval: {ds} ===")
        ds_config = copy.deepcopy(bam_config)
        ds_config["data"]["test_path"]   = test_path
        ds_config["data"]["corpus_path"] = corpus_path
        evaluator = FullEvaluator(ds_config)

        print("  -- MRL --")
        mrl_metrics = evaluator.evaluate_model(
            mrl_model, test_path, corpus_path, tokenizer, device,
            mrl_truncation_dims=mrl_dims,
        )
        print("  -- BAM-PQ --")
        bam_metrics = evaluator.evaluate_model(
            bam_model, test_path, corpus_path, tokenizer, device,
        )
        all_results[ds] = {"MRL": mrl_metrics, "BAM-PQ": bam_metrics}

    # ── Save raw JSON ────────────────────────────────────────────────────────
    def _serializable(v):
        if isinstance(v, float): return float(v)
        if isinstance(v, (int, str, bool, list, type(None))): return v
        return str(v)

    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump({
            ds: {model: {k: _serializable(v) for k, v in metrics.items()}
                 for model, metrics in res.items()}
            for ds, res in all_results.items()
        }, f, indent=2)
    print(f"\nRaw metrics saved to {out_path}")

    # ── Console table ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ZERO-SHOT BEIR — R@10 / R@50")
    print("=" * 70)
    header = f"{'Dataset':<12} {'Model':<22} {'Dims':>6} {'R@10':>8} {'R@50':>8}"
    print(header)
    print("-" * len(header))
    for ds, res in all_results.items():
        mrl = res["MRL"]
        for d in mrl_dims:
            r10 = mrl.get(f"mrl_d{d}_recall@10", 0.0)
            r50 = mrl.get(f"mrl_d{d}_recall@50", 0.0)
            tag = f"MRL @ {d}"
            print(f"{ds:<12} {tag:<22} {d:>6} {r10:>8.4f} {r50:>8.4f}")
        bam = res["BAM-PQ"]
        avg = int(round(bam.get("avg_active_dims", 0)))
        r10 = bam.get("recall@10", 0.0)
        r50 = bam.get("recall@50", 0.0)
        print(f"{ds:<12} {'BAM-PQ (routed)':<22} {avg:>6} {r10:>8.4f} {r50:>8.4f}")
        print("-" * len(header))

    # ── LaTeX table ──────────────────────────────────────────────────────────
    latex_path = os.path.join(args.output_dir, "zero_shot_table.tex")
    with open(latex_path, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\\small\n")
        f.write("\\begin{tabular}{@{}llrrr@{}}\n\\toprule\n")
        f.write("Dataset & Model & Dims & R@10 & R@50 \\\\\n\\midrule\n")
        for ds, res in all_results.items():
            mrl = res["MRL"]
            first = True
            for d in mrl_dims:
                r10 = mrl.get(f"mrl_d{d}_recall@10", 0.0)
                r50 = mrl.get(f"mrl_d{d}_recall@50", 0.0)
                ds_cell = f"\\multirow{{{len(mrl_dims) + 1}}}{{*}}{{{ds}}}" if first else ""
                first = False
                f.write(f"{ds_cell} & MRL @ {d} & {d} & {r10:.4f} & {r50:.4f} \\\\\n")
            bam = res["BAM-PQ"]
            avg = int(round(bam.get("avg_active_dims", 0)))
            r10 = bam.get("recall@10", 0.0)
            r50 = bam.get("recall@50", 0.0)
            f.write(f" & \\textbf{{BAM-PQ (routed)}} & {avg} & "
                    f"\\textbf{{{r10:.4f}}} & \\textbf{{{r50:.4f}}} \\\\\n")
            f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Zero-shot BEIR retrieval. MRL is reported at each "
                "prefix-truncation dimension; BAM-PQ uses the model's average "
                "routed active dimensions. \\textbf{Bold} = BAM-PQ row.}\n")
        f.write("\\label{tab:zero-shot-beir}\n")
        f.write("\\end{table}\n")
    print(f"LaTeX table saved to {latex_path}")


if __name__ == "__main__":
    main()
