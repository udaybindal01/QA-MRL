"""
Collect BSR (Bloom Stratified Recall) scores for all ablation variants.

For each variant, either reads the pre-computed epoch_results_bsr.json
from best_bsr/ dir, or runs find_best_epoch_bsr.py if not present.

Usage:
    python scripts/collect_ablation_bsr.py
    python scripts/collect_ablation_bsr.py --ckpt_root /tmp/bam-pq-ckpts
    python scripts/collect_ablation_bsr.py --recompute   # re-run BSR for all
    python scripts/collect_ablation_bsr.py --latex        # print LaTeX table
"""

import argparse
import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VARIANTS = [
    ("full",             "Full model (all losses)"),
    ("no_sparsity",      "w/o Mask Sparsity"),
    ("no_diversity",     "w/o Mask Diversity"),
    ("no_variance",      "w/o Mask Variance"),
    ("no_distill",       "w/o Mask Distillation"),
    ("no_query_div",     "w/o Query Routing Div."),
    ("contrastive_only", "Contrastive Only"),
]

BLOOM_NAMES = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]


def load_bsr_json(bsr_json_path):
    """
    Load epoch_results_bsr.json and return the best-epoch entry.
    The file is a list of per-epoch dicts each containing bsr, quality,
    efficiency, per_bloom_r10, per_bloom_dim, recall@10, ndcg@10, etc.
    """
    with open(bsr_json_path) as f:
        epochs = json.load(f)

    if isinstance(epochs, dict):
        # Single-epoch or summary dict
        return epochs

    # List of epoch dicts — pick the one with max bsr
    best = max(epochs, key=lambda e: e.get("bsr", 0.0))
    return best


def run_bsr(variant, ckpt_root, config_root, alpha, data_path, corpus_path):
    """Run find_best_epoch_bsr.py for a variant and return output dir."""
    ckpt_dir = os.path.join(ckpt_root, f"abl_{variant}")
    cfg_path  = os.path.join(config_root, f"abl_{variant}.yaml")
    out_dir   = os.path.join(ckpt_dir, "best_bsr")

    if not os.path.isdir(ckpt_dir):
        print(f"  [{variant}] checkpoint dir not found: {ckpt_dir}")
        return None
    if not os.path.exists(cfg_path):
        print(f"  [{variant}] config not found: {cfg_path}")
        return None

    print(f"  [{variant}] running find_best_epoch_bsr.py ...")
    cmd = [
        sys.executable, "scripts/find_best_epoch_bsr.py",
        "--config",         cfg_path,
        "--checkpoint_dir", ckpt_dir,
        "--output_dir",     out_dir,
        "--alpha",          str(alpha),
    ]
    if data_path:
        cmd += ["--data_path", data_path]
    if corpus_path:
        cmd += ["--corpus_path", corpus_path]

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [{variant}] find_best_epoch_bsr.py failed (exit {result.returncode})")
        return None
    return out_dir


def collect(ckpt_root, config_root, alpha, data_path, corpus_path, recompute):
    rows = []
    for name, label in VARIANTS:
        bsr_dir  = os.path.join(ckpt_root, f"abl_{name}", "best_bsr")
        bsr_json = os.path.join(bsr_dir, "epoch_results_bsr.json")

        if recompute or not os.path.exists(bsr_json):
            out = run_bsr(name, ckpt_root, config_root, alpha, data_path, corpus_path)
            if out is None:
                rows.append({"name": name, "label": label, "missing": True})
                continue
            bsr_json = os.path.join(out, "epoch_results_bsr.json")

        if not os.path.exists(bsr_json):
            print(f"  [{name}] epoch_results_bsr.json not found at {bsr_json}")
            rows.append({"name": name, "label": label, "missing": True})
            continue

        data = load_bsr_json(bsr_json)
        row = {
            "name":       name,
            "label":      label,
            "missing":    False,
            "bsr":        data.get("bsr",        data.get("best_bsr",   0.0)),
            "quality":    data.get("quality",    data.get("best_quality", 0.0)),
            "efficiency": data.get("efficiency", data.get("best_efficiency", 0.0)),
            "r10":        data.get("recall@10",  data.get("r10",        0.0)),
            "ndcg10":     data.get("ndcg@10",    data.get("ndcg10",     0.0)),
            "avg_dim":    data.get("avg_dim",    data.get("avg_active_dims", 0.0)),
        }

        # Per-bloom dims
        bloom_dims = []
        for b_name in BLOOM_NAMES:
            d = data.get(f"bloom_{b_name}_avg_dim") or data.get(f"dim_{b_name}")
            bloom_dims.append(d)
        row["bloom_dims"] = bloom_dims

        rows.append(row)
    return rows


def print_table(rows, alpha):
    print(f"\n{'='*80}")
    print(f"  ABLATION BSR TABLE  (α={alpha})")
    print(f"{'='*80}")
    header = f"  {'Variant':<30}  {'BSR':>7}  {'Quality':>7}  {'Effic.':>7}  {'R@10':>7}  {'NDCG@10':>8}  {'AvgDim':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Find best per column for highlighting
    present = [r for r in rows if not r.get("missing")]
    best_bsr   = max((r["bsr"]    for r in present), default=0)
    best_r10   = max((r["r10"]    for r in present), default=0)
    best_ndcg  = max((r["ndcg10"] for r in present), default=0)

    for row in rows:
        if row.get("missing"):
            print(f"  {row['label']:<30}  {'--':>7}  {'--':>7}  {'--':>7}  {'--':>7}  {'--':>8}  {'--':>7}")
            continue
        bsr_s   = f"{'*' if abs(row['bsr']   - best_bsr)  < 1e-6 else ' '}{row['bsr']:.4f}"
        r10_s   = f"{'*' if abs(row['r10']   - best_r10)  < 1e-6 else ' '}{row['r10']:.4f}"
        ndcg_s  = f"{'*' if abs(row['ndcg10']- best_ndcg) < 1e-6 else ' '}{row['ndcg10']:.4f}"
        dim_s   = f"{row['avg_dim']:.0f}" if row['avg_dim'] else "--"
        print(f"  {row['label']:<30}  {bsr_s:>7}  {row['quality']:7.4f}  {row['efficiency']:7.4f}  {r10_s:>7}  {ndcg_s:>8}  {dim_s:>7}")

    print()
    print("  Per-Bloom dims:")
    dim_header = f"  {'Variant':<30}" + "".join(f"  {n[:4]:>6}" for n in BLOOM_NAMES)
    print(dim_header)
    print("  " + "-" * (len(dim_header) - 2))
    for row in rows:
        if row.get("missing"):
            continue
        dims_str = "".join(
            f"  {int(d):>6}" if d is not None else f"  {'--':>6}"
            for d in row.get("bloom_dims", [None]*6)
        )
        print(f"  {row['label']:<30}{dims_str}")


def print_latex(rows, alpha):
    print(f"\n% ── Ablation table (α={alpha}) ──────────────────────────────────────")
    print(r"\begin{table}[t]")
    print(r"\centering\small")
    print(r"\begin{tabular}{@{}lrrrrr@{}}")
    print(r"\toprule")
    print(r"Variant & BSR & Quality & Efficiency & R@10 & NDCG@10 \\")
    print(r"\midrule")

    present = [r for r in rows if not r.get("missing")]
    best_bsr  = max((r["bsr"]    for r in present), default=0)
    best_r10  = max((r["r10"]    for r in present), default=0)
    best_ndcg = max((r["ndcg10"] for r in present), default=0)

    def bf(v, best):
        s = f"{v:.4f}"
        return f"\\textbf{{{s}}}" if abs(v - best) < 1e-5 else s

    for row in rows:
        if row.get("missing"):
            print(f"{row['label']} & -- & -- & -- & -- & -- \\\\")
            continue
        print(f"{row['label']} & {bf(row['bsr'], best_bsr)} & {row['quality']:.4f} & "
              f"{row['efficiency']:.4f} & {bf(row['r10'], best_r10)} & "
              f"{bf(row['ndcg10'], best_ndcg)} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Loss ablation study (BSR = quality $\times$ (1 + $\alpha \times$ efficiency), $\alpha="
          + str(alpha) + r"$). * = best per column.}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_root",    default="/tmp/bam-pq-ckpts",
                        help="Parent dir containing abl_<variant>/ subdirs")
    parser.add_argument("--config_root",  default="configs/ablations",
                        help="Dir containing abl_<variant>.yaml configs")
    parser.add_argument("--data_path",    default="./data/real/test.jsonl",
                        help="Test JSONL for evaluation (used when recomputing)")
    parser.add_argument("--corpus_path",  default="./data/real/corpus.jsonl",
                        help="Corpus JSONL (used when recomputing)")
    parser.add_argument("--alpha",        type=float, default=0.5,
                        help="Efficiency weight in BSR formula")
    parser.add_argument("--recompute",    action="store_true",
                        help="Re-run find_best_epoch_bsr.py even if cache exists")
    parser.add_argument("--latex",        action="store_true",
                        help="Also print LaTeX table")
    parser.add_argument("--save",         default=None,
                        help="Save collected results to this JSON path")
    args = parser.parse_args()

    rows = collect(
        ckpt_root=args.ckpt_root,
        config_root=args.config_root,
        alpha=args.alpha,
        data_path=args.data_path,
        corpus_path=args.corpus_path,
        recompute=args.recompute,
    )

    print_table(rows, args.alpha)

    if args.latex:
        print_latex(rows, args.alpha)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\n  Saved → {args.save}")


if __name__ == "__main__":
    main()
