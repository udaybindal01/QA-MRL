"""
Collect loss ablation results and print a comparison table + LaTeX.

Reads JSON metric files from results/ablations/ (written by evaluate.py).
Prints console table and LaTeX table suitable for paper.

Usage:
    python scripts/collect_ablation_results.py
    python scripts/collect_ablation_results.py --results_dir results/ablations/
"""

import argparse
import json
import os
import glob

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

METRICS = ["recall@1", "recall@10", "recall@50", "ndcg@10", "mrr"]
METRIC_LABELS = ["R@1", "R@10", "R@50", "NDCG@10", "MRR"]


def load_result(results_dir, variant_name):
    pattern = os.path.join(results_dir, f"bam_pq_*{variant_name}*_metrics.json")
    matches = glob.glob(pattern)
    if not matches:
        pattern2 = os.path.join(results_dir, f"bam_pq_best_bsr_metrics.json")
        matches = glob.glob(os.path.join(results_dir, f"*{variant_name}*"))
    if not matches:
        return None
    with open(matches[0]) as f:
        return json.load(f)


def fmt(v, bold=False):
    s = f"{v:.4f}"
    return f"\\textbf{{{s}}}" if bold else s


def best_per_col(rows, col_idx):
    vals = [r[col_idx] for r in rows if r[col_idx] is not None]
    return max(vals) if vals else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/ablations/")
    args = parser.parse_args()

    results = {}
    for name, label in VARIANTS:
        r = load_result(args.results_dir, name)
        results[name] = (label, r)

    # ── Console table ────────────────────────────────────────────────────
    col_w = 30
    print()
    print("LOSS ABLATION — BAM-PQ (e5-large, test set)")
    print("=" * (col_w + len(METRIC_LABELS) * 10))
    hdr = f"{'Variant':<{col_w}}" + "".join(f"{m:>10}" for m in METRIC_LABELS)
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for name, label in VARIANTS:
        _, r = results[name]
        if r is None:
            print(f"  {label:<{col_w-2}}  [not found]")
            rows.append([None] * len(METRICS))
            continue
        vals = [r.get(m) for m in METRICS]
        rows.append(vals)
        row_str = f"{label:<{col_w}}"
        for v in vals:
            row_str += f"{v:>10.4f}" if v is not None else f"{'—':>10}"
        print(row_str)

    print()

    # ── Bloom-stratified table ────────────────────────────────────────────
    print("BLOOM-STRATIFIED R@10")
    print("=" * (col_w + len(BLOOM_NAMES) * 12))
    bhdr = f"{'Variant':<{col_w}}" + "".join(f"{b:>12}" for b in BLOOM_NAMES)
    print(bhdr)
    print("-" * len(bhdr))
    for name, label in VARIANTS:
        _, r = results[name]
        if r is None:
            print(f"  {label:<{col_w-2}}  [not found]")
            continue
        row_str = f"{label:<{col_w}}"
        for b in BLOOM_NAMES:
            v = r.get(f"bloom_{b}_recall@10")
            row_str += f"{v:>12.4f}" if v is not None else f"{'—':>12}"
        print(row_str)

    print()

    # ── LaTeX table ───────────────────────────────────────────────────────
    col_vals = []
    for ci in range(len(METRICS)):
        col_vals.append(best_per_col(rows, ci))

    print("% ── LaTeX table ─────────────────────────────────────────────────")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Loss ablation study on the educational test set (e5-large backbone).")
    print(r"Each row removes one loss component from the full BAM-PQ model.")
    print(r"\textbf{Bold} = best per column.}")
    print(r"\label{tab:loss_ablation}")
    ncols = "l" + "r" * len(METRICS)
    print(rf"\begin{{tabular}}{{{ncols}}}")
    print(r"\toprule")
    print("Variant & " + " & ".join(METRIC_LABELS) + r" \\")
    print(r"\midrule")

    for i, (name, label) in enumerate(VARIANTS):
        _, r = results[name]
        if r is None:
            print(f"{label} & " + " & ".join(["—"] * len(METRICS)) + r" \\")
            continue
        vals = [r.get(m) for m in METRICS]
        cells = []
        for ci, v in enumerate(vals):
            if v is None:
                cells.append("—")
            else:
                is_best = (col_vals[ci] is not None and abs(v - col_vals[ci]) < 1e-7)
                cells.append(fmt(v, bold=is_best))
        if name == "full":
            print(r"\midrule")
        print(f"{label} & " + " & ".join(cells) + r" \\")
        if name == "full":
            print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()

    # ── Delta table (drop vs full model) ─────────────────────────────────
    _, full_r = results.get("full", (None, None))
    if full_r:
        print("% ── Delta vs Full Model ─────────────────────────────────────────")
        print(r"\begin{table}[t]")
        print(r"\centering")
        print(r"\caption{$\Delta$ R@10 when removing each loss from BAM-PQ full model.")
        print(r"Negative = degradation. Losses with largest drop are most critical.}")
        print(r"\label{tab:loss_ablation_delta}")
        print(r"\begin{tabular}{lrr}")
        print(r"\toprule")
        print(r"Removed Loss & R@10 & $\Delta$ R@10 \\")
        print(r"\midrule")
        full_r10 = full_r.get("recall@10", 0)
        for name, label in VARIANTS:
            if name in ("full", "contrastive_only"):
                continue
            _, r = results[name]
            if r is None:
                print(f"{label} & — & — \\\\")
                continue
            v = r.get("recall@10", 0)
            delta = v - full_r10
            delta_str = f"{delta:+.4f}"
            print(f"{label} & {v:.4f} & {delta_str} \\\\")
        print(r"\midrule")
        _, co_r = results.get("contrastive_only", (None, None))
        if co_r:
            v = co_r.get("recall@10", 0)
            delta = v - full_r10
            print(f"Contrastive Only & {v:.4f} & {delta:+.4f} \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(r"\end{table}")


if __name__ == "__main__":
    main()
