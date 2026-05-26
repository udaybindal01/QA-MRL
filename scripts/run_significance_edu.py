"""
Paired significance testing on the educational benchmark.

Compares BAM-PQ against the MRL baseline and Standard FT baseline on
R@1, R@10, R@50, and NDCG@10 across the four backbones present on
disk (e5-large, bge-large, bge-base, arctic). mxbai-L is intentionally
excluded — no checkpoint trained for it. Bonferroni-Holm corrects
across backbones for each metric. R@1 uses McNemar's exact test
(binary outcome); R@10, R@50, NDCG@10 use the paired Wilcoxon
signed-rank test. Tests are one-sided ("BAM-PQ > baseline"); flip
`--two_sided` for two-sided reporting.

Outputs:
    {output_dir}/per_query/{system}_{backbone}.csv  -- cached per-query metrics
    {output_dir}/significance_table.csv             -- final significance table
    {output_dir}/significance_table.tex             -- LaTeX-ready row dump

Usage:
    python scripts/run_significance_edu.py \\
        --output_dir /scratch/ishaan.karan/significance/ \\
        --test_path  data/real/test.jsonl \\
        --corpus_path data/real/corpus.jsonl
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from evaluation.evaluator import FullEvaluator
from scripts.eval_bam import load_bam, load_mrl


# ---------------------------------------------------------------------------
# Cluster paths confirmed from `ls /scratch/ishaan.karan/bampq-checkpoints/educational/`.
# mxbai-L is excluded (no checkpoint on disk).
# ---------------------------------------------------------------------------
CKPT_ROOT = "/scratch/ishaan.karan/bampq-checkpoints/educational"

BACKBONES = {
    # "e5-large": {
    #     "bampq_config":  "configs/bam_pq.yaml",
    #     "mrl_config":    "configs/mrl_e5large.yaml",
    #     "ft_config":     "configs/standard_ft_e5large.yaml",
    #     "bampq_ckpt":    f"{CKPT_ROOT}/bam_pq_e5large/best_bsr/",
    #     "mrl_ckpt":      f"{CKPT_ROOT}/mrl_e5large/best/",
    #     "ft_ckpt":       f"{CKPT_ROOT}/standard_ft_e5large/best/",
    # },
    # "bge-large": {
    #     "bampq_config":  "configs/bam_pq_bge_large.yaml",
    #     "mrl_config":    "configs/mrl_bge_large.yaml",
    #     "ft_config":     "configs/standard_ft_bge.yaml",
    #     "bampq_ckpt":    f"{CKPT_ROOT}/bam_pq_bge/best_bsr/",
    #     "mrl_ckpt":      f"{CKPT_ROOT}/mrl_bge/best/",
    #     "ft_ckpt":       f"{CKPT_ROOT}/standard_ft_bge/best/",
    # },
    # "bge-base": {
    #     "bampq_config":  "configs/bam_pq_bge_base.yaml",
    #     "mrl_config":    "configs/mrl_bge_base.yaml",
    #     "ft_config":     "configs/standard_ft_bge_base.yaml",
    #     "bampq_ckpt":    f"{CKPT_ROOT}/bam_pq_bge_base/best_bsr/",
    #     "mrl_ckpt":      f"{CKPT_ROOT}/mrl_bge_base/best/",
    #     "ft_ckpt":       f"{CKPT_ROOT}/standard_ft_bge_base/best/",
    # },
    # "arctic": {
    #     "bampq_config":  "configs/bam_pq_arctic.yaml",
    #     "mrl_config":    "configs/mrl_arctic.yaml",
    #     "ft_config":     "configs/standard_ft_arctic.yaml",
    #     "bampq_ckpt":    f"{CKPT_ROOT}/bam_pq_arctic/best_bsr/",
    #     "mrl_ckpt":      f"{CKPT_ROOT}/mrl_arctic/best/",
    #     "ft_ckpt":       f"{CKPT_ROOT}/standard_ft_arctic/best/",
    # },
    "mxbai": {
        "bampq_config":  "configs/bam_pq_mxbai.yaml",
        "mrl_config":    "configs/mrl_mxbai.yaml",
        "ft_config":     "configs/standard_ft_mxbai.yaml",
        "bampq_ckpt":    f"{CKPT_ROOT}/bam_pq_mxbai/best_bsr/",
        "mrl_ckpt":      f"{CKPT_ROOT}/mrl_mxbai/best/",
        "ft_ckpt":       f"{CKPT_ROOT}/standard_ft_mxbai/best/",
    },
}

METRICS = ["recall@1", "recall@10", "recall@50", "ndcg@10"]
METRIC_LABELS = {
    "recall@1":  "R@1",
    "recall@10": "R@10",
    "recall@50": "R@50",
    "ndcg@10":   "NDCG@10",
}


# ---------------------------------------------------------------------------
# Score one (system, backbone) combo using the existing FullEvaluator.
# Per-query metric arrays are read off the returned dict (added in
# evaluator.py as `per_query_recall@{k}` and `per_query_ndcg@10`).
# ---------------------------------------------------------------------------
def score_combo(system, backbone, paths, test_path, corpus_path, device):
    if system == "BAMPQ":
        cfg = load_config(paths["bampq_config"])
        cfg["training"]["loss"].setdefault("bloom_frequencies", [1/6] * 6)
        model = load_bam(cfg, paths["bampq_ckpt"], device)
    elif system == "MRL":
        cfg = load_config(paths["mrl_config"])
        model = load_mrl(cfg, paths["mrl_ckpt"], device)
    elif system == "StandardFT":
        cfg = load_config(paths["ft_config"])
        model = load_mrl(cfg, paths["ft_ckpt"], device)
    else:
        raise ValueError(system)

    set_seed(cfg["training"]["seed"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])
    evaluator = FullEvaluator(cfg)
    result = evaluator.evaluate_model(
        model, test_path, corpus_path, tokenizer, device,
        compute_bootstrap=False,
    )

    bloom_1idx = result["_query_blooms_1idx"]
    df = pd.DataFrame({
        "bloom_1idx":  bloom_1idx,
        "recall@1":    result["per_query_recall@1"],
        "recall@10":   result["per_query_recall@10"],
        "recall@50":   result["per_query_recall@50"],
        "ndcg@10":     result["per_query_ndcg@10"],
    })

    # Aggregate sanity numbers (echo what evaluator already prints).
    aggregates = {m: float(df[m].mean()) for m in METRICS}

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return df, aggregates


# ---------------------------------------------------------------------------
# Paired test: McNemar for R@1 (binary), Wilcoxon for the rest.
# ---------------------------------------------------------------------------
def paired_test(a, b, metric, two_sided=False):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mean_delta = float((a - b).mean())

    if metric == "recall@1":
        n11 = int(((a == 1) & (b == 1)).sum())
        n10 = int(((a == 1) & (b == 0)).sum())
        n01 = int(((a == 0) & (b == 1)).sum())
        n00 = int(((a == 0) & (b == 0)).sum())
        table = [[n11, n10], [n01, n00]]
        if n10 + n01 == 0:
            return mean_delta, 1.0
        # Two-sided exact McNemar p-value.
        result = mcnemar(table, exact=True)
        p_two = float(result.pvalue)
        if two_sided:
            return mean_delta, p_two
        # One-sided 'a > b': only confirms when more queries flipped from miss to hit
        # under A than under B (i.e., n10 > n01).
        if n10 > n01:
            return mean_delta, p_two / 2.0
        return mean_delta, 1.0 - p_two / 2.0

    # Wilcoxon signed-rank (R@10, R@50, NDCG@10).
    if np.allclose(a, b):
        return mean_delta, 1.0
    alt = "two-sided" if two_sided else "greater"
    try:
        _, p = wilcoxon(a, b, alternative=alt, zero_method="wilcox")
    except ValueError:
        p = 1.0
    return mean_delta, float(p)


def sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir",  default="/scratch/ishaan.karan/significance/")
    ap.add_argument("--test_path",   default="data/real/test.jsonl")
    ap.add_argument("--corpus_path", default="data/real/corpus.jsonl")
    ap.add_argument("--two_sided", action="store_true",
                    help="Two-sided tests (default: one-sided 'BAM-PQ > baseline').")
    ap.add_argument("--correction", default="holm",
                    choices=["holm", "bonferroni", "fdr_bh"],
                    help="Multiple-comparisons method across backbones (per metric).")
    args = ap.parse_args()

    per_query_dir = os.path.join(args.output_dir, "per_query")
    os.makedirs(per_query_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Step 1: encode + score each (system, backbone), caching per-query CSV ===
    cache = {}
    aggregates_log = []
    for bb_name, paths in BACKBONES.items():
        for system in ("BAMPQ", "MRL", "StandardFT"):
            csv_path = os.path.join(per_query_dir, f"{system}_{bb_name}.csv")
            if os.path.exists(csv_path):
                print(f"[cached] {system} / {bb_name} -> {csv_path}")
                cache[(system, bb_name)] = pd.read_csv(csv_path)
                continue
            print(f"\n=== Scoring {system} / {bb_name} ===")
            df, agg = score_combo(system, bb_name, paths,
                                  args.test_path, args.corpus_path, device)
            df.to_csv(csv_path, index=False)
            cache[(system, bb_name)] = df
            aggregates_log.append({"system": system, "backbone": bb_name, **agg})

    if aggregates_log:
        agg_df = pd.DataFrame(aggregates_log)
        print("\n--- Aggregate sanity (means across queries) ---")
        print(agg_df.to_string(index=False))

    # === Step 2: paired tests ===
    rows = []
    comparisons = [
        ("BAM-PQ vs MRL",         "BAMPQ", "MRL"),
        ("BAM-PQ vs Standard FT", "BAMPQ", "StandardFT"),
    ]
    for cmp_label, sys_a, sys_b in comparisons:
        for metric in METRICS:
            p_raw_list, delta_list = [], []
            backbone_order = list(BACKBONES.keys())
            for bb in backbone_order:
                a = cache[(sys_a, bb)][metric].values
                b = cache[(sys_b, bb)][metric].values
                if len(a) != len(b):
                    raise RuntimeError(
                        f"Length mismatch for {sys_a} vs {sys_b} on {bb}: "
                        f"{len(a)} vs {len(b)}"
                    )
                d, p = paired_test(a, b, metric, two_sided=args.two_sided)
                p_raw_list.append(p)
                delta_list.append(d)

            # Multiple-comparison correction across the 4 backbones for this metric.
            reject, p_corr, _, _ = multipletests(
                p_raw_list, alpha=0.05, method=args.correction
            )
            for bb, d, p, pc in zip(backbone_order, delta_list, p_raw_list, p_corr):
                rows.append({
                    "comparison":    cmp_label,
                    "metric":        METRIC_LABELS[metric],
                    "backbone":      bb,
                    "delta":         d,
                    "p_raw":         float(p),
                    "p_corrected":   float(pc),
                    "sig":           sig_stars(pc),
                })

    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(args.output_dir, "significance_table.csv")
    out_df.to_csv(out_csv, index=False)

    # === Step 3: pretty-print ===
    print("\n" + "=" * 88)
    test_label = ("two-sided" if args.two_sided else "one-sided")
    print(f"PAIRED SIGNIFICANCE — educational benchmark ({test_label}, "
          f"{args.correction} across backbones)")
    print("=" * 88)
    for cmp_label in out_df["comparison"].unique():
        print(f"\n{cmp_label}")
        sub = out_df[out_df["comparison"] == cmp_label]
        print(f"  {'metric':<8} {'backbone':<10} {'Δ':>10} "
              f"{'p (raw)':>12} {'p (corr)':>12}   sig")
        for _, r in sub.iterrows():
            print(f"  {r['metric']:<8} {r['backbone']:<10} "
                  f"{r['delta']:+.4f}    {r['p_raw']:.2e}    "
                  f"{r['p_corrected']:.2e}   {r['sig']}")

    # === Step 4: LaTeX-ready row dump for Table 3 caption append ===
    tex_path = os.path.join(args.output_dir, "significance_table.tex")
    with open(tex_path, "w") as f:
        f.write("% Generated by scripts/run_significance_edu.py\n")
        f.write("% Paired Wilcoxon (R@10/R@50/NDCG@10), McNemar exact (R@1).\n")
        f.write(f"% Correction: {args.correction}, alpha=0.05, "
                f"{test_label}.\n")
        for _, r in out_df.iterrows():
            f.write(f"% {r['comparison']:<25} {r['metric']:<8} "
                    f"{r['backbone']:<10} Δ={r['delta']:+.4f}  "
                    f"p_corr={r['p_corrected']:.2e}  {r['sig']}\n")
    print(f"\nWrote: {out_csv}")
    print(f"Wrote: {tex_path}")
    print(f"Per-query CSVs:  {per_query_dir}/")


if __name__ == "__main__":
    main()
