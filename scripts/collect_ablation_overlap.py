"""
Compute per-variant Bloom-mask dimension-overlap diagnostics for the loss ablations.

For each variant's best_bsr checkpoint, this:
  - Encodes all test queries
  - Computes the per-Bloom mean hard mask
  - Computes the 6x6 pairwise dimension-overlap matrix
    (cell (i,j) = fraction of level i's active dims also active at level j)
  - Reports per-Bloom active dim counts + dim spread (max-min)
  - Reports mean off-diagonal overlap, min/max, Remember<->Evaluate overlap
  - Also outputs mean off-diagonal cosine similarity (for completeness)

Output:
  results/ablations/overlap_summary.json  — full matrices per variant
  Console comparison table                — one row per variant

Usage:
  python3 scripts/collect_ablation_overlap.py \
      --ckpt_root   /scratch/.../bampq-checkpoints/loss_ablations \
      --config_root configs/ablations \
      --output      results/ablations/overlap_summary.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config                              # noqa: E402
from models.bam import BloomAlignedMRL                          # noqa: E402

BLOOM_NAMES = ["Remember", "Understand", "Apply",
               "Analyze", "Evaluate", "Create"]

VARIANTS = [
    "full",
    "no_sparsity",
    "no_diversity",
    "no_variance",
    "no_distill",
    "no_query_div",
    "contrastive_only",
]


@torch.no_grad()
def compute_overlap(model, config, tokenizer, device):
    samples = []
    with open(config["data"]["test_path"]) as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    bloom_hard_masks = {b: [] for b in range(6)}
    for i in range(0, len(samples), 64):
        batch = samples[i:i + 64]
        enc = tokenizer([s["query"] for s in batch], padding=True,
                        truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        bloom_labels = torch.tensor(
            [s["bloom_level"] - 1 for s in batch],
            dtype=torch.long, device=device,
        )
        out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                   bloom_labels=bloom_labels)
        hard = (out["mask"] > 0.5).float().cpu()
        for j, bl in enumerate(bloom_labels.cpu().tolist()):
            bloom_hard_masks[bl].append(hard[j])

    mean_hard = {}
    dims_per_level = {}
    for b in range(6):
        if bloom_hard_masks[b]:
            mh = torch.stack(bloom_hard_masks[b]).mean(dim=0)
            mean_hard[b] = mh
            dims_per_level[b] = int((mh > 0.5).sum().item())

    levels = sorted(mean_hard.keys())
    n = len(levels)

    # Overlap (asymmetric): cell (i,j) = |Ai ∩ Aj| / |Ai|
    overlap = np.zeros((n, n))
    for i, bi in enumerate(levels):
        ai = (mean_hard[bi] > 0.5).float()
        denom = max(1.0, ai.sum().item())
        for j, bj in enumerate(levels):
            aj = (mean_hard[bj] > 0.5).float()
            overlap[i, j] = float((ai * aj).sum().item()) / denom

    # Cosine similarity on the mean hard masks (for completeness)
    means = torch.stack([mean_hard[b] for b in levels])
    normed = F.normalize(means, p=2, dim=-1)
    cos_sim = torch.mm(normed, normed.t()).numpy()

    off_overlap = [overlap[i, j] for i in range(n) for j in range(n) if i != j]
    off_cos     = [cos_sim[i, j] for i in range(n) for j in range(n) if i != j]

    idx = {b: i for i, b in enumerate(levels)}
    rem_eval = float(overlap[idx[0], idx[4]]) if 0 in idx and 4 in idx else None
    eval_rem = float(overlap[idx[4], idx[0]]) if 0 in idx and 4 in idx else None

    return {
        "active_dims_per_level": {BLOOM_NAMES[b]: dims_per_level.get(b, 0)
                                   for b in range(6)},
        "dim_min":     int(min(dims_per_level.values())) if dims_per_level else 0,
        "dim_max":     int(max(dims_per_level.values())) if dims_per_level else 0,
        "dim_spread":  int(max(dims_per_level.values()) - min(dims_per_level.values()))
                       if dims_per_level else 0,
        "mean_off_diag_overlap": float(np.mean(off_overlap)),
        "min_overlap":           float(min(off_overlap)),
        "max_overlap":           float(max(off_overlap)),
        "remember_evaluate_overlap": rem_eval,
        "evaluate_remember_overlap": eval_rem,
        "mean_off_diag_cosine":  float(np.mean(off_cos)),
        "overlap_matrix":        overlap.tolist(),
        "cosine_matrix":         cos_sim.tolist(),
        "levels":                [BLOOM_NAMES[b] for b in levels],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_root",   default="/tmp/bam-pq-ckpts",
                        help="Parent dir containing abl_<variant>/best_bsr/ checkpoints")
    parser.add_argument("--config_root", default="configs/ablations",
                        help="Dir with abl_<variant>.yaml configs")
    parser.add_argument("--output",      default="results/ablations/overlap_summary.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    summary = {}
    for variant in VARIANTS:
        ckpt_path = os.path.join(args.ckpt_root, f"abl_{variant}", "best_bsr")
        cfg_path  = os.path.join(args.config_root, f"abl_{variant}.yaml")
        ckpt_file = os.path.join(ckpt_path, "checkpoint.pt")

        if not (os.path.exists(ckpt_file) and os.path.exists(cfg_path)):
            print(f"\n=== {variant} === SKIP — missing files")
            print(f"  ckpt: {ckpt_file}")
            print(f"  cfg : {cfg_path}")
            continue

        print(f"\n=== {variant} ===")
        config = load_config(cfg_path)
        config["training"]["loss"].setdefault("bloom_frequencies", [1 / 6] * 6)
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])

        model = BloomAlignedMRL(config)
        ckpt = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.to(device).eval()

        if not model.use_mask_routing:
            print(f"  [{variant}] not Option B (use_mask_routing=False) — skip")
            del model, ckpt
            continue

        try:
            result = compute_overlap(model, config, tokenizer, device)
        finally:
            del model, ckpt
            if device.type == "cuda":
                torch.cuda.empty_cache()

        summary[variant] = result

        dims = result["active_dims_per_level"]
        print(f"  Active dims/level: " + "  ".join(
            f"{name[:5]}={dims[name]}" for name in BLOOM_NAMES))
        print(f"  Dim spread (max-min): {result['dim_spread']}")
        print(f"  Mean off-diag overlap: {result['mean_off_diag_overlap']:.3f}")
        print(f"  Mean off-diag cosine:  {result['mean_off_diag_cosine']:.3f}")
        if result['remember_evaluate_overlap'] is not None:
            print(f"  Remember<->Evaluate overlap: "
                  f"{result['remember_evaluate_overlap']*100:.1f}% / "
                  f"{result['evaluate_remember_overlap']*100:.1f}%")

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {args.output}")

    # ── Comparison table ────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("ABLATION MASK SPECIALIZATION — DIMENSION OVERLAP")
    print("=" * 90)
    hdr = (f"{'Variant':<22} {'Dim range':>12} {'Spread':>7} "
           f"{'MeanOverlap':>12} {'MeanCos':>9} {'Rem->Eval':>11} {'Eval->Rem':>11}")
    print(hdr)
    print("-" * len(hdr))
    for v in VARIANTS:
        if v not in summary:
            print(f"{v:<22} (not run)")
            continue
        r = summary[v]
        dim_range = f"{r['dim_min']}-{r['dim_max']}"
        spread = r['dim_spread']
        mo = r['mean_off_diag_overlap']
        mc = r['mean_off_diag_cosine']
        re_ov = r['remember_evaluate_overlap']
        er_ov = r['evaluate_remember_overlap']
        re_s = f"{re_ov*100:.1f}%" if re_ov is not None else "  n/a"
        er_s = f"{er_ov*100:.1f}%" if er_ov is not None else "  n/a"
        print(f"{v:<22} {dim_range:>12} {spread:>7} "
              f"{mo:>12.3f} {mc:>9.3f} {re_s:>11} {er_s:>11}")

    # ── LaTeX snippet ───────────────────────────────────────────────────────
    latex_path = os.path.join(os.path.dirname(args.output), "overlap_table.tex")
    with open(latex_path, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\\small\n")
        f.write("\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n")
        f.write("Variant & Dim range & Spread & "
                "Mean overlap & Rem$\\to$Eval & Eval$\\to$Rem \\\\\n\\midrule\n")
        for v in VARIANTS:
            v_tex = v.replace("_", "\\_")
            if v not in summary:
                f.write(v_tex + " & -- & -- & -- & -- & -- \\\\\n")
                continue
            r = summary[v]
            dim_range = f"{r['dim_min']}--{r['dim_max']}"
            mo = r['mean_off_diag_overlap']
            re_ov = r['remember_evaluate_overlap']
            er_ov = r['evaluate_remember_overlap']
            re_s = f"{re_ov*100:.1f}\\%" if re_ov is not None else "--"
            er_s = f"{er_ov*100:.1f}\\%" if er_ov is not None else "--"
            f.write(f"{v_tex} & {dim_range} & {r['dim_spread']} & "
                    f"{mo*100:.1f}\\% & {re_s} & {er_s} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Mask specialisation diagnostics across loss ablations. "
                "\\textit{Dim range / Spread}: per-Bloom-level active dimension "
                "count and its max$-$min spread (wide = differentiated routing). "
                "\\textit{Mean overlap}: mean off-diagonal pairwise dimension "
                "overlap across the 6 Bloom levels (lower = more specialised). "
                "\\textit{Rem$\\to$Eval / Eval$\\to$Rem}: overlap between the "
                "most cognitively distant pair (lower = more specialised).}\n")
        f.write("\\label{tab:ablation-overlap}\n")
        f.write("\\end{table}\n")
    print(f"LaTeX table -> {latex_path}")


if __name__ == "__main__":
    main()
