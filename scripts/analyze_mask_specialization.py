"""
Analyze whether BAM v4 Option B learns genuinely different dimension subsets per Bloom level.

Computes:
  - Per-Bloom average hard mask (mean over all test queries at that level)
  - Pairwise cosine similarity between Bloom-level average masks
  - Pairwise overlap: % of dims shared between level pairs
  - 6×6 heatmap of mask similarity

Low similarity between levels (e.g. Remember vs Evaluate) is qualitative evidence
that Option B learns genuinely different feature subsets per cognitive level.

Output:
  mask_specialization.pdf         — 6×6 heatmap of cosine similarity
  mask_specialization_table.json  — quantitative results for paper

Usage:
    python scripts/analyze_mask_specialization.py \
        --config configs/bam_v4.yaml \
        --checkpoint /tmp/bam-v4-ckpts/best/ \
        --output_dir results/analysis/
"""

import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from models.bam import BloomAlignedMRL
from transformers import AutoTokenizer

BLOOM_NAMES = {0: "Remember", 1: "Understand", 2: "Apply",
               3: "Analyze", 4: "Evaluate", 5: "Create"}
BLOOM_DISPLAY = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]


@torch.no_grad()
def run_analysis(args):
    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])
    os.makedirs(args.output_dir, exist_ok=True)

    config["training"]["loss"]["bloom_frequencies"] = [1/6] * 6
    model = BloomAlignedMRL(config)

    if not model.use_mask_routing:
        print("ERROR: This script requires a model trained with use_mask_routing=True (Option B).")
        print("       Use --config configs/bam_v4.yaml")
        return

    ckpt = os.path.join(args.checkpoint, "checkpoint.pt")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"], strict=False)
        print(f"Loaded from {ckpt}")
    model.to(device).eval()

    # Load test queries
    samples = []
    with open(config["data"]["test_path"]) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    print(f"Test queries: {len(samples)}")

    # Collect masks per Bloom level
    bloom_hard_masks = {b: [] for b in range(6)}
    bloom_soft_masks = {b: [] for b in range(6)}

    for i in tqdm(range(0, len(samples), 64), desc="Encoding queries"):
        batch = samples[i:i + 64]
        enc = tokenizer([s["query"] for s in batch], padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        bloom_labels = torch.tensor(
            [s["bloom_level"] - 1 for s in batch],
            dtype=torch.long, device=device,
        )
        out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                   bloom_labels=bloom_labels)
        hard = (out["mask"] > 0.5).float().cpu()
        soft = out["soft_mask"].cpu()
        for j, bl in enumerate(bloom_labels.cpu().tolist()):
            bloom_hard_masks[bl].append(hard[j])
            bloom_soft_masks[bl].append(soft[j])

    # Compute per-Bloom mean masks
    mean_hard = {}
    mean_soft = {}
    n_per_level = {}
    for b in range(6):
        if bloom_hard_masks[b]:
            mean_hard[b] = torch.stack(bloom_hard_masks[b]).mean(dim=0)  # [768]
            mean_soft[b] = torch.stack(bloom_soft_masks[b]).mean(dim=0)
            n_per_level[b] = len(bloom_hard_masks[b])

    present = sorted(mean_hard.keys())
    if len(present) < 2:
        print("ERROR: Fewer than 2 Bloom levels present in test data. Cannot compute pairwise stats.")
        return

    names = [BLOOM_DISPLAY[b] for b in present]
    L = len(present)

    # Pairwise cosine similarity on mean soft masks
    means_t = torch.stack([mean_soft[b] for b in present])  # [L, 768]
    normed = F.normalize(means_t, p=2, dim=-1)
    sim_matrix = torch.mm(normed, normed.t()).numpy()  # [L, L]

    # Pairwise overlap: % of dims shared (both active) out of level i's active dims
    overlap_matrix = np.zeros((L, L))
    for i, bi in enumerate(present):
        for j, bj in enumerate(present):
            mi = (mean_hard[bi] > 0.5).float()
            mj = (mean_hard[bj] > 0.5).float()
            shared = (mi * mj).sum().item()
            total_i = mi.sum().clamp(min=1).item()
            overlap_matrix[i, j] = shared / total_i

    # Mean active dims per level
    active_dims = {b: float((mean_hard[b] > 0.5).sum().item()) for b in present}

    # Summary stats
    off_diag_sim = []
    for i in range(L):
        for j in range(L):
            if i != j:
                off_diag_sim.append(sim_matrix[i, j])
    mean_off_diag_sim = float(np.mean(off_diag_sim))

    # Print results
    print("\n=== Mask Specialization (Option B) ===")
    print(f"\nActive dims per Bloom level (out of 768):")
    for b in present:
        print(f"  {BLOOM_DISPLAY[b]:12s}: {active_dims[b]:.0f}  ({active_dims[b]/768*100:.1f}%)")

    print(f"\nPairwise cosine similarity (mean soft masks):")
    header = f"{'':14s}" + "".join(f"{n:>12s}" for n in names)
    print(header)
    print("-" * len(header))
    for i, ni in enumerate(names):
        row = f"{ni:14s}"
        for j in range(L):
            row += f"{sim_matrix[i, j]:>12.3f}"
        print(row)

    print(f"\nPairwise overlap (row_level's dims that are also active in col_level):")
    print(header)
    print("-" * len(header))
    for i, ni in enumerate(names):
        row = f"{ni:14s}"
        for j in range(L):
            row += f"{overlap_matrix[i, j]:>12.1%}"
        print(row)

    print(f"\nMean off-diagonal cosine similarity: {mean_off_diag_sim:.3f}")
    print("(lower = more specialized; 0 = fully orthogonal, 1 = identical)")

    # Save
    result = {
        "levels": names,
        "n_per_level": {BLOOM_DISPLAY[b]: n_per_level.get(b, 0) for b in present},
        "active_dims": {BLOOM_DISPLAY[b]: active_dims[b] for b in present},
        "cosine_similarity": sim_matrix.tolist(),
        "overlap_matrix": overlap_matrix.tolist(),
        "mean_off_diagonal_cosine_similarity": mean_off_diag_sim,
    }
    out_path = os.path.join(args.output_dir, "mask_specialization_table.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Plot heatmap
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, matrix, title, fmt in [
            (axes[0], sim_matrix, "Cosine Similarity\n(mean soft masks per level)", ".2f"),
            (axes[1], overlap_matrix, "Mask Overlap\n(% of row level's dims in col level)", ".0%"),
        ]:
            im = ax.imshow(matrix, cmap="RdYlGn_r" if "Overlap" in title else "coolwarm_r",
                           vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(L)); ax.set_xticklabels(names, rotation=45, ha="right")
            ax.set_yticks(range(L)); ax.set_yticklabels(names)
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
            for i in range(L):
                for j in range(L):
                    val = matrix[i, j]
                    ax.text(j, i, f"{val:{fmt}}", ha="center", va="center", fontsize=8,
                            color="white" if abs(val - 0.5) > 0.3 else "black")

        plt.suptitle("BAM v4 Option B: Mask Specialization per Bloom Level", fontsize=13)
        plt.tight_layout()
        pdf_path = os.path.join(args.output_dir, "mask_specialization.pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved heatmap to {pdf_path}")
        plt.close()
    except ImportError:
        print("matplotlib not available — skipping heatmap")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bam_v4.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="results/analysis/")
    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
