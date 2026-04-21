"""
Bloom-Level Dataset Analysis for Balanced Educational Training Data.

Loads question/query texts from a wide range of educational QA datasets,
classifies each with MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
(zero-shot NLI — much better Bloom distribution than cip29 which collapses
to 82% Remember on non-educational queries), and reports
per-dataset Bloom distribution. This tells you which datasets to mix
for a balanced 6-level training set.

Datasets analyzed:
  Already used in this repo:
    - SciQ              (allenai/sciq)
    - ARC-Easy          (allenai/ai2_arc, ARC-Easy)
    - ARC-Challenge     (allenai/ai2_arc, ARC-Challenge)
    - OpenBookQA        (allenai/openbookqa)
    - QASC              (allenai/qasc)

  Additional educational QA:
    - ScienceQA         (derek-thomas/ScienceQA)
    - RACE (middle)     (ehovy/race, middle)
    - RACE (high)       (ehovy/race, high)
    - CommonsenseQA      (tau/commonsense_qa)
    - PIQA              (ybisk/piqa)
    - WinoGrande        (allenai/winogrande, winogrande_xl)
    - BoolQ             (google/boolq)
    - CosmosQA          (cosmos_qa)
    - DREAM             (dataset-org/dream)
    - QuAIL             (text-machine-lab/quail)
    - Jeopardy          (jeopardy)
    - TriviaQA          (trivia_qa, rc.nocontext)
    - NaturalQuestions   (google-research-datasets/natural_questions)
    - MMLU              (cais/mmlu, all)

Usage:
    python scripts/analyze_bloom_datasets.py
    python scripts/analyze_bloom_datasets.py --max_per_dataset 2000 --output results/bloom_analysis.json
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.annotate_bloom_local import load_classifier, classify_batch, BLOOM_NAMES


# ─────────────────────── Dataset Loaders ───────────────────────
# Each returns (dataset_name, list_of_query_strings)

def load_sciq(max_n):
    from datasets import load_dataset
    ds = load_dataset("allenai/sciq", split="train")
    queries = [row["question"] for row in ds if row.get("question")]
    return "SciQ", queries[:max_n]


def load_arc_easy(max_n):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
    queries = [row["question"] for row in ds if row.get("question")]
    return "ARC-Easy", queries[:max_n]


def load_arc_challenge(max_n):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    queries = [row["question"] for row in ds if row.get("question")]
    return "ARC-Challenge", queries[:max_n]


def load_openbookqa(max_n):
    from datasets import load_dataset
    ds = load_dataset("allenai/openbookqa", "main", split="train")
    queries = [row["question_stem"] for row in ds if row.get("question_stem")]
    return "OpenBookQA", queries[:max_n]


def load_qasc(max_n):
    from datasets import load_dataset
    ds = load_dataset("allenai/qasc", split="train")
    queries = [row["question"] for row in ds if row.get("question")]
    return "QASC", queries[:max_n]


def load_scienceqa(max_n):
    from datasets import load_dataset
    try:
        ds = load_dataset("derek-thomas/ScienceQA", split="train", trust_remote_code=True)
    except Exception:
        ds = load_dataset("tasksource/ScienceQA", split="train", trust_remote_code=True)
    queries = [row["question"] for row in ds if row.get("question")]
    return "ScienceQA", queries[:max_n]


def load_race_middle(max_n):
    from datasets import load_dataset
    ds = load_dataset("ehovy/race", "middle", split="train")
    queries = [row["question"] for row in ds if row.get("question")]
    return "RACE-Middle", queries[:max_n]


def load_race_high(max_n):
    from datasets import load_dataset
    ds = load_dataset("ehovy/race", "high", split="train")
    queries = [row["question"] for row in ds if row.get("question")]
    return "RACE-High", queries[:max_n]


def load_commonsenseqa(max_n):
    from datasets import load_dataset
    ds = load_dataset("tau/commonsense_qa", split="train")
    queries = [row["question"] for row in ds if row.get("question")]
    return "CommonsenseQA", queries[:max_n]


def load_piqa(max_n):
    from datasets import load_dataset
    ds = load_dataset("ybisk/piqa", split="train", trust_remote_code=True)
    queries = [row["goal"] for row in ds if row.get("goal")]
    return "PIQA", queries[:max_n]


def load_boolq(max_n):
    from datasets import load_dataset
    ds = load_dataset("google/boolq", split="train")
    queries = [row["question"] for row in ds if row.get("question")]
    return "BoolQ", queries[:max_n]


def load_cosmosqa(max_n):
    from datasets import load_dataset
    ds = load_dataset("cosmos_qa", split="train", trust_remote_code=True)
    queries = [row["question"] for row in ds if row.get("question")]
    return "CosmosQA", queries[:max_n]


def load_dream(max_n):
    from datasets import load_dataset
    ds = load_dataset("dataset-org/dream", split="train")
    queries = [row["question"] for row in ds if row.get("question")]
    return "DREAM", queries[:max_n]


def load_quail(max_n):
    from datasets import load_dataset
    ds = load_dataset("text-machine-lab/quail", split="train")
    queries = [row["question"] for row in ds if row.get("question")]
    return "QuAIL", queries[:max_n]


def load_triviaqa(max_n):
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.nocontext", split="train")
    queries = [row["question"] for row in ds if row.get("question")]
    return "TriviaQA", queries[:max_n]


def load_mmlu(max_n):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    queries = [row["question"] for row in ds if row.get("question")]
    return "MMLU", queries[:max_n]


def load_winogrande(max_n):
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="train")
    queries = [row["sentence"] for row in ds if row.get("sentence")]
    return "WinoGrande", queries[:max_n]


ALL_LOADERS = [
    # Already used
    load_sciq, load_arc_easy, load_arc_challenge, load_openbookqa, load_qasc,
    # Additional educational
    load_scienceqa, load_race_middle, load_race_high,
    load_commonsenseqa, load_piqa, load_boolq, load_cosmosqa,
    load_dream, load_quail, load_triviaqa, load_mmlu, load_winogrande,
]


# ─────────────────────── Analysis ───────────────────────

def analyze_dataset(name: str, queries: List[str], clf) -> dict:
    """Classify queries and return Bloom distribution stats."""
    if not queries:
        return {"name": name, "total": 0, "distribution": {}, "percentages": {}}

    levels = classify_batch(queries, clf)
    dist = Counter(levels)

    pcts = {}
    for b in range(1, 7):
        pcts[BLOOM_NAMES[b]] = dist.get(b, 0) / len(levels) * 100

    return {
        "name": name,
        "total": len(queries),
        "distribution": {BLOOM_NAMES[b]: dist.get(b, 0) for b in range(1, 7)},
        "percentages": pcts,
        "samples": {BLOOM_NAMES[b]: [] for b in range(1, 7)},
        "_levels": levels,
        "_queries": queries,
    }


def collect_samples(result: dict, n_samples: int = 3):
    """Collect example queries per Bloom level for inspection."""
    levels = result.pop("_levels", [])
    queries = result.pop("_queries", [])
    by_level = defaultdict(list)
    for q, l in zip(queries, levels):
        by_level[l].append(q)
    for b in range(1, 7):
        result["samples"][BLOOM_NAMES[b]] = by_level[b][:n_samples]


def print_report(results: List[dict]):
    """Print a readable comparison table."""
    print("\n" + "=" * 110)
    print("BLOOM LEVEL DISTRIBUTION ACROSS EDUCATIONAL DATASETS")
    print("=" * 110)

    header = f"{'Dataset':20s} {'Total':>7s}"
    for b in range(1, 7):
        header += f" {BLOOM_NAMES[b]:>12s}"
    header += f" {'Entropy':>9s}"
    print(header)
    print("-" * 110)

    import math
    for r in results:
        if r["total"] == 0:
            continue
        row = f"{r['name']:20s} {r['total']:>7d}"
        pcts = r["percentages"]
        entropy = 0.0
        for b in range(1, 7):
            p = pcts[BLOOM_NAMES[b]] / 100
            row += f" {pcts[BLOOM_NAMES[b]]:>10.1f}%"
            if p > 0:
                entropy -= p * math.log2(p)
        # Entropy: max = log2(6) ≈ 2.585 for perfectly balanced
        row += f" {entropy:>8.3f}"
        print(row)

    print("-" * 110)
    print(f"  Entropy: higher = more balanced (max {math.log2(6):.3f} for uniform)")

    # Identify best sources per level
    print("\n" + "=" * 110)
    print("BEST SOURCES PER BLOOM LEVEL (highest % for that level)")
    print("=" * 110)
    for b in range(1, 7):
        bname = BLOOM_NAMES[b]
        ranked = sorted(
            [(r["name"], r["percentages"].get(bname, 0), r["distribution"].get(bname, 0))
             for r in results if r["total"] > 0],
            key=lambda x: -x[1]
        )
        print(f"\n  {bname} (level {b}):")
        for name, pct, count in ranked[:5]:
            print(f"    {name:20s}  {pct:5.1f}%  ({count:>5d} queries)")

    # Suggest a mix
    print("\n" + "=" * 110)
    print("SUGGESTED MIX FOR BALANCED DATASET")
    print("=" * 110)
    print("  Strategy: for each Bloom level, pick the dataset(s) with the")
    print("  highest absolute count at that level. Then compute how many")
    print("  queries to sample from each dataset to hit a target per-level count.\n")

    # Find which dataset contributes most queries at each level
    target_per_level = 2000  # reasonable default
    print(f"  Target: ~{target_per_level} queries per Bloom level ({target_per_level * 6} total)\n")

    # Greedy allocation
    allocation = defaultdict(lambda: defaultdict(int))  # dataset -> level -> count
    for b in range(1, 7):
        bname = BLOOM_NAMES[b]
        # Rank datasets by absolute count at this level
        sources = sorted(
            [(r["name"], r["distribution"].get(bname, 0), r["total"],
              r["distribution"].get(bname, 0) / r["total"] if r["total"] > 0 else 0)
             for r in results if r["total"] > 0 and r["distribution"].get(bname, 0) > 0],
            key=lambda x: -x[1]
        )
        remaining = target_per_level
        for name, count, total, frac in sources:
            if remaining <= 0:
                break
            take = min(remaining, count)
            allocation[name][b] = take
            remaining -= take

        if remaining > 0:
            print(f"  WARNING: {bname} — only {target_per_level - remaining}/{target_per_level}"
                  f" available across all datasets")

    print(f"  {'Dataset':20s} {'Total':>7s}", end="")
    for b in range(1, 7):
        print(f" {BLOOM_NAMES[b]:>10s}", end="")
    print()
    print("  " + "-" * 90)
    for name in sorted(allocation.keys()):
        levels = allocation[name]
        total = sum(levels.values())
        print(f"  {name:20s} {total:>7d}", end="")
        for b in range(1, 7):
            print(f" {levels.get(b, 0):>10d}", end="")
        print()

    grand_total = sum(sum(v.values()) for v in allocation.values())
    print("  " + "-" * 90)
    print(f"  {'TOTAL':20s} {grand_total:>7d}", end="")
    for b in range(1, 7):
        col_total = sum(allocation[name].get(b, 0) for name in allocation)
        print(f" {col_total:>10d}", end="")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Bloom-level distribution across educational datasets")
    parser.add_argument("--max_per_dataset", type=int, default=5000,
                        help="Max queries to sample per dataset (default 5000)")
    parser.add_argument("--output", default=None,
                        help="Save JSON results (default: print only)")
    parser.add_argument("--target_per_level", type=int, default=2000,
                        help="Target queries per Bloom level for mix suggestion")
    parser.add_argument("--device", default=None,
                        help="Device: 'cuda', 'cpu', or int GPU index. Auto-detected if omitted.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Inference batch size (reduce if OOM)")
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = int(args.device) if args.device.isdigit() else args.device

    print("=" * 70)
    print("BLOOM DATASET ANALYSIS")
    print(f"Classifier: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    print(f"Device: {device}")
    print(f"Max per dataset: {args.max_per_dataset}")
    print("=" * 70)

    clf = load_classifier("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", device)

    results = []
    for loader in ALL_LOADERS:
        try:
            name, queries = loader(args.max_per_dataset)
            print(f"\n  {name}: {len(queries)} queries loaded")
            r = analyze_dataset(name, queries, clf)
            collect_samples(r, n_samples=3)
            results.append(r)
        except Exception as e:
            fname = loader.__name__.replace("load_", "")
            print(f"\n  {fname}: FAILED — {e}")
            results.append({"name": fname, "total": 0,
                            "distribution": {}, "percentages": {}, "samples": {}})

    print_report(results)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        # Strip non-serializable fields
        for r in results:
            r.pop("_levels", None)
            r.pop("_queries", None)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
