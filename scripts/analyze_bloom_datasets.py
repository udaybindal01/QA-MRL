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


# ─────────────── Evaluate-heavy datasets ───────────────

def load_arg_quality(max_n):
    """IBM argument quality ranking — "is this argument convincing?" = Evaluate."""
    from datasets import load_dataset
    ds = load_dataset("ibm/argument_quality_ranking_30k", "argument_quality_ranking",
                      split="train")
    queries = []
    for row in ds:
        arg = row.get("argument", "").strip()
        topic = row.get("topic", "").strip()
        if arg and topic:
            queries.append(f"Evaluate the strength of this argument about {topic}: {arg}")
        if len(queries) >= max_n:
            break
    return "ArgQuality-30k", queries[:max_n]


def load_prosocial(max_n):
    """Prosocial dialog — evaluate social behavior, judge appropriateness."""
    from datasets import load_dataset
    ds = load_dataset("allenai/prosocial-dialog", split="train")
    queries = []
    for row in ds:
        ctx = row.get("context", "").strip()
        rots = row.get("rots", [])
        if ctx and rots:
            # Rules-of-thumb are moral judgments — frame as evaluation
            queries.append(f"Evaluate the appropriateness of this behavior: {ctx[:200]}")
        if len(queries) >= max_n:
            break
    return "ProsocialDialog", queries[:max_n]


def load_strategyqa(max_n):
    """StrategyQA — multi-hop yes/no requiring strategic judgment."""
    from datasets import load_dataset
    ds = load_dataset("ChilleD/StrategyQA", split="train")
    queries = [row["question"].strip() for row in ds if row.get("question")]
    return "StrategyQA", queries[:max_n]


# ─────────────── Create-heavy datasets ───────────────

def load_writingprompts(max_n):
    """Reddit WritingPrompts — creative writing tasks = Create."""
    from datasets import load_dataset
    ds = load_dataset("euclaise/writingprompts", split="train", trust_remote_code=True)
    queries = []
    for row in ds:
        prompt = row.get("prompt", "").strip()
        if prompt and len(prompt) > 20:
            queries.append(prompt[:300])
        if len(queries) >= max_n:
            break
    return "WritingPrompts", queries[:max_n]


def load_dolly_creative(max_n):
    """Dolly brainstorming + creative_writing categories → Create."""
    from datasets import load_dataset
    ds = load_dataset("argilla/databricks-dolly-15k-curated-en", split="train")
    queries = []
    for row in ds:
        cat = row.get("category", "")
        instr = row.get("original-instruction", "").strip()
        if cat in ("brainstorming", "creative_writing") and instr:
            queries.append(instr)
        if len(queries) >= max_n:
            break
    return "Dolly-Creative", queries[:max_n]


def load_dolly_all(max_n):
    """Dolly full dataset — all categories for distribution analysis."""
    from datasets import load_dataset
    ds = load_dataset("argilla/databricks-dolly-15k-curated-en", split="train")
    queries = [row["original-instruction"].strip() for row in ds
               if row.get("original-instruction", "").strip()]
    return "Dolly-All", queries[:max_n]


def load_alpaca(max_n):
    """Alpaca instructions — diverse tasks including design, create, generate."""
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    queries = []
    for row in ds:
        instr = row.get("instruction", "").strip()
        inp = row.get("input", "").strip()
        if instr:
            q = f"{instr} {inp}" if inp else instr
            queries.append(q[:300])
        if len(queries) >= max_n:
            break
    return "Alpaca", queries[:max_n]


def load_alpaca_create(max_n):
    """Alpaca — filtered to instructions with create/design/generate/write verbs."""
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    create_verbs = {"create", "design", "generate", "write", "compose", "develop",
                    "propose", "invent", "construct", "formulate", "draft", "build",
                    "produce", "synthesize", "imagine", "devise"}
    queries = []
    for row in ds:
        instr = row.get("instruction", "").strip()
        if instr:
            first_word = instr.split()[0].lower().rstrip(".,!?")
            if first_word in create_verbs:
                inp = row.get("input", "").strip()
                q = f"{instr} {inp}" if inp else instr
                queries.append(q[:300])
        if len(queries) >= max_n:
            break
    return "Alpaca-Create", queries[:max_n]


def load_alpaca_evaluate(max_n):
    """Alpaca — filtered to instructions with evaluate/judge/assess/critique verbs."""
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    eval_verbs = {"evaluate", "judge", "assess", "critique", "rate", "rank",
                  "compare", "review", "justify", "defend", "argue", "debate",
                  "determine", "decide", "weigh"}
    queries = []
    for row in ds:
        instr = row.get("instruction", "").strip()
        if instr:
            first_word = instr.split()[0].lower().rstrip(".,!?")
            if first_word in eval_verbs:
                inp = row.get("input", "").strip()
                q = f"{instr} {inp}" if inp else instr
                queries.append(q[:300])
        if len(queries) >= max_n:
            break
    return "Alpaca-Evaluate", queries[:max_n]


def load_flan_v2(max_n):
    """FLAN v2 — diverse NLP tasks, broad Bloom coverage."""
    from datasets import load_dataset
    ds = load_dataset("SirNeural/flan_v2", split="train", streaming=True)
    queries = []
    for row in ds:
        text = row.get("inputs", "").strip()
        if text and 20 < len(text) < 500:
            queries.append(text[:300])
        if len(queries) >= max_n:
            break
    return "FLAN-v2", queries[:max_n]


def load_oasst_prompts(max_n):
    """OpenAssistant — human prompts (first messages) span all Bloom levels."""
    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    queries = []
    for row in ds:
        if row.get("role") == "prompter" and row.get("parent_id") is None:
            text = row.get("text", "").strip()
            if text and len(text) > 15:
                queries.append(text[:300])
        if len(queries) >= max_n:
            break
    return "OASST-Prompts", queries[:max_n]


ALL_LOADERS = [
    # Already used in repo
    load_sciq, load_arc_easy, load_arc_challenge, load_openbookqa, load_qasc,
    # Additional educational QA
    load_scienceqa, load_race_middle, load_race_high,
    load_commonsenseqa, load_piqa, load_boolq, load_cosmosqa,
    load_dream, load_quail, load_triviaqa, load_mmlu, load_winogrande,
    # Evaluate-heavy
    load_arg_quality, load_prosocial, load_strategyqa,
    # Create-heavy
    load_writingprompts, load_dolly_creative, load_dolly_all,
    load_alpaca, load_alpaca_create, load_alpaca_evaluate,
    # Broad coverage
    load_flan_v2, load_oasst_prompts,
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
