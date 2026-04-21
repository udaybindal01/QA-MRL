"""
Local Bloom Taxonomy Annotation using zero-shot NLI (no API key needed).

Uses MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli — one of the strongest
zero-shot NLI models on HuggingFace. Replaces the educational BERT classifier
(cip29/bert-blooms-taxonomy-classifier) which collapses to 82% "Remember" on BEIR
keyword queries because it was trained only on Kaggle educational data.

Zero-shot NLI frames each Bloom level as a natural-language hypothesis and scores
entailment probability, picking the highest-scoring level. No fine-tuning needed.

Outputs:
  - Updated JSONL with new bloom_level field
  - Updated .bloom_cache.json (used by EducationalRetrievalDataset at train time)

Usage:
    # Annotate a single file
    python data/annotate_bloom_local.py --input /tmp/data/beir/nfcorpus/train.jsonl

    # Annotate all BEIR splits
    python data/annotate_bloom_local.py \
        --beir_root /tmp/data/beir \
        --datasets scifact nfcorpus fiqa

    # Use a lighter/faster model
    python data/annotate_bloom_local.py \
        --beir_root /tmp/data/beir \
        --model facebook/bart-large-mnli
"""

import argparse
import json
import os
import sys
from collections import Counter
from typing import List

BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply",
               4: "Analyze",  5: "Evaluate",   6: "Create"}

# Hypothesis templates for each Bloom level.
# Phrased as "This query is about X" — works well with NLI entailment framing.
# Kept concrete and distinct to avoid NLI model conflating adjacent levels.
BLOOM_HYPOTHESES = [
    "This query is asking to recall or retrieve a specific fact, name, or definition.",                    # 1 Remember
    "This query is asking to explain, describe, or summarize how something works.",                        # 2 Understand
    "This query is asking how to use or apply knowledge to solve a practical problem.",                    # 3 Apply
    "This query is asking to compare, contrast, or examine the relationship between things.",              # 4 Analyze
    "This query requires making a decision or forming an opinion about the worth or validity of something.",  # 5 Evaluate
    "This query is asking to design, propose, or synthesize something new.",                               # 6 Create
]

# Model options:
#   Strong  : MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (~440M, best quality)
#   Balanced: cross-encoder/nli-deberta-v3-base (~180M, good speed/quality)
#   Fast    : facebook/bart-large-mnli (~400M, widely used, slightly weaker)
DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def load_classifier(model_name: str, device):
    """Load zero-shot classification pipeline. device: 'cpu', 'cuda', or int GPU index."""
    from transformers import pipeline
    print(f"  Loading model: {model_name} on {device}...")
    clf = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
        batch_size=32,
    )
    print("  Model loaded.")
    return clf


def classify_batch(queries: List[str], clf, batch_size: int = 64) -> List[int]:
    """
    Classify a list of queries into Bloom levels 1-6 using zero-shot NLI.
    Returns 1-indexed Bloom levels.
    """
    levels = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        results = clf(
            batch,
            candidate_labels=BLOOM_HYPOTHESES,
            hypothesis_template="{}",   # hypotheses are already full sentences
            multi_label=False,
        )
        # results is a list of dicts when batch > 1, single dict when batch == 1
        if isinstance(results, dict):
            results = [results]
        for res in results:
            # res["labels"] are ordered by score descending; find which hypothesis won
            top_label = res["labels"][0]
            level = BLOOM_HYPOTHESES.index(top_label) + 1   # 1-indexed
            levels.append(level)

        if (i // batch_size) % 10 == 0:
            print(f"    {min(i + batch_size, len(queries))}/{len(queries)}", end="\r", flush=True)

    return levels


def annotate_file(input_path: str, clf, batch_size: int = 64,
                  overwrite: bool = False) -> dict:
    """
    Annotate all queries in a JSONL file.
    Updates bloom_level in-place and writes .bloom_cache.json alongside.
    Returns a Counter of old vs new distribution.
    """
    cache_path = input_path + ".bloom_cache.json"

    samples = [json.loads(l) for l in open(input_path)]
    queries = [s["query"] for s in samples]
    old_dist = Counter(s.get("bloom_level", 0) for s in samples)

    if os.path.exists(cache_path) and not overwrite:
        with open(cache_path) as f:
            cached = json.load(f)
        if len(cached) == len(samples):
            print(f"  Cache exists ({len(cached)} entries) — skipping. "
                  f"Use --overwrite to re-annotate.")
            new_dist = Counter(b + 1 for b in cached)
            return {"old": old_dist, "new": new_dist, "skipped": True}

    print(f"  Classifying {len(queries)} queries...")
    all_levels = classify_batch(queries, clf, batch_size)
    print(f"    Done: {len(all_levels)} queries classified.       ")

    for s, level in zip(samples, all_levels):
        s["bloom_level"] = level

    with open(input_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    # 0-indexed cache matching EducationalRetrievalDataset
    cache = [l - 1 for l in all_levels]
    with open(cache_path, "w") as f:
        json.dump(cache, f)

    new_dist = Counter(all_levels)
    return {"old": old_dist, "new": new_dist, "skipped": False}


def print_distribution(name: str, old: Counter, new: Counter):
    print(f"\n  {name}:")
    print(f"    {'Level':12s}  {'Old':>6s}  {'New':>6s}")
    print(f"    {'-'*30}")
    total_new = max(sum(new.values()), 1)
    for level in range(1, 7):
        o = old.get(level, 0)
        n = new.get(level, 0)
        pct = 100 * n / total_new
        bar = "█" * int(pct / 5)
        print(f"    {BLOOM_NAMES[level]:12s}  {o:>6d}  {n:>6d}  {pct:5.1f}%  {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None,
                        help="Single JSONL file to annotate")
    parser.add_argument("--beir_root", default=None,
                        help="Root dir of BEIR data (annotates train+val+test for each dataset)")
    parser.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus", "fiqa"],
                        help="BEIR datasets to annotate")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="HuggingFace zero-shot NLI model")
    parser.add_argument("--device", default=None,
                        help="Device: 'cuda', 'cpu', or int GPU index. Auto-detected if omitted.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Inference batch size (reduce if OOM)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-annotate even if cache exists")
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        try:
            import torch
            device = 0 if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # Install transformers if missing
    try:
        import transformers  # noqa
    except ImportError:
        print("Installing transformers...")
        os.system("pip install transformers -q")

    clf = load_classifier(args.model, device)

    files_to_annotate = []
    if args.input:
        files_to_annotate.append(args.input)

    if args.beir_root:
        for ds in args.datasets:
            for split in ["train", "val", "test"]:
                path = os.path.join(args.beir_root, ds, f"{split}.jsonl")
                if os.path.exists(path):
                    files_to_annotate.append(path)
                else:
                    print(f"  Skipping {path} (not found)")

    if not files_to_annotate:
        print("No files to annotate. Use --input or --beir_root.")
        sys.exit(1)

    print(f"\nAnnotating {len(files_to_annotate)} files...")
    for path in files_to_annotate:
        print(f"\n{'='*60}")
        print(f"  {path}")
        print(f"{'='*60}")
        result = annotate_file(path, clf, args.batch_size, args.overwrite)
        if not result.get("skipped"):
            print_distribution(os.path.basename(path), result["old"], result["new"])

    print("\nDone.")


if __name__ == "__main__":
    main()
