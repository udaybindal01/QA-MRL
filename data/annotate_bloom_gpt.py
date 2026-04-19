"""
GPT-4 Bloom Taxonomy Annotation.

Replaces the pretrained BERT classifier with GPT-4o-mini for BEIR and other
non-educational datasets where the BERT classifier collapses (e.g. 82% Remember
on NFCorpus). GPT-4 handles keyword queries, scientific claims, and financial
questions that are out-of-distribution for the BERT classifier.

Outputs:
  - Updated JSONL with new bloom_level field
  - Updated .bloom_cache.json (used by EducationalRetrievalDataset at train time)

Usage:
    # Annotate a single file
    python data/annotate_bloom_gpt.py \
        --input /tmp/data/beir/nfcorpus/train.jsonl \
        --model gpt-4o-mini

    # Annotate all BEIR splits for all datasets
    python data/annotate_bloom_gpt.py \
        --beir_root /tmp/data/beir \
        --datasets scifact nfcorpus fiqa \
        --model gpt-4o-mini

    export OPENAI_API_KEY=sk-...
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import List

BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply",
               4: "Analyze",  5: "Evaluate",   6: "Create"}

SYSTEM_PROMPT = """You are an expert in Bloom's Taxonomy for classifying information retrieval queries.

Bloom's Taxonomy levels for IR queries:
1 = Remember  — recall/retrieve a fact, definition, or list (e.g. "What is DNA?", "types of rocks")
2 = Understand — explain, describe, or summarize a concept (e.g. "How does photosynthesis work?")
3 = Apply     — use knowledge to solve, calculate, or demonstrate (e.g. "What drug treats condition X?")
4 = Analyze   — compare, contrast, examine relationships (e.g. "What is the relationship between A and B?")
5 = Evaluate  — judge, assess, or critique evidence (e.g. "What is the evidence for/against X?")
6 = Create    — synthesize or design something novel (rare in retrieval queries)

Important: even short keyword queries have a Bloom level.
  "obesity treatment" → 3 (Apply: looking for what to do)
  "BRCA1 cancer" → 2 (Understand: what is the relationship)
  "define mitosis" → 1 (Remember)
  "compare growth factors" → 4 (Analyze)

Return ONLY a JSON array of integers (1-6), one per query, in the same order."""


def classify_batch_gpt(queries: List[str], client, model: str,
                        retries: int = 6, backoff: float = 5.0) -> List[int]:
    """Call GPT to classify a batch of queries. Returns list of 1-indexed Bloom levels.
    Handles rate-limit (429) with exponential backoff up to 5 minutes."""
    import re
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))
    user_msg = f"Classify these {len(queries)} queries:\n\n{numbered}"

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=len(queries) * 4 + 20,
            )
            text = response.choices[0].message.content.strip()

            match = re.search(r'\[[\d,\s]+\]', text)
            if match:
                levels = json.loads(match.group())
            else:
                levels = [int(x) for x in re.findall(r'\b[1-6]\b', text)]

            if len(levels) == len(queries):
                return [max(1, min(6, l)) for l in levels]

            print(f"  WARNING: got {len(levels)} labels for {len(queries)} queries, retrying...")

        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "rate_limit" in err.lower() or "Rate limit" in err
            wait = backoff * (2 ** attempt)   # exponential: 5, 10, 20, 40, 80, 160s
            if is_rate_limit:
                print(f"  Rate limit hit — waiting {wait:.0f}s before retry "
                      f"({attempt+1}/{retries})...")
            else:
                print(f"  API error (attempt {attempt+1}/{retries}): {e} — retrying in {wait:.0f}s")
            if attempt < retries - 1:
                time.sleep(wait)

    # Fallback: classify one at a time with conservative pacing
    print("  Falling back to single-query classification (1 req/s)...")
    results = []
    for q in queries:
        for attempt in range(retries):
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",
                         "content": f"Classify this query (return only 1-6): {q}"},
                    ],
                    temperature=0.0,
                    max_tokens=5,
                )
                nums = re.findall(r'\b[1-6]\b', r.choices[0].message.content)
                results.append(int(nums[0]) if nums else 1)
                break
            except Exception as e:
                wait = backoff * (2 ** attempt)
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    print(f"  Rate limit — waiting {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    results.append(1)
                    break
        time.sleep(1.0)   # 1 req/s hard floor for fallback path
    return results


def annotate_file(input_path: str, model: str, client,
                  batch_size: int = 25, overwrite: bool = False) -> dict:
    """
    Annotate all queries in a JSONL file with GPT Bloom levels.
    Updates bloom_level in-place and writes .bloom_cache.json alongside.
    Returns a Counter of old vs new distribution.
    """
    cache_path = input_path + ".bloom_cache.json"

    samples = [json.loads(l) for l in open(input_path)]
    queries = [s["query"] for s in samples]
    old_dist = Counter(s.get("bloom_level", 0) for s in samples)

    # Check if cache already done
    if os.path.exists(cache_path) and not overwrite:
        with open(cache_path) as f:
            cached = json.load(f)
        if len(cached) == len(samples):
            print(f"  Cache already exists ({len(cached)} entries) — skipping. "
                  f"Use --overwrite to re-annotate.")
            new_dist = Counter(b + 1 for b in cached)
            return {"old": old_dist, "new": new_dist, "skipped": True}

    print(f"  Annotating {len(queries)} queries with {model}...")
    all_levels = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        levels = classify_batch_gpt(batch, client, model)
        all_levels.extend(levels)
        if (i // batch_size) % 5 == 0:
            print(f"    {len(all_levels)}/{len(queries)}", end="\r", flush=True)
        time.sleep(0.05)  # gentle rate limiting

    print(f"    Done: {len(all_levels)} queries classified.         ")

    # Update samples in-place
    for s, level in zip(samples, all_levels):
        s["bloom_level"] = level

    # Write updated JSONL
    with open(input_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    # Write bloom_cache (0-indexed, matching EducationalRetrievalDataset)
    cache = [l - 1 for l in all_levels]
    with open(cache_path, "w") as f:
        json.dump(cache, f)

    new_dist = Counter(all_levels)
    return {"old": old_dist, "new": new_dist, "skipped": False}


def print_distribution(name: str, old: Counter, new: Counter):
    print(f"\n  {name}:")
    print(f"    {'Level':12s}  {'Old':>6s}  {'New':>6s}")
    print(f"    {'-'*28}")
    for level in range(1, 7):
        o = old.get(level, 0)
        n = new.get(level, 0)
        bar = "█" * min(20, n // max(1, max(new.values()) // 20))
        print(f"    {BLOOM_NAMES[level]:12s}  {o:>6d}  {n:>6d}  {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None,
                        help="Single JSONL file to annotate")
    parser.add_argument("--beir_root", default=None,
                        help="Root dir of BEIR data (annotates train+val+test for each dataset)")
    parser.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus", "fiqa"],
                        help="BEIR datasets to annotate (used with --beir_root)")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="OpenAI model (gpt-4o-mini recommended for cost/speed)")
    parser.add_argument("--batch_size", type=int, default=25,
                        help="Queries per API call")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-annotate even if cache exists")
    args = parser.parse_args()

    # Init OpenAI client
    try:
        from openai import OpenAI
    except ImportError:
        print("Installing openai...")
        os.system("pip install openai -q")
        from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    print(f"Using model: {args.model}")

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
        result = annotate_file(path, args.model, client,
                               args.batch_size, args.overwrite)
        if not result.get("skipped"):
            print_distribution(os.path.basename(path), result["old"], result["new"])

    print("\nDone. Delete *.bloom_cache.json files to force re-annotation.")


if __name__ == "__main__":
    main()
