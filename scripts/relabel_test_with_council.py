"""
Re-annotate a test.jsonl's bloom_level fields with the trained Bloom
Council, so per-Bloom stratification reflects Council labels (~87% acc)
instead of the fallback classifier's ~57%.

Writes a new jsonl next to the original with `.council.jsonl` suffix,
plus a companion `.bloom_cache.json` (used by dataset.py for routing).

Does NOT touch the corpus. Does NOT change the queries or gold-passage
IDs — only the `bloom_level` field. All eval numbers (R@k, NDCG) are
unaffected; only the per-Bloom breakdown labels change.

Usage:
    export COUNCIL_DIR=$HOME/bloom-council
    python scripts/relabel_test_with_council.py \\
        --input  data/real/test.jsonl \\
        --output data/real/test.council.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.bloom_classifier import classify_bloom_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    with open(args.input) as f:
        samples = [json.loads(l) for l in f]
    print(f"Loaded {len(samples)} queries from {args.input}")

    queries = [s["query"] for s in samples]
    print(f"Classifying with Bloom Council (COUNCIL_DIR="
          f"{os.environ.get('COUNCIL_DIR', '/tmp/bloom-council')})...")
    new_levels = classify_bloom_batch(queries, batch_size=args.batch_size)

    # Report before/after
    from collections import Counter
    old = Counter(s.get("bloom_level", 0) for s in samples)
    new = Counter(new_levels)
    print("\nBloom-level distribution (before → after):")
    for b in range(1, 7):
        o = old.get(b, 0)
        n = new.get(b, 0)
        print(f"  L{b}: {o:>5} → {n:>5}   ({(n-o):+d})")

    # Write new jsonl with updated bloom_level
    with open(args.output, "w") as f:
        for s, lv in zip(samples, new_levels):
            s2 = dict(s)
            s2["bloom_level"] = int(lv)
            f.write(json.dumps(s2) + "\n")
    print(f"\nWrote {len(samples)} lines to {args.output}")

    # Also write a matching bloom_cache.json (0-indexed, in file order)
    # so downstream evaluators using the cache see the Council labels.
    cache_path = args.output + ".bloom_cache.json"
    with open(cache_path, "w") as f:
        json.dump([int(lv) - 1 for lv in new_levels], f)
    print(f"Wrote cache to {cache_path}")


if __name__ == "__main__":
    main()
