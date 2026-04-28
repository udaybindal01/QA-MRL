"""
Annotate bloom_level field in JSONL files using the trained council.

Reads each line, classifies the 'query' field, writes bloom_level back.
Uses bloom_classifier.py which auto-loads trained council from
/tmp/bloom-council/ or falls back to cip29 if not trained yet.

Usage:
    python3 data/annotate_with_council.py --input data/real/train.jsonl
    python3 data/annotate_with_council.py --input a.jsonl b.jsonl --overwrite
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def annotate_file(path: str, overwrite: bool = True, batch_size: int = 64):
    from data.bloom_classifier import classify_bloom_batch

    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    if not overwrite:
        rows = [r for r in rows if not r.get("bloom_level")]
        if not rows:
            print(f"  {path}: all rows already annotated — skipping.")
            return

    queries = [r.get("query", r.get("question", "")) for r in rows]
    levels  = classify_bloom_batch(queries, batch_size=batch_size)

    with open(path) as f:
        all_rows = [json.loads(l) for l in f if l.strip()]

    q_to_level = {q: l for q, l in zip(queries, levels)}
    updated = 0
    for row in all_rows:
        q = row.get("query", row.get("question", ""))
        if q in q_to_level:
            row["bloom_level"] = q_to_level[q]
            updated += 1

    with open(path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    print(f"  {os.path.basename(path)}: {updated}/{len(all_rows)} rows annotated.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      nargs="+", required=True)
    parser.add_argument("--overwrite",  action="store_true", default=True)
    parser.add_argument("--no_overwrite", dest="overwrite", action="store_false")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    for path in args.input:
        if not os.path.exists(path):
            print(f"  SKIP (not found): {path}")
            continue
        print(f"Annotating {path} ...")
        annotate_file(path, overwrite=args.overwrite, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
