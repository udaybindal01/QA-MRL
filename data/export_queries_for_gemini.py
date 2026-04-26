"""
Collect all unique training queries from SciQ, ARC, OpenBookQA, QASC.
Export as numbered batch text files (50 per batch) ready to paste into Gemini.

Usage:
    python3 data/export_queries_for_gemini.py --output_dir gemini_batches
"""

import argparse
import os
import random
from datasets import load_dataset

random.seed(42)

def collect_queries():
    queries = []

    print("Loading SciQ...")
    ds = load_dataset("allenai/sciq", split="train")
    for row in ds:
        if len(row.get("support", "")) >= 40:
            queries.append(row["question"].strip())
    print(f"  SciQ: {len(queries)}")

    n0 = len(queries)
    print("Loading ARC...")
    for arc_name in ["ARC-Easy", "ARC-Challenge"]:
        for split in ["train", "validation", "test"]:
            try:
                ds = load_dataset("allenai/ai2_arc", arc_name, split=split)
                for row in ds:
                    queries.append(row["question"].strip())
            except Exception:
                pass
    print(f"  ARC: {len(queries) - n0}")

    n0 = len(queries)
    print("Loading OpenBookQA...")
    for split in ["train", "validation", "test"]:
        try:
            ds = load_dataset("allenai/openbookqa", "main", split=split)
            for row in ds:
                queries.append(row["question_stem"].strip())
        except Exception:
            pass
    print(f"  OpenBookQA: {len(queries) - n0}")

    n0 = len(queries)
    print("Loading QASC...")
    for split in ["train", "validation"]:
        try:
            ds = load_dataset("allenai/qasc", split=split)
            for row in ds:
                q = row.get("question", "").strip()
                if q:
                    queries.append(q)
        except Exception:
            pass
    print(f"  QASC: {len(queries) - n0}")

    # Deduplicate preserving order
    seen = set()
    unique = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)

    print(f"\nTotal unique queries: {len(unique)}")
    return unique


def export_batches(queries, output_dir, batch_size=50):
    os.makedirs(output_dir, exist_ok=True)

    batches = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]
    print(f"Creating {len(batches)} batches of {batch_size} in '{output_dir}/'")

    for b_idx, batch in enumerate(batches):
        path = os.path.join(output_dir, f"batch_{b_idx+1:03d}.txt")
        # Global offset so IDs are unique across batches
        offset = b_idx * batch_size
        lines = []
        for i, q in enumerate(batch):
            qid = f"Q{offset + i + 1:05d}"
            # Escape quotes inside question
            q_safe = q.replace('"', "'")
            lines.append(f'{qid},"{q_safe}"')
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # Also write a master ID→query CSV for merging labels later
    master_path = os.path.join(output_dir, "all_queries.csv")
    with open(master_path, "w", encoding="utf-8") as f:
        f.write("id,question\n")
        for i, q in enumerate(queries):
            q_safe = q.replace('"', "'")
            f.write(f'Q{i+1:05d},"{q_safe}"\n')

    print(f"Master CSV: {master_path}")
    print(f"\nPaste each batch_{{}}.txt into Gemini with the system prompt.")
    print("Gemini should output: QID,bloom_label  (one per line)")
    print(f"Collect all outputs into a single file, then run merge_gemini_labels.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="gemini_batches")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--max_queries", type=int, default=2000)
    args = parser.parse_args()

    queries = collect_queries()
    if args.max_queries and args.max_queries < len(queries):
        random.shuffle(queries)
        queries = queries[:args.max_queries]
        print(f"Subsampled to {len(queries)} queries")
    export_batches(queries, args.output_dir, args.batch_size)
