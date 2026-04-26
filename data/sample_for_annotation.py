"""
Sample 150 questions from SciQ, OpenBookQA, QASC, ARC for Bloom annotation.
Exports two files:
  - bloom_annotation_input.csv  : id, source, question  (paste into Gemini)
  - bloom_annotation_template.csv : id, source, question, bloom_label (fill in)
"""

import csv
import random
import sys

random.seed(42)

try:
    from datasets import load_dataset
except ImportError:
    print("pip install datasets")
    sys.exit(1)

SAMPLES_PER_SOURCE = {
    "sciq": 40,
    "openbookqa": 35,
    "qasc": 40,
    "arc_easy": 20,
    "arc_challenge": 15,
}

def load_sciq(n):
    ds = load_dataset("allenai/sciq", split="train")
    rows = [{"source": "sciq", "question": r["question"]} for r in ds if r["question"].strip()]
    return random.sample(rows, min(n, len(rows)))

def load_openbookqa(n):
    ds = load_dataset("allenai/openbookqa", "main", split="train")
    rows = [{"source": "openbookqa", "question": r["question_stem"]} for r in ds if r["question_stem"].strip()]
    return random.sample(rows, min(n, len(rows)))

def load_qasc(n):
    ds = load_dataset("allenai/qasc", split="train")
    rows = [{"source": "qasc", "question": r["question"]} for r in ds if r["question"].strip()]
    return random.sample(rows, min(n, len(rows)))

def load_arc(n_easy, n_challenge):
    out = []
    for arc_name, n in [("ARC-Easy", n_easy), ("ARC-Challenge", n_challenge)]:
        ds = load_dataset("allenai/ai2_arc", arc_name, split="train")
        rows = [{"source": arc_name.lower().replace("-", "_"), "question": r["question"]}
                for r in ds if r["question"].strip()]
        out.extend(random.sample(rows, min(n, len(rows))))
    return out

print("Loading datasets...")
all_rows = []
all_rows.extend(load_sciq(SAMPLES_PER_SOURCE["sciq"]))
all_rows.extend(load_openbookqa(SAMPLES_PER_SOURCE["openbookqa"]))
all_rows.extend(load_qasc(SAMPLES_PER_SOURCE["qasc"]))
all_rows.extend(load_arc(SAMPLES_PER_SOURCE["arc_easy"], SAMPLES_PER_SOURCE["arc_challenge"]))

random.shuffle(all_rows)

for i, row in enumerate(all_rows):
    row["id"] = f"Q{i+1:03d}"

# Input CSV — paste questions into Gemini
input_path = "bloom_annotation_input.csv"
with open(input_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "source", "question"])
    writer.writeheader()
    writer.writerows(all_rows)

# Template CSV — fill in bloom_label column after Gemini annotation
template_path = "bloom_annotation_template.csv"
with open(template_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "source", "question", "bloom_label"])
    writer.writeheader()
    for row in all_rows:
        writer.writerow({**row, "bloom_label": ""})

print(f"Saved {len(all_rows)} questions:")
print(f"  {input_path}  ← paste questions into Gemini")
print(f"  {template_path}  ← fill bloom_label column (1-6), then use for council validation")

# Also print source distribution
from collections import Counter
counts = Counter(r["source"] for r in all_rows)
for src, cnt in sorted(counts.items()):
    print(f"  {src}: {cnt}")
