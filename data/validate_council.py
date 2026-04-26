"""
Validate council accuracy against Gemini-annotated ground truth.
Uses bloom_annotation_template.csv as held-out test set.

Usage:
    python3 data/validate_council.py --batch_size 8 --pipe_batch_size 4
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_ground_truth(path="bloom_annotation_template.csv"):
    queries, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["bloom_label"].strip():
                queries.append(row["question"])
                labels.append(int(row["bloom_label"]) - 1)  # 0-indexed
    return queries, labels

def accuracy(preds, labels):
    return sum(p == l for p, l in zip(preds, labels)) / len(labels)

def per_class_accuracy(preds, labels, n_classes=6):
    correct = defaultdict(int)
    total = defaultdict(int)
    for p, l in zip(preds, labels):
        total[l] += 1
        if p == l:
            correct[l] += 1
    return {c: correct[c] / total[c] if total[c] else None for c in range(n_classes)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="bloom_annotation_template.csv")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pipe_batch_size", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    queries, labels = load_ground_truth(args.csv)
    print(f"Loaded {len(queries)} annotated queries")
    dist = Counter(labels)
    print("Label distribution (0-indexed):", dict(sorted(dist.items())))

    # Import council members
    from data.annotate_bloom_council import (
        NLIMember, ClassifierMember, KNNMember, LexicalMember, COUNCIL_SPECS,
        load_calibration_datasets
    )

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load calibration data for fitting data-driven members
    print("Loading calibration datasets for fitting k-NN and lexical members...")
    cal_texts, cal_labels = load_calibration_datasets(n=0)
    print(f"  {len(cal_texts)} calibration examples loaded\n")

    results = {}

    for spec in COUNCIL_SPECS:
        name = spec["name"]
        mtype = spec["type"]
        print(f"── {name} ({mtype})")

        try:
            if mtype == "nli":
                member = NLIMember(name, device=device, pipe_batch_size=args.pipe_batch_size)
                member.load()
                probs = member.predict_proba(queries, batch_size=args.pipe_batch_size)

            elif mtype == "classifier":
                member = ClassifierMember(name, device=device)
                member.load()
                probs = member.predict_proba(queries, batch_size=args.pipe_batch_size)

            elif mtype == "knn":
                member = KNNMember(spec.get("encoder_name", name), device=device)
                member.load()
                member.fit(cal_texts, cal_labels)
                probs = member.predict_proba(queries, batch_size=args.batch_size)

            elif mtype == "lexical":
                member = LexicalMember()
                member.fit(cal_texts, cal_labels)
                probs = member.predict_proba(queries, batch_size=args.batch_size)

            preds = [int(np.argmax(p)) for p in probs]
            acc = accuracy(preds, labels)
            per_cls = per_class_accuracy(preds, labels)

            results[name] = {"accuracy": acc, "per_class": per_cls, "preds": preds}
            print(f"   Overall accuracy: {acc:.4f}")
            for c in range(6):
                level_names = ["Remember","Understand","Apply","Analyze","Evaluate","Create"]
                if per_cls[c] is not None:
                    n = dist[c]
                    print(f"   Level {c+1} {level_names[c]:10s} (n={n:3d}): {per_cls[c]:.3f}")
            print()

        except Exception as e:
            print(f"   FAILED: {e}\n")
            results[name] = {"accuracy": None, "error": str(e)}

    # Summary table
    print("=" * 55)
    print("VALIDATION SUMMARY (Gemini ground truth, n=150)")
    print("=" * 55)
    print(f"{'Member':<40} {'Accuracy':>10}")
    print("-" * 55)
    for name, r in results.items():
        short = name.split("/")[-1][:38]
        acc_str = f"{r['accuracy']:.4f}" if r["accuracy"] is not None else "FAILED"
        print(f"{short:<40} {acc_str:>10}")

    # Save results
    out_path = "bloom_council_validation.json"
    with open(out_path, "w") as f:
        save = {k: {kk: vv for kk, vv in v.items() if kk != "preds"} for k, v in results.items()}
        json.dump(save, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
