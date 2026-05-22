"""
Evaluate trained Bloom council on held-out test set and Gemini-labeled ground truth.

Usage:
    python3 data/eval_bloom_council.py                      # uses built-in test split
    python3 data/eval_bloom_council.py --gemini             # also runs on Gemini labels
    python3 data/eval_bloom_council.py --gemini --only      # only Gemini labels
"""

import argparse
import csv
import json
import os
import pickle
import random
import sys
from collections import defaultdict, Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COUNCIL_DIR  = os.environ.get("COUNCIL_DIR", "/tmp/bloom-council")
WEIGHTS_FILE = os.path.join(COUNCIL_DIR, "council_weights.json")
BLOOM_LABELS = {0: "Remember", 1: "Understand", 2: "Apply",
                3: "Analyze",  4: "Evaluate",   5: "Create"}


def load_council():
    assert os.path.exists(WEIGHTS_FILE), \
        f"Council not trained yet. Run: python3 data/train_bloom_council.py"
    with open(WEIGHTS_FILE) as f:
        info = json.load(f)

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    council, weights = {}, info["weights"]
    paths = info["models"]

    for name in ["deberta", "roberta", "bert"]:
        path = paths.get(name)
        if path and os.path.exists(path):
            print(f"  Loading {name} ...")
            tok   = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path)
            model.to(device).eval()
            council[name] = {"type": "transformer", "model": model,
                             "tokenizer": tok, "device": device}

    svm_path = paths.get("svm")
    if svm_path and os.path.exists(svm_path):
        print(f"  Loading svm ...")
        with open(svm_path, "rb") as f:
            council["svm"] = {"type": "svm", **pickle.load(f)}

    total = sum(weights[k] for k in council if k in weights)
    norm_w = {k: weights[k] / total for k in council if k in weights}
    print(f"  Members: {list(council.keys())}")
    print(f"  Weights: { {k: round(v,3) for k,v in norm_w.items()} }\n")
    return council, norm_w


def transformer_proba(member, texts, batch_size=32):
    import torch
    model, tok, device = member["model"], member["tokenizer"], member["device"]
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, padding=True, truncation=True,
                  max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(all_probs)


def svm_proba(member, texts):
    from scipy.sparse import hstack, csr_matrix
    BLOOM_VERBS = {
        1: ["define","list","recall","name","identify","state","memorize","repeat","recognize"],
        2: ["explain","summarize","interpret","classify","compare","describe","paraphrase"],
        3: ["apply","calculate","demonstrate","solve","use","illustrate","compute","show"],
        4: ["analyze","differentiate","distinguish","examine","contrast","investigate","why"],
        5: ["evaluate","assess","judge","justify","critique","defend","argue","recommend"],
        6: ["create","design","construct","develop","formulate","propose","invent","compose"],
    }
    def verb_feats(texts):
        f = np.zeros((len(texts), 6))
        for i, t in enumerate(texts):
            tl = t.lower()
            for lv, verbs in BLOOM_VERBS.items():
                f[i, lv-1] = sum(1 for v in verbs if v in tl)
        return f
    X = hstack([member["tfidf"].transform(texts),
                csr_matrix(verb_feats(texts))])
    scores = member["svm"].decision_function(X)
    scores -= scores.max(axis=1, keepdims=True)
    exp_s = np.exp(scores)
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    aligned = np.zeros((len(texts), 6))
    for col, cls in enumerate(member["svm"].classes_):
        t = int(cls) - 1
        if 0 <= t < 6:
            aligned[:, t] = probs[:, col]
    return aligned


def predict(council, weights, texts, batch_size=32):
    combined = np.zeros((len(texts), 6))
    for name, member in council.items():
        w = weights.get(name, 0.0)
        if w == 0: continue
        if member["type"] == "transformer":
            combined += w * transformer_proba(member, texts, batch_size)
        else:
            combined += w * svm_proba(member, texts)
    return combined.argmax(axis=1)   # 0-indexed


def report(name, preds, labels_0, label_dist=None):
    acc = float(np.mean(preds == np.array(labels_0)))
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Overall accuracy : {acc:.4f}  ({int(acc*len(labels_0))}/{len(labels_0)})")

    majority = Counter(labels_0).most_common(1)[0][1] / len(labels_0)
    print(f"  Majority baseline: {majority:.4f}")

    print(f"\n  {'Level':<12} {'Name':10} {'n':>5} {'Acc':>7}")
    print(f"  {'-'*38}")
    for lv in range(6):
        mask = np.array(labels_0) == lv
        if mask.sum() == 0: continue
        cls_acc = float(np.mean(preds[mask] == lv))
        print(f"  Level {lv+1:<6} {BLOOM_LABELS[lv]:10} {mask.sum():>5} {cls_acc:>7.3f}")

    # Per-member breakdown if available
    return acc


# ─── Test split from Kaggle data ──────────────────────────────────────────────

def eval_on_kaggle_test(council, weights, n_test=100):
    print("\n[Evaluating on Kaggle held-out test split]")
    from data.annotate_bloom_council import _load_kaggle_dataset
    random.seed(42)

    datasets = [
        "vijaydevane/blooms-taxonomy-dataset",
        "abhaygotmare/blooms-taxonomy-questions-level",
        "dineshsheelam/blooms-taxonomy-dataset",
    ]
    all_texts, all_labels = [], []
    for name in datasets:
        r = _load_kaggle_dataset(name)
        if r:
            all_texts.extend(r[0])
            all_labels.extend(r[1])

    # Same balanced split as training (reproduce test split)
    buckets = defaultdict(list)
    for t, l in zip(all_texts, all_labels):
        buckets[l].append(t)

    test_texts, test_labels = [], []
    for lv in range(1, 7):
        pool = buckets[lv][:]
        random.shuffle(pool)
        # train=800, val=100, test=100 — take last 100
        offset = 900  # 800 train + 100 val
        test_texts  += pool[offset:offset+n_test]
        test_labels += [lv] * min(n_test, len(pool) - offset)

    test_labels_0 = [l-1 for l in test_labels]
    print(f"  Test set: {len(test_texts)} examples (100/class)")

    preds = predict(council, weights, test_texts)
    report("Kaggle held-out test split", preds, test_labels_0)

    # Per-member accuracy
    print(f"\n  Per-member accuracy:")
    for name, member in council.items():
        if member["type"] == "transformer":
            p = transformer_proba(member, test_texts).argmax(axis=1)
        else:
            p = svm_proba(member, test_texts).argmax(axis=1)
        acc = float(np.mean(p == np.array(test_labels_0)))
        print(f"    {name:<12}: {acc:.4f}")


# ─── Gemini ground truth ──────────────────────────────────────────────────────

def eval_on_gemini(council, weights, csv_path="bloom_annotation_template.csv"):
    print(f"\n[Evaluating on Gemini-labeled ground truth: {csv_path}]")
    if not os.path.exists(csv_path):
        print(f"  Not found: {csv_path}")
        return

    texts, labels_0 = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["bloom_label"].strip():
                texts.append(row["question"])
                labels_0.append(int(row["bloom_label"]) - 1)

    print(f"  Ground truth: {len(texts)} examples")
    preds = predict(council, weights, texts)
    report("Gemini ground truth (n=150)", preds, labels_0)

    print(f"\n  Per-member accuracy:")
    for name, member in council.items():
        if member["type"] == "transformer":
            p = transformer_proba(member, texts).argmax(axis=1)
        else:
            p = svm_proba(member, texts).argmax(axis=1)
        acc = float(np.mean(p == np.array(labels_0)))
        w   = weights.get(name, 0)
        print(f"    {name:<12}: {acc:.4f}  (weight={w:.3f})")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemini",      action="store_true", help="also eval on Gemini labels")
    parser.add_argument("--only",        action="store_true", help="only eval on Gemini labels")
    parser.add_argument("--gemini_csv",  default="bloom_annotation_template.csv")
    parser.add_argument("--batch_size",  type=int, default=32)
    args = parser.parse_args()

    council, weights = load_council()

    if not args.only:
        eval_on_kaggle_test(council, weights)

    if args.gemini or args.only:
        eval_on_gemini(council, weights, args.gemini_csv)


if __name__ == "__main__":
    main()
