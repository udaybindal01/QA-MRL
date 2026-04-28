"""
Train 4-model Bloom taxonomy council on 3 Kaggle datasets.

Models:
  - microsoft/deberta-v3-large   (transformer, strongest)
  - roberta-large                (transformer)
  - bert-base-uncased            (transformer, fastest)
  - tfidf-svm                    (classical ML, strong on lexical patterns)

Data: 800 train + 100 val + 100 test per class (balanced, no oversampling needed).
Output saved to: /tmp/bloom-council/

Usage:
    python3 data/train_bloom_council.py
    python3 data/train_bloom_council.py --output_dir /tmp/bloom-council --n_train 800
"""

import argparse
import json
import os
import pickle
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BLOOM_LABELS = {1: "Remember", 2: "Understand", 3: "Apply",
                4: "Analyze", 5: "Evaluate", 6: "Create"}

TRANSFORMER_SPECS = [
    {"name": "deberta", "hf_name": "microsoft/deberta-v3-large",
     "batch_size": 8,  "grad_accum": 4, "epochs": 5, "lr": 1e-5,  "fp16": False},
    {"name": "roberta", "hf_name": "roberta-large",
     "batch_size": 16, "grad_accum": 2, "epochs": 5, "lr": 2e-5,  "fp16": True},
    {"name": "bert",    "hf_name": "bert-base-uncased",
     "batch_size": 32, "grad_accum": 1, "epochs": 5, "lr": 2e-5,  "fp16": True},
]

BLOOM_VERBS = {
    1: ["define","list","recall","name","identify","state","label","match",
        "memorize","repeat","recognize","describe","who","what","when","where"],
    2: ["explain","summarize","interpret","classify","compare","describe",
        "discuss","outline","paraphrase","predict","review","translate"],
    3: ["apply","calculate","demonstrate","solve","use","illustrate",
        "compute","modify","prepare","produce","show","operate"],
    4: ["analyze","differentiate","distinguish","examine","compare",
        "contrast","investigate","breakdown","relate","why","how does"],
    5: ["evaluate","assess","judge","justify","critique","defend","argue",
        "recommend","support","appraise","debate","prioritize","select"],
    6: ["create","design","construct","develop","formulate","propose",
        "invent","compose","generate","plan","produce","build","synthesize"],
}


# ─── Data Loading ──────────────────────────────────────────────────────────────

def load_kaggle_datasets() -> Tuple[List[str], List[int]]:
    from data.annotate_bloom_council import _load_kaggle_dataset
    datasets = [
        "vijaydevane/blooms-taxonomy-dataset",
        "abhaygotmare/blooms-taxonomy-questions-level",
        "dineshsheelam/blooms-taxonomy-dataset",
    ]
    all_texts, all_labels = [], []
    for name in datasets:
        result = _load_kaggle_dataset(name)
        if result:
            all_texts.extend(result[0])
            all_labels.extend(result[1])
    print(f"Total examples loaded: {len(all_texts)}")
    return all_texts, all_labels


def balance_and_split(texts, labels, n_train=800, n_val=100, n_test=100):
    """Split into balanced train/val/test with n_per_class each."""
    random.seed(42)
    buckets = defaultdict(list)
    for t, l in zip(texts, labels):
        buckets[l].append(t)

    train_t, train_l = [], []
    val_t,   val_l   = [], []
    test_t,  test_l  = [], []

    for lv in range(1, 7):
        pool = buckets[lv][:]
        random.shuffle(pool)
        needed = n_train + n_val + n_test
        if len(pool) < needed:
            print(f"  WARNING: Level {lv} has only {len(pool)} examples, needed {needed}")
            # oversample if necessary
            while len(pool) < needed:
                pool.extend(buckets[lv])
            pool = pool[:needed]

        train_t += pool[:n_train];              train_l += [lv]*n_train
        val_t   += pool[n_train:n_train+n_val]; val_l   += [lv]*n_val
        test_t  += pool[n_train+n_val:needed];  test_l  += [lv]*n_test

    print(f"Split: train={len(train_t)}, val={len(val_t)}, test={len(test_t)}")
    return (train_t, train_l), (val_t, val_l), (test_t, test_l)


# ─── SVM ───────────────────────────────────────────────────────────────────────

def _bloom_verb_features(texts: List[str]) -> np.ndarray:
    feats = np.zeros((len(texts), 6))
    for i, text in enumerate(texts):
        t = text.lower()
        for lv, verbs in BLOOM_VERBS.items():
            feats[i, lv-1] = sum(1 for v in verbs if v in t)
    return feats


def train_svm(train_texts, train_labels, val_texts, val_labels,
              test_texts, test_labels, output_dir: str):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder
    from scipy.sparse import hstack, csr_matrix

    print("\n[SVM] Training...")
    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=50000,
                            sublinear_tf=True, min_df=2)
    X_train_tfidf = tfidf.fit_transform(train_texts)
    X_val_tfidf   = tfidf.transform(val_texts)
    X_test_tfidf  = tfidf.transform(test_texts)

    X_train = hstack([X_train_tfidf, csr_matrix(_bloom_verb_features(train_texts))])
    X_val   = hstack([X_val_tfidf,   csr_matrix(_bloom_verb_features(val_texts))])
    X_test  = hstack([X_test_tfidf,  csr_matrix(_bloom_verb_features(test_texts))])

    svm = LinearSVC(C=1.0, max_iter=3000, class_weight="balanced")
    svm.fit(X_train, train_labels)

    val_acc  = float(np.mean(svm.predict(X_val)  == np.array(val_labels)))
    test_acc = float(np.mean(svm.predict(X_test) == np.array(test_labels)))
    print(f"  Val acc: {val_acc:.4f}  |  Test acc: {test_acc:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "svm.pkl"), "wb") as f:
        pickle.dump({"tfidf": tfidf, "svm": svm}, f)
    print(f"  Saved to {output_dir}/svm.pkl")
    return test_acc


# ─── Transformer Training ──────────────────────────────────────────────────────

def train_transformer(spec: dict, train_data, val_data, test_data, output_dir: str):
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                               TrainingArguments, Trainer, DataCollatorWithPadding,
                               EarlyStoppingCallback)
    from torch.utils.data import Dataset

    train_texts, train_labels = train_data
    val_texts,   val_labels   = val_data
    test_texts,  test_labels  = test_data
    hf_name   = spec["hf_name"]
    model_dir = os.path.join(output_dir, spec["name"])

    # Skip if already trained
    if os.path.exists(os.path.join(model_dir, "config.json")):
        print(f"\n[{spec['name']}] Already trained — loading for eval.")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    else:
        print(f"\n[{spec['name']}] Loading {hf_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_name, num_labels=6, ignore_mismatched_sizes=True)

    # Convert labels to 0-indexed
    train_labels_0 = [l-1 for l in train_labels]
    val_labels_0   = [l-1 for l in val_labels]
    test_labels_0  = [l-1 for l in test_labels]

    class BloomDataset(Dataset):
        def __init__(self, texts, labels, tokenizer):
            self.enc = tokenizer(texts, padding=True, truncation=True,
                                 max_length=128, return_tensors="pt")
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            return {k: v[i] for k, v in self.enc.items()} | {"labels": self.labels[i]}

    train_ds = BloomDataset(train_texts, train_labels_0, tokenizer)
    val_ds   = BloomDataset(val_texts,   val_labels_0,   tokenizer)
    test_ds  = BloomDataset(test_texts,  test_labels_0,  tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float(np.mean(preds == labels))}

    if not os.path.exists(os.path.join(model_dir, "config.json")):
        training_args = TrainingArguments(
            output_dir=model_dir,
            num_train_epochs=spec["epochs"],
            per_device_train_batch_size=spec["batch_size"],
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=spec["grad_accum"],
            learning_rate=spec["lr"],
            warmup_ratio=0.1,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            logging_steps=50,
            fp16=spec.get("fp16", True) and torch.cuda.is_available(),
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        print(f"  Training {spec['name']} for up to {spec['epochs']} epochs...")
        trainer.train()
        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"  Saved to {model_dir}")

    # Evaluate on test set
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    all_preds = []
    loader = torch.utils.data.DataLoader(test_ds, batch_size=32)
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**batch).logits
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())

    test_acc = float(np.mean(np.array(all_preds) == np.array(test_labels_0)))
    print(f"  Test accuracy: {test_acc:.4f}")

    # Per-class accuracy
    for lv in range(6):
        mask = np.array(test_labels_0) == lv
        if mask.sum():
            acc = float(np.mean(np.array(all_preds)[mask] == lv))
            print(f"    Level {lv+1} {BLOOM_LABELS[lv+1]:10s}: {acc:.3f} (n={mask.sum()})")

    return test_acc


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/tmp/bloom-council")
    parser.add_argument("--n_train",    type=int, default=800)
    parser.add_argument("--n_val",      type=int, default=100)
    parser.add_argument("--n_test",     type=int, default=100)
    parser.add_argument("--skip_svm",   action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("BLOOM COUNCIL TRAINING")
    print("=" * 60)

    # Load and split data
    print("\n[1/6] Loading Kaggle datasets...")
    texts, labels = load_kaggle_datasets()
    train_data, val_data, test_data = balance_and_split(
        texts, labels, args.n_train, args.n_val, args.n_test)

    weights: Dict[str, float] = {}

    # Train SVM
    if not args.skip_svm:
        print("\n[2/6] Training SVM...")
        svm_acc = train_svm(*train_data, *val_data, *test_data,
                            os.path.join(args.output_dir, "svm"))
        weights["svm"] = svm_acc

    # Train transformers
    for i, spec in enumerate(TRANSFORMER_SPECS):
        step = i + 3
        print(f"\n[{step}/6] Training {spec['name']} ({spec['hf_name']})...")
        acc = train_transformer(spec, train_data, val_data, test_data, args.output_dir)
        weights[spec["name"]] = acc

    # Normalize weights
    total = sum(weights.values())
    normalized = {k: round(v / total, 4) for k, v in weights.items()}

    # Save
    council_info = {
        "weights": normalized,
        "raw_test_accuracy": weights,
        "n_train_per_class": args.n_train,
        "n_test_per_class": args.n_test,
        "models": {
            "deberta": os.path.join(args.output_dir, "deberta"),
            "roberta": os.path.join(args.output_dir, "roberta"),
            "bert":    os.path.join(args.output_dir, "bert"),
            "svm":     os.path.join(args.output_dir, "svm", "svm.pkl"),
        }
    }
    weights_path = os.path.join(args.output_dir, "council_weights.json")
    with open(weights_path, "w") as f:
        json.dump(council_info, f, indent=2)

    print("\n" + "=" * 60)
    print("COUNCIL SUMMARY")
    print("=" * 60)
    print(f"{'Model':<12} {'Test Acc':>10} {'Weight':>10}")
    print("-" * 35)
    for name, acc in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"{name:<12} {acc:>10.4f} {normalized[name]:>10.4f}")
    print(f"\nWeights saved to: {weights_path}")
    print("Run build_real_data.py to annotate training data with this council.")


if __name__ == "__main__":
    main()
