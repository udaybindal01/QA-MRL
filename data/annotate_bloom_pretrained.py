"""
Bloom Level Annotation — Auto-downloads everything.

Three approaches (all auto-download, no manual steps):

  1. pretrained: Use cip29/bert-blooms-taxonomy-classifier from HuggingFace
  2. finetune:   Download Kaggle Bloom datasets, fine-tune BERT, then annotate
  3. both:       Fine-tune on Kaggle data, then annotate (most accurate)

Usage:
    # Approach 1: Pre-trained HuggingFace model (fastest, ~2 min)
    python data/annotate_bloom_pretrained.py \
        --data_dir /tmp/data/real \
        --method pretrained

    # Approach 2: Auto-download Kaggle data + fine-tune + annotate (~15 min)
    python data/annotate_bloom_pretrained.py \
        --data_dir /tmp/data/real \
        --method finetune

    # Just test set
    python data/annotate_bloom_pretrained.py \
        --data_dir /tmp/data/real \
        --splits test \
        --method pretrained \
        --compare

Setup (one-time):
    pip install transformers datasets kagglehub scikit-learn pandas --break-system-packages

    # For Kaggle downloads, set up API token:
    # 1. Go to https://www.kaggle.com/settings → API → Create New Token
    # 2. Save kaggle.json to ~/.kaggle/kaggle.json
    #    OR set environment variables:
    export KAGGLE_USERNAME=your_username
    export KAGGLE_KEY=your_api_key
"""

import argparse
import json
import os
import glob
import shutil
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter
from tqdm import tqdm


BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply",
               4: "Analyze", 5: "Evaluate", 6: "Create"}


# ──────────────────── Dataset Download ────────────────────

def download_kaggle_bloom_datasets(output_dir="/tmp/bloom_kaggle_data"):
    """
    Auto-download all available Bloom taxonomy datasets from Kaggle.

    Returns path to a merged CSV with columns: text, bloom_level (1-6)
    """
    os.makedirs(output_dir, exist_ok=True)
    merged_path = os.path.join(output_dir, "merged_bloom_data.csv")

    # Skip if already downloaded and merged
    if os.path.exists(merged_path):
        import pandas as pd
        df = pd.read_csv(merged_path)
        print(f"  Using cached merged dataset: {len(df)} rows from {merged_path}")
        return merged_path

    try:
        import kagglehub
    except ImportError:
        print("  Installing kagglehub...")
        os.system("pip install kagglehub --break-system-packages -q")
        import kagglehub

    import pandas as pd

    all_dfs = []

    # Dataset 1: vijaydevane/blooms-taxonomy-dataset (~7000 questions)
    print("  Downloading vijaydevane/blooms-taxonomy-dataset...")
    try:
        path = kagglehub.dataset_download("vijaydevane/blooms-taxonomy-dataset")
        print(f"    Downloaded to: {path}")
        csvs = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)
        for csv_path in csvs:
            df = pd.read_csv(csv_path)
            print(f"    Found CSV: {os.path.basename(csv_path)} ({len(df)} rows, cols: {list(df.columns)})")
            parsed = _parse_bloom_csv(df, source="vijaydevane")
            if parsed is not None and len(parsed) > 0:
                all_dfs.append(parsed)
                print(f"    Parsed: {len(parsed)} rows")
    except Exception as e:
        print(f"    Warning: {e}")

    # Dataset 2: abhaygotmare/blooms-taxonomy-questions-level
    print("  Downloading abhaygotmare/blooms-taxonomy-questions-level...")
    try:
        path = kagglehub.dataset_download("abhaygotmare/blooms-taxonomy-questions-level")
        print(f"    Downloaded to: {path}")
        csvs = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)
        for csv_path in csvs:
            df = pd.read_csv(csv_path)
            print(f"    Found CSV: {os.path.basename(csv_path)} ({len(df)} rows, cols: {list(df.columns)})")
            parsed = _parse_bloom_csv(df, source="abhaygotmare")
            if parsed is not None and len(parsed) > 0:
                all_dfs.append(parsed)
                print(f"    Parsed: {len(parsed)} rows")
    except Exception as e:
        print(f"    Warning: {e}")

    # Dataset 3: dineshsheelam/blooms-taxonomy-dataset
    print("  Downloading dineshsheelam/blooms-taxonomy-dataset...")
    try:
        path = kagglehub.dataset_download("dineshsheelam/blooms-taxonomy-dataset")
        print(f"    Downloaded to: {path}")
        csvs = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)
        for csv_path in csvs:
            df = pd.read_csv(csv_path)
            print(f"    Found CSV: {os.path.basename(csv_path)} ({len(df)} rows, cols: {list(df.columns)})")
            parsed = _parse_bloom_csv(df, source="dineshsheelam")
            if parsed is not None and len(parsed) > 0:
                all_dfs.append(parsed)
                print(f"    Parsed: {len(parsed)} rows")
    except Exception as e:
        print(f"    Warning: {e}")

    if not all_dfs:
        print("  ERROR: No Kaggle datasets could be downloaded.")
        print("  Make sure you have Kaggle credentials set up:")
        print("    export KAGGLE_USERNAME=your_username")
        print("    export KAGGLE_KEY=your_api_key")
        print("  Or place kaggle.json in ~/.kaggle/")
        return None

    # Merge and deduplicate
    merged = pd.concat(all_dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["text"])
    merged = merged.dropna(subset=["text", "bloom_level"])
    merged = merged[merged["bloom_level"].between(1, 6)]

    print(f"\n  Merged dataset: {len(merged)} unique rows")
    print(f"  Bloom distribution:")
    for level in sorted(merged["bloom_level"].unique()):
        count = (merged["bloom_level"] == level).sum()
        print(f"    {BLOOM_NAMES.get(int(level), level)}: {count}")

    merged.to_csv(merged_path, index=False)
    print(f"  Saved to {merged_path}")
    return merged_path


def _parse_bloom_csv(df, source="unknown"):
    """Parse a Bloom taxonomy CSV into standardized format: text, bloom_level (1-6)."""
    import pandas as pd

    # Find text column
    text_col = None
    for col in ["Question", "question", "Text", "text", "Sentence", "sentence",
                "query", "Query", "Question_Text", "question_text"]:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        for col in df.columns:
            if df[col].dtype == "object" and df[col].str.len().mean() > 20:
                text_col = col
                break
    if text_col is None:
        return None

    # Find label column
    label_col = None
    for col in ["Bloom's Taxonomy Level", "bloom_level", "Bloom Level", "Level",
                "level", "label", "Label", "cognitive_level", "Cognitive Level",
                "bloom", "Bloom", "Category", "category", "class", "Class"]:
        if col in df.columns:
            label_col = col
            break
    if label_col is None:
        for col in df.columns:
            if col != text_col:
                label_col = col
                break
    if label_col is None:
        return None

    result = pd.DataFrame()
    result["text"] = df[text_col].astype(str).str.strip()
    result["bloom_level"] = df[label_col].apply(_parse_bloom_label)
    result["source"] = source
    result = result[result["bloom_level"].between(1, 6)]
    result = result[result["text"].str.len() > 10]
    return result


def _parse_bloom_label(val):
    """Convert any Bloom label format to integer 1-6."""
    if isinstance(val, (int, float)):
        v = int(val)
        if 1 <= v <= 6:
            return v
        if 0 <= v <= 5:
            return v + 1
        return -1

    name = str(val).lower().strip()
    mapping = {
        "remember": 1, "remembering": 1, "knowledge": 1, "recall": 1,
        "understand": 2, "understanding": 2, "comprehension": 2,
        "apply": 3, "applying": 3, "application": 3,
        "analyze": 4, "analyse": 4, "analyzing": 4, "analysing": 4, "analysis": 4,
        "evaluate": 5, "evaluating": 5, "evaluation": 5,
        "create": 6, "creating": 6, "synthesis": 6, "synthesize": 6,
    }
    for key, level in mapping.items():
        if key in name:
            return level

    try:
        v = int(name)
        if 1 <= v <= 6: return v
        if 0 <= v <= 5: return v + 1
    except ValueError:
        pass

    return -1


# ──────────────────── Model Loading ────────────────────

def load_pretrained_classifier(model_name="cip29/bert-blooms-taxonomy-classifier",
                                device="cuda"):
    """Load pre-trained Bloom classifier from HuggingFace (auto-downloads)."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"  Loading model: {model_name} (auto-download if needed)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device).eval()

    num_labels = model.config.num_labels
    id2label = model.config.id2label if hasattr(model.config, "id2label") else None
    print(f"  Num labels: {num_labels}")
    if id2label:
        print(f"  Label mapping: {id2label}")

    return model, tokenizer, id2label


def predict_bloom(queries, model, tokenizer, device="cuda",
                  batch_size=64, id2label=None):
    """Predict Bloom levels. Returns 1-indexed (1=Remember, ..., 6=Create)."""
    results = []
    for i in tqdm(range(0, len(queries), batch_size), desc="  Classifying"):
        batch = queries[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                       max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits

        preds = logits.argmax(dim=-1).cpu().tolist()
        for pred in preds:
            if id2label:
                label_name = id2label.get(str(pred), id2label.get(pred, str(pred)))
                bloom = _parse_bloom_label(label_name)
                if bloom < 1:
                    bloom = pred + 1
            else:
                bloom = pred + 1
            results.append(max(1, min(6, bloom)))

    return results


# ──────────────────── Fine-tuning ────────────────────

def finetune_bloom_classifier(csv_path, output_dir="/tmp/bloom_classifier_finetuned",
                               base_model="bert-base-uncased", epochs=5,
                               batch_size=16, device="cuda"):
    """Fine-tune BERT on a Bloom taxonomy CSV dataset."""
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer)
    from torch.utils.data import Dataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    import pandas as pd

    print(f"\n  Fine-tuning on {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows")

    # Standardize columns
    if "text" not in df.columns or "bloom_level" not in df.columns:
        print("  ERROR: CSV must have 'text' and 'bloom_level' columns")
        return None

    df = df.dropna(subset=["text", "bloom_level"])
    df["label_idx"] = df["bloom_level"].astype(int) - 1  # 0-indexed
    df = df[df["label_idx"].between(0, 5)]

    print(f"  Label distribution:")
    for idx in range(6):
        count = (df["label_idx"] == idx).sum()
        print(f"    {BLOOM_NAMES[idx+1]}: {count}")

    # Split
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42,
                                         stratify=df["label_idx"])
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    class BloomDS(Dataset):
        def __init__(self, texts, labels, tok, max_len=128):
            self.enc = tok(texts, truncation=True, padding=True,
                          max_length=max_len, return_tensors="pt")
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.enc.items()}
            item["labels"] = self.labels[idx]
            return item

    train_ds = BloomDS(train_df["text"].tolist(), train_df["label_idx"].tolist(), tokenizer)
    val_ds = BloomDS(val_df["text"].tolist(), val_df["label_idx"].tolist(), tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=6,
        id2label={str(i): BLOOM_NAMES[i+1] for i in range(6)},
        label2id={BLOOM_NAMES[i+1]: i for i in range(6)},
    )

    args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="accuracy",
        logging_steps=50, report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds),
                "f1_macro": f1_score(labels, preds, average="macro")}

    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, compute_metrics=compute_metrics)

    print(f"\n  Training for {epochs} epochs...")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = trainer.evaluate()
    print(f"  Val accuracy: {metrics.get('eval_accuracy', 0):.4f}")
    print(f"  Val F1 macro: {metrics.get('eval_f1_macro', 0):.4f}")
    print(f"  Saved to {output_dir}")
    return output_dir


# ──────────────────── Annotation ────────────────────

def annotate_file(input_path, output_path, model, tokenizer,
                  device, id2label=None, batch_size=64):
    """Annotate a JSONL file with predicted Bloom levels."""
    pairs = []
    with open(input_path) as f:
        for line in f:
            pairs.append(json.loads(line.strip()))

    print(f"\n  Annotating {len(pairs)} queries from {os.path.basename(input_path)}")
    queries = [p["query"] for p in pairs]

    old_dist = Counter(p["bloom_level"] for p in pairs)
    print(f"  Old (regex): {dict(sorted(old_dist.items()))}")

    new_blooms = predict_bloom(queries, model, tokenizer, device, batch_size, id2label)

    changed = 0
    for i, pair in enumerate(pairs):
        if pair["bloom_level"] != new_blooms[i]:
            changed += 1
        pair["bloom_level"] = new_blooms[i]
        pair["bloom_source"] = "pretrained_model"

    new_dist = Counter(p["bloom_level"] for p in pairs)
    print(f"  New (model): {dict(sorted(new_dist.items()))}")
    print(f"  Changed: {changed}/{len(pairs)} ({changed/len(pairs):.1%})")

    with open(output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    return old_dist, new_dist


def compare_annotations(data_dir):
    """Compare model vs regex on test set."""
    backup = os.path.join(data_dir, "test_regex_backup.jsonl")
    current = os.path.join(data_dir, "test.jsonl")
    if not os.path.exists(backup):
        return

    regex_pairs = [json.loads(l) for l in open(backup)]
    model_pairs = [json.loads(l) for l in open(current)]
    agree = sum(1 for r, m in zip(regex_pairs, model_pairs)
                if r["bloom_level"] == m["bloom_level"])

    print(f"\n  === Model vs Regex Agreement ===")
    print(f"  Agreement: {agree}/{len(regex_pairs)} ({agree/len(regex_pairs):.1%})")

    changes = Counter()
    for r, m in zip(regex_pairs, model_pairs):
        if r["bloom_level"] != m["bloom_level"]:
            changes[(r["bloom_level"], m["bloom_level"])] += 1
    print(f"  Top disagreements (regex→model):")
    for (rb, mb), count in changes.most_common(8):
        print(f"    {BLOOM_NAMES.get(rb,'?')}→{BLOOM_NAMES.get(mb,'?')}: {count}")


# ──────────────────── Main ────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bloom annotation with auto-download")
    parser.add_argument("--data_dir", default="/tmp/data/real")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--method", default="pretrained",
                        choices=["pretrained", "finetune"],
                        help="pretrained: HuggingFace model. finetune: download Kaggle data + fine-tune.")
    parser.add_argument("--model_name", default="cip29/bert-blooms-taxonomy-classifier")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--kaggle_csv", default=None,
                        help="Path to your own Bloom CSV (skips Kaggle download)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Bloom Level Annotation (Auto-Download)")
    print("=" * 60)
    print(f"  Method: {args.method}")
    print(f"  Device: {device}")
    print(f"  Data dir: {args.data_dir}")

    # ── Load or train model ──
    if args.method == "pretrained":
        print(f"\n[1/2] Loading pre-trained model from HuggingFace...")
        model, tokenizer, id2label = load_pretrained_classifier(
            args.model_name, device)

    elif args.method == "finetune":
        # Step 1: Get Kaggle data
        if args.kaggle_csv and os.path.exists(args.kaggle_csv):
            csv_path = args.kaggle_csv
            print(f"\n[1/3] Using provided CSV: {csv_path}")
        else:
            print(f"\n[1/3] Downloading Bloom taxonomy datasets from Kaggle...")
            csv_path = download_kaggle_bloom_datasets()
            if csv_path is None:
                print("  FATAL: Could not download Kaggle data. Falling back to pretrained.")
                model, tokenizer, id2label = load_pretrained_classifier(
                    args.model_name, device)
                csv_path = None

        # Step 2: Fine-tune
        if csv_path is not None:
            print(f"\n[2/3] Fine-tuning BERT on Kaggle Bloom data...")
            model_dir = finetune_bloom_classifier(csv_path, device=device)
            if model_dir:
                model, tokenizer, id2label = load_pretrained_classifier(model_dir, device)
            else:
                print("  Fine-tuning failed. Falling back to pretrained.")
                model, tokenizer, id2label = load_pretrained_classifier(
                    args.model_name, device)

    # ── Annotate ──
    step = "[2/2]" if args.method == "pretrained" else "[3/3]"
    print(f"\n{step} Annotating query splits...")

    all_old, all_new = Counter(), Counter()
    for split in args.splits:
        path = os.path.join(args.data_dir, f"{split}.jsonl")
        if not os.path.exists(path):
            print(f"  Skipping {split} — not found")
            continue

        # Backup original regex annotations
        backup = os.path.join(args.data_dir, f"{split}_regex_backup.jsonl")
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
            print(f"  Backed up regex labels → {os.path.basename(backup)}")

        old_dist, new_dist = annotate_file(
            path, path, model, tokenizer, device, id2label, args.batch_size)
        all_old.update(old_dist)
        all_new.update(new_dist)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Level':<12s} {'Regex':>8s} {'Model':>8s} {'Change':>8s}")
    print(f"  {'-'*40}")
    for level in sorted(set(list(all_old.keys()) + list(all_new.keys()))):
        name = BLOOM_NAMES.get(level, str(level))
        old = all_old.get(level, 0)
        new = all_new.get(level, 0)
        print(f"  {name:<12s} {old:>8d} {new:>8d} {new-old:>+8d}")

    if args.compare:
        compare_annotations(args.data_dir)

    print(f"\n  Done! Backups saved as *_regex_backup.jsonl")
    print(f"\n  NEXT STEPS:")
    print(f"    1. Re-mine curriculum negatives:")
    print(f"       python data/curriculum_negatives.py \\")
    print(f"           --pairs /tmp/data/real/train.jsonl \\")
    print(f"           --corpus /tmp/data/real/corpus.jsonl \\")
    print(f"           --output /tmp/data/real/train_curriculum.jsonl \\")
    print(f"           --stage 0.7 --num_neg 3")
    print(f"    2. Update config: sed -i 's|train.jsonl|train_curriculum.jsonl|' configs/bam.yaml")
    print(f"    3. Retrain: python scripts/train_baseline_mrl.py --config configs/bam.yaml")
    print(f"    4. Train BAM: python scripts/train_bam.py --config configs/bam.yaml \\")
    print(f"           --init_encoder /tmp/bam-ckpts/mrl_baseline_best/")


if __name__ == "__main__":
    main()