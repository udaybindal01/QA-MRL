"""
Bloom Level Annotation — Council of 5 Models (SOTA ensemble).

Five models vote via weighted soft-voting. Weights are calibrated automatically
by evaluating each model on 3 held-out Kaggle Bloom datasets and averaging
per-dataset accuracy. Cached to disk after first calibration run.

Council members (model, role):
  1. MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli  — SOTA zero-shot NLI (440M)
  2. MoritzLaurer/deberta-v3-base-zeroshot-v2                   — Improved base zero-shot NLI (180M)
  3. cross-encoder/nli-deberta-v3-base                          — Cross-encoder NLI for diversity (180M)
  4. facebook/bart-large-mnli                                    — BART NLI, architectural diversity (400M)
  5. cip29/bert-blooms-taxonomy-classifier                       — BERT fine-tuned on Bloom data (110M)

Weight calibration:
  - Downloads 3 Kaggle Bloom taxonomy datasets (vijaydevane, abhaygotmare, dineshsheelam)
  - Evaluates each model on a 200-sample stratified hold-out per dataset
  - Per-model weight = mean accuracy across 3 datasets
  - Cached to /tmp/bloom_council_weights.json — re-run with --recalibrate to refresh

Voting:
  - Each model returns a 6-dim probability vector (one per Bloom level)
  - Final prediction = argmax(weighted average of all five vectors)
  - Confidence = max probability in final vector (reflects council agreement)

Output fields added per record:
  bloom_level       — 1-indexed Bloom level (1=Remember … 6=Create)
  bloom_source      — "council_v2"
  bloom_confidence  — float 0-1, council ensemble confidence
  bloom_votes       — per-model predictions (for debugging)
  bloom_weights     — per-model weights used (for reproducibility)

Calibration datasets:
  vijaydevane/blooms-taxonomy-dataset
  abhaygotmare/blooms-taxonomy-questions-level
  dineshsheelam/blooms-taxonomy-dataset

Usage:
    # Annotate educational data (auto-calibrates weights on first run)
    python data/annotate_bloom_council.py --data_dir /tmp/data/real

    # Annotate BEIR datasets
    python data/annotate_bloom_council.py \
        --beir_root /tmp/data/beir \
        --datasets scifact nfcorpus fiqa

    # Force weight recalibration
    python data/annotate_bloom_council.py --data_dir /tmp/data/real --recalibrate

    # Calibrate only (no annotation)
    python data/annotate_bloom_council.py --calibrate_only

    # Annotate a single file
    python data/annotate_bloom_council.py --input /tmp/data/beir/scifact/test.jsonl
"""

import argparse
import json
import os
import shutil
from collections import Counter
from typing import List, Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply",
               4: "Analyze",  5: "Evaluate",   6: "Create"}

# NLI hypothesis templates — one per Bloom level.
# Concrete and mutually distinct to avoid NLI conflating adjacent levels.
BLOOM_HYPOTHESES = [
    "This query is asking to recall or retrieve a specific fact, name, or definition.",
    "This query is asking to explain, describe, or summarize how something works.",
    "This query is asking how to use or apply knowledge to solve a practical problem.",
    "This query is asking to compare, contrast, or examine the relationship between things.",
    "This query requires making a decision or forming an opinion about the worth or validity of something.",
    "This query is asking to design, propose, or synthesize something new.",
]

WEIGHTS_CACHE = "/tmp/bloom_council_weights.json"

# ─── Council member definitions ───────────────────────────────────────────────

COUNCIL_MEMBERS = [
    {
        "name": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "type": "nli",
        "description": "SOTA zero-shot NLI (440M)",
        "fallback": None,
    },
    {
        "name": "MoritzLaurer/deberta-v3-base-zeroshot-v2",
        "type": "nli",
        "description": "Improved base zero-shot NLI (180M)",
        # deberta-v3-base-zeroshot-v2 can be gated; fall back to a small public model
        "fallback": "cross-encoder/nli-deberta-v3-small",
    },
    {
        "name": "cross-encoder/nli-deberta-v3-base",
        "type": "nli",
        "description": "Cross-encoder DeBERTa NLI (180M)",
        "fallback": None,
    },
    {
        "name": "facebook/bart-large-mnli",
        "type": "nli",
        "description": "BART NLI, architectural diversity (400M)",
        "fallback": None,
    },
    {
        "name": "cip29/bert-blooms-taxonomy-classifier",
        "type": "classifier",
        "description": "BERT fine-tuned on Bloom taxonomy data (110M)",
        "fallback": None,
    },
]


# ─── NLI Member ───────────────────────────────────────────────────────────────

class NLIMember:
    """Zero-shot NLI model. Returns (N, 6) probability matrix."""

    def __init__(self, model_name: str, device, pipe_batch_size: int = 4):
        from transformers import pipeline
        print(f"  [NLI] Loading {model_name} (pipe_batch={pipe_batch_size}) ...")
        self.name = model_name
        self.weight: float = 1.0          # set after calibration
        # pipe_batch_size controls NLI pairs per forward pass.
        # Each query expands to len(BLOOM_HYPOTHESES)=6 pairs, so actual
        # GPU memory per step = pipe_batch_size * 6 * seq_len * hidden.
        # Default 4 → 24 pairs/step, safe on near-full GPUs.
        self._pipe = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device,
            batch_size=pipe_batch_size,
        )
        print(f"  [NLI] Ready: {model_name}")

    def _to_cpu(self):
        """Move pipeline to CPU (called when CUDA OOM occurs during inference)."""
        import torch
        torch.cuda.empty_cache()
        self._pipe.model = self._pipe.model.to("cpu")
        self._pipe.device = torch.device("cpu")
        print(f"\n  [NLI] {self.name.split('/')[-1]} moved to CPU after inference OOM")

    def predict_proba(self, queries: List[str], batch_size: int = 8) -> np.ndarray:
        all_probs = []
        tag = self.name.split("/")[-1][:30]
        for i in tqdm(range(0, len(queries), batch_size),
                      desc=f"    [{tag}]", leave=False):
            batch = queries[i:i + batch_size]
            for attempt in range(2):
                try:
                    results = self._pipe(batch, BLOOM_HYPOTHESES, multi_label=False)
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and attempt == 0:
                        self._to_cpu()
                    else:
                        raise
            if isinstance(results, dict):
                results = [results]
            for r in results:
                score_map = dict(zip(r["labels"], r["scores"]))
                probs = np.array([score_map.get(h, 0.0) for h in BLOOM_HYPOTHESES],
                                 dtype=np.float32)
                probs /= probs.sum() + 1e-9
                all_probs.append(probs)
        return np.stack(all_probs)   # (N, 6)


# ─── Classifier Member ────────────────────────────────────────────────────────

class ClassifierMember:
    """Fine-tuned sequence classifier. Returns (N, 6) probability matrix."""

    def __init__(self, model_name: str, device):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        print(f"  [CLS] Loading {model_name} ...")
        self.name = model_name
        self.weight: float = 1.0
        self._device = device
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(device).eval()

        id2label = getattr(self._model.config, "id2label", None)
        self._idx_to_bloom = _build_classifier_idx_map(id2label)
        print(f"  [CLS] Ready: {model_name} | idx→bloom: {self._idx_to_bloom}")

    def _to_cpu(self):
        """Move model to CPU (called when CUDA OOM occurs during inference)."""
        import torch
        torch.cuda.empty_cache()
        self._model = self._model.to("cpu")
        self._device = "cpu"
        print(f"\n  [CLS] {self.name.split('/')[-1]} moved to CPU after inference OOM")

    def predict_proba(self, queries: List[str], batch_size: int = 16) -> np.ndarray:
        import torch
        all_probs = []
        tag = self.name.split("/")[-1][:30]
        for i in tqdm(range(0, len(queries), batch_size),
                      desc=f"    [{tag}]", leave=False):
            batch = queries[i:i + batch_size]
            for attempt in range(2):
                try:
                    enc = self._tok(batch, padding=True, truncation=True,
                                    max_length=128, return_tensors="pt")
                    enc = {k: v.to(self._device) for k, v in enc.items()}
                    with torch.no_grad():
                        logits = self._model(**enc).logits
                    probs_raw = torch.softmax(logits, dim=-1).cpu().numpy()
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and attempt == 0:
                        self._to_cpu()
                    else:
                        raise
            for row in probs_raw:
                bloom_probs = np.zeros(6, dtype=np.float32)
                for idx, p in enumerate(row):
                    bl = self._idx_to_bloom.get(idx, idx + 1)
                    if 1 <= bl <= 6:
                        bloom_probs[bl - 1] += float(p)
                bloom_probs /= bloom_probs.sum() + 1e-9
                all_probs.append(bloom_probs)
        return np.stack(all_probs)   # (N, 6)


def _build_classifier_idx_map(id2label) -> Dict[int, int]:
    _name_map = {
        "remember": 1, "remembering": 1, "knowledge": 1,
        "understand": 2, "understanding": 2, "comprehension": 2,
        "apply": 3, "applying": 3, "application": 3,
        "analyze": 4, "analyse": 4, "analysis": 4,
        "evaluate": 5, "evaluating": 5, "evaluation": 5,
        "create": 6, "creating": 6, "synthesis": 6,
    }
    if id2label is None:
        return {i: i + 1 for i in range(6)}
    mapping = {}
    for idx, name in id2label.items():
        n = str(name).lower().strip()
        bloom = next((v for k, v in _name_map.items() if k in n), None)
        if bloom is None:
            try:
                v = int(n)
                bloom = v if 1 <= v <= 6 else (v + 1 if 0 <= v <= 5 else None)
            except ValueError:
                pass
        if bloom is not None:
            mapping[int(idx)] = bloom
    return mapping if len(mapping) >= 3 else {i: i + 1 for i in range(6)}


# ─── Weight Calibration ───────────────────────────────────────────────────────

def _load_kaggle_dataset(name: str, output_dir: str) -> Optional[Tuple[List[str], List[int]]]:
    """Download one Kaggle Bloom dataset; return (texts, 1-indexed bloom labels)."""
    import glob
    import pandas as pd
    try:
        import kagglehub
    except ImportError:
        os.system("pip install kagglehub --break-system-packages -q")
        import kagglehub

    _name_map = {
        "remember": 1, "remembering": 1, "knowledge": 1,
        "understand": 2, "understanding": 2, "comprehension": 2,
        "apply": 3, "applying": 3, "application": 3,
        "analyze": 4, "analyse": 4, "analysis": 4,
        "evaluate": 5, "evaluating": 5, "evaluation": 5,
        "create": 6, "creating": 6, "synthesis": 6,
    }

    def parse_label(val):
        # Drop NaN before any conversion
        try:
            import math
            if isinstance(val, float) and math.isnan(val):
                return -1
        except Exception:
            pass
        if isinstance(val, (int, float)):
            v = int(val)
            return v if 1 <= v <= 6 else (v + 1 if 0 <= v <= 5 else -1)
        n = str(val).lower().strip()
        if n in ("nan", "none", ""):
            return -1
        for k, lv in _name_map.items():
            if k in n:
                return lv
        # Roman numerals
        roman = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6}
        if n in roman:
            return roman[n]
        # "bt1"…"bt6" (vijaydevane dataset format)
        import re
        m = re.match(r"bt[ _]?([1-6])$", n)
        if m:
            return int(m.group(1))
        # "l1"…"l6" or "level1"…"level6"
        m = re.match(r"l(?:evel)?[ _]?([1-6])$", n)
        if m:
            return int(m.group(1))
        try:
            v = int(float(n))
            return v if 1 <= v <= 6 else (v + 1 if 0 <= v <= 5 else -1)
        except ValueError:
            return -1

    try:
        print(f"    Downloading {name} ...")
        path = kagglehub.dataset_download(name)
        texts, labels = [], []
        for csv_path in glob.glob(os.path.join(path, "**/*.csv"), recursive=True):
            try:
                df = pd.read_csv(csv_path, on_bad_lines="skip")
            except Exception:
                df = pd.read_csv(csv_path, error_bad_lines=False)

            # Debug: show columns when nothing was detected previously
            # Find text column — broad search
            text_candidates = ["Question", "question", "Text", "text", "Sentence",
                               "sentence", "query", "Query", "Question_Text",
                               "question_text", "Questions", "questions"]
            text_col = next((c for c in text_candidates if c in df.columns), None)
            if text_col is None:
                # Pick longest average-length object column
                obj_cols = [c for c in df.columns if df[c].dtype == "object"]
                if obj_cols:
                    text_col = max(obj_cols,
                                   key=lambda c: df[c].dropna().astype(str).str.len().mean())

            # Find label column — broad search
            label_candidates = [
                "Bloom's Taxonomy Level", "bloom_level", "Bloom Level",
                "Bloom's Level", "blooms_level", "Bloom_Level",
                "Level", "level", "label", "Label", "cognitive_level",
                "Cognitive Level", "Category", "category", "class", "Class",
                "Taxonomy", "taxonomy", "BT_Level", "bt_level",
            ]
            label_col = next((c for c in label_candidates if c in df.columns), None)
            if label_col is None:
                # Pick non-text column with fewest unique values (likely the label)
                other_cols = [c for c in df.columns if c != text_col]
                if other_cols:
                    label_col = min(other_cols, key=lambda c: df[c].nunique())

            if text_col is None or label_col is None:
                print(f"      Skipping {os.path.basename(csv_path)} "
                      f"— could not detect text/label columns. "
                      f"Columns: {list(df.columns)}")
                continue

            # Drop NaN rows in label column before iterating
            df = df.dropna(subset=[label_col])
            parsed_count = 0
            for _, row in df.iterrows():
                lv = parse_label(row[label_col])
                txt = str(row[text_col]).strip()
                if lv != -1 and len(txt) > 10:
                    texts.append(txt)
                    labels.append(lv)
                    parsed_count += 1

            if parsed_count == 0:
                # Show a sample of label values to aid debugging
                sample_labels = df[label_col].dropna().unique()[:10].tolist()
                print(f"      Skipping {os.path.basename(csv_path)} "
                      f"— 0 valid labels parsed. "
                      f"Sample label values: {sample_labels}")

        print(f"    {name}: {len(texts)} examples")
        return (texts, labels) if texts else None
    except Exception as e:
        print(f"    WARNING: Could not download {name}: {e}")
        return None


def calibrate_weights(members: list, device, n_per_dataset: int = 0,
                      cache_path: str = WEIGHTS_CACHE,
                      batch_size: int = 8) -> Dict[str, float]:
    """
    Evaluate each council member on 3 Kaggle Bloom datasets.
    Returns {model_name: weight} where weight = mean accuracy across datasets.
    Caches results to cache_path.
    """
    print("\n" + "=" * 65)
    print("CALIBRATING COUNCIL WEIGHTS ON 3 KAGGLE BLOOM DATASETS")
    print("=" * 65)

    kaggle_datasets = [
        "vijaydevane/blooms-taxonomy-dataset",
        "abhaygotmare/blooms-taxonomy-questions-level",
        "dineshsheelam/blooms-taxonomy-dataset",
    ]

    # Load all 3 datasets
    cal_data: List[Optional[Tuple[List[str], List[int]]]] = []
    for ds_name in kaggle_datasets:
        result = _load_kaggle_dataset(ds_name, "/tmp/bloom_kaggle_cal")
        cal_data.append(result)

    valid = [r for r in cal_data if r is not None]
    if not valid:
        print("  WARNING: No calibration data available. Falling back to uniform weights.")
        return {m.name: 1.0 / len(members) for m in members}

    # Build stratified hold-out per dataset.
    # n <= 0 means use all examples (no subsampling).
    def stratified_sample(texts, labels, n):
        if n <= 0 or n >= len(texts):
            return list(texts), list(labels)
        from collections import defaultdict
        import random
        random.seed(42)
        buckets = defaultdict(list)
        for t, l in zip(texts, labels):
            buckets[l].append(t)
        sampled_t, sampled_l = [], []
        per_class = max(1, n // 6)
        for lv in range(1, 7):
            pool = buckets[lv]
            random.shuffle(pool)
            for t in pool[:per_class]:
                sampled_t.append(t)
                sampled_l.append(lv)
        return sampled_t, sampled_l

    # Evaluate each member
    # ACC_CAP: any per-dataset accuracy above this is capped before averaging.
    # Without capping, a fine-tuned classifier that was trained on one of the
    # Kaggle datasets will score ~1.0 on it, inflating its weight to dominate
    # the ensemble. The cap keeps weights honest across generalist models too.
    ACC_CAP = 0.75

    weights: Dict[str, float] = {}
    for member in members:
        print(f"\n  Evaluating: {member.name}")
        dataset_accs = []
        for ds_idx, (ds_name, data) in enumerate(zip(kaggle_datasets, cal_data)):
            if data is None:
                print(f"    Dataset {ds_idx+1} ({ds_name.split('/')[0]}): SKIPPED (download failed)")
                continue
            texts, labels = data
            sample_t, sample_l = stratified_sample(texts, labels, n_per_dataset)
            if not sample_t:
                continue
            proba = member.predict_proba(sample_t, batch_size=batch_size)
            preds = proba.argmax(axis=1) + 1          # 1-indexed
            raw_acc = float(np.mean(np.array(preds) == np.array(sample_l)))
            acc = min(raw_acc, ACC_CAP)
            note = f" (capped from {raw_acc:.3f})" if raw_acc > ACC_CAP else ""
            dataset_accs.append(acc)
            print(f"    Dataset {ds_idx+1} ({ds_name.split('/')[0]}): "
                  f"acc={acc:.3f}{note}  (n={len(sample_t)})")
        mean_acc = float(np.mean(dataset_accs)) if dataset_accs else 0.10
        print(f"  → Mean accuracy (capped): {mean_acc:.4f}")
        weights[member.name] = mean_acc

    # Normalize so weights sum to 1
    total = sum(weights.values()) or 1.0
    weights = {k: v / total for k, v in weights.items()}

    # Cache to disk
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"\n  Weights saved to {cache_path}")

    print("\n  Final calibrated weights:")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"    {name.split('/')[-1]:<45s} {w:.4f}")

    return weights


# ─── Council ──────────────────────────────────────────────────────────────────

class BloomCouncil:
    """Ensemble of 5 council members with calibrated accuracy-based weights."""

    def __init__(self, members: list, weights: Dict[str, float]):
        self.members = members
        # Assign weights; fall back to uniform if a member isn't in the weights dict
        raw = [weights.get(m.name, 1.0 / len(members)) for m in members]
        total = sum(raw) or 1.0
        self._weights = [w / total for w in raw]
        for m, w in zip(members, self._weights):
            m.weight = w

        print(f"\n  Council of {len(members)} — calibrated weights:")
        for m, w in sorted(zip(members, self._weights), key=lambda x: -x[1]):
            print(f"    {m.name.split('/')[-1]:<45s} weight={w:.4f}")

    def predict(self, queries: List[str], batch_size: int = 32):
        """
        Returns:
          levels      : List[int]        1-indexed Bloom level per query
          confidences : List[float]      max probability in ensemble distribution
          votes       : List[List[int]]  per-member predictions (debugging)
          weights_used: List[float]      weights in same order as votes
        """
        member_probs = []
        member_preds = []
        for m, w in zip(self.members, self._weights):
            proba = m.predict_proba(queries, batch_size)   # (N, 6)
            member_probs.append(proba * w)
            member_preds.append((proba.argmax(axis=1) + 1).tolist())

        ensemble = sum(member_probs)                        # (N, 6) weighted avg
        levels = (ensemble.argmax(axis=1) + 1).tolist()
        confidences = ensemble.max(axis=1).tolist()

        votes = [
            [int(member_preds[mi][qi]) for mi in range(len(self.members))]
            for qi in range(len(queries))
        ]
        return levels, confidences, votes, self._weights


# ─── Annotation helpers ───────────────────────────────────────────────────────

def annotate_file(input_path: str, output_path: str, council: BloomCouncil,
                  batch_size: int = 32):
    with open(input_path) as f:
        records = [json.loads(l.strip()) for l in f]

    queries = [r["query"] for r in records]
    print(f"\n  Annotating {len(queries)} queries from {os.path.basename(input_path)}")

    old_dist = Counter(r.get("bloom_level") for r in records)
    print(f"  Old distribution: {dict(sorted(old_dist.items()))}")

    levels, confidences, votes, weights_used = council.predict(queries, batch_size)

    changed = 0
    for r, lvl, conf, vote in zip(records, levels, confidences, votes):
        if r.get("bloom_level") != lvl:
            changed += 1
        r["bloom_level"] = lvl
        r["bloom_source"] = "council_v2"
        r["bloom_confidence"] = round(float(conf), 4)
        r["bloom_votes"] = vote
        r["bloom_weights"] = [round(w, 4) for w in weights_used]

    new_dist = Counter(r["bloom_level"] for r in records)
    print(f"  New distribution: {dict(sorted(new_dist.items()))}")
    print(f"  Changed: {changed}/{len(records)} ({changed/len(records):.1%})")
    print(f"  Avg confidence: {float(np.mean(confidences)):.3f}")

    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Update cache file used by EducationalRetrievalDataset at train time
    cache_path = input_path + ".bloom_cache.json"
    if os.path.exists(cache_path):
        cache = {r["query_id"]: r["bloom_level"]
                 for r in records if "query_id" in r}
        with open(cache_path, "w") as f:
            json.dump(cache, f)
        print(f"  Updated cache: {os.path.basename(cache_path)}")

    return old_dist, new_dist


def collect_files(args) -> List[str]:
    paths = []
    if getattr(args, "input", None):
        return [args.input]
    if getattr(args, "data_dir", None):
        for split in args.splits:
            p = os.path.join(args.data_dir, f"{split}.jsonl")
            if os.path.exists(p):
                paths.append(p)
            else:
                print(f"  Skipping {split} — not found at {p}")
    if getattr(args, "beir_root", None) and getattr(args, "datasets", None):
        for ds in args.datasets:
            for split in args.splits:
                p = os.path.join(args.beir_root, ds, f"{split}.jsonl")
                if os.path.exists(p):
                    paths.append(p)
                else:
                    print(f"  Skipping {ds}/{split} — not found")
    return paths


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bloom annotation — council of 5 with calibrated weights"
    )
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--beir_root", default=None)
    parser.add_argument("--datasets", nargs="+",
                        default=["scifact", "nfcorpus", "fiqa"])
    parser.add_argument("--input", default=None,
                        help="Annotate a single JSONL file")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Queries per outer predict loop iteration (default 8)")
    parser.add_argument("--pipe_batch_size", type=int, default=4,
                        help="NLI pairs per GPU forward pass — each query expands to "
                             "6 pairs, so GPU load = pipe_batch_size*6 (default 4)")
    parser.add_argument("--cal_samples", type=int, default=0,
                        help="Samples per Kaggle dataset for calibration. "
                             "0 = use all examples (default). "
                             "Set to e.g. 60 to cap and speed up calibration.")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Re-run weight calibration even if cache exists")
    parser.add_argument("--calibrate_only", action="store_true",
                        help="Only calibrate weights, skip annotation")
    parser.add_argument("--weights_cache", default=WEIGHTS_CACHE,
                        help=f"Path to weights JSON cache (default: {WEIGHTS_CACHE})")
    parser.add_argument("--no_backup", action="store_true")
    args = parser.parse_args()

    if not args.calibrate_only and not args.data_dir \
            and not args.beir_root and not args.input:
        parser.error("Provide --data_dir, --beir_root, --input, or --calibrate_only")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print("Bloom Level Annotation — Council of 5 Models")
    print("=" * 65)
    print(f"  Device : {device}")
    print(f"  Members: {len(COUNCIL_MEMBERS)}")

    # ── Load all council members ──
    # Each candidate is tried on `device` first; on CUDA OOM we clear the cache
    # and retry on CPU so the council still runs even on a near-full GPU.
    members = []
    for spec in COUNCIL_MEMBERS:
        loaded = False
        candidates = [spec["name"]] + ([spec["fallback"]] if spec["fallback"] else [])
        for candidate in candidates:
            for try_device in ([device, "cpu"] if device != "cpu" else ["cpu"]):
                try:
                    if spec["type"] == "nli":
                        m = NLIMember(candidate, try_device,
                                      pipe_batch_size=args.pipe_batch_size)
                    else:
                        m = ClassifierMember(candidate, try_device)
                    if candidate != spec["name"]:
                        print(f"  NOTE: Using fallback {candidate} for {spec['name']}")
                    if try_device != device:
                        print(f"  NOTE: {candidate} loaded on CPU (CUDA OOM)")
                    members.append(m)
                    loaded = True
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and try_device != "cpu":
                        import torch
                        torch.cuda.empty_cache()
                        print(f"  CUDA OOM for {candidate} — retrying on CPU ...")
                    else:
                        print(f"  WARNING: Could not load {candidate} on {try_device}: "
                              f"{str(e)[:120]}")
                        break  # non-OOM error — don't retry on CPU, try next candidate
                except Exception as e:
                    print(f"  WARNING: Could not load {candidate}: {str(e)[:120]}")
                    break  # auth error etc. — skip to fallback model
            if loaded:
                break
        if not loaded:
            print(f"  SKIPPING {spec['name']} (all candidates failed)")

    if len(members) < 2:
        raise RuntimeError(f"Only {len(members)} member(s) loaded — need at least 2.")

    # ── Calibrate weights ──
    loaded_names = set(m.name for m in members)
    use_cache = os.path.exists(args.weights_cache) and not args.recalibrate
    if use_cache:
        with open(args.weights_cache) as f:
            cached = json.load(f)
        cached_names = set(cached.keys())
        # Invalidate cache when the loaded members don't match what was calibrated
        if not loaded_names.issubset(cached_names):
            missing = loaded_names - cached_names
            print(f"\n  Cache miss for members: {missing}")
            print("  Stale cache — running recalibration ...")
            use_cache = False
        else:
            weights = cached
            print(f"\n  Loaded cached weights from {args.weights_cache}")
            print("  (Use --recalibrate to refresh)\n")
            print("  Cached weights (for loaded members):")
            for name in sorted(loaded_names, key=lambda n: -weights.get(n, 0)):
                print(f"    {name.split('/')[-1]:<45s} {weights.get(name, 0):.4f}")
    if not use_cache:
        weights = calibrate_weights(members, device,
                                    n_per_dataset=args.cal_samples,
                                    cache_path=args.weights_cache,
                                    batch_size=args.batch_size)

    if args.calibrate_only:
        print("\n  --calibrate_only set. Done.")
        return

    # ── Build council and annotate ──
    council = BloomCouncil(members, weights)

    files = collect_files(args)
    if not files:
        print("No files found to annotate.")
        return

    all_old: Counter = Counter()
    all_new: Counter = Counter()

    for path in files:
        if not args.no_backup:
            backup = path + ".pre_council_backup"
            if not os.path.exists(backup):
                shutil.copy2(path, backup)
                print(f"  Backed up → {os.path.basename(backup)}")

        old_d, new_d = annotate_file(path, path, council, args.batch_size)
        all_old.update(old_d)
        all_new.update(new_d)

    # ── Summary ──
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Level':<12s} {'Before':>8s} {'After':>8s} {'Δ':>6s}")
    print(f"  {'-'*38}")
    for level in range(1, 7):
        name = BLOOM_NAMES[level]
        old = all_old.get(level, 0)
        new = all_new.get(level, 0)
        print(f"  {name:<12s} {old:>8d} {new:>8d} {new - old:>+6d}")

    print(f"\n  NEXT STEPS:")
    print(f"    1. Re-mine curriculum negatives:")
    print(f"       python data/curriculum_negatives.py \\")
    print(f"           --pairs /tmp/data/real/train.jsonl \\")
    print(f"           --corpus /tmp/data/real/corpus.jsonl \\")
    print(f"           --output /tmp/data/real/train_curriculum.jsonl \\")
    print(f"           --stage 0.7 --num_neg 3")
    print(f"    2. Retrain: python scripts/train_bam.py --config configs/bam.yaml")


if __name__ == "__main__":
    main()
