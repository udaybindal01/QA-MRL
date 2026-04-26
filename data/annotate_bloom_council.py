"""
Bloom Level Annotation — Diverse Council of 5 (SOTA ensemble).

Five paradigmatically distinct members vote via weighted soft-voting.
Weights are calibrated by evaluating each member on a held-out split of
3 Kaggle Bloom taxonomy datasets, then normalising by mean accuracy.

Council members:
  1. MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli  [NLI]
     SOTA zero-shot NLI (440M). Best open-weight zero-shot classifier.

  2. facebook/bart-large-mnli                                    [NLI]
     BART NLI (400M). Different encoder-decoder architecture from DeBERTa
     — makes different errors, adds genuine diversity.

  3. cip29/bert-blooms-taxonomy-classifier                       [Classifier]
     BERT fine-tuned on educational Bloom data (110M).
     Domain knowledge; capped at 0.75 to prevent training-set weight inflation.

  4. sentence-transformers/all-MiniLM-L6-v2  +  sklearn k-NN    [Semantic k-NN]
     Embeds the labelled Kaggle corpus with a 22M sentence encoder then
     classifies each query by k=7 weighted nearest neighbours.
     Completely data-driven — no hypothesis engineering needed.

  5. TF-IDF (1-2gram) + Bloom verb features + LogReg             [Lexical]
     Trained on Bloom verb lists and Kaggle corpus.
     Grounding mechanism: reliably catches queries with explicit cognitive
     verbs (define/explain/analyze/evaluate/create) without overthinking.

Calibration protocol (per dataset, 80/20 split):
  - Data-driven members (k-NN, Lexical) are FIT on the 80% train split.
  - ALL 5 members are EVALUATED on the 20% held-out split.
  - Final weight = mean accuracy across 3 datasets, capped at 0.75.
  - Cached to /tmp/bloom_council_weights.json.

Output fields per record:
  bloom_level       — 1-indexed Bloom level (1=Remember … 6=Create)
  bloom_source      — "council_v3"
  bloom_confidence  — float 0-1, ensemble confidence
  bloom_votes       — per-member predictions (debugging)
  bloom_weights     — per-member weights used

Usage:
    python data/annotate_bloom_council.py --data_dir /tmp/data/real

    python data/annotate_bloom_council.py \\
        --beir_root /tmp/data/beir --datasets scifact nfcorpus fiqa

    python data/annotate_bloom_council.py --data_dir /tmp/data/real \\
        --pipe_batch_size 2 --batch_size 4

    python data/annotate_bloom_council.py --data_dir /tmp/data/real \\
        --recalibrate --cal_samples 0     # 0 = use all Kaggle samples
"""

import argparse
import json
import os
import re
import shutil
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply",
               4: "Analyze",  5: "Evaluate",   6: "Create"}

BLOOM_HYPOTHESES = [
    "This query is asking to recall or retrieve a specific fact, name, or definition.",
    "This query is asking to explain, describe, or summarize how something works.",
    "This query is asking how to use or apply knowledge to solve a practical problem.",
    "This query is asking to compare, contrast, or examine the relationship between things.",
    "This query requires making a decision or forming an opinion about the worth or validity of something.",
    "This query is asking to design, propose, or synthesize something new.",
]

WEIGHTS_CACHE = "/tmp/bloom_council_weights.json"


# ─── Council member specs ─────────────────────────────────────────────────────

COUNCIL_SPECS = [
    {"name": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
     "type": "nli",      "fallback": None},
    {"name": "facebook/bart-large-mnli",
     "type": "nli",      "fallback": None},
    {"name": "cip29/bert-blooms-taxonomy-classifier",
     "type": "classifier","fallback": None},
    {"name": "sentence-transformers/all-MiniLM-L6-v2",
     "type": "knn",      "fallback": "sentence-transformers/paraphrase-MiniLM-L3-v2"},
    {"name": "lexical-tfidf-svm",
     "type": "lexical",  "fallback": None},
]


# ─── NLI member ───────────────────────────────────────────────────────────────

class NLIMember:
    def __init__(self, model_name: str, device, pipe_batch_size: int = 4):
        from transformers import pipeline
        print(f"  [NLI] Loading {model_name} (pipe_batch={pipe_batch_size}) ...")
        self.name = model_name
        self.weight: float = 1.0
        self._pipe = pipeline("zero-shot-classification", model=model_name,
                               device=device, batch_size=pipe_batch_size)
        print(f"  [NLI] Ready: {model_name}")

    def _to_cpu(self):
        import torch
        torch.cuda.empty_cache()
        self._pipe.model = self._pipe.model.to("cpu")
        self._pipe.device = torch.device("cpu")
        print(f"\n  [NLI] {self.name.split('/')[-1]} → CPU (inference OOM)")

    def predict_proba(self, queries: List[str], batch_size: int = 8) -> np.ndarray:
        all_probs = []
        tag = self.name.split("/")[-1][:28]
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
        return np.stack(all_probs)


# ─── Fine-tuned classifier member ─────────────────────────────────────────────

class ClassifierMember:
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
        self._idx_to_bloom = _build_cls_idx_map(id2label)
        print(f"  [CLS] Ready: {model_name}")

    def _to_cpu(self):
        import torch
        torch.cuda.empty_cache()
        self._model = self._model.to("cpu")
        self._device = "cpu"
        print(f"\n  [CLS] {self.name.split('/')[-1]} → CPU (inference OOM)")

    def predict_proba(self, queries: List[str], batch_size: int = 16) -> np.ndarray:
        import torch
        all_probs = []
        tag = self.name.split("/")[-1][:28]
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
        return np.stack(all_probs)


def _build_cls_idx_map(id2label) -> Dict[int, int]:
    _nm = {"remember": 1, "remembering": 1, "knowledge": 1,
           "understand": 2, "understanding": 2, "comprehension": 2,
           "apply": 3, "applying": 3, "application": 3,
           "analyze": 4, "analyse": 4, "analysis": 4,
           "evaluate": 5, "evaluating": 5, "evaluation": 5,
           "create": 6, "creating": 6, "synthesis": 6}
    if id2label is None:
        return {i: i + 1 for i in range(6)}
    mapping = {}
    for idx, name in id2label.items():
        n = str(name).lower().strip()
        bloom = next((v for k, v in _nm.items() if k in n), None)
        if bloom is None:
            try:
                v = int(n)
                bloom = v if 1 <= v <= 6 else (v + 1 if 0 <= v <= 5 else None)
            except ValueError:
                pass
        if bloom is not None:
            mapping[int(idx)] = bloom
    return mapping if len(mapping) >= 3 else {i: i + 1 for i in range(6)}


# ─── Semantic k-NN member ─────────────────────────────────────────────────────

class KNNMember:
    """
    Semantic k-NN voter.
    Encodes the labelled Kaggle corpus with a lightweight sentence encoder
    (all-MiniLM-L6-v2, 22M params), then classifies each query by finding
    its 7 nearest neighbours via cosine similarity (dot product on L2-normalised
    embeddings). Votes are similarity-weighted for soft probabilities.
    """
    K = 7

    def __init__(self, encoder_name: str, device):
        self.name = encoder_name
        self.weight: float = 1.0
        self._device = device
        self._encoder = None
        self._index_embs: Optional[np.ndarray] = None
        self._index_labels: Optional[np.ndarray] = None

    def fit(self, texts: List[str], labels: List[int]):
        from sentence_transformers import SentenceTransformer
        print(f"  [KNN] Loading encoder {self.name} ...")
        self._encoder = SentenceTransformer(self.name, device=self._device)
        print(f"  [KNN] Encoding {len(texts)} reference examples ...")
        self._index_embs = self._encoder.encode(
            texts, batch_size=256, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)
        self._index_labels = np.array(labels, dtype=np.int32)
        print(f"  [KNN] Index ready: {self._index_embs.shape}")

    def predict_proba(self, queries: List[str], batch_size: int = 256) -> np.ndarray:
        if self._encoder is None:
            raise RuntimeError("KNNMember not fitted — run calibrate_weights first")
        query_embs = self._encoder.encode(
            queries, batch_size=batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)                             # (Q, dim)
        sims = query_embs @ self._index_embs.T           # (Q, N) cosine sim
        k = min(self.K, self._index_embs.shape[0])
        top_k_idx = np.argpartition(-sims, k, axis=1)[:, :k]
        all_probs = []
        for qi in range(len(queries)):
            nn_labels = self._index_labels[top_k_idx[qi]]
            nn_sims   = sims[qi, top_k_idx[qi]]
            probs = np.zeros(6, dtype=np.float32)
            for lv, sim in zip(nn_labels, nn_sims):
                if 1 <= lv <= 6:
                    probs[lv - 1] += max(0.0, float(sim))
            probs /= probs.sum() + 1e-9
            all_probs.append(probs)
        return np.stack(all_probs)


# ─── Lexical TF-IDF + SVM member ──────────────────────────────────────────────

class LexicalMember:
    """
    Lexical grounding voter.
    TF-IDF (1-2gram, 20k features) + explicit Bloom action-verb counts,
    trained on the Kaggle Bloom corpus via Logistic Regression.
    Reliable for queries with clear cognitive-level verbs; prevents deep
    models from overthinking simple factual queries.
    """
    BLOOM_VERBS = {
        1: ["define", "list", "recall", "name", "identify", "state", "label",
            "match", "recognize", "select", "locate", "cite", "who", "when", "what"],
        2: ["explain", "summarize", "interpret", "classify", "describe",
            "discuss", "paraphrase", "translate", "predict", "outline", "review",
            "restate", "convert", "infer", "why", "how"],
        3: ["apply", "use", "solve", "demonstrate", "calculate", "compute", "show",
            "operate", "execute", "implement", "modify", "prepare", "produce", "chart"],
        4: ["analyze", "compare", "contrast", "differentiate", "examine", "test",
            "distinguish", "relate", "breakdown", "organize", "attribute", "separate",
            "deconstruct", "inspect", "investigate"],
        5: ["evaluate", "judge", "justify", "critique", "assess", "defend", "argue",
            "debate", "rate", "recommend", "prioritize", "conclude", "appraise",
            "choose", "support"],
        6: ["create", "design", "construct", "develop", "formulate", "plan",
            "compose", "generate", "invent", "produce", "propose", "build", "devise",
            "hypothesize", "originate"],
    }

    def __init__(self):
        self.name = "lexical-tfidf-svm"
        self.weight: float = 1.0
        self._model = None
        self._vectorizer = None

    def _verb_features(self, texts: List[str]) -> np.ndarray:
        feats = np.zeros((len(texts), 6), dtype=np.float32)
        for i, text in enumerate(texts):
            words = set(re.findall(r'\b\w+\b', text.lower()))
            for lv, verbs in self.BLOOM_VERBS.items():
                feats[i, lv - 1] = sum(1 for v in verbs if v in words)
        return feats

    def fit(self, texts: List[str], labels: List[int]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from scipy.sparse import hstack, csr_matrix
        print(f"  [LEX] Training on {len(texts)} examples ...")
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000,
                                           sublinear_tf=True, min_df=2)
        tfidf = self._vectorizer.fit_transform(texts)
        verb  = csr_matrix(self._verb_features(texts))
        X = hstack([tfidf, verb])
        y = np.array(labels) - 1     # 0-indexed
        self._model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                                         solver="lbfgs", multi_class="multinomial")
        self._model.fit(X, y)
        print(f"  [LEX] Trained. Classes present: {(self._model.classes_ + 1).tolist()}")

    def predict_proba(self, queries: List[str], batch_size: int = 4096) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LexicalMember not fitted — run calibrate_weights first")
        from scipy.sparse import hstack, csr_matrix
        tfidf = self._vectorizer.transform(queries)
        verb  = csr_matrix(self._verb_features(queries))
        X = hstack([tfidf, verb])
        raw = self._model.predict_proba(X)          # (N, n_classes)
        full = np.zeros((len(queries), 6), dtype=np.float32)
        for i, cls in enumerate(self._model.classes_):
            if 0 <= cls < 6:
                full[:, cls] = raw[:, i]
        full /= full.sum(axis=1, keepdims=True) + 1e-9
        return full


# ─── Kaggle data loading ──────────────────────────────────────────────────────

def _parse_label(val) -> int:
    import math
    if isinstance(val, float) and math.isnan(val):
        return -1
    if isinstance(val, (int, float)):
        v = int(val)
        return v if 1 <= v <= 6 else (v + 1 if 0 <= v <= 5 else -1)
    n = str(val).lower().strip()
    if n in ("nan", "none", ""):
        return -1
    _nm = {"remember": 1, "remembering": 1, "knowledge": 1,
           "understand": 2, "understanding": 2, "comprehension": 2,
           "apply": 3, "applying": 3, "application": 3,
           "analyze": 4, "analyse": 4, "analysis": 4,
           "evaluate": 5, "evaluating": 5, "evaluation": 5,
           "create": 6, "creating": 6, "synthesis": 6}
    for k, lv in _nm.items():
        if k in n:
            return lv
    roman = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6}
    if n in roman:
        return roman[n]
    m = re.match(r"bt[ _]?([1-6])$", n)
    if m:
        return int(m.group(1))
    m = re.match(r"l(?:evel)?[ _]?([1-6])$", n)
    if m:
        return int(m.group(1))
    try:
        v = int(float(n))
        return v if 1 <= v <= 6 else (v + 1 if 0 <= v <= 5 else -1)
    except ValueError:
        return -1


def _load_kaggle_dataset(name: str) -> Optional[Tuple[List[str], List[int]]]:
    import glob
    import pandas as pd
    try:
        import kagglehub
    except ImportError:
        os.system("pip install kagglehub --break-system-packages -q")
        import kagglehub
    try:
        print(f"    Downloading {name} ...")
        path = kagglehub.dataset_download(name)
        texts, labels = [], []
        for csv_path in glob.glob(os.path.join(path, "**/*.csv"), recursive=True):
            try:
                df = pd.read_csv(csv_path, on_bad_lines="skip")
            except Exception:
                try:
                    df = pd.read_csv(csv_path, error_bad_lines=False)
                except Exception:
                    continue
            text_candidates = ["Question", "question", "Text", "text", "Sentence",
                               "sentence", "query", "Query", "Question_Text",
                               "question_text", "Questions", "questions"]
            text_col = next((c for c in text_candidates if c in df.columns), None)
            if text_col is None:
                obj_cols = [c for c in df.columns if df[c].dtype == "object"]
                if obj_cols:
                    text_col = max(obj_cols,
                                   key=lambda c: df[c].dropna().astype(str).str.len().mean())
            label_candidates = ["Bloom's Taxonomy Level", "bloom_level", "Bloom Level",
                                "Bloom's Level", "blooms_level", "Bloom_Level",
                                "Level", "level", "label", "Label", "cognitive_level",
                                "Category", "category", "class", "Class",
                                "Taxonomy", "taxonomy", "BT_Level", "bt_level"]
            label_col = next((c for c in label_candidates if c in df.columns), None)
            if label_col is None:
                other = [c for c in df.columns if c != text_col]
                if other:
                    label_col = min(other, key=lambda c: df[c].nunique())
            if text_col is None or label_col is None:
                continue
            df = df.dropna(subset=[label_col])
            parsed = 0
            for _, row in df.iterrows():
                lv = _parse_label(row[label_col])
                txt = str(row[text_col]).strip()
                if lv != -1 and len(txt) > 10:
                    texts.append(txt)
                    labels.append(lv)
                    parsed += 1
            if parsed == 0:
                sample = df[label_col].dropna().unique()[:8].tolist()
                print(f"      0 valid labels in {os.path.basename(csv_path)}. "
                      f"Sample values: {sample}")
        print(f"    {name}: {len(texts)} examples")
        return (texts, labels) if texts else None
    except Exception as e:
        print(f"    WARNING: {name}: {e}")
        return None


# ─── Calibration ──────────────────────────────────────────────────────────────

def calibrate_weights(members: list, n_per_dataset: int = 0,
                      cache_path: str = WEIGHTS_CACHE,
                      batch_size: int = 8) -> Dict[str, float]:
    """
    Download 3 Kaggle Bloom datasets. For each dataset split 80/20:
      - Fit data-driven members (KNN, Lexical) on 80% train
      - Evaluate ALL members on 20% held-out
    Weight = mean accuracy (capped at 0.75) across 3 datasets.
    """
    import random
    random.seed(42)

    print("\n" + "=" * 65)
    print("CALIBRATING COUNCIL WEIGHTS ON 3 KAGGLE BLOOM DATASETS")
    print("=" * 65)

    kaggle_datasets = [
        "vijaydevane/blooms-taxonomy-dataset",
        "abhaygotmare/blooms-taxonomy-questions-level",
        "dineshsheelam/blooms-taxonomy-dataset",
    ]

    cal_data: List[Optional[Tuple[List[str], List[int]]]] = []
    for ds_name in kaggle_datasets:
        cal_data.append(_load_kaggle_dataset(ds_name))

    if not any(r is not None for r in cal_data):
        print("  WARNING: No calibration data. Falling back to uniform weights.")
        return {m.name: 1.0 / len(members) for m in members}

    ACC_CAP = 0.75
    weights: Dict[str, float] = {m.name: [] for m in members}  # accumulate per-ds

    for ds_idx, (ds_name, data) in enumerate(zip(kaggle_datasets, cal_data)):
        if data is None:
            print(f"\n  Dataset {ds_idx+1} ({ds_name.split('/')[0]}): SKIPPED")
            continue

        texts, labels = data
        # Optional cap
        if n_per_dataset > 0 and n_per_dataset < len(texts):
            from collections import defaultdict
            buckets: Dict = defaultdict(list)
            for t, l in zip(texts, labels):
                buckets[l].append(t)
            texts_s, labels_s = [], []
            per_class = max(1, n_per_dataset // 6)
            for lv in range(1, 7):
                pool = buckets[lv]
                random.shuffle(pool)
                for t in pool[:per_class]:
                    texts_s.append(t)
                    labels_s.append(lv)
            texts, labels = texts_s, labels_s

        # 80 / 20 split (stratified by label)
        from collections import defaultdict
        buckets = defaultdict(list)
        for t, l in zip(texts, labels):
            buckets[l].append((t, l))
        train_t, train_l, test_t, test_l = [], [], [], []
        for lv in range(1, 7):
            pool = buckets[lv]
            random.shuffle(pool)
            split = max(1, int(len(pool) * 0.8))
            for t, l in pool[:split]:
                train_t.append(t); train_l.append(l)
            for t, l in pool[split:]:
                test_t.append(t); test_l.append(l)

        print(f"\n  Dataset {ds_idx+1} ({ds_name.split('/')[0]}): "
              f"train={len(train_t)}, test={len(test_t)}")

        if not test_t:
            print("    Too few examples for evaluation — skipping.")
            continue

        # Fit data-driven members on the TRAIN split of this dataset
        for member in members:
            if isinstance(member, (KNNMember, LexicalMember)):
                member.fit(train_t, train_l)

        # Evaluate ALL members on held-out TEST split
        test_labels_arr = np.array(test_l)
        for member in members:
            print(f"    Evaluating: {member.name.split('/')[-1][:40]}")
            proba = member.predict_proba(test_t, batch_size=batch_size)
            preds = proba.argmax(axis=1) + 1
            raw_acc = float(np.mean(preds == test_labels_arr))
            acc = min(raw_acc, ACC_CAP)
            note = f" (capped from {raw_acc:.3f})" if raw_acc > ACC_CAP else ""
            print(f"      acc={acc:.3f}{note}")
            weights[member.name].append(acc)

    # Average across datasets
    final: Dict[str, float] = {}
    print("\n  Final per-member mean accuracy:")
    for member in members:
        accs = weights[member.name]
        mean = float(np.mean(accs)) if accs else 0.10
        final[member.name] = mean
        print(f"    {member.name.split('/')[-1]:<45s} {mean:.4f}  "
              f"(datasets: {[f'{a:.3f}' for a in accs]})")

    # Normalize
    total = sum(final.values()) or 1.0
    final = {k: v / total for k, v in final.items()}

    os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".",
                exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n  Weights saved → {cache_path}")
    print("\n  Normalised weights:")
    for name, w in sorted(final.items(), key=lambda x: -x[1]):
        print(f"    {name.split('/')[-1]:<45s} {w:.4f}")

    return final


# ─── Council ──────────────────────────────────────────────────────────────────

class BloomCouncil:
    def __init__(self, members: list, weights: Dict[str, float]):
        self.members = members
        raw = [weights.get(m.name, 1.0 / len(members)) for m in members]
        total = sum(raw) or 1.0
        self._weights = [w / total for w in raw]
        for m, w in zip(members, self._weights):
            m.weight = w
        print(f"\n  Council of {len(members)} assembled:")
        for m, w in sorted(zip(members, self._weights), key=lambda x: -x[1]):
            print(f"    {m.name.split('/')[-1]:<45s} weight={w:.4f}")

    def predict(self, queries: List[str], batch_size: int = 8):
        member_probs, member_preds = [], []
        for m, w in zip(self.members, self._weights):
            proba = m.predict_proba(queries, batch_size)
            member_probs.append(proba * w)
            member_preds.append((proba.argmax(axis=1) + 1).tolist())
        ensemble   = sum(member_probs)
        levels     = (ensemble.argmax(axis=1) + 1).tolist()
        confidences = ensemble.max(axis=1).tolist()
        votes = [[int(member_preds[mi][qi]) for mi in range(len(self.members))]
                 for qi in range(len(queries))]
        return levels, confidences, votes, self._weights


# ─── Annotation ───────────────────────────────────────────────────────────────

def annotate_file(path: str, council: BloomCouncil, batch_size: int = 8):
    with open(path) as f:
        records = [json.loads(l.strip()) for l in f]
    queries = [r["query"] for r in records]
    print(f"\n  Annotating {len(queries)} queries: {os.path.basename(path)}")
    old_dist = Counter(r.get("bloom_level") for r in records)
    print(f"  Old: {dict(sorted(old_dist.items()))}")

    levels, confidences, votes, weights_used = council.predict(queries, batch_size)

    changed = 0
    for r, lvl, conf, vote in zip(records, levels, confidences, votes):
        if r.get("bloom_level") != lvl:
            changed += 1
        r["bloom_level"]      = lvl
        r["bloom_source"]     = "council_v3"
        r["bloom_confidence"] = round(float(conf), 4)
        r["bloom_votes"]      = vote
        r["bloom_weights"]    = [round(w, 4) for w in weights_used]

    new_dist = Counter(r["bloom_level"] for r in records)
    print(f"  New: {dict(sorted(new_dist.items()))}")
    print(f"  Changed: {changed}/{len(records)} ({changed/len(records):.1%})  "
          f"Avg confidence: {float(np.mean(confidences)):.3f}")

    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    cache_path = path + ".bloom_cache.json"
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
        description="Bloom annotation — diverse council of 5 with calibrated weights"
    )
    parser.add_argument("--data_dir",   default=None)
    parser.add_argument("--splits",     nargs="+", default=["train", "val", "test"])
    parser.add_argument("--beir_root",  default=None)
    parser.add_argument("--datasets",   nargs="+", default=["scifact", "nfcorpus", "fiqa"])
    parser.add_argument("--input",      default=None)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Queries per predict-loop iteration (default 8)")
    parser.add_argument("--pipe_batch_size", type=int, default=4,
                        help="NLI pairs per GPU forward (each query → 6 pairs, default 4)")
    parser.add_argument("--cal_samples", type=int, default=0,
                        help="Kaggle samples per dataset for calibration; 0=all (default)")
    parser.add_argument("--recalibrate",    action="store_true")
    parser.add_argument("--calibrate_only", action="store_true")
    parser.add_argument("--weights_cache",  default=WEIGHTS_CACHE)
    parser.add_argument("--no_backup",      action="store_true")
    args = parser.parse_args()

    if not args.calibrate_only and not args.data_dir \
            and not args.beir_root and not args.input:
        parser.error("Provide --data_dir, --beir_root, --input, or --calibrate_only")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print("Bloom Level Annotation — Diverse Council of 5")
    print("=" * 65)
    print(f"  Device        : {device}")
    print(f"  pipe_batch    : {args.pipe_batch_size}")
    print(f"  batch_size    : {args.batch_size}")
    print(f"  cal_samples   : {'all' if args.cal_samples <= 0 else args.cal_samples}")

    # ── Load members ──
    members = []
    for spec in COUNCIL_SPECS:
        if spec["type"] in ("knn", "lexical"):
            # Data-driven: constructed directly (no load needed yet)
            if spec["type"] == "knn":
                members.append(KNNMember(spec["name"], device))
            else:
                members.append(LexicalMember())
            print(f"  [{spec['type'].upper()}] Ready: {spec['name']}")
            continue

        loaded = False
        candidates = [spec["name"]] + ([spec["fallback"]] if spec["fallback"] else [])
        for candidate in candidates:
            for try_device in ([device, "cpu"] if device != "cpu" else ["cpu"]):
                try:
                    if spec["type"] == "nli":
                        m = NLIMember(candidate, try_device, args.pipe_batch_size)
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
                        import torch as _t
                        _t.cuda.empty_cache()
                        print(f"  CUDA OOM for {candidate} — retrying on CPU ...")
                    else:
                        print(f"  WARNING: {candidate} on {try_device}: {str(e)[:100]}")
                        break
                except Exception as e:
                    print(f"  WARNING: {candidate}: {str(e)[:100]}")
                    break
            if loaded:
                break
        if not loaded:
            print(f"  SKIPPING {spec['name']}")

    if len(members) < 2:
        raise RuntimeError(f"Only {len(members)} member(s) loaded — need at least 2.")

    # ── Calibrate weights ──
    loaded_names = {m.name for m in members}
    use_cache = os.path.exists(args.weights_cache) and not args.recalibrate
    if use_cache:
        with open(args.weights_cache) as f:
            cached = json.load(f)
        if not loaded_names.issubset(set(cached.keys())):
            missing = loaded_names - set(cached.keys())
            print(f"\n  Stale cache (missing {missing}) — recalibrating ...")
            use_cache = False
        else:
            weights = cached
            print(f"\n  Loaded cached weights from {args.weights_cache}")
            print("  (Use --recalibrate to refresh)\n")
            for n in sorted(loaded_names, key=lambda n: -weights.get(n, 0)):
                print(f"    {n.split('/')[-1]:<45s} {weights.get(n, 0):.4f}")
    if not use_cache:
        weights = calibrate_weights(
            members,
            n_per_dataset=args.cal_samples,
            cache_path=args.weights_cache,
            batch_size=args.batch_size,
        )

    if args.calibrate_only:
        print("\n  --calibrate_only done.")
        return

    # ── Annotate ──
    council = BloomCouncil(members, weights)
    files   = collect_files(args)
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
        old_d, new_d = annotate_file(path, council, args.batch_size)
        all_old.update(old_d)
        all_new.update(new_d)

    print(f"\n{'='*65}\nSUMMARY\n{'='*65}")
    print(f"  {'Level':<12s} {'Before':>8s} {'After':>8s} {'Δ':>6s}")
    print(f"  {'-'*38}")
    for level in range(1, 7):
        name = BLOOM_NAMES[level]
        old  = all_old.get(level, 0)
        new  = all_new.get(level, 0)
        print(f"  {name:<12s} {old:>8d} {new:>8d} {new-old:>+6d}")

    print(f"\n  NEXT STEPS:")
    print(f"    python data/curriculum_negatives.py \\")
    print(f"        --pairs /tmp/data/real/train.jsonl \\")
    print(f"        --corpus /tmp/data/real/corpus.jsonl \\")
    print(f"        --output /tmp/data/real/train_curriculum.jsonl")
    print(f"    python scripts/train_bam.py --config configs/bam.yaml")


if __name__ == "__main__":
    main()
