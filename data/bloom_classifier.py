"""
Bloom taxonomy classifier.

Primary path: 4-model trained council from /tmp/bloom-council/ (train first via
              data/train_bloom_council.py). Expected accuracy: ~85-88%.

Fallback: cip29/bert-blooms-taxonomy-classifier from HuggingFace (~57% honest).

Public API (unchanged):
    from data.bloom_classifier import classify_bloom_batch, classify_bloom
    levels = classify_bloom_batch(["What is photosynthesis?", ...])  # list of int 1-6
"""

import json
import os
import pickle
from typing import List

import numpy as np
import torch
from tqdm import tqdm

# COUNCIL_DIR is overridable via env so users on servers without /tmp
# persistence (or without root access to write there) can point at their
# own trained-council directory.
COUNCIL_DIR   = os.environ.get("COUNCIL_DIR", "/tmp/bloom-council")
WEIGHTS_FILE  = os.path.join(COUNCIL_DIR, "council_weights.json")
FALLBACK_MODEL = "cip29/bert-blooms-taxonomy-classifier"

# ─── Lazy singletons ──────────────────────────────────────────────────────────
_council       = None   # dict: name → model/tokenizer or svm artifacts
_council_weights = None # dict: name → float
_fallback_model    = None
_fallback_tokenizer = None
_fallback_id2label  = None
_device        = None


def _get_device():
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


# ─── Council loader ───────────────────────────────────────────────────────────

def _load_council():
    global _council, _council_weights
    if _council is not None:
        return True

    if not os.path.exists(WEIGHTS_FILE):
        return False

    with open(WEIGHTS_FILE) as f:
        info = json.load(f)

    weights = info["weights"]
    models_paths = info["models"]
    device = _get_device()

    council = {}
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    for name in ["deberta", "roberta", "bert"]:
        path = models_paths.get(name)
        if not path or not os.path.exists(path):
            print(f"  [council] {name} not found at {path}, skipping.")
            continue
        print(f"  [council] Loading {name} from {path} ...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
        except AttributeError:
            # transformers >=4.50 changed extra_special_tokens format (list vs dict).
            # Fix the saved tokenizer config in-place and retry.
            tok_cfg_path = os.path.join(path, "tokenizer_config.json")
            if os.path.exists(tok_cfg_path):
                with open(tok_cfg_path) as f:
                    tok_cfg = json.load(f)
                if isinstance(tok_cfg.get("extra_special_tokens"), list):
                    tok_cfg["extra_special_tokens"] = {}
                    with open(tok_cfg_path, "w") as f:
                        json.dump(tok_cfg, f, indent=2)
                    print(f"  [council] Fixed extra_special_tokens in {tok_cfg_path}, retrying.")
            tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.to(device).eval()
        council[name] = {"type": "transformer", "model": model, "tokenizer": tokenizer}

    svm_path = models_paths.get("svm")
    if svm_path and os.path.exists(svm_path):
        print(f"  [council] Loading SVM from {svm_path} ...")
        with open(svm_path, "rb") as f:
            svm_artifacts = pickle.load(f)
        council["svm"] = {"type": "svm", **svm_artifacts}

    if not council:
        print("  [council] No models loaded — falling back to cip29.")
        return False

    # Only keep weights for models that loaded successfully
    total = sum(weights[k] for k in council if k in weights)
    _council_weights = {k: weights[k] / total for k in council if k in weights}
    _council = council
    print(f"  [council] Loaded {len(council)} members: {list(council.keys())}")
    return True


# ─── Fallback loader ──────────────────────────────────────────────────────────

def _load_fallback():
    global _fallback_model, _fallback_tokenizer, _fallback_id2label
    if _fallback_model is not None:
        return
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = _get_device()
    print(f"  [bloom] Loading fallback {FALLBACK_MODEL} on {device}...")
    _fallback_tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
    _fallback_model = AutoModelForSequenceClassification.from_pretrained(FALLBACK_MODEL)
    _fallback_model.to(device).eval()
    _fallback_id2label = getattr(_fallback_model.config, "id2label", None)


def _label_to_int(val) -> int:
    if isinstance(val, (int, float)):
        v = int(val)
        if 1 <= v <= 6: return v
        if 0 <= v <= 5: return v + 1
        return -1
    name = str(val).lower().strip()
    mapping = {
        "remember": 1, "remembering": 1, "knowledge": 1, "recall": 1,
        "understand": 2, "understanding": 2, "comprehension": 2,
        "apply": 3, "applying": 3, "application": 3,
        "analyze": 4, "analyse": 4, "analyzing": 4, "analysis": 4,
        "evaluate": 5, "evaluating": 5, "evaluation": 5,
        "create": 6, "creating": 6, "synthesis": 6, "synthesize": 6,
    }
    for key, level in mapping.items():
        if key in name:
            return level
    try:
        v = int(name)
        return v if 1 <= v <= 6 else (v+1 if 0 <= v <= 5 else -1)
    except ValueError:
        return -1


# ─── Inference ────────────────────────────────────────────────────────────────

def _transformer_proba(member: dict, texts: List[str], batch_size: int) -> np.ndarray:
    from typing import List
    model, tokenizer = member["model"], member["tokenizer"]
    device = _get_device()
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


def _svm_proba(member: dict, texts: List[str]) -> np.ndarray:
    from typing import List
    from scipy.sparse import hstack, csr_matrix

    BLOOM_VERBS = {
        1: ["define","list","recall","name","identify","state","label","memorize","repeat","recognize"],
        2: ["explain","summarize","interpret","classify","compare","describe","discuss","paraphrase"],
        3: ["apply","calculate","demonstrate","solve","use","illustrate","compute","modify","show"],
        4: ["analyze","differentiate","distinguish","examine","contrast","investigate","relate","why"],
        5: ["evaluate","assess","judge","justify","critique","defend","argue","recommend","appraise"],
        6: ["create","design","construct","develop","formulate","propose","invent","compose","generate"],
    }

    def verb_features(texts):
        feats = np.zeros((len(texts), 6))
        for i, text in enumerate(texts):
            t = text.lower()
            for lv, verbs in BLOOM_VERBS.items():
                feats[i, lv-1] = sum(1 for v in verbs if v in t)
        return feats

    tfidf, svm = member["tfidf"], member["svm"]
    X_tfidf = tfidf.transform(texts)
    X = hstack([X_tfidf, csr_matrix(verb_features(texts))])

    # LinearSVC has no predict_proba — use decision_function + softmax
    scores = svm.decision_function(X)
    if scores.ndim == 1:
        # Binary fallback — shouldn't happen with 6 classes
        scores = scores.reshape(-1, 1)
    # Softmax over decision scores
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_s = np.exp(scores)
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)

    # Align columns to classes 1-6 (LinearSVC stores classes in svm.classes_)
    classes = list(svm.classes_)  # e.g. [1,2,3,4,5,6]
    aligned = np.zeros((len(texts), 6))
    for col_idx, cls in enumerate(classes):
        target_col = int(cls) - 1  # 0-indexed
        if 0 <= target_col < 6:
            aligned[:, target_col] = probs[:, col_idx]
    return aligned


def _council_predict(queries: List[str], batch_size: int) -> List[int]:
    from typing import List
    # Weighted average of probability distributions
    combined = np.zeros((len(queries), 6))
    for name, member in _council.items():
        w = _council_weights.get(name, 0.0)
        if w == 0:
            continue
        if member["type"] == "transformer":
            probs = _transformer_proba(member, queries, batch_size)
        else:
            probs = _svm_proba(member, queries)
        combined += w * probs

    preds = combined.argmax(axis=1) + 1  # 0-indexed → 1-indexed
    return [max(1, min(6, int(p))) for p in preds]


def _fallback_predict(queries: List[str], batch_size: int) -> List[int]:
    from typing import List
    _load_fallback()
    device = _get_device()
    results = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        enc = _fallback_tokenizer(batch, padding=True, truncation=True,
                                  max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = _fallback_model(**enc).logits
        preds = logits.argmax(dim=-1).cpu().tolist()
        for pred in preds:
            if _fallback_id2label:
                name = _fallback_id2label.get(str(pred), _fallback_id2label.get(pred, str(pred)))
                bloom = _label_to_int(name)
                if bloom < 1:
                    bloom = pred + 1
            else:
                bloom = pred + 1
            results.append(max(1, min(6, bloom)))
    return results


# ─── Public API ───────────────────────────────────────────────────────────────

def classify_bloom_batch(queries: list, batch_size: int = 32) -> list:
    """Classify queries into Bloom levels 1-6. Uses trained council if available."""
    if not queries:
        return []

    # Try trained council first
    council_loaded = _load_council()

    results = []
    desc = "Bloom council" if council_loaded else "Bloom classifier (fallback)"
    for i in tqdm(range(0, len(queries), batch_size),
                  desc=f"  {desc}", leave=False):
        batch = queries[i:i+batch_size]
        if council_loaded:
            results.extend(_council_predict(batch, batch_size=batch_size))
        else:
            results.extend(_fallback_predict(batch, batch_size=batch_size))
    return results


def classify_bloom(query: str) -> int:
    """Classify a single query. Returns Bloom level 1-6."""
    return classify_bloom_batch([query])[0]
