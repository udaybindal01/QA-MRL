"""Shared Bloom's taxonomy classifier backed by a HuggingFace pretrained model.

Model: cip29/bert-blooms-taxonomy-classifier
Returns integer levels 1-6 (1=Remember, 2=Understand, 3=Apply,
                              4=Analyze, 5=Evaluate, 6=Create).

Usage:
    from data.bloom_classifier import classify_bloom_batch, classify_bloom

    levels = classify_bloom_batch(["What is photosynthesis?", "Compare mitosis and meiosis"])
    # => [1, 4]

    level = classify_bloom("Evaluate the impact of climate change")
    # => 5
"""

import torch
from tqdm import tqdm

MODEL_NAME = "cip29/bert-blooms-taxonomy-classifier"

_model = None
_tokenizer = None
_id2label = None
_device = None


def _ensure_loaded():
    global _model, _tokenizer, _id2label, _device
    if _model is not None:
        return
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading Bloom classifier ({MODEL_NAME}) on {_device}...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    _model.to(_device).eval()
    _id2label = getattr(_model.config, "id2label", None)


def _label_to_int(val) -> int:
    """Convert a model label (name string or raw int) to a 1-indexed Bloom level."""
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
        if 1 <= v <= 6:
            return v
        if 0 <= v <= 5:
            return v + 1
    except ValueError:
        pass
    return -1


def classify_bloom_batch(queries: list, batch_size: int = 64) -> list:
    """Classify a list of query strings. Returns list of Bloom levels (int, 1-6)."""
    _ensure_loaded()
    results = []
    for i in tqdm(range(0, len(queries), batch_size),
                  desc="  Bloom classification", leave=False):
        batch = queries[i : i + batch_size]
        enc = _tokenizer(batch, padding=True, truncation=True,
                         max_length=128, return_tensors="pt")
        enc = {k: v.to(_device) for k, v in enc.items()}
        with torch.no_grad():
            logits = _model(**enc).logits
        preds = logits.argmax(dim=-1).cpu().tolist()
        for pred in preds:
            if _id2label:
                label_name = _id2label.get(str(pred), _id2label.get(pred, str(pred)))
                bloom = _label_to_int(label_name)
                if bloom < 1:
                    bloom = pred + 1
            else:
                bloom = pred + 1
            results.append(max(1, min(6, bloom)))
    return results


def classify_bloom(query: str) -> int:
    """Classify a single query. Returns Bloom level (int, 1-6)."""
    return classify_bloom_batch([query])[0]
