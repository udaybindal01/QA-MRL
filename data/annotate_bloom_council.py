"""
Bloom Level Annotation — Council of Models (SOTA ensemble).

Three models vote via weighted soft-voting (averaged probability distributions):

  1. MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli  (weight 3)
     SOTA zero-shot NLI (~440M). Best zero-shot Bloom accuracy among local models.

  2. cross-encoder/nli-deberta-v3-base  (weight 2)
     Smaller DeBERTa NLI (~180M). Adds architectural diversity at lower compute cost.

  3. cip29/bert-blooms-taxonomy-classifier  (weight 1)
     BERT fine-tuned on educational Bloom data. Domain knowledge but narrow training set.

Voting: each model returns a 6-dim probability vector (one per Bloom level).
        Final prediction = argmax(weighted average of all three vectors).
        Confidence = max probability in final vector (reflects council agreement).

Output fields added to each record:
  bloom_level       — 1-indexed Bloom level (1=Remember … 6=Create)
  bloom_source      — "council"
  bloom_confidence  — float 0-1, council agreement confidence
  bloom_votes       — per-model predictions, for debugging

Usage:
    # Annotate educational dataset
    python data/annotate_bloom_council.py --data_dir /tmp/data/real

    # Annotate BEIR datasets
    python data/annotate_bloom_council.py \
        --beir_root /tmp/data/beir \
        --datasets scifact nfcorpus fiqa

    # Use only 2 models (e.g. skip pretrained classifier)
    python data/annotate_bloom_council.py \
        --data_dir /tmp/data/real \
        --skip_classifier

    # Annotate a single file
    python data/annotate_bloom_council.py --input /tmp/data/beir/scifact/test.jsonl
"""

import argparse
import json
import os
import shutil
from collections import Counter
from typing import List, Optional, Dict, Any

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


# ──────────────────── NLI Council Member ────────────────────

class NLIMember:
    """
    Zero-shot NLI model as a council member.
    Returns a 6-dim probability vector for each query using BLOOM_HYPOTHESES.
    """

    def __init__(self, model_name: str, weight: float, device):
        from transformers import pipeline
        print(f"  [NLI] Loading {model_name} ...")
        self.name = model_name
        self.weight = weight
        self._pipe = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device,
            batch_size=16,
        )
        print(f"  [NLI] Loaded {model_name}")

    def predict_proba(self, queries: List[str], batch_size: int = 32) -> np.ndarray:
        """Returns (N, 6) array of probabilities, columns = Bloom levels 1-6."""
        all_probs = []
        for i in tqdm(range(0, len(queries), batch_size),
                      desc=f"  [{self.name.split('/')[-1]}]", leave=False):
            batch = queries[i:i + batch_size]
            results = self._pipe(batch, BLOOM_HYPOTHESES, multi_label=False)
            if isinstance(results, dict):
                results = [results]
            for r in results:
                # r["labels"] and r["scores"] are sorted by score descending.
                # Re-align to hypothesis order (index = Bloom level - 1).
                score_map = dict(zip(r["labels"], r["scores"]))
                probs = np.array([score_map.get(h, 0.0) for h in BLOOM_HYPOTHESES],
                                 dtype=np.float32)
                probs /= probs.sum() + 1e-9
                all_probs.append(probs)
        return np.stack(all_probs)  # (N, 6)


# ──────────────────── Classifier Council Member ────────────────────

class ClassifierMember:
    """
    Fine-tuned sequence-classification model as a council member.
    Returns a 6-dim probability vector via softmax over logits.
    """

    def __init__(self, model_name: str, weight: float, device):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        print(f"  [CLS] Loading {model_name} ...")
        self.name = model_name
        self.weight = weight
        self._device = device
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(device).eval()

        # Build index → Bloom-level-1 mapping
        id2label = getattr(self._model.config, "id2label", None)
        self._idx_to_bloom = self._build_idx_map(id2label)
        print(f"  [CLS] Loaded {model_name} | idx→bloom: {self._idx_to_bloom}")

    @staticmethod
    def _build_idx_map(id2label) -> Dict[int, int]:
        """Map model output indices to 1-6 Bloom levels."""
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
        # Fill gaps if mapping is incomplete
        if len(mapping) < 6:
            mapping = {i: i + 1 for i in range(6)}
        return mapping

    def predict_proba(self, queries: List[str], batch_size: int = 64) -> np.ndarray:
        """Returns (N, 6) array of probabilities aligned to Bloom levels 1-6."""
        import torch
        all_probs = []
        for i in tqdm(range(0, len(queries), batch_size),
                      desc=f"  [{self.name.split('/')[-1]}]", leave=False):
            batch = queries[i:i + batch_size]
            enc = self._tok(batch, padding=True, truncation=True,
                            max_length=128, return_tensors="pt")
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self._model(**enc).logits  # (B, num_labels)
            probs_raw = torch.softmax(logits, dim=-1).cpu().numpy()  # (B, num_labels)

            for row in probs_raw:
                # Re-map to fixed 6-dim Bloom vector
                bloom_probs = np.zeros(6, dtype=np.float32)
                for idx, p in enumerate(row):
                    bl = self._idx_to_bloom.get(idx, idx + 1)
                    if 1 <= bl <= 6:
                        bloom_probs[bl - 1] += p
                bloom_probs /= bloom_probs.sum() + 1e-9
                all_probs.append(bloom_probs)
        return np.stack(all_probs)  # (N, 6)


# ──────────────────── Council ────────────────────

class BloomCouncil:
    """
    Ensemble of council members. Combines predictions via weighted soft voting.
    """

    def __init__(self, members: list):
        self.members = members
        total_w = sum(m.weight for m in members)
        self._weights = [m.weight / total_w for m in members]
        print(f"\n  Council assembled: {len(members)} members")
        for m, w in zip(members, self._weights):
            print(f"    {m.name.split('/')[-1]}  weight={w:.3f}")

    def predict(self, queries: List[str], batch_size: int = 32):
        """
        Returns:
            levels      : List[int], 1-indexed Bloom levels
            confidences : List[float], max probability in the ensemble distribution
            votes       : List[List[int]], per-member predictions (for debugging)
        """
        member_probs = []
        member_preds = []
        for m, w in zip(self.members, self._weights):
            proba = m.predict_proba(queries, batch_size)   # (N, 6)
            member_probs.append(proba * w)
            member_preds.append(proba.argmax(axis=1) + 1)  # 1-indexed per member

        ensemble = sum(member_probs)                   # weighted average, (N, 6)
        levels = (ensemble.argmax(axis=1) + 1).tolist()
        confidences = ensemble.max(axis=1).tolist()

        votes = [
            [int(member_preds[m_idx][q_idx])
             for m_idx in range(len(self.members))]
            for q_idx in range(len(queries))
        ]
        return levels, confidences, votes


# ──────────────────── Annotation helpers ────────────────────

def annotate_file(input_path: str, output_path: str, council: BloomCouncil,
                  batch_size: int = 32):
    with open(input_path) as f:
        records = [json.loads(l.strip()) for l in f]

    queries = [r["query"] for r in records]
    print(f"\n  Annotating {len(queries)} queries from {os.path.basename(input_path)}")

    old_dist = Counter(r.get("bloom_level") for r in records)
    print(f"  Old distribution: {dict(sorted(old_dist.items()))}")

    levels, confidences, votes = council.predict(queries, batch_size)

    changed = 0
    for r, lvl, conf, vote in zip(records, levels, confidences, votes):
        if r.get("bloom_level") != lvl:
            changed += 1
        r["bloom_level"] = lvl
        r["bloom_source"] = "council"
        r["bloom_confidence"] = round(float(conf), 4)
        r["bloom_votes"] = vote

    new_dist = Counter(r["bloom_level"] for r in records)
    print(f"  New distribution: {dict(sorted(new_dist.items()))}")
    print(f"  Changed: {changed}/{len(records)} ({changed/len(records):.1%})")

    avg_conf = float(np.mean(confidences))
    print(f"  Avg council confidence: {avg_conf:.3f}")

    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Update cache file if present (used by EducationalRetrievalDataset at train time)
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
    if args.input:
        paths.append(args.input)
        return paths
    if args.data_dir:
        for split in args.splits:
            p = os.path.join(args.data_dir, f"{split}.jsonl")
            if os.path.exists(p):
                paths.append(p)
            else:
                print(f"  Skipping {split} — not found at {p}")
    if args.beir_root and args.datasets:
        for ds in args.datasets:
            for split in args.splits:
                p = os.path.join(args.beir_root, ds, f"{split}.jsonl")
                if os.path.exists(p):
                    paths.append(p)
                else:
                    print(f"  Skipping {ds}/{split} — not found")
    return paths


# ──────────────────── Main ────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bloom annotation via council of models (SOTA ensemble)"
    )
    parser.add_argument("--data_dir", default=None,
                        help="Annotate train/val/test.jsonl under this directory")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--beir_root", default=None,
                        help="Root directory of BEIR datasets (e.g. /tmp/data/beir)")
    parser.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus", "fiqa"],
                        help="BEIR dataset names to annotate (used with --beir_root)")
    parser.add_argument("--input", default=None,
                        help="Annotate a single JSONL file")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--skip_classifier", action="store_true",
                        help="Skip the fine-tuned BERT classifier (use 2 NLI models only)")
    parser.add_argument("--skip_secondary_nli", action="store_true",
                        help="Skip cross-encoder NLI (use DeBERTa-large + classifier only)")
    parser.add_argument("--no_backup", action="store_true",
                        help="Do not back up original annotations")
    args = parser.parse_args()

    if not args.data_dir and not args.beir_root and not args.input:
        parser.error("Provide --data_dir, --beir_root, or --input")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print("Bloom Level Annotation — Council of Models")
    print("=" * 65)
    print(f"  Device: {device}")

    # ── Assemble council ──
    members = []

    # Member 1: DeBERTa-v3-large NLI — SOTA zero-shot (weight 3)
    try:
        members.append(NLIMember(
            "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
            weight=3.0, device=device
        ))
    except Exception as e:
        print(f"  WARNING: Could not load DeBERTa-v3-large NLI: {e}")
        print("  Falling back to facebook/bart-large-mnli as primary NLI")
        members.append(NLIMember("facebook/bart-large-mnli", weight=3.0, device=device))

    # Member 2: cross-encoder/nli-deberta-v3-base — secondary NLI (weight 2)
    if not args.skip_secondary_nli:
        try:
            members.append(NLIMember(
                "cross-encoder/nli-deberta-v3-base",
                weight=2.0, device=device
            ))
        except Exception as e:
            print(f"  WARNING: Could not load cross-encoder NLI: {e}")
            print("  Skipping secondary NLI member.")

    # Member 3: fine-tuned BERT Bloom classifier (weight 1)
    if not args.skip_classifier:
        try:
            members.append(ClassifierMember(
                "cip29/bert-blooms-taxonomy-classifier",
                weight=1.0, device=device
            ))
        except Exception as e:
            print(f"  WARNING: Could not load BERT classifier: {e}")
            print("  Skipping classifier member.")

    if not members:
        raise RuntimeError("No council members could be loaded.")

    council = BloomCouncil(members)

    # ── Collect files ──
    files = collect_files(args)
    if not files:
        print("No files found to annotate.")
        return

    # ── Annotate ──
    all_old: Counter = Counter()
    all_new: Counter = Counter()

    for path in files:
        if not args.no_backup:
            backup = path + ".pre_council_backup"
            if not os.path.exists(backup):
                shutil.copy2(path, backup)
                print(f"  Backed up original → {os.path.basename(backup)}")

        old_dist, new_dist = annotate_file(path, path, council, args.batch_size)
        all_old.update(old_dist)
        all_new.update(new_dist)

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
        print(f"  {name:<12s} {old:>8d} {new:>8d} {new-old:>+6d}")

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
