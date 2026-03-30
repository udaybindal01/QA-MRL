"""
Real Educational Data Pipeline v3.

v3 changes:
- Documents NO LONGER have bloom_level. Bloom is query-only.
  A passage is just content — its cognitive demand comes from the query.
- Fixed OpenBookQA loading: use "fact1" field correctly, handle missing.
- Negative mining is topic-based only (no Bloom-distance tiers on docs).
- Bloom classification uses cip29/bert-blooms-taxonomy-classifier (HuggingFace),
  not regex patterns.

Usage:
    python data/build_real_data.py --config configs/bam.yaml --output_dir data/real
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional
from collections import defaultdict
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.bloom_classifier import classify_bloom_batch


# ─────────────────────── Annotation helpers ───────────────────────

STOPWORDS = {"the", "and", "for", "are", "was", "were", "that", "this",
             "with", "from", "have", "has", "had", "not", "but", "what",
             "which", "when", "where", "how", "who", "why", "can", "will",
             "would", "could", "should", "does", "did", "been", "being",
             "than", "then", "them", "they", "their", "there", "these",
             "those", "into", "about", "between", "through", "during",
             "before", "after", "above", "below", "each", "every",
             "some", "such", "only", "other", "also", "most", "more"}


def assign_subject(text: str) -> str:
    t = text.lower()
    scores = {
        "biology": sum(1 for k in ["cell", "organism", "photosynthesis", "dna", "gene",
                                     "protein", "species", "evolution", "ecosystem", "plant",
                                     "animal", "bacteria", "blood", "organ", "tissue"] if k in t),
        "chemistry": sum(1 for k in ["atom", "molecule", "element", "compound", "reaction",
                                       "acid", "base", "electron", "bond", "ion", "solution",
                                       "chemical", "periodic", "oxidation", "ph"] if k in t),
        "physics": sum(1 for k in ["force", "energy", "gravity", "velocity", "acceleration",
                                     "wave", "light", "electric", "magnetic", "circuit",
                                     "current", "heat", "temperature", "pressure"] if k in t),
        "earth_science": sum(1 for k in ["earth", "rock", "mineral", "volcano", "weather",
                                           "climate", "atmosphere", "ocean", "fossil", "planet",
                                           "star", "sun", "orbit", "soil"] if k in t),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general_science"


def extract_topic(text: str, subject: str) -> str:
    t = text.lower()
    topics = {
        "biology": {"photosynthesis": ["photosynthesis", "chlorophyll"],
                     "genetics": ["gene", "dna", "chromosome", "heredity"],
                     "evolution": ["evolution", "natural selection", "adaptation"],
                     "cells": ["cell", "membrane", "organelle", "nucleus"],
                     "ecology": ["ecosystem", "habitat", "food chain", "biome"],
                     "human_body": ["organ", "tissue", "blood", "heart", "brain"]},
        "chemistry": {"atomic_structure": ["atom", "electron", "proton", "neutron"],
                       "reactions": ["reaction", "reactant", "product", "catalyst"],
                       "acids_bases": ["acid", "base", "ph"],
                       "bonding": ["bond", "ionic", "covalent"]},
        "physics": {"forces_motion": ["force", "newton", "acceleration", "velocity"],
                     "energy": ["energy", "kinetic", "potential"],
                     "electricity": ["electric", "circuit", "current"],
                     "waves_light": ["wave", "light", "frequency"]},
        "earth_science": {"geology": ["rock", "mineral", "volcano"],
                           "weather": ["weather", "climate", "atmosphere"],
                           "astronomy": ["planet", "star", "solar system"]},
    }
    for topic, kws in topics.get(subject, {}).items():
        if any(kw in t for kw in kws):
            return topic
    return "general"


def _query_type(question: str, bloom: int) -> str:
    q = question.lower()
    if any(w in q for w in ["what is", "define", "name", "list"]):
        return "factual"
    elif any(w in q for w in ["explain", "why", "how does"]):
        return "conceptual"
    elif any(w in q for w in ["calculate", "solve", "predict", "what would"]):
        return "procedural"
    elif any(w in q for w in ["compare", "analyze", "evaluate"]):
        return "metacognitive"
    return ["factual", "factual", "conceptual", "procedural", "metacognitive", "metacognitive"][bloom - 1]


def _get_keywords(text: str) -> set:
    words = text.lower().split()
    return {w.strip(".,!?;:\"'()[]{}") for w in words
            if len(w) >= 3 and w.strip(".,!?;:\"'()[]{}").isalpha()} - STOPWORDS


# ─────────────────────── Corpus Building ──────────────────────────

def build_corpus() -> List[Dict]:
    """
    Build passage corpus from real educational text.

    v3: Documents have NO bloom_level. A passage is content, not a
    cognitive task. Bloom level is determined by the query.
    """
    corpus = []
    pid = 0

    # SciQ support passages (richest source)
    print("  Loading SciQ support passages...")
    for split in ["train", "validation", "test"]:
        try:
            ds = load_dataset("allenai/sciq", split=split)
            for row in ds:
                sup = row.get("support", "").strip()
                if len(sup) < 40:
                    continue
                subj = assign_subject(sup)
                corpus.append({
                    "id": f"p_{pid}", "text": sup,
                    "subject": subj, "topic": extract_topic(sup, subj),
                    "source": "sciq_support", "difficulty": "medium",
                })
                pid += 1
        except Exception:
            pass
    print(f"    SciQ: {pid} passages")

    # OpenBookQA facts
    # v3 fix: properly handle the dataset structure
    print("  Loading OpenBookQA facts...")
    n0 = len(corpus)
    try:
        ds = load_dataset("allenai/openbookqa", "additional", split="train")
        for row in ds:
            # The "additional" config has "fact1" field
            fact = row.get("fact1", "").strip()
            if len(fact) < 20:
                continue
            subj = assign_subject(fact)
            corpus.append({
                "id": f"p_{pid}", "text": fact,
                "subject": subj, "topic": extract_topic(fact, subj),
                "source": "obqa_fact", "difficulty": "easy",
            })
            pid += 1
    except Exception as e:
        print(f"    Warning loading 'additional': {e}")
        # Fallback: try "main" config
        try:
            ds = load_dataset("allenai/openbookqa", "main", split="train")
            for row in ds:
                # "main" config may not have fact1, check available fields
                fact = ""
                for field_name in ["fact1", "fact", "stem"]:
                    fact = row.get(field_name, "").strip()
                    if fact:
                        break
                if len(fact) < 20:
                    continue
                subj = assign_subject(fact)
                corpus.append({
                    "id": f"p_{pid}", "text": fact,
                    "subject": subj, "topic": extract_topic(fact, subj),
                    "source": "obqa_fact", "difficulty": "easy",
                })
                pid += 1
        except Exception as e2:
            print(f"    Warning loading 'main': {e2}")
    print(f"    OBQA: {len(corpus) - n0} passages")

    # QASC facts
    print("  Loading QASC facts...")
    n0 = len(corpus)
    try:
        ds = load_dataset("allenai/qasc", split="train")
        for row in ds:
            for fact in [row.get("fact1", ""), row.get("fact2", "")]:
                fact = fact.strip()
                if len(fact) >= 20:
                    subj = assign_subject(fact)
                    corpus.append({
                        "id": f"p_{pid}", "text": fact,
                        "subject": subj, "topic": extract_topic(fact, subj),
                        "source": "qasc_fact", "difficulty": "easy",
                    })
                    pid += 1

            combined = row.get("combinedfact", "").strip()
            if len(combined) >= 30:
                subj = assign_subject(combined)
                corpus.append({
                    "id": f"p_{pid}", "text": combined,
                    "subject": subj, "topic": extract_topic(combined, subj),
                    "source": "qasc_combined", "difficulty": "hard",
                })
                pid += 1
    except Exception as e:
        print(f"    Warning: {e}")
    print(f"    QASC: {len(corpus) - n0} passages")

    print(f"  Total corpus: {len(corpus)} passages")
    return corpus


# ─────────────────────── Query-Passage Matching ───────────────────

def _find_best_passage(query_kw: set, corpus_kw: List[set], min_overlap: int = 3) -> Optional[int]:
    """Find corpus passage with highest keyword overlap to query."""
    best_idx, best_score = None, 0
    for i, ckw in enumerate(corpus_kw):
        overlap = len(query_kw & ckw)
        if overlap > best_score and overlap >= min_overlap:
            best_score = overlap
            best_idx = i
    return best_idx


def _mine_negatives(pos_idx, corpus, by_topic, by_subject, topic, subject, num_neg):
    """
    Topic-based hard negatives only (no Bloom-distance tiers).

    v3: Documents don't have Bloom levels, so negatives are mined by
    topical confusion:
      Tier 1 (hardest): Same topic, different passage
      Tier 2: Same subject, different topic
      Tier 3: Random from corpus
    """
    negs = []

    # Tier 1: Same topic, different passage (hardest — topically confusing)
    cands = [j for j in by_topic.get(topic, []) if j != pos_idx]
    if cands:
        negs.extend(random.sample(cands, min(num_neg, len(cands))))

    # Tier 2: Same subject, different topic
    if len(negs) < num_neg:
        cands = [j for j in by_subject.get(subject, [])
                 if j != pos_idx and j not in negs and corpus[j]["topic"] != topic]
        if cands:
            negs.extend(random.sample(cands, min(num_neg - len(negs), len(cands))))

    # Tier 3: Random fill
    if len(negs) < num_neg:
        others = [j for j in range(len(corpus)) if j != pos_idx and j not in negs]
        if others:
            negs.extend(random.sample(others, min(num_neg - len(negs), len(others))))

    return negs[:num_neg]


def build_pairs(corpus: List[Dict], num_neg: int = 3) -> List[Dict]:
    """Match questions from QA datasets to corpus passages by keyword overlap,
    then batch-classify all query Bloom levels with the HuggingFace model."""

    # Precompute keyword index
    corpus_kw = [_get_keywords(p["text"]) for p in corpus]
    by_topic = defaultdict(list)
    by_subject = defaultdict(list)
    for i, p in enumerate(corpus):
        by_topic[p["topic"]].append(i)
        by_subject[p["subject"]].append(i)

    # ── Collect (query, corpus_idx, bloom_boost) for all sources ──

    raw_matches = []  # (query, corpus_idx, bloom_boost)

    # SciQ questions
    print("  Collecting SciQ questions...")
    n0 = len(raw_matches)
    ds = load_dataset("allenai/sciq", split="train")
    for row in ds:
        q = row["question"]
        ans = row.get("correct_answer", "")
        if len(row.get("support", "")) < 40:
            continue
        qkw = _get_keywords(q + " " + ans)
        idx = _find_best_passage(qkw, corpus_kw)
        if idx is None:
            continue
        raw_matches.append((q, idx, 0))
    print(f"    SciQ: {len(raw_matches) - n0} candidates")

    # ARC questions
    print("  Collecting ARC questions...")
    n0 = len(raw_matches)
    for split_name, boost in [("ARC-Easy", 0), ("ARC-Challenge", 2)]:
        try:
            ds = load_dataset("allenai/ai2_arc", split_name, split="train")
        except Exception:
            continue
        for row in ds:
            q = row["question"]
            ans = ""
            for lb, tx in zip(row["choices"]["label"], row["choices"]["text"]):
                if lb == row["answerKey"]:
                    ans = tx
                    break
            qkw = _get_keywords(q + " " + ans)
            idx = _find_best_passage(qkw, corpus_kw, min_overlap=2)
            if idx is None:
                continue
            raw_matches.append((q, idx, boost))
    print(f"    ARC: {len(raw_matches) - n0} candidates")

    # OpenBookQA questions
    print("  Collecting OpenBookQA questions...")
    n0 = len(raw_matches)
    try:
        ds = load_dataset("allenai/openbookqa", "main", split="train")
        for row in ds:
            q = row["question_stem"]
            ans = ""
            for lb, tx in zip(row["choices"]["label"], row["choices"]["text"]):
                if lb == row["answerKey"]:
                    ans = tx
                    break
            qkw = _get_keywords(q + " " + ans)
            idx = _find_best_passage(qkw, corpus_kw, min_overlap=2)
            if idx is None:
                continue
            raw_matches.append((q, idx, 0))
    except Exception:
        pass
    print(f"    OBQA: {len(raw_matches) - n0} candidates")

    # ── Batch classify all queries at once ──
    print(f"  Classifying {len(raw_matches)} queries with HuggingFace Bloom model...")
    bloom_levels = classify_bloom_batch([q for q, _, _ in raw_matches])

    # ── Build final pairs ──
    pairs = []
    for (q, idx, boost), bl in zip(raw_matches, bloom_levels):
        bl = min(6, max(1, bl + boost))
        negs = _mine_negatives(idx, corpus, by_topic, by_subject,
                               corpus[idx]["topic"], corpus[idx]["subject"], num_neg)
        pairs.append({
            "query": q, "positive_text": corpus[idx]["text"], "positive_id": corpus[idx]["id"],
            "negative_texts": [corpus[j]["text"] for j in negs],
            "negative_ids": [corpus[j]["id"] for j in negs],
            "bloom_level": bl, "subject": corpus[idx]["subject"],
            "topic": corpus[idx]["topic"], "query_type": _query_type(q, bl),
        })

    print(f"  Total: {len(pairs)} pairs")
    return pairs


# ─────────────────────── Main ─────────────────────────────────────

def build_real_dataset(config: dict, output_dir: str = "data/real"):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(config["training"]["seed"])
    num_neg = config["data"]["num_hard_negatives"]

    print("=" * 60)
    print("Building REAL Educational Dataset v3")
    print("(Query-only Bloom — documents have no Bloom labels)")
    print("=" * 60)

    print("\n[1/3] Building passage corpus...")
    corpus = build_corpus()

    subject_dist = defaultdict(int)
    source_dist = defaultdict(int)
    for p in corpus:
        subject_dist[p["subject"]] += 1
        source_dist[p["source"]] += 1
    print(f"  Subjects: {dict(sorted(subject_dist.items(), key=lambda x: -x[1])[:6])}")
    print(f"  Sources: {dict(sorted(source_dist.items(), key=lambda x: -x[1]))}")

    with open(os.path.join(output_dir, "corpus.jsonl"), "w") as f:
        for p in corpus:
            f.write(json.dumps(p) + "\n")

    print(f"\n[2/3] Building query-passage pairs (num_neg={num_neg})...")
    pairs = build_pairs(corpus, num_neg=num_neg)

    pair_bloom = defaultdict(int)
    for p in pairs:
        pair_bloom[p["bloom_level"]] += 1
    print(f"  Query Bloom distribution: {dict(sorted(pair_bloom.items()))}")

    print(f"\n[3/3] Splitting and saving...")
    random.shuffle(pairs)
    n = len(pairs)
    splits = {"train": pairs[:int(0.8*n)], "val": pairs[int(0.8*n):int(0.9*n)], "test": pairs[int(0.9*n):]}
    for name, sp in splits.items():
        with open(os.path.join(output_dir, f"{name}.jsonl"), "w") as f:
            for p in sp:
                f.write(json.dumps(p) + "\n")
        print(f"  {name}: {len(sp)} pairs")

    meta = {"num_corpus": len(corpus), "num_train": len(splits["train"]),
            "num_val": len(splits["val"]), "num_test": len(splits["test"]),
            "corpus_has_bloom": False,
            "pair_bloom": dict(sorted(pair_bloom.items())),
            "source_dist": dict(source_dist)}
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Corpus={len(corpus)}, Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_data.yaml")
    parser.add_argument("--output_dir", default="data/real")
    args = parser.parse_args()

    from utils.misc import load_config
    build_real_dataset(load_config(args.config), args.output_dir)
