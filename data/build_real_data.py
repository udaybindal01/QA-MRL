"""
Real Educational Data Pipeline v2.

Key fix: Passages are NOT constructed from questions.
- Corpus = real educational text (SciQ support, OBQA facts, QASC facts)
- Queries = questions from QA datasets
- Matched by semantic relevance, NOT lexical overlap

Usage:
    python data/build_real_data.py --config configs/real_data.yaml --output_dir data/real
"""

import argparse
import json
import os
import random
import re
from typing import Dict, List, Optional
from collections import defaultdict
from datasets import load_dataset


# ─────────────────────── Annotation helpers ───────────────────────

BLOOM_PATTERNS = {
    1: [r"\bwhat is\b", r"\bdefine\b", r"\bname\b", r"\blist\b",
        r"\bidentify\b", r"\bwhich of the following\b", r"\bwhat are\b"],
    2: [r"\bwhy\b", r"\bexplain\b", r"\bdescribe\b", r"\bhow does\b",
        r"\bwhat happens when\b", r"\bwhat causes\b"],
    3: [r"\bwhat would happen\b", r"\bpredict\b", r"\bcalculate\b",
        r"\bif .+ then\b", r"\bhow would\b", r"\bhow can\b"],
    4: [r"\bcompare\b", r"\bcontrast\b", r"\bdifference between\b",
        r"\brelationship between\b", r"\banalyze\b"],
    5: [r"\bevaluate\b", r"\bjudge\b", r"\bjustify\b",
        r"\bwhich .+ best\b", r"\bmost likely\b", r"\bevidence\b"],
}

STOPWORDS = {"the", "and", "for", "are", "was", "were", "that", "this",
             "with", "from", "have", "has", "had", "not", "but", "what",
             "which", "when", "where", "how", "who", "why", "can", "will",
             "would", "could", "should", "does", "did", "been", "being",
             "than", "then", "them", "they", "their", "there", "these",
             "those", "into", "about", "between", "through", "during",
             "before", "after", "above", "below", "each", "every",
             "some", "such", "only", "other", "also", "most", "more"}


def assign_bloom(question: str, source: str = "") -> int:
    q = question.lower().strip()
    for level in [5, 4, 3, 2, 1]:
        for pat in BLOOM_PATTERNS[level]:
            if re.search(pat, q):
                return level
    return {"sciq": 2, "arc_easy": 1, "arc_challenge": 3,
            "openbookqa": 3, "qasc": 4}.get(source, 2)


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
    return set(re.findall(r'\b[a-z]{3,}\b', text.lower())) - STOPWORDS


# ─────────────────────── Corpus Building ──────────────────────────

def build_corpus() -> List[Dict]:
    """Build passage corpus from real educational text (no question text)."""
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
                    "bloom_level": 2, "source": "sciq_support", "difficulty": "medium",
                })
                pid += 1
        except Exception:
            pass
    print(f"    SciQ: {pid} passages")

    # OpenBookQA facts (short factual = Bloom 1)
    print("  Loading OpenBookQA facts...")
    n0 = len(corpus)
    try:
        ds = load_dataset("allenai/openbookqa", "main", split="train")
        for row in ds:
            fact = row.get("fact1", "").strip()
            if len(fact) < 20:
                continue
            subj = assign_subject(fact)
            corpus.append({
                "id": f"p_{pid}", "text": fact,
                "subject": subj, "topic": extract_topic(fact, subj),
                "bloom_level": 1, "source": "obqa_fact", "difficulty": "easy",
            })
            pid += 1
    except Exception as e:
        print(f"    Warning: {e}")
    print(f"    OBQA: {len(corpus) - n0} passages")

    # QASC facts (individual = Bloom 1, combined = Bloom 4)
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
                        "bloom_level": 1, "source": "qasc_fact", "difficulty": "easy",
                    })
                    pid += 1

            combined = row.get("combinedfact", "").strip()
            if len(combined) >= 30:
                subj = assign_subject(combined)
                corpus.append({
                    "id": f"p_{pid}", "text": combined,
                    "subject": subj, "topic": extract_topic(combined, subj),
                    "bloom_level": 4, "source": "qasc_combined", "difficulty": "hard",
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


def _mine_negatives(pos_idx, corpus, by_topic, by_subject, topic, subject, bloom, num_neg):
    """Hard negatives: same-topic-diff-bloom > same-subject > random."""
    negs = []

    # Same topic, different Bloom
    cands = [j for j in by_topic.get(topic, []) if j != pos_idx and corpus[j]["bloom_level"] != bloom]
    if cands:
        negs.extend(random.sample(cands, min(num_neg, len(cands))))

    # Same subject, different topic
    if len(negs) < num_neg:
        cands = [j for j in by_subject.get(subject, [])
                 if j != pos_idx and j not in negs and corpus[j]["topic"] != topic]
        if cands:
            negs.extend(random.sample(cands, min(num_neg - len(negs), len(cands))))

    # Random fill
    if len(negs) < num_neg:
        others = [j for j in range(len(corpus)) if j != pos_idx and j not in negs]
        if others:
            negs.extend(random.sample(others, min(num_neg - len(negs), len(others))))

    return negs[:num_neg]


def build_pairs(corpus: List[Dict], num_neg: int = 3) -> List[Dict]:
    """Match questions from QA datasets to corpus passages by keyword overlap."""

    # Precompute keyword index
    corpus_kw = [_get_keywords(p["text"]) for p in corpus]
    by_topic = defaultdict(list)
    by_subject = defaultdict(list)
    for i, p in enumerate(corpus):
        by_topic[p["topic"]].append(i)
        by_subject[p["subject"]].append(i)

    pairs = []

    # SciQ questions
    print("  Matching SciQ questions...")
    n0 = len(pairs)
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
        bl = assign_bloom(q, "sciq")
        negs = _mine_negatives(idx, corpus, by_topic, by_subject,
                                corpus[idx]["topic"], corpus[idx]["subject"], bl, num_neg)
        pairs.append({
            "query": q, "positive_text": corpus[idx]["text"], "positive_id": corpus[idx]["id"],
            "negative_texts": [corpus[j]["text"] for j in negs],
            "negative_ids": [corpus[j]["id"] for j in negs],
            "bloom_level": bl, "subject": corpus[idx]["subject"],
            "topic": corpus[idx]["topic"], "query_type": _query_type(q, bl),
        })
    print(f"    SciQ: {len(pairs) - n0} pairs")

    # ARC questions
    print("  Matching ARC questions...")
    n0 = len(pairs)
    for split_name, src, boost in [("ARC-Easy", "arc_easy", 0), ("ARC-Challenge", "arc_challenge", 2)]:
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
            bl = min(6, max(1, assign_bloom(q, src) + boost))
            negs = _mine_negatives(idx, corpus, by_topic, by_subject,
                                    corpus[idx]["topic"], corpus[idx]["subject"], bl, num_neg)
            pairs.append({
                "query": q, "positive_text": corpus[idx]["text"], "positive_id": corpus[idx]["id"],
                "negative_texts": [corpus[j]["text"] for j in negs],
                "negative_ids": [corpus[j]["id"] for j in negs],
                "bloom_level": bl, "subject": corpus[idx]["subject"],
                "topic": corpus[idx]["topic"], "query_type": _query_type(q, bl),
            })
    print(f"    ARC: {len(pairs) - n0} pairs")

    # OpenBookQA questions
    print("  Matching OpenBookQA questions...")
    n0 = len(pairs)
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
            bl = assign_bloom(q, "openbookqa")
            negs = _mine_negatives(idx, corpus, by_topic, by_subject,
                                    corpus[idx]["topic"], corpus[idx]["subject"], bl, num_neg)
            pairs.append({
                "query": q, "positive_text": corpus[idx]["text"], "positive_id": corpus[idx]["id"],
                "negative_texts": [corpus[j]["text"] for j in negs],
                "negative_ids": [corpus[j]["id"] for j in negs],
                "bloom_level": bl, "subject": corpus[idx]["subject"],
                "topic": corpus[idx]["topic"], "query_type": _query_type(q, bl),
            })
    except Exception:
        pass
    print(f"    OBQA: {len(pairs) - n0} pairs")

    print(f"  Total: {len(pairs)} pairs")
    return pairs


# ─────────────────────── Main ─────────────────────────────────────

def build_real_dataset(config: dict, output_dir: str = "data/real"):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(config["training"]["seed"])
    num_neg = config["data"]["num_hard_negatives"]

    print("=" * 60)
    print("Building REAL Educational Dataset v2")
    print("(Decoupled queries and passages)")
    print("=" * 60)

    print("\n[1/3] Building passage corpus...")
    corpus = build_corpus()

    bloom_dist = defaultdict(int)
    subject_dist = defaultdict(int)
    for p in corpus:
        bloom_dist[p["bloom_level"]] += 1
        subject_dist[p["subject"]] += 1
    print(f"  Bloom: {dict(sorted(bloom_dist.items()))}")
    print(f"  Subjects: {dict(sorted(subject_dist.items(), key=lambda x: -x[1])[:6])}")

    with open(os.path.join(output_dir, "corpus.jsonl"), "w") as f:
        for p in corpus:
            f.write(json.dumps(p) + "\n")

    print(f"\n[2/3] Building query-passage pairs (num_neg={num_neg})...")
    pairs = build_pairs(corpus, num_neg=num_neg)

    pair_bloom = defaultdict(int)
    for p in pairs:
        pair_bloom[p["bloom_level"]] += 1
    print(f"  Pair Bloom: {dict(sorted(pair_bloom.items()))}")

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
            "corpus_bloom": dict(sorted(bloom_dist.items())),
            "pair_bloom": dict(sorted(pair_bloom.items()))}
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Corpus={len(corpus)}, Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_data.yaml")
    parser.add_argument("--output_dir", default="data/real")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.misc import load_config
    build_real_dataset(load_config(args.config), args.output_dir)
