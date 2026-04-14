"""
MS MARCO Data Pipeline for BAM training.

Downloads MS MARCO passage ranking dataset from HuggingFace and converts it
to the same JSONL format expected by EducationalRetrievalDataset:
  - corpus.jsonl: one passage per line with {id, text, subject, topic, source, difficulty}
  - train.jsonl / val.jsonl / test.jsonl: {query, positive_text, positive_id,
    negative_texts, negative_ids, bloom_level, subject, query_type}

Bloom levels are assigned by cip29/bert-blooms-taxonomy-classifier.
Hard negatives are mined from the official BM25 negatives in MS MARCO.

MS MARCO stats:
  - ~8.8M passages in corpus
  - ~500k training queries (each with 1 relevant passage)
  - ~6900 dev queries
  - We subsample training to --max_train (default 100k) for tractability

Usage:
    python data/build_msmarco_data.py --output_dir /tmp/data/msmarco --max_train 100000
    python data/build_msmarco_data.py --output_dir /tmp/data/msmarco --max_train 50000 --num_neg 7
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List

from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_msmarco_corpus(max_passages: int = 0) -> Dict[str, Dict]:
    """Download MS MARCO passages and build corpus dict.

    Returns: {passage_id_str: {id, text, subject, topic, source, difficulty}}
    """
    print("  Loading MS MARCO passages from HuggingFace...")
    ds = load_dataset("microsoft/ms_marco", "v1.1", split="train")

    # MS MARCO v1.1 has passages embedded in each query row
    # We need to extract unique passages
    corpus = {}
    pid = 0
    seen_texts = set()

    for row in tqdm(ds, desc="  Extracting passages"):
        for passage_dict in row.get("passages", {}).get("passage_text", []):
            text = passage_dict if isinstance(passage_dict, str) else str(passage_dict)
            text = text.strip()
            if len(text) < 20:
                continue
            # Deduplicate by first 200 chars
            key = text[:200]
            if key in seen_texts:
                continue
            seen_texts.add(key)

            corpus[str(pid)] = {
                "id": f"p_{pid}",
                "text": text,
                "subject": "general",
                "topic": "web",
                "source": "msmarco",
                "difficulty": "medium",
            }
            pid += 1
            if max_passages and pid >= max_passages:
                break
        if max_passages and pid >= max_passages:
            break

    print(f"  Corpus: {len(corpus)} unique passages")
    return corpus


def build_msmarco_corpus_beir() -> Dict[str, Dict]:
    """Download MS MARCO corpus via BEIR format (cleaner, passage IDs match)."""
    print("  Loading MS MARCO corpus via BeIR...")
    try:
        from beir import util as beir_util
        from beir.datasets.data_loader import GenericDataLoader

        data_path = os.path.join("data/beir", "msmarco")
        if not os.path.exists(data_path):
            url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip"
            print("  Downloading msmarco (~1GB)...")
            beir_util.download_and_unzip(url, "data/beir")

        beir_corpus, _, _ = GenericDataLoader(data_path).load(split="dev")

        corpus = {}
        for cid, doc in tqdm(beir_corpus.items(), desc="  Building corpus"):
            text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
            if len(text) < 20:
                continue
            corpus[cid] = {
                "id": cid,
                "text": text,
                "subject": "general",
                "topic": "web",
                "source": "msmarco",
                "difficulty": "medium",
            }
        print(f"  Corpus: {len(corpus)} passages")
        return corpus

    except ImportError:
        print("  beir not installed, falling back to HuggingFace loader")
        return None


def build_msmarco_pairs_beir(corpus: Dict, max_train: int, num_neg: int,
                              val_size: int = 2000) -> Dict[str, List[Dict]]:
    """Build train/val/test pairs from BEIR MS MARCO format."""
    from beir.datasets.data_loader import GenericDataLoader

    data_path = os.path.join("data/beir", "msmarco")

    # Load dev split (MS MARCO has no public test labels)
    _, queries, qrels = GenericDataLoader(data_path).load(split="dev")

    print(f"  Dev queries: {len(queries)}, Qrels: {len(qrels)}")

    # Build pairs from qrels
    pairs = []
    corpus_texts = list(corpus.values())
    corpus_ids = list(corpus.keys())

    for qid, rels in tqdm(qrels.items(), desc="  Building pairs"):
        if qid not in queries:
            continue
        query = queries[qid]

        # Find positive passage(s)
        pos_ids = [did for did, score in rels.items() if score > 0]
        if not pos_ids:
            continue

        pos_id = pos_ids[0]
        if pos_id not in corpus:
            continue

        pos_text = corpus[pos_id]["text"]

        pairs.append({
            "query": query,
            "positive_text": pos_text,
            "positive_id": pos_id,
            "negative_texts": [],  # Will be filled by curriculum_negatives.py
            "negative_ids": [],
            "bloom_level": 1,  # Placeholder — will be overwritten by classifier
            "subject": "general",
            "topic": "web",
            "query_type": "factual",
        })

    print(f"  Raw pairs from dev qrels: {len(pairs)}")

    # For training, we need the train split too
    print("  Loading MS MARCO train split for training pairs...")
    try:
        _, train_queries, train_qrels = GenericDataLoader(data_path).load(split="train")
        print(f"  Train queries: {len(train_queries)}, Train qrels: {len(train_qrels)}")

        train_pairs = []
        for qid, rels in tqdm(train_qrels.items(), desc="  Building train pairs"):
            if qid not in train_queries:
                continue
            pos_ids = [did for did, score in rels.items() if score > 0]
            if not pos_ids:
                continue
            pos_id = pos_ids[0]
            if pos_id not in corpus:
                continue

            train_pairs.append({
                "query": train_queries[qid],
                "positive_text": corpus[pos_id]["text"],
                "positive_id": pos_id,
                "negative_texts": [],
                "negative_ids": [],
                "bloom_level": 1,
                "subject": "general",
                "topic": "web",
                "query_type": "factual",
            })

        print(f"  Train pairs from train qrels: {len(train_pairs)}")
    except Exception as e:
        print(f"  Could not load train split: {e}")
        print("  Using dev pairs only — will split into train/val/test")
        train_pairs = []

    # Combine and split
    if train_pairs:
        random.shuffle(train_pairs)
        train_set = train_pairs[:max_train]
        random.shuffle(pairs)
        val_set = pairs[:val_size]
        test_set = pairs[val_size:]
    else:
        # Only dev pairs available — split them
        random.shuffle(pairs)
        n = len(pairs)
        train_set = pairs[:int(0.8 * n)][:max_train]
        val_set = pairs[int(0.8 * n):int(0.9 * n)]
        test_set = pairs[int(0.9 * n):]

    return {"train": train_set, "val": val_set, "test": test_set}


def build_msmarco_pairs_hf(max_train: int, num_neg: int,
                            val_size: int = 2000) -> tuple:
    """Build pairs directly from HuggingFace MS MARCO dataset.

    Returns (corpus_dict, splits_dict).
    """
    print("  Loading MS MARCO v2.1 from HuggingFace...")
    ds_train = load_dataset("ms_marco", "v2.1", split="train")
    ds_val = load_dataset("ms_marco", "v2.1", split="validation")

    corpus = {}
    pid = 0
    seen = set()

    def extract_pairs(dataset, max_pairs, desc):
        nonlocal pid
        pairs = []
        for row in tqdm(dataset, desc=desc):
            query = row["query"]
            passages = row.get("passages", {})
            texts = passages.get("passage_text", [])
            is_selected = passages.get("is_selected", [])

            # Find positive
            pos_text = None
            neg_texts = []
            for t, sel in zip(texts, is_selected):
                t = t.strip()
                if len(t) < 20:
                    continue
                # Add to corpus
                key = t[:200]
                if key not in seen:
                    seen.add(key)
                    corpus[str(pid)] = {
                        "id": f"p_{pid}", "text": t,
                        "subject": "general", "topic": "web",
                        "source": "msmarco", "difficulty": "medium",
                    }
                    pid += 1
                if sel == 1 and pos_text is None:
                    pos_text = t
                elif sel == 0:
                    neg_texts.append(t)

            if pos_text is None:
                continue

            pairs.append({
                "query": query,
                "positive_text": pos_text,
                "positive_id": "",  # Will resolve later
                "negative_texts": neg_texts[:num_neg],
                "negative_ids": [],
                "bloom_level": 1,
                "subject": "general",
                "topic": "web",
                "query_type": "factual",
            })

            if max_pairs and len(pairs) >= max_pairs:
                break
        return pairs

    train_pairs = extract_pairs(ds_train, max_train, "  Train pairs")
    val_pairs = extract_pairs(ds_val, val_size, "  Val pairs")

    # Use rest of val as test
    test_pairs = val_pairs[val_size // 2:] if len(val_pairs) > val_size // 2 else []
    val_pairs = val_pairs[:val_size // 2]

    return corpus, {"train": train_pairs, "val": val_pairs, "test": test_pairs}


def annotate_bloom_levels(pairs: List[Dict], device="cuda") -> List[Dict]:
    """Classify queries into Bloom taxonomy levels using pretrained classifier."""
    from data.annotate_bloom_pretrained import load_pretrained_classifier, predict_bloom

    queries = [p["query"] for p in pairs]
    print(f"  Annotating {len(queries)} queries with Bloom levels...")

    bloom_model, bloom_tok, id2label = load_pretrained_classifier(device=device)
    labels_1idx = predict_bloom(queries, bloom_model, bloom_tok,
                                device=device, id2label=id2label)

    for p, bl in zip(pairs, labels_1idx):
        p["bloom_level"] = bl  # 1-indexed for data files

    # Print distribution
    from collections import Counter
    dist = Counter(labels_1idx)
    bloom_names = {1: "Remember", 2: "Understand", 3: "Apply",
                   4: "Analyze", 5: "Evaluate", 6: "Create"}
    print("  Bloom distribution:")
    for b in sorted(dist):
        print(f"    {bloom_names.get(b, b)}: {dist[b]} ({dist[b] / len(labels_1idx):.1%})")

    import torch
    del bloom_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Build MS MARCO data for BAM training")
    parser.add_argument("--output_dir", default="/tmp/data/msmarco")
    parser.add_argument("--max_train", type=int, default=100000,
                        help="Max training pairs (default 100k)")
    parser.add_argument("--num_neg", type=int, default=7,
                        help="Hard negatives per query")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_beir", action="store_true",
                        help="Use BEIR loader (cleaner passage IDs, auto-downloads ~1GB)")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Building MS MARCO Dataset for BAM Training")
    print("=" * 60)

    if args.use_beir:
        # BEIR path: cleaner corpus, standard passage IDs
        corpus = build_msmarco_corpus_beir()
        if corpus is None:
            print("  Falling back to HuggingFace loader...")
            corpus, splits = build_msmarco_pairs_hf(args.max_train, args.num_neg)
        else:
            splits = build_msmarco_pairs_beir(corpus, args.max_train, args.num_neg)
    else:
        # Direct HuggingFace loader — has negatives built in
        corpus, splits = build_msmarco_pairs_hf(args.max_train, args.num_neg)

    # Save corpus
    print(f"\n  Saving corpus ({len(corpus)} passages)...")
    with open(os.path.join(args.output_dir, "corpus.jsonl"), "w") as f:
        for cid, p in corpus.items():
            f.write(json.dumps(p) + "\n")

    # Annotate all splits with Bloom levels
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for split_name in ["train", "val", "test"]:
        if splits[split_name]:
            splits[split_name] = annotate_bloom_levels(splits[split_name], device=device)

    # Save splits
    for split_name, pairs in splits.items():
        path = os.path.join(args.output_dir, f"{split_name}.jsonl")
        with open(path, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"  {split_name}: {len(pairs)} pairs → {path}")

    # Metadata
    meta = {
        "num_corpus": len(corpus),
        "num_train": len(splits["train"]),
        "num_val": len(splits["val"]),
        "num_test": len(splits["test"]),
        "max_train": args.max_train,
        "num_neg": args.num_neg,
        "source": "msmarco",
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Corpus={len(corpus)}, "
          f"Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    print(f"Output: {args.output_dir}/")
    print(f"\nNext: mine hard negatives with:")
    print(f"  python data/curriculum_negatives.py \\")
    print(f"    --pairs {args.output_dir}/train.jsonl \\")
    print(f"    --corpus {args.output_dir}/corpus.jsonl \\")
    print(f"    --output {args.output_dir}/train_curriculum.jsonl \\")
    print(f"    --num_neg {args.num_neg} --stage 0.7")


if __name__ == "__main__":
    main()
