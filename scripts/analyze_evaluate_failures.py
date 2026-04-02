"""
Analyze failures for Bloom=Evaluate queries (Challenge 3).

For queries where Bloom=Evaluate and R@10=0, categorizes the failure mode:
  - RANK_11_50:   positive is ranked 11-50 (near miss)
  - RANK_51_200:  positive is ranked 51-200 (recall failure)
  - RANK_200+:    positive is ranked >200 (deep failure)
  - NOT_FOUND:    positive not in top-1000 (complete miss)

Also checks whether top-10 retrieved passages are topically related
(via lexical overlap with the positive) to distinguish:
  - Precision failure: retrieves related but not exact passage
  - Recall failure:   retrieves unrelated passages entirely

Output:
  evaluate_failures.json  — failure breakdown table for paper

Usage:
    python scripts/analyze_evaluate_failures.py \
        --config configs/bam.yaml \
        --checkpoint /tmp/bam-ckpts/best/ \
        --output_dir results/analysis/
"""

import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from models.bam import BloomAlignedMRL
from transformers import AutoTokenizer

BLOOM_EVALUATE = 5  # 1-indexed level 5 = Evaluate


def tokenize_text(text: str):
    """Simple whitespace tokenizer for overlap computation."""
    return set(text.lower().split())


def lexical_overlap(text_a: str, text_b: str) -> float:
    """Jaccard similarity between word sets."""
    a, b = tokenize_text(text_a), tokenize_text(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@torch.no_grad()
def run_analysis(args):
    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])
    os.makedirs(args.output_dir, exist_ok=True)

    config["training"]["loss"]["bloom_frequencies"] = [1/6] * 6
    model = BloomAlignedMRL(config)
    ckpt = os.path.join(args.checkpoint, "checkpoint.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(
            torch.load(ckpt, map_location=device)["model_state_dict"], strict=False
        )
        print(f"Loaded from {ckpt}")
    model.to(device).eval()

    # Load corpus
    corpus = []
    with open(config["data"]["corpus_path"]) as f:
        for line in f:
            corpus.append(json.loads(line.strip()))
    corpus_id_to_idx = {p["id"]: i for i, p in enumerate(corpus)}
    corpus_texts = [p["text"] for p in corpus]

    # Load test data, filter to Evaluate queries
    samples = []
    with open(config["data"]["test_path"]) as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    evaluate_samples = [
        s for s in samples
        if s.get("bloom_level") == BLOOM_EVALUATE
        and s.get("positive_id", "") in corpus_id_to_idx
    ]
    print(f"Evaluate queries (Bloom=Evaluate, positive in corpus): {len(evaluate_samples)}")

    if not evaluate_samples:
        print("No Evaluate queries found.")
        return

    # Encode corpus
    print("Encoding corpus...")
    corpus_embs = []
    for i in tqdm(range(0, len(corpus), 128), desc="Corpus"):
        batch = [c["text"] for c in corpus[i:i + 128]]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
        corpus_embs.append(out["full_embedding"].cpu())
    corpus_embs = torch.cat(corpus_embs)

    # Encode Evaluate queries
    TOP_K = 1000  # check this many candidates for rank
    query_masked_list = []
    for i in tqdm(range(0, len(evaluate_samples), 64), desc="Evaluate queries"):
        batch = evaluate_samples[i:i + 64]
        enc = tokenizer([s["query"] for s in batch], padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        bloom_labels = torch.full((len(batch),), BLOOM_EVALUATE - 1, dtype=torch.long, device=device)
        out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                   bloom_labels=bloom_labels)
        query_masked_list.append(out["masked_embedding"].cpu())

    query_masked = torch.cat(query_masked_list)
    N = len(evaluate_samples)
    gt_indices = np.array([corpus_id_to_idx[s["positive_id"]] for s in evaluate_samples])

    # Retrieve top-K for each query
    print(f"Retrieving top-{TOP_K}...")
    all_rankings = []
    for i in range(0, N, 256):
        q = query_masked[i:i + 256].to(device)
        sim = torch.mm(q, corpus_embs.to(device).t())
        topk = sim.topk(min(TOP_K, len(corpus)), dim=-1).indices.cpu().numpy()
        all_rankings.append(topk)
    all_rankings = np.concatenate(all_rankings, axis=0)  # [N, TOP_K]

    # Identify failures: queries where positive not in top-10
    fail_indices = [i for i in range(N) if gt_indices[i] not in all_rankings[i, :10]]
    print(f"\nTotal Evaluate queries: {N}")
    print(f"R@10 = {1 - len(fail_indices)/N:.4f}  ({N - len(fail_indices)}/{N} hits)")
    print(f"Failures (R@10=0): {len(fail_indices)}")

    # Categorize failures by rank
    categories = Counter()
    failure_details = []
    for i in fail_indices:
        pos_id = gt_indices[i]
        ranking = all_rankings[i].tolist()
        try:
            rank = ranking.index(pos_id) + 1  # 1-indexed
        except ValueError:
            rank = TOP_K + 1  # beyond our search window

        if rank <= 50:
            cat = "RANK_11_50"
        elif rank <= 200:
            cat = "RANK_51_200"
        elif rank <= TOP_K:
            cat = "RANK_200+"
        else:
            cat = "NOT_FOUND"
        categories[cat] += 1

        # Check top-10 lexical overlap with positive
        pos_text = corpus_texts[pos_id]
        query_text = evaluate_samples[i]["query"]
        top10_texts = [corpus_texts[j] for j in all_rankings[i, :10]]
        top10_overlaps = [lexical_overlap(pos_text, t) for t in top10_texts]
        max_overlap = max(top10_overlaps)
        avg_overlap = float(np.mean(top10_overlaps))

        failure_details.append({
            "query": query_text,
            "positive_id": evaluate_samples[i]["positive_id"],
            "rank": rank,
            "category": cat,
            "top10_max_lexical_overlap_with_positive": round(max_overlap, 3),
            "top10_avg_lexical_overlap_with_positive": round(avg_overlap, 3),
            "top_retrieved_text_snippet": top10_texts[0][:200],
        })

    # Precision vs recall failure breakdown
    # If top-10 overlap is high but retrieval fails → precision failure (retrieves similar but not exact)
    # If top-10 overlap is low → recall failure (retrieves unrelated passages)
    OVERLAP_THRESHOLD = 0.15
    precision_failures = [d for d in failure_details
                          if d["top10_max_lexical_overlap_with_positive"] >= OVERLAP_THRESHOLD]
    recall_failures = [d for d in failure_details
                       if d["top10_max_lexical_overlap_with_positive"] < OVERLAP_THRESHOLD]

    print(f"\nFailure categories:")
    for cat, count in sorted(categories.items()):
        pct = count / len(fail_indices) * 100
        print(f"  {cat:15s}: {count:4d} ({pct:.1f}%)")

    print(f"\nFailure type (lexical overlap threshold={OVERLAP_THRESHOLD}):")
    print(f"  Precision failures (top-10 topically related): {len(precision_failures)} "
          f"({len(precision_failures)/max(len(fail_indices),1)*100:.1f}%)")
    print(f"  Recall failures   (top-10 unrelated):          {len(recall_failures)} "
          f"({len(recall_failures)/max(len(fail_indices),1)*100:.1f}%)")

    # Save
    result = {
        "bloom_level": "Evaluate",
        "total_queries": N,
        "hits_at_10": N - len(fail_indices),
        "recall_at_10": float(1 - len(fail_indices) / N),
        "n_failures": len(fail_indices),
        "rank_categories": dict(categories),
        "failure_type": {
            "precision_failures": len(precision_failures),
            "recall_failures": len(recall_failures),
            "threshold": OVERLAP_THRESHOLD,
        },
        "failure_details": failure_details[:50],  # top-50 examples for paper
    }
    out_path = os.path.join(args.output_dir, "evaluate_failures.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bam.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="results/analysis/")
    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
