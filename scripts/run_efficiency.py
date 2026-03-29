"""
Run efficiency benchmark: full vs MRL-truncated vs QA-MRL sparse retrieval.

Produces the efficiency table showing actual FLOPS and latency savings.

Usage:
    python scripts/run_efficiency.py --config configs/neurips.yaml \
        --checkpoint /tmp/qa-mrl-ckpts/best/ \
        --corpus_path data/real/corpus.jsonl
"""

import argparse
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from models.bam import BloomAlignedMRL
from evaluation.efficient_retrieval import SparseRetriever, EfficiencyBenchmark
from transformers import AutoTokenizer
from tqdm import tqdm


@torch.no_grad()
def encode_corpus_and_queries(model, corpus_path, test_path, tokenizer, device):
    """Encode corpus and test queries, return embeddings and masks."""
    model.eval()

    # Corpus
    print("Encoding corpus...")
    corpus_texts = []
    with open(corpus_path) as f:
        for line in f:
            d = json.loads(line)
            corpus_texts.append(d["text"])

    corpus_embs = []
    for i in tqdm(range(0, len(corpus_texts), 128), desc="  corpus"):
        batch = corpus_texts[i:i+128]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        if hasattr(model, "encode_documents"):
            out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
            corpus_embs.append(out["full_embedding"].cpu().numpy())
        else:
            out = model(enc["input_ids"], enc["attention_mask"])
            corpus_embs.append(out["full"].cpu().numpy())
    corpus_embs = np.concatenate(corpus_embs)

    # Queries
    print("Encoding queries...")
    query_texts = []
    with open(test_path) as f:
        for line in f:
            d = json.loads(line)
            query_texts.append(d["query"])

    query_embs, query_masks = [], []
    for i in tqdm(range(0, len(query_texts), 64), desc="  queries"):
        batch = query_texts[i:i+64]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        if hasattr(model, "encode_queries"):
            # out = model.encode_queries(enc["input_ids"], enc["attention_mask"])
            lf = None
            if hasattr(model.query_router, "learner_proj"):
                lf = torch.zeros(enc["input_ids"].size(0), 6).to(device)
            out = model.encode_queries(enc["input_ids"], enc["attention_mask"], learner_features=lf)
            query_embs.append(out["masked_embedding"].cpu().numpy())
            query_masks.append(out["mask"].cpu().numpy())
        else:
            out = model(enc["input_ids"], enc["attention_mask"])
            query_embs.append(out["full"].cpu().numpy())

    query_embs = np.concatenate(query_embs)
    query_masks_np = np.concatenate(query_masks) if query_masks else None

    return corpus_embs, query_embs, query_masks_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/neurips.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="results/efficiency/")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])

    # Load model
    model = QAMRL(config)
    ckpt = os.path.join(args.checkpoint, "checkpoint.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
    model.to(device).eval()

    # Encode
    corpus_path = config["data"]["corpus_path"]
    test_path = config["data"]["test_path"]
    corpus_embs, query_embs, query_masks = encode_corpus_and_queries(
        model, corpus_path, test_path, tokenizer, device
    )

    print(f"\nCorpus: {corpus_embs.shape}, Queries: {query_embs.shape}")
    if query_masks is not None:
        avg_active = (query_masks > 0.5).sum(axis=1).mean()
        print(f"Avg active dims: {avg_active:.0f}/{corpus_embs.shape[1]}")

    # Run benchmark
    benchmark = EfficiencyBenchmark(corpus_embs, group_size=config["model"]["router"]["group_size"])
    results = benchmark.benchmark(query_embs, query_masks, k=100)
    benchmark.print_benchmark(results)

    # Also test sparse retrieval quality
    if query_masks is not None:
        print("\nSparse retrieval quality check...")
        retriever = SparseRetriever(corpus_embs, config["model"]["router"]["group_size"])
        scores, indices, stats = retriever.search(query_embs, query_masks, k=100)
        retriever.print_stats(stats)

    # Save
    with open(os.path.join(args.output_dir, "efficiency_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()