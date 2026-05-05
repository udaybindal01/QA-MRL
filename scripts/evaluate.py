"""
General-purpose evaluation script for BAM-PQ, MRL, and Standard FT models.

Usage:
    python scripts/evaluate.py \
        --corpus_path /tmp/data/real/corpus.jsonl \
        --test_path   /tmp/data/real/test.jsonl \
        --model_type  bam_pq \
        --config      configs/bam_pq.yaml \
        --checkpoint  /tmp/uday/multi-domain/educational/bam_pq_e5large/best_bsr \
        --output_dir  results/eval/

    python scripts/evaluate.py \
        --corpus_path /tmp/data/real/corpus.jsonl \
        --test_path   /tmp/data/real/test.jsonl \
        --model_type  mrl \
        --config      configs/mrl_e5large.yaml \
        --checkpoint  /tmp/uday/multi-domain/educational/mrl_e5large/best \
        --output_dir  results/eval/

    python scripts/evaluate.py \
        --corpus_path /tmp/data/real/corpus.jsonl \
        --test_path   /tmp/data/real/test.jsonl \
        --model_type  standard_ft \
        --backbone    intfloat/e5-large-v2 \
        --checkpoint  /tmp/uday/multi-domain/educational/standard_ft_e5large/best \
        --output_dir  results/eval/
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from transformers import AutoTokenizer, AutoModel


BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply",
               4: "Analyze",  5: "Evaluate",   6: "Create"}


# ──────────────────────────── Data loading ────────────────────────────────

def load_corpus(corpus_path):
    corpus = []
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if line:
                corpus.append(json.loads(line))
    return corpus


def load_queries(test_path, corpus_id_to_idx):
    samples = []
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    valid = [s for s in samples if s.get("positive_id", "") in corpus_id_to_idx]
    skipped = len(samples) - len(valid)
    if skipped:
        print(f"  WARNING: skipped {skipped} queries with missing positive_id in corpus")
    return valid


# ──────────────────────────── Metrics ─────────────────────────────────────

def compute_metrics(query_embs, corpus_embs, gt_indices, query_blooms, device, ks=(1, 5, 10, 50)):
    N = len(query_embs)
    query_embs  = query_embs.float()
    corpus_embs = corpus_embs.float()

    rankings = []
    for i in range(0, N, 256):
        sim = torch.mm(query_embs[i:i+256].to(device), corpus_embs.t().to(device))
        topk = sim.topk(100, dim=-1).indices.cpu().numpy()
        rankings.append(topk)
    rankings = np.concatenate(rankings)

    metrics = {}
    for k in ks:
        hits = np.array([gt_indices[i] in rankings[i, :k] for i in range(N)])
        metrics[f"recall@{k}"] = float(hits.mean())

    mrrs = []
    for i in range(N):
        where = np.where(rankings[i] == gt_indices[i])[0]
        mrrs.append(1.0 / (where[0] + 1) if len(where) > 0 else 0.0)
    metrics["mrr"] = float(np.mean(mrrs))

    ndcgs = []
    for i in range(N):
        for j, idx in enumerate(rankings[i, :10]):
            if idx == gt_indices[i]:
                ndcgs.append(1.0 / np.log2(j + 2))
                break
        else:
            ndcgs.append(0.0)
    metrics["ndcg@10"] = float(np.mean(ndcgs))

    for level in range(1, 7):
        mask = query_blooms == level
        if mask.sum() == 0:
            continue
        lr = rankings[mask]
        lg = gt_indices[mask]
        nl = int(mask.sum())
        hits = np.array([lg[i] in lr[i, :10] for i in range(nl)])
        metrics[f"bloom_{BLOOM_NAMES[level]}_recall@10"] = float(hits.mean())
        metrics[f"bloom_{BLOOM_NAMES[level]}_n"] = nl

    return metrics


# ──────────────────────────── Encoders ────────────────────────────────────

@torch.no_grad()
def encode_bam_pq(config, checkpoint, corpus, queries, device):
    from models.bam import BloomAlignedMRL

    model = BloomAlignedMRL(config)
    ckpt_file = os.path.join(checkpoint, "checkpoint.pt")
    if os.path.exists(ckpt_file):
        state = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
        print(f"  Loaded BAM-PQ from {ckpt_file}")
    else:
        print(f"  WARNING: checkpoint not found at {ckpt_file}")
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])

    corpus_embs = []
    for i in tqdm(range(0, len(corpus), 128), desc="  corpus", leave=False):
        batch = [c["text"] for c in corpus[i:i+128]]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
        corpus_embs.append(out["full_embedding"].float().cpu())
    corpus_embs = torch.cat(corpus_embs)

    query_embs = []
    avg_dims_list = []
    for i in tqdm(range(0, len(queries), 64), desc="  queries", leave=False):
        batch = queries[i:i+64]
        enc = tokenizer([q["query"] for q in batch], padding=True,
                        truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        bloom_labels = torch.tensor(
            [q["bloom_level"] - 1 for q in batch], dtype=torch.long, device=device
        )
        out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                   bloom_labels=bloom_labels)
        query_embs.append(out["masked_embedding"].float().cpu())
        if "active_dims" in out:
            avg_dims_list.append(out["active_dims"].float().cpu())

    query_embs = torch.cat(query_embs)
    avg_dims = float(torch.cat(avg_dims_list).mean().item()) if avg_dims_list else float(query_embs.shape[-1])

    del model
    torch.cuda.empty_cache()
    return query_embs, corpus_embs, avg_dims


@torch.no_grad()
def encode_mrl(config, checkpoint, corpus, queries, device):
    from models.encoder import MRLEncoder

    mc = config["model"]
    model = MRLEncoder(model_name=mc["backbone"],
                       embedding_dim=mc["embedding_dim"],
                       mrl_dims=mc["mrl_dims"])
    ckpt_file = os.path.join(checkpoint, "checkpoint.pt")
    if os.path.exists(ckpt_file):
        state = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
        print(f"  Loaded MRL from {ckpt_file}")
    else:
        print(f"  WARNING: checkpoint not found at {ckpt_file}")
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(mc["backbone"])

    def encode(texts, max_len):
        all_embs = []
        for i in tqdm(range(0, len(texts), 128), desc="  encoding", leave=False):
            enc = tokenizer(texts[i:i+128], padding=True, truncation=True,
                            max_length=max_len, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(enc["input_ids"], enc["attention_mask"])
            all_embs.append(out["full"].float().cpu())
        return torch.cat(all_embs)

    corpus_embs = encode([c["text"] for c in corpus], 256)
    query_embs  = encode([q["query"] for q in queries], 128)
    avg_dims = float(mc["embedding_dim"])

    del model
    torch.cuda.empty_cache()
    return query_embs, corpus_embs, avg_dims


@torch.no_grad()
def encode_standard_ft(backbone, checkpoint, corpus, queries, device):
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = AutoModel.from_pretrained(backbone).to(device).eval()

    ckpt_file = os.path.join(checkpoint, "checkpoint.pt")
    if os.path.exists(ckpt_file):
        state = torch.load(ckpt_file, map_location=device)
        sd = state.get("model_state_dict", state)
        try:
            model.load_state_dict(sd, strict=False)
            print(f"  Loaded standard FT from {ckpt_file}")
        except Exception as e:
            print(f"  WARNING: partial load ({e})")
    else:
        print(f"  WARNING: checkpoint not found at {ckpt_file}; using pretrained weights")

    def encode(texts, max_len):
        all_embs = []
        for i in tqdm(range(0, len(texts), 128), desc="  encoding", leave=False):
            enc = tokenizer(texts[i:i+128], padding=True, truncation=True,
                            max_length=max_len, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            emb = F.normalize(out.last_hidden_state[:, 0].float(), p=2, dim=-1)
            all_embs.append(emb.cpu())
        return torch.cat(all_embs)

    corpus_embs = encode([c["text"] for c in corpus], 256)
    query_embs  = encode([q["query"] for q in queries], 128)
    avg_dims = float(query_embs.shape[-1])

    del model
    torch.cuda.empty_cache()
    return query_embs, corpus_embs, avg_dims


# ──────────────────────────── Main ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval model on a given test set")
    parser.add_argument("--corpus_path", required=True, help="Path to corpus.jsonl")
    parser.add_argument("--test_path",   required=True, help="Path to test/val jsonl")
    parser.add_argument("--model_type",  required=True,
                        choices=["bam_pq", "mrl", "standard_ft"],
                        help="Model type to evaluate")
    parser.add_argument("--config",      default=None,
                        help="Path to YAML config (required for bam_pq / mrl)")
    parser.add_argument("--checkpoint",  required=True,
                        help="Directory containing checkpoint.pt")
    parser.add_argument("--backbone",    default=None,
                        help="HuggingFace model name (for standard_ft; overrides config)")
    parser.add_argument("--output_dir",  default="results/eval/",
                        help="Directory to save metrics JSON")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print(f"  Model:  {args.model_type}")

    # Load data
    print(f"\n  Loading corpus from {args.corpus_path}")
    corpus = load_corpus(args.corpus_path)
    corpus_id_to_idx = {c["id"]: i for i, c in enumerate(corpus)}
    print(f"  Loading queries from {args.test_path}")
    queries = load_queries(args.test_path, corpus_id_to_idx)
    gt_indices   = np.array([corpus_id_to_idx[q["positive_id"]] for q in queries])
    query_blooms = np.array([q.get("bloom_level", 0) for q in queries])
    print(f"  Corpus: {len(corpus)}, Queries: {len(queries)}")

    # Load config if needed
    config = None
    if args.config:
        config = load_config(args.config)

    # Encode
    print(f"\n  Encoding with {args.model_type} ...")
    if args.model_type == "bam_pq":
        if config is None:
            parser.error("--config is required for model_type=bam_pq")
        query_embs, corpus_embs, avg_dims = encode_bam_pq(
            config, args.checkpoint, corpus, queries, device)

    elif args.model_type == "mrl":
        if config is None:
            parser.error("--config is required for model_type=mrl")
        query_embs, corpus_embs, avg_dims = encode_mrl(
            config, args.checkpoint, corpus, queries, device)

    else:  # standard_ft
        backbone = args.backbone
        if backbone is None and config is not None:
            backbone = config["model"]["backbone"]
        if backbone is None:
            parser.error("--backbone is required for model_type=standard_ft when no config given")
        query_embs, corpus_embs, avg_dims = encode_standard_ft(
            backbone, args.checkpoint, corpus, queries, device)

    # Metrics
    print("  Computing metrics ...")
    metrics = compute_metrics(query_embs, corpus_embs, gt_indices, query_blooms, device)
    metrics["avg_dims"] = avg_dims
    metrics["n_queries"] = len(queries)
    metrics["model_type"] = args.model_type
    metrics["checkpoint"] = args.checkpoint

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    tag = os.path.basename(args.checkpoint.rstrip("/"))
    out_path = os.path.join(args.output_dir, f"{args.model_type}_{tag}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    # Print
    print(f"\n{'═'*70}")
    print(f"  {args.model_type.upper()}  —  {os.path.basename(args.test_path)}")
    print(f"{'═'*70}")
    print(f"  {'Metric':<20}  {'Value':>10}")
    print(f"  {'─'*32}")
    for k in ["recall@1", "recall@5", "recall@10", "recall@50", "ndcg@10", "mrr", "avg_dims"]:
        if k in metrics:
            print(f"  {k:<20}  {metrics[k]:>10.4f}")

    print(f"\n  {'─'*70}")
    print(f"  Bloom-stratified R@10")
    print(f"  {'─'*70}")
    for level in range(1, 7):
        key = f"bloom_{BLOOM_NAMES[level]}_recall@10"
        n_key = f"bloom_{BLOOM_NAMES[level]}_n"
        if key in metrics:
            n = metrics.get(n_key, "?")
            print(f"  {BLOOM_NAMES[level]:<12}  R@10={metrics[key]:.4f}  (n={n})")

    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
