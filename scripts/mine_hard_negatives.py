"""
Hard Negative Mining for BAM Fine-tuning.

Uses the trained BAM model to retrieve top-K passages from the full corpus
for each training query. Passages that rank highly but are NOT the positive
are hard negatives — the model found them similar but they're wrong.

Replaces BM25 negatives in the training JSONL with model-mined hard negatives.
Run this after initial BAM training, then fine-tune for 3-5 more epochs.

Usage:
    python scripts/mine_hard_negatives.py \\
        --config configs/bam.yaml \\
        --checkpoint /tmp/bam-ckpts/best/ \\
        --input  /tmp/data/real/train.jsonl \\
        --corpus /tmp/data/real/corpus.jsonl \\
        --output /tmp/data/real/train_hard.jsonl \\
        --top_k 100 --num_hard 5

    # Then fine-tune BAM on hard negatives:
    python scripts/train_bam.py \\
        --config configs/bam.yaml \\
        --resume /tmp/bam-ckpts/best/ \\
        --train_path /tmp/data/real/train_hard.jsonl \\
        --checkpoint_dir /tmp/bam-ckpts-hard/ \\
        --num_epochs 5
"""
import argparse, json, sys, os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config
from models.bam import BloomAlignedMRL
from transformers import AutoTokenizer


def encode_texts(model, texts, tokenizer, device, is_query=False,
                 bloom_labels=None, batch_size=128):
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  encoding", leave=False):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            if is_query and bloom_labels is not None:
                bl = torch.tensor(bloom_labels[i:i + len(batch)], device=device) - 1  # 0-indexed
                out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                           bloom_labels=bl)
                all_embs.append(out["full_embedding"].cpu())  # use full emb for mining
            else:
                out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
                all_embs.append(out["full_embedding"].cpu())
    return torch.cat(all_embs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/bam.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input",      required=True, help="Training JSONL")
    parser.add_argument("--corpus",     required=True, help="Corpus JSONL")
    parser.add_argument("--output",     required=True, help="Output JSONL with hard negatives")
    parser.add_argument("--top_k",      type=int, default=100,
                        help="Retrieve top-K, then exclude positive")
    parser.add_argument("--num_hard",   type=int, default=5,
                        help="Number of hard negatives to keep per query")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])

    # Load model
    model = BloomAlignedMRL(config)
    ckpt_path = os.path.join(args.checkpoint, "checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()
    print(f"Loaded BAM from {ckpt_path}")

    # Load corpus
    print("Loading corpus...")
    corpus = []
    with open(args.corpus) as f:
        for line in f:
            corpus.append(json.loads(line.strip()))
    corpus_id_to_idx = {p["id"]: i for i, p in enumerate(corpus)}
    print(f"  {len(corpus)} passages")

    # Load training samples
    print("Loading training queries...")
    samples = []
    with open(args.input) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    print(f"  {len(samples)} samples")

    # Encode corpus (full dim — mining uses full encoder quality)
    print("Encoding corpus...")
    corpus_embs = encode_texts(model, [p["text"] for p in corpus], tokenizer, device,
                               is_query=False)
    corpus_embs = F.normalize(corpus_embs, p=2, dim=-1)
    print(f"  Corpus embeddings: {corpus_embs.shape}")

    # Encode queries
    print("Encoding queries...")
    query_texts  = [s["query"] for s in samples]
    query_blooms = [s.get("bloom_level", 1) for s in samples]
    query_embs = encode_texts(model, query_texts, tokenizer, device,
                              is_query=True, bloom_labels=query_blooms)
    query_embs = F.normalize(query_embs, p=2, dim=-1)
    print(f"  Query embeddings: {query_embs.shape}")

    # Retrieve hard negatives
    print(f"Mining top-{args.top_k} candidates per query...")
    top_k = min(args.top_k, len(corpus))
    chunk = 256
    all_topk_ids = []
    for i in tqdm(range(0, len(query_embs), chunk), desc="  retrieving"):
        q_chunk = query_embs[i:i + chunk].to(device)
        sim = torch.mm(q_chunk, corpus_embs.to(device).t())
        topk = sim.topk(top_k, dim=-1).indices.cpu().numpy()
        all_topk_ids.append(topk)
    all_topk_ids = np.concatenate(all_topk_ids)  # [N, top_k]

    # Build output JSONL: replace negatives with hard negatives
    print("Writing hard negative examples...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    n_found = 0
    n_fallback = 0

    with open(args.output, "w") as fout:
        for i, sample in enumerate(tqdm(samples, desc="  writing")):
            pos_id = sample.get("positive_id", "")
            pos_idx = corpus_id_to_idx.get(pos_id, -1)

            # Collect hard negatives: retrieved but not the positive
            hard_neg_ids = []
            for retrieved_idx in all_topk_ids[i]:
                if retrieved_idx != pos_idx:
                    hard_neg_ids.append(corpus[retrieved_idx]["id"])
                if len(hard_neg_ids) >= args.num_hard:
                    break

            if hard_neg_ids:
                n_found += 1
                sample["negative_ids"] = hard_neg_ids
            else:
                # Fallback: keep existing negatives
                n_fallback += 1

            fout.write(json.dumps(sample) + "\n")

    print(f"\nDone. Hard negatives mined for {n_found}/{len(samples)} queries "
          f"({n_fallback} fallbacks).")
    print(f"Output: {args.output}")
    print(f"\nNext step — fine-tune BAM on hard negatives:")
    print(f"  python scripts/train_bam.py \\")
    print(f"      --config configs/bam.yaml \\")
    print(f"      --resume /tmp/bam-ckpts/best/ \\")
    print(f"      --checkpoint_dir /tmp/bam-ckpts-hard/")
    print(f"  (Update data.train_path in config to {args.output} first)")


if __name__ == "__main__":
    main()
