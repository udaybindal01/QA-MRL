"""
Dynamic Hard Negative Refresh.

Re-mines hard negatives against the current model's embedding space.
Static negatives mined at initialization become trivially easy as training
progresses — this script refreshes them using the current checkpoint.

Strategy:
  For each training query, encode it with the current model, then find the
  top-K corpus passages by cosine similarity that are NOT the labeled positive.
  These are the new hard negatives: the model currently considers them relevant
  but they are not.

Usage (standalone):
    python scripts/refresh_hard_negatives.py \
        --config configs/bam.yaml \
        --checkpoint /tmp/bam-ckpts/epoch_4/ \
        --pairs ./data/real/train.jsonl \
        --corpus ./data/real/corpus.jsonl \
        --output ./data/real/train_refreshed.jsonl \
        --num_neg 3 \
        --margin 0.1

Called automatically by BAMTrainer when hard_neg_refresh_epochs > 0.
"""

import argparse, json, os, sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from models.bam import BloomAlignedMRL
from models.encoder import MRLEncoder
from transformers import AutoTokenizer


@torch.no_grad()
def encode_texts(model, texts, tokenizer, device, batch_size=128,
                 is_query=False, max_length=256):
    """Encode texts returning normalized full embeddings."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        if hasattr(model, "encode_queries") and is_query:
            out = model.encode_queries(enc["input_ids"], enc["attention_mask"])
            all_embs.append(out["full_embedding"].cpu())
        elif hasattr(model, "encode_documents") and not is_query:
            out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
            all_embs.append(out["full_embedding"].cpu())
        else:
            out = model(enc["input_ids"], enc["attention_mask"])
            all_embs.append(out["full"].cpu())

    return torch.cat(all_embs)  # [N, D] normalized


def refresh_hard_negatives(
    model,
    tokenizer,
    pairs_path: str,
    corpus_path: str,
    output_path: str,
    device,
    num_neg: int = 3,
    margin: float = 0.05,
    batch_size: int = 128,
):
    """
    Re-mine hard negatives for each training query.

    margin: how much below the positive similarity a negative must be to qualify.
    Negatives with sim > positive_sim - margin are excluded (too hard / false negatives).
    This avoids mining the positive itself or passages so similar they may be true positives.

    Writes updated JSONL to output_path.
    """
    # Load corpus
    corpus = []
    with open(corpus_path) as f:
        for line in f:
            corpus.append(json.loads(line.strip()))
    corpus_id_to_idx = {p["id"]: i for i, p in enumerate(corpus)}

    # Load pairs
    pairs = []
    with open(pairs_path) as f:
        for line in f:
            pairs.append(json.loads(line.strip()))

    print(f"  Encoding {len(corpus)} corpus passages...")
    corpus_embs = encode_texts(
        model, [p["text"] for p in corpus], tokenizer, device,
        batch_size=batch_size, is_query=False, max_length=256,
    )  # [C, D]

    print(f"  Encoding {len(pairs)} training queries...")
    query_embs = encode_texts(
        model, [p["query"] for p in pairs], tokenizer, device,
        batch_size=batch_size, is_query=True, max_length=128,
    )  # [Q, D]

    print(f"  Mining hard negatives (num_neg={num_neg}, margin={margin})...")
    chunk = 256
    updated = 0
    out_pairs = []

    for i in tqdm(range(0, len(pairs), chunk), desc="  mining"):
        q_chunk = query_embs[i:i + chunk].to(device)
        sim = torch.mm(q_chunk, corpus_embs.to(device).t())  # [chunk, C]

        for j, pair in enumerate(pairs[i:i + chunk]):
            sims_j = sim[j].cpu()

            pos_id  = pair.get("positive_id", "")
            pos_idx = corpus_id_to_idx.get(pos_id, -1)
            pos_sim = sims_j[pos_idx].item() if pos_idx >= 0 else 1.0

            # Zero out the positive to exclude it from negative selection
            sims_j[pos_idx] = -1.0

            # Filter candidates: must be below (pos_sim - margin) to avoid false negatives
            upper_bound = pos_sim - margin
            candidate_mask = sims_j <= upper_bound
            candidate_sims = sims_j.clone()
            candidate_sims[~candidate_mask] = -2.0

            # Top-K by similarity among valid candidates (hardest valid negatives)
            top_k = min(num_neg, int(candidate_mask.sum()))
            if top_k == 0:
                # Fallback: just take the top-num_neg excluding positive
                sims_j[pos_idx] = -1.0
                top_k = num_neg
                top_idx = sims_j.topk(top_k).indices.tolist()
            else:
                top_idx = candidate_sims.topk(top_k).indices.tolist()

            new_neg_texts = [corpus[idx]["text"] for idx in top_idx]
            new_neg_ids   = [corpus[idx]["id"]   for idx in top_idx]

            # Pad to num_neg if needed (edge case: corpus too small)
            while len(new_neg_texts) < num_neg:
                new_neg_texts.append(new_neg_texts[-1])
                new_neg_ids.append(new_neg_ids[-1])

            updated_pair = dict(pair)
            updated_pair["negative_texts"] = new_neg_texts[:num_neg]
            updated_pair["negative_ids"]   = new_neg_ids[:num_neg]
            out_pairs.append(updated_pair)
            updated += 1

    with open(output_path, "w") as f:
        for p in out_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"  Updated {updated} training pairs → {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True,
                        help="Checkpoint dir (epoch_N/ or best/)")
    parser.add_argument("--pairs",      required=True,
                        help="Input training pairs JSONL")
    parser.add_argument("--corpus",     required=True,
                        help="Corpus JSONL")
    parser.add_argument("--output",     required=True,
                        help="Output JSONL with refreshed negatives")
    parser.add_argument("--num_neg",    type=int, default=3)
    parser.add_argument("--margin",     type=float, default=0.05,
                        help="Negatives with sim > pos_sim - margin are excluded.")
    parser.add_argument("--model_type", choices=["bam", "mrl"], default="bam")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])

    config["training"]["loss"].setdefault("bloom_frequencies", [1/6] * 6)
    if args.model_type == "bam":
        model = BloomAlignedMRL(config)
    else:
        mc = config["model"]
        model = MRLEncoder(mc["backbone"], mc["embedding_dim"], mc["mrl_dims"])

    ckpt = os.path.join(args.checkpoint, "checkpoint.pt")
    if os.path.exists(ckpt):
        result = model.load_state_dict(
            torch.load(ckpt, map_location=device)["model_state_dict"], strict=False
        )
        if result.missing_keys:
            print(f"WARNING missing keys: {result.missing_keys[:5]}")
    model.to(device).eval()

    refresh_hard_negatives(
        model, tokenizer,
        pairs_path=args.pairs,
        corpus_path=args.corpus,
        output_path=args.output,
        device=device,
        num_neg=args.num_neg,
        margin=args.margin,
    )


if __name__ == "__main__":
    main()
