"""
eval_pretrained_truncation.py

Evaluates a pretrained backbone at multiple truncation dimensions WITHOUT
any fine-tuning. Serves as the zero-shot truncation baseline for all models.

For MRL-native models (Qwen3-Embedding): truncation is already meaningful
since the model was pretrained with Matryoshka loss by its authors.
For others (e5-large, BGE-large, LLM2Vec, GritLM): shows performance
degradation from naive truncation — these need MRL fine-tuning to be useful.

Usage:
    python scripts/eval_pretrained_truncation.py \
        --config configs/mrl_e5large.yaml \
        --test_path ./data/real/test.jsonl \
        --corpus_path ./data/real/corpus.jsonl \
        --output_dir results/pretrained_baselines/e5large/
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

from models.encoder import MRLEncoder
from utils.misc import load_config


BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply",
               4: "Analyze", 5: "Evaluate", 6: "Create"}

# Models natively pretrained with Matryoshka loss by their authors
MRL_NATIVE_BACKBONES = {
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",
}


@torch.no_grad()
def encode_texts(model, texts, tokenizer, device,
                 is_query=False, batch_size=128, max_length=256):
    model.eval()
    _query_instr = getattr(model, "query_instruction", None) if is_query else None

    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size),
                  desc="  queries" if is_query else "  corpus", leave=False):
        batch = texts[i:i + batch_size]
        if _query_instr:
            batch = [_query_instr + t for t in batch]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        all_embs.append(out["full"].cpu())
    return torch.cat(all_embs, dim=0)


def retrieval_metrics(query_embs, corpus_embs, gt_indices, ks, device, chunk_size=256):
    N = len(query_embs)
    max_k = max(ks)
    rankings = []

    c = corpus_embs.to(device)
    for i in range(0, N, chunk_size):
        chunk = query_embs[i:i + chunk_size].to(device)
        sim = torch.mm(chunk, c.t())
        topk = sim.topk(max_k, dim=-1).indices.cpu().numpy()
        rankings.append(topk)
    rankings = np.concatenate(rankings, axis=0)

    metrics = {}
    for k in ks:
        topk = rankings[:, :k]
        hits = np.array([gt_indices[i] in topk[i] for i in range(N)])
        metrics[f"recall@{k}"] = float(hits.mean())

        ndcg_scores = []
        for i in range(N):
            row = topk[i].tolist()
            if gt_indices[i] in row:
                rank = row.index(gt_indices[i]) + 1
                ndcg_scores.append(1.0 / np.log2(rank + 1))
            else:
                ndcg_scores.append(0.0)
        metrics[f"ndcg@{k}"] = float(np.mean(ndcg_scores))

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained model at multiple truncation dims (no fine-tuning)"
    )
    parser.add_argument("--config", required=True,
                        help="MRL config YAML — used for model architecture only, no checkpoint loaded")
    parser.add_argument("--test_path", default=None,
                        help="Override test path from config")
    parser.add_argument("--corpus_path", default=None,
                        help="Override corpus path from config")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--checkpoint", default=None,
                        help="Optional checkpoint dir (standard FT or MRL). "
                             "If omitted, evaluates raw pretrained weights.")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 20, 50, 100])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = load_config(args.config)
    mc = config["model"]
    dc = config["data"]

    test_path   = args.test_path   or dc["test_path"]
    corpus_path = args.corpus_path or dc["corpus_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = mc["backbone"]
    is_mrl_native = backbone in MRL_NATIVE_BACKBONES
    mode = "standard_ft" if args.checkpoint else "pretrained"
    print(f"Model   : {backbone}")
    print(f"Mode    : {mode}")
    print(f"MRL-native (pretrained by authors): {is_mrl_native}")
    print(f"Dims    : {mc['mrl_dims']}")
    print(f"Device  : {device}")

    model = MRLEncoder(
        model_name=backbone,
        embedding_dim=mc["embedding_dim"],
        mrl_dims=mc["mrl_dims"],
        pooling=mc["pooling"],
        normalize=mc["normalize_embeddings"],
        torch_dtype=mc.get("torch_dtype", None),
        gradient_checkpointing=False,
        backbone_type=mc.get("backbone_type", "standard"),
        query_instruction=mc.get("query_instruction", None),
        peft_model_name=mc.get("peft_model_name", None),
    ).to(device)
    # ── Load fine-tuned checkpoint if provided ────────────────────────────────
    if args.checkpoint:
        ckpt_file = os.path.join(args.checkpoint, "checkpoint.pt")
        if os.path.exists(ckpt_file):
            ckpt = torch.load(ckpt_file, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            print(f"  Loaded checkpoint: {ckpt_file}")
        else:
            print(f"  WARNING: checkpoint not found at {ckpt_file}, using pretrained weights")

    model.eval()
    tokenizer = model.get_tokenizer()

    # ── Load corpus ───────────────────────────────────────────────────────────
    print("\nLoading corpus...")
    corpus = [json.loads(l) for l in open(corpus_path)]
    corpus_id_to_idx = {p["id"]: i for i, p in enumerate(corpus)}
    print(f"  {len(corpus)} passages")

    print("Encoding corpus (full dim)...")
    corpus_embs = encode_texts(model, [p["text"] for p in corpus], tokenizer,
                                device, is_query=False,
                                batch_size=args.batch_size, max_length=256)

    # ── Load test queries ─────────────────────────────────────────────────────
    print("Loading test queries...")
    test_samples = [json.loads(l) for l in open(test_path)]
    valid = [s for s in test_samples
             if s.get("positive_id", "") in corpus_id_to_idx]
    print(f"  {len(valid)} valid queries")

    print("Encoding queries (full dim)...")
    query_embs = encode_texts(model, [s["query"] for s in valid], tokenizer,
                               device, is_query=True,
                               batch_size=args.batch_size, max_length=128)

    gt_indices  = np.array([corpus_id_to_idx[s["positive_id"]] for s in valid])
    bloom_levels = np.array([s.get("bloom_level", 0) for s in valid])

    # ── Evaluate at each truncation dim ───────────────────────────────────────
    print(f"\nEvaluating at dims: {mc['mrl_dims']}")
    print(f"  {'Dim':>6}  {'R@10':>8}  {'NDCG@10':>10}")
    print("  " + "─" * 30)

    results_by_dim = {}
    for d in mc["mrl_dims"]:
        q_d = F.normalize(query_embs[:, :d],  p=2, dim=-1)
        c_d = F.normalize(corpus_embs[:, :d], p=2, dim=-1)
        m   = retrieval_metrics(q_d, c_d, gt_indices, args.ks, device)
        results_by_dim[d] = m
        print(f"  {d:>6}  {m.get('recall@10', 0):>8.4f}  {m.get('ndcg@10', 0):>10.4f}")

    # ── Bloom-stratified at full dim ──────────────────────────────────────────
    q_full = F.normalize(query_embs, p=2, dim=-1)
    c_full = F.normalize(corpus_embs, p=2, dim=-1)

    bloom_results = {}
    for level in range(1, 7):
        mask = bloom_levels == level
        if mask.sum() == 0:
            continue
        m = retrieval_metrics(q_full[mask], c_full, gt_indices[mask], args.ks, device)
        bloom_results[BLOOM_NAMES[level]] = m

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "model":         backbone,
        "backbone_type": mc.get("backbone_type", "standard"),
        "is_mrl_native": is_mrl_native,
        "embedding_dim": mc["embedding_dim"],
        "dims_evaluated": mc["mrl_dims"],
        "results_by_dim": results_by_dim,
        "bloom_stratified_full_dim": bloom_results,
        "num_queries": len(valid),
        "corpus_size": len(corpus),
    }

    out_path = os.path.join(args.output_dir, "pretrained_truncation.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
