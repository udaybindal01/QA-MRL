"""
eval_cutoff_norm.py

Cutoff and Norm baselines evaluated on an existing MRL checkpoint.
No additional training required — post-hoc evaluation only.

Cutoff: Fixed truncation of the full-dim MRL embedding to a preset budget.
        All queries use the same d dimensions (the first d via prefix slicing).
        Equivalent to "MRL at truncation dim d" but framed as a baseline.

Norm:   Per-query dimension selection by top-d absolute magnitude.
        Each query picks its own d dims from its embedding by |e_q[i]| ranking.
        Query-adaptive, unsupervised, no training required.
        Corpus is projected to the same query-selected dims before dot-product.

Outputs per-dim and Bloom-stratified recall for both baselines.

Usage:
    python scripts/eval_cutoff_norm.py \\
        --config   configs/mrl_e5large.yaml \\
        --checkpoint /tmp/multi-domain/educational/mrl_e5large/best \\
        --output_dir results/cutoff_norm/e5large/

    # Evaluate at all MRL dims + 30% of embedding_dim:
    python scripts/eval_cutoff_norm.py \\
        --config   configs/mrl_bge_large.yaml \\
        --checkpoint /tmp/multi-domain/educational/mrl_bge/best \\
        --output_dir results/cutoff_norm/bge/ \\
        --add_30pct
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoder import MRLEncoder
from utils.misc import load_config

BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply",
               4: "Analyze", 5: "Evaluate", 6: "Create"}


# ── Encoding ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_texts(model: MRLEncoder, texts: List[str], tokenizer,
                 device, is_query: bool = False,
                 batch_size: int = 128, max_length: int = 256) -> torch.Tensor:
    model.eval()
    instr = getattr(model, "query_instruction", None) if is_query else None
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size),
                  desc="  queries" if is_query else "  corpus", leave=False):
        batch = texts[i:i + batch_size]
        if instr:
            batch = [instr + t for t in batch]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        all_embs.append(out["full"].cpu())
    return torch.cat(all_embs, dim=0)   # [N, D], normalized


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def recall_at_k_prefix(q_embs: torch.Tensor, c_embs: torch.Tensor,
                        gt_indices: np.ndarray, k: int,
                        device, chunk: int = 512) -> np.ndarray:
    """Standard prefix-truncation retrieval. Both tensors already normalized."""
    N = len(q_embs)
    hits = np.zeros(N, dtype=float)
    c = c_embs.to(device)
    for i in range(0, N, chunk):
        q = q_embs[i:i + chunk].to(device)
        sim = torch.mm(q, c.t())
        topk = sim.topk(k, dim=-1).indices.cpu().numpy()
        for j, row in enumerate(topk):
            hits[i + j] = float(gt_indices[i + j] in row)
    return hits


def recall_at_k_norm(q_embs: torch.Tensor, c_embs: torch.Tensor,
                      gt_indices: np.ndarray, d: int, k: int,
                      device) -> np.ndarray:
    """
    Norm baseline retrieval: each query selects its own top-d dims by |e_q[i]|,
    then both query and corpus are projected and normalized to those d dims.

    c_embs: [C, D] full-dim corpus (NOT pre-normalized — we normalize per mask)
    q_embs: [N, D] full-dim query  (NOT pre-normalized — we normalize per mask)
    """
    N = len(q_embs)
    hits = np.zeros(N, dtype=float)

    # Pre-sort dim indices per query by descending |e_q[i]|: [N, D]
    sorted_dims = q_embs.abs().argsort(dim=-1, descending=True)   # [N, D]
    top_d_idx = sorted_dims[:, :d]   # [N, d] — top-d dim indices per query

    c_full = c_embs.to(device)   # [C, D]

    for i in range(N):
        idx = top_d_idx[i].to(device)   # [d]

        # Query: project to top-d dims and normalize
        q_proj = q_embs[i].to(device)[idx].unsqueeze(0)   # [1, d]
        q_proj = F.normalize(q_proj, p=2, dim=-1)          # [1, d]

        # Corpus: project to same dims and normalize
        c_proj = c_full[:, idx]                             # [C, d]
        c_proj = F.normalize(c_proj, p=2, dim=-1)          # [C, d]

        sim = torch.mm(q_proj, c_proj.t()).squeeze(0)      # [C]
        topk = sim.topk(k, dim=-1).indices.cpu().numpy()
        hits[i] = float(gt_indices[i] in topk)

    return hits


# ── Per-level aggregation ─────────────────────────────────────────────────────

def bloom_stratified(hits: np.ndarray, bloom_levels: np.ndarray
                     ) -> Dict[str, float]:
    result = {}
    for level in range(1, 7):
        mask = bloom_levels == level
        if mask.sum() == 0:
            continue
        result[BLOOM_NAMES[level]] = float(hits[mask].mean())
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True,
                        help="MRL config YAML (model architecture + data paths)")
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to best/ checkpoint dir (must contain checkpoint.pt)")
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--test_path",   default=None,
                        help="Override data.test_path from config")
    parser.add_argument("--corpus_path", default=None,
                        help="Override data.corpus_path from config")
    parser.add_argument("--k",           type=int,   default=10)
    parser.add_argument("--batch_size",  type=int,   default=128)
    parser.add_argument("--add_30pct",   action="store_true",
                        help="Also evaluate at 30%% of embedding_dim (rounded to nearest MRL dim)")
    parser.add_argument("--dims",        nargs="+",  type=int, default=None,
                        help="Override dim budgets (default: mrl_dims from config)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config  = load_config(args.config)
    mc      = config["model"]
    dc      = config["data"]

    test_path   = args.test_path   or dc["test_path"]
    corpus_path = args.corpus_path or dc["corpus_path"]
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build dim budget list ─────────────────────────────────────────────────
    mrl_dims = args.dims or mc["mrl_dims"]
    if args.add_30pct:
        pct30 = int(round(mc["embedding_dim"] * 0.30))
        if pct30 not in mrl_dims:
            mrl_dims = sorted(set(mrl_dims) | {pct30})

    print(f"Backbone      : {mc['backbone']}")
    print(f"Checkpoint    : {args.checkpoint}")
    print(f"Dim budgets   : {mrl_dims}")
    print(f"Eval metric   : R@{args.k}")
    print(f"Device        : {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt_file = os.path.join(args.checkpoint, "checkpoint.pt")
    if not os.path.exists(ckpt_file):
        print(f"ERROR: checkpoint not found at {ckpt_file}")
        sys.exit(1)

    model = MRLEncoder(
        model_name=mc["backbone"],
        embedding_dim=mc["embedding_dim"],
        mrl_dims=mc["mrl_dims"],
        pooling=mc.get("pooling", "cls"),
        normalize=mc.get("normalize_embeddings", True),
        torch_dtype=mc.get("torch_dtype", None),
        gradient_checkpointing=False,
        backbone_type=mc.get("backbone_type", "standard"),
        query_instruction=mc.get("query_instruction", None),
        peft_model_name=mc.get("peft_model_name", None),
    ).to(device).eval()

    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"  Loaded checkpoint ({ckpt_file})")
    tokenizer = model.get_tokenizer()

    # ── Load corpus + queries ─────────────────────────────────────────────────
    print("\nLoading corpus...")
    corpus = [json.loads(l) for l in open(corpus_path)]
    corpus_id_to_idx = {p["id"]: i for i, p in enumerate(corpus)}
    print(f"  {len(corpus)} passages")

    print("Loading test queries...")
    test_samples = [json.loads(l) for l in open(test_path)]
    valid = [s for s in test_samples
             if s.get("positive_id", "") in corpus_id_to_idx]
    print(f"  {len(valid)} valid queries")

    gt_indices   = np.array([corpus_id_to_idx[s["positive_id"]] for s in valid])
    bloom_levels = np.array([s.get("bloom_level", 0) for s in valid])

    print("\nEncoding corpus (full dim)...")
    # Keep un-normalized copy for Norm baseline (need raw magnitudes per dim)
    model_normalize = model.normalize
    model.normalize = False
    corpus_embs_raw = encode_texts(model, [p["text"] for p in corpus], tokenizer,
                                   device, is_query=False,
                                   batch_size=args.batch_size, max_length=256)
    print("\nEncoding queries (full dim)...")
    query_embs_raw  = encode_texts(model, [s["query"] for s in valid], tokenizer,
                                   device, is_query=True,
                                   batch_size=args.batch_size, max_length=128)
    model.normalize = model_normalize   # restore

    # Normalized versions for Cutoff (prefix truncation with re-normalize per dim)
    corpus_embs_norm = F.normalize(corpus_embs_raw, p=2, dim=-1)   # [C, D]
    query_embs_norm  = F.normalize(query_embs_raw,  p=2, dim=-1)   # [N, D]

    # ── Evaluate ──────────────────────────────────────────────────────────────
    k = args.k
    cutoff_results: Dict[int, Dict] = {}
    norm_results:   Dict[int, Dict] = {}

    header = f"  {'Dims':>6}  {'Cutoff R@'+str(k):>12}  {'Norm R@'+str(k):>12}"
    print(f"\n{'='*50}")
    print(f"  Cutoff vs Norm Baseline  (R@{k})")
    print(f"{'='*50}")
    print(header)
    print("  " + "─" * (len(header) - 2))

    for d in mrl_dims:
        d = min(d, mc["embedding_dim"])

        # ── Cutoff ──
        q_cut = F.normalize(query_embs_norm[:, :d],  p=2, dim=-1)
        c_cut = F.normalize(corpus_embs_norm[:, :d], p=2, dim=-1)
        hits_cut = recall_at_k_prefix(q_cut, c_cut, gt_indices, k, device)
        r_cut = float(hits_cut.mean())
        cutoff_results[d] = {
            f"recall@{k}": r_cut,
            "bloom": bloom_stratified(hits_cut, bloom_levels),
        }

        # ── Norm ──
        hits_norm = recall_at_k_norm(query_embs_raw, corpus_embs_raw,
                                      gt_indices, d, k, device)
        r_norm = float(hits_norm.mean())
        norm_results[d] = {
            f"recall@{k}": r_norm,
            "bloom": bloom_stratified(hits_norm, bloom_levels),
        }

        tag = "  ← 30%" if (d == int(round(mc["embedding_dim"] * 0.30))) else ""
        print(f"  {d:>6}  {r_cut:>12.4f}  {r_norm:>12.4f}{tag}")

    # ── Bloom-stratified table ────────────────────────────────────────────────
    print(f"\n  Bloom-stratified R@{k} at each dim budget")
    print(f"  {'Level':14s}", end="")
    for d in mrl_dims:
        print(f"  {'Cut-'+str(d):>8}  {'Nrm-'+str(d):>8}", end="")
    print()
    print("  " + "─" * (14 + len(mrl_dims) * 20))

    for level in range(1, 7):
        name = BLOOM_NAMES.get(level, str(level))
        print(f"  {name:14s}", end="")
        for d in mrl_dims:
            r_c = cutoff_results[d]["bloom"].get(name, float("nan"))
            r_n = norm_results[d]["bloom"].get(name, float("nan"))
            print(f"  {r_c:>8.4f}  {r_n:>8.4f}", end="")
        print()

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "backbone":       mc["backbone"],
        "checkpoint":     args.checkpoint,
        "embedding_dim":  mc["embedding_dim"],
        "dims_evaluated": mrl_dims,
        "k":              k,
        "num_queries":    len(valid),
        "corpus_size":    len(corpus),
        "cutoff":  {str(d): v for d, v in cutoff_results.items()},
        "norm":    {str(d): v for d, v in norm_results.items()},
    }
    out_path = os.path.join(args.output_dir, "cutoff_norm.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
