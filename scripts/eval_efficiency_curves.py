"""
Efficiency-quality curves: R@10 vs embedding dimensions for MRL and BAM-B.

For each dataset, plots/prints how retrieval quality changes as we vary the
dimension budget. Shows:
  - MRL at dims [64, 128, 256, 384, 512, 768] (or full model dims)
  - BAM-B per-Bloom operating points (one point per level)
  - MRL at BAM-B's per-level budgets (fair comparison baseline)

The key figure for the paper: if BAM-B operating points sit ABOVE the MRL
curve at the same budget, routing is genuinely adding value — not just
benefiting from more dimensions.

Usage:
    python scripts/eval_efficiency_curves.py \
        --config configs/bam.yaml \
        --bam_checkpoint /tmp/multi-domain/educational/bam_b/best_bsr \
        --mrl_checkpoint /tmp/multi-domain/educational/mrl/best \
        --output_dir results/efficiency_curves/educational/

    # All 4 datasets at once
    python scripts/eval_efficiency_curves.py --all_datasets
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
from models.bam import BloomAlignedMRL
from models.encoder import MRLEncoder
from evaluation.evaluator import BLOOM_NAMES, bootstrap_ci
from transformers import AutoTokenizer


# ── helpers ──────────────────────────────────────────────────────────────────

def load_bam(config, ckpt_path, device):
    config["training"]["loss"].setdefault("bloom_frequencies", [1/6]*6)
    model = BloomAlignedMRL(config)
    f = os.path.join(ckpt_path, "checkpoint.pt")
    if os.path.exists(f):
        ckpt = torch.load(f, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return model.to(device).eval()


def load_mrl(config, ckpt_path, device):
    mc = config["model"]
    model = MRLEncoder(model_name=mc["backbone"],
                       embedding_dim=mc["embedding_dim"],
                       mrl_dims=mc["mrl_dims"])
    f = os.path.join(ckpt_path, "checkpoint.pt")
    if os.path.exists(f):
        ckpt = torch.load(f, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return model.to(device).eval()


def encode_all(model, texts, tokenizer, device, batch_size=128,
               is_query=False, bloom_labels=None):
    """Encode texts. Returns [N, D] full embeddings."""
    all_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="encoding", leave=False):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=128 if is_query else 256, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}

            if is_query and hasattr(model, "encode_queries"):
                bls = None
                if bloom_labels is not None:
                    bls = torch.tensor(bloom_labels[i:i+batch_size],
                                       dtype=torch.long, device=device)
                out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                           bloom_labels=bls)
                # Return full_embedding for MRL-style truncation comparisons
                emb = out.get("full_embedding", out["masked_embedding"])
            elif hasattr(model, "encode_documents"):
                out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
                emb = out["masked_embedding"]
            else:
                out = model(enc["input_ids"], enc["attention_mask"])
                emb = out["full"]
            all_embs.append(emb.cpu())
    return torch.cat(all_embs)


def recall_at_k(q_norm, c_norm, gt_indices, k, device, chunk=512):
    hits = []
    for i in range(0, len(q_norm), chunk):
        q = q_norm[i:i+chunk].to(device)
        sim = torch.mm(q, c_norm.to(device).t())
        topk = sim.topk(k, dim=-1).indices.cpu().numpy()
        for j, row in enumerate(topk):
            hits.append(int(gt_indices[i+j] in row))
    return float(np.mean(hits))


def run_for_dataset(config, bam_ckpt, mrl_ckpt, output_dir, device, k=10):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])
    test_path   = config["data"]["test_path"]
    corpus_path = config["data"]["corpus_path"]
    emb_dim     = config["model"]["embedding_dim"]

    # Dim budgets to sweep
    mrl_dims = sorted(set(config["model"]["mrl_dims"] +
                          [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768]))
    mrl_dims = [d for d in mrl_dims if d <= emb_dim]

    # ── Load data ─────────────────────────────────────────────────────────────
    corpus = [json.loads(l) for l in open(corpus_path)]
    corpus_id_to_idx = {p["id"]: i for i, p in enumerate(corpus)}
    samples = [json.loads(l) for l in open(test_path)]
    valid   = [s for s in samples if s.get("positive_id","") in corpus_id_to_idx]
    gt_indices   = np.array([corpus_id_to_idx[s["positive_id"]] for s in valid])
    query_blooms = np.array([s["bloom_level"] for s in valid])
    query_texts  = [s["query"] for s in valid]
    corpus_texts = [p["text"] for p in corpus]
    bloom_labels_0 = [s["bloom_level"]-1 for s in valid]
    print(f"  Corpus: {len(corpus)}  Valid queries: {len(valid)}")

    # ── MRL ───────────────────────────────────────────────────────────────────
    print("\nEncoding with MRL...")
    mrl_model = load_mrl(config, mrl_ckpt, device)
    mrl_q = encode_all(mrl_model, query_texts, tokenizer, device, is_query=False)
    mrl_c = encode_all(mrl_model, corpus_texts, tokenizer, device, is_query=False)
    del mrl_model; torch.cuda.empty_cache() if device.type == "cuda" else None

    mrl_curve = {}
    print("  MRL curve over dims...")
    for d in mrl_dims:
        q_d = F.normalize(mrl_q[:, :d], p=2, dim=-1)
        c_d = F.normalize(mrl_c[:, :d], p=2, dim=-1)
        mrl_curve[d] = recall_at_k(q_d, c_d, gt_indices, k, device)

    # ── BAM-B ─────────────────────────────────────────────────────────────────
    print("\nEncoding with BAM-B...")
    bam_model = load_bam(config, bam_ckpt, device)
    bam_q_full = encode_all(bam_model, query_texts, tokenizer, device,
                            is_query=True, bloom_labels=bloom_labels_0)
    bam_c_full = encode_all(bam_model, corpus_texts, tokenizer, device, is_query=False)

    # Get per-query masks and dims from BAM
    all_masks, all_dims = [], []
    with torch.no_grad():
        for i in range(0, len(valid), 64):
            batch = valid[i:i+64]
            texts = [s["query"] for s in batch]
            bls   = [s["bloom_level"]-1 for s in batch]
            enc = tokenizer(texts, padding=True, truncation=True,
                            max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            bl_t = torch.tensor(bls, dtype=torch.long, device=device)
            out  = bam_model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                            bloom_labels=bl_t)
            if "mask" in out:
                all_masks.append((out["mask"] > 0.5).float().cpu())
            if "discrete_dim" in out:
                all_dims.append(out["discrete_dim"].cpu())
            elif "active_dims" in out:
                all_dims.append(out["active_dims"].cpu())

    is_prefix = len(all_dims) > 0 and "discrete_dim" in out
    masks = torch.cat(all_masks) if all_masks else None
    dims  = torch.cat(all_dims)  if all_dims  else None
    del bam_model; torch.cuda.empty_cache() if device.type == "cuda" else None

    # BAM-B overall R@k (using its own masks/dims)
    bam_overall, bam_per_level = {}, {}
    if is_prefix and dims is not None:
        for b in range(6):
            lmask = query_blooms == (b+1)
            if lmask.sum() == 0: continue
            level_idx = np.where(lmask)[0]
            level_gt  = gt_indices[level_idx]
            avg_d = int(dims[level_idx].float().mean().item())
            hits  = []
            for j, qi in enumerate(level_idx):
                d = max(1, min(int(dims[qi].item()), bam_c_full.shape[1]))
                q_v = F.normalize(bam_q_full[qi:qi+1, :d], p=2, dim=-1).to(device)
                c_v = F.normalize(bam_c_full[:, :d], p=2, dim=-1).to(device)
                sim = torch.mm(q_v, c_v.t())
                topk = sim.topk(k, dim=-1).indices.cpu().numpy()[0]
                hits.append(int(level_gt[j] in topk))
            bam_per_level[BLOOM_NAMES[b+1]] = {
                "avg_dims": avg_d, f"recall@{k}": float(np.mean(hits)),
                "n": int(lmask.sum())
            }
    elif masks is not None:
        for b in range(6):
            lmask = query_blooms == (b+1)
            if lmask.sum() == 0: continue
            level_idx = np.where(lmask)[0]
            level_gt  = gt_indices[level_idx]
            avg_d = int((masks[level_idx] > 0.5).float().sum(dim=-1).mean().item())
            hits  = []
            for j, qi in enumerate(level_idx):
                m = masks[qi].to(device)
                q_v = F.normalize(bam_q_full[qi:qi+1].to(device) * m, p=2, dim=-1)
                c_v = F.normalize(bam_c_full.to(device) * m, p=2, dim=-1)
                sim = torch.mm(q_v, c_v.t())
                topk = sim.topk(k, dim=-1).indices.cpu().numpy()[0]
                hits.append(int(level_gt[j] in topk))
            bam_per_level[BLOOM_NAMES[b+1]] = {
                "avg_dims": avg_d, f"recall@{k}": float(np.mean(hits)),
                "n": int(lmask.sum())
            }

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n  MRL R@{k} vs dims:")
    print(f"  {'Dims':>6}  {'MRL':>8}  {'BAM-B':>8}  {'ΔBAM-MRL':>10}")
    print(f"  {'-'*40}")

    # For each BAM-B per-level point, find MRL R@k at the same budget
    bam_operating_points = []
    for name, lv in bam_per_level.items():
        d     = lv["avg_dims"]
        r_bam = lv[f"recall@{k}"]
        # Interpolate MRL curve at this dim
        lower_d = max((dd for dd in mrl_dims if dd <= d), default=mrl_dims[0])
        r_mrl = mrl_curve.get(lower_d, mrl_curve[mrl_dims[0]])
        bam_operating_points.append((d, r_bam, r_mrl, name))

    # Print MRL curve
    for d in mrl_dims:
        r = mrl_curve[d]
        print(f"  {d:>6}  {r:>8.4f}")

    # Print BAM-B operating points
    print(f"\n  BAM-B per-Bloom operating points (k={k}):")
    print(f"  {'Level':12s}  {'Dims':>6}  {'N':>5}  {'BAM-B':>8}  {'MRL@same':>10}  {'Δ':>8}")
    print(f"  {'-'*54}")
    for d, r_bam, r_mrl, name in sorted(bam_operating_points, key=lambda x: x[0]):
        delta = r_bam - r_mrl
        sign  = "+" if delta >= 0 else ""
        n = bam_per_level[name]["n"]
        print(f"  {name:12s}  {d:>6}  {n:>5}  {r_bam:>8.4f}  {r_mrl:>10.4f}  {sign}{delta*100:.1f}%")

    # ── Save ─────────────────────────────────────────────────────────────────
    out = {
        "mrl_curve":          {str(d): v for d, v in mrl_curve.items()},
        "bam_per_level":      bam_per_level,
        "bam_operating_pts":  bam_operating_points,
        "k": k,
    }
    path = os.path.join(output_dir, "efficiency_curves.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved → {path}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          default=None)
    parser.add_argument("--bam_checkpoint",  default=None)
    parser.add_argument("--mrl_checkpoint",  default=None)
    parser.add_argument("--output_dir",      default="results/efficiency_curves/")
    parser.add_argument("--k",               type=int, default=10)
    parser.add_argument("--all_datasets",    action="store_true",
                        help="Run for all 4 multi-domain datasets")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.all_datasets:
        datasets = ["educational", "scifact", "nfcorpus", "fiqa"]
        ckpt_root   = "/tmp/multi-domain"
        results_root = "./results/multi_domain"

        summary = {}
        for ds in datasets:
            print(f"\n{'='*60}")
            print(f"  Dataset: {ds}")
            print(f"{'='*60}")
            bam_ckpt = f"{ckpt_root}/{ds}/bam_b/best_bsr"
            mrl_ckpt = f"{ckpt_root}/{ds}/mrl/best"
            cfg_path = f"{results_root}/{ds}/configs/bam_b.yaml"
            out_dir  = f"{results_root}/{ds}/efficiency_curves"

            if not all(os.path.exists(p) for p in [bam_ckpt, mrl_ckpt, cfg_path]):
                print(f"  Skipping {ds} — missing checkpoints or config.")
                continue

            config = load_config(cfg_path)
            set_seed(config["training"]["seed"])
            result = run_for_dataset(config, bam_ckpt, mrl_ckpt, out_dir, device, args.k)
            summary[ds] = result

        # Cross-dataset summary: BAM-B average dim vs MRL@same
        print(f"\n{'='*70}")
        print("  CROSS-DATASET EFFICIENCY SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Dataset':14s}  {'BAM avg dims':>14s}  {'BAM R@10':>10s}  {'MRL@same':>10s}  {'Δ':>8s}")
        print(f"  {'-'*60}")
        for ds, res in summary.items():
            if not res["bam_operating_pts"]:
                continue
            pts = res["bam_operating_pts"]
            # Weight average by N (number of queries per level)
            total_n = sum(res["bam_per_level"][name]["n"] for _, _, _, name in pts)
            avg_dim = sum(d * res["bam_per_level"][name]["n"]
                          for d, _, _, name in pts) / max(total_n, 1)
            avg_bam = sum(r_bam * res["bam_per_level"][name]["n"]
                          for _, r_bam, _, name in pts) / max(total_n, 1)
            avg_mrl = sum(r_mrl * res["bam_per_level"][name]["n"]
                          for _, _, r_mrl, name in pts) / max(total_n, 1)
            delta = avg_bam - avg_mrl
            sign  = "+" if delta >= 0 else ""
            print(f"  {ds:14s}  {avg_dim:>14.0f}  {avg_bam:>10.4f}  {avg_mrl:>10.4f}  {sign}{delta*100:.1f}%")

    else:
        if not all([args.config, args.bam_checkpoint, args.mrl_checkpoint]):
            print("Provide --config, --bam_checkpoint, --mrl_checkpoint "
                  "or use --all_datasets")
            return
        config = load_config(args.config)
        set_seed(config["training"]["seed"])
        run_for_dataset(config, args.bam_checkpoint, args.mrl_checkpoint,
                        args.output_dir, device, args.k)


if __name__ == "__main__":
    main()
