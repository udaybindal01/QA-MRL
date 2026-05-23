"""
Zero-shot BEIR evaluation with PER-QUERY matched dimension budget.

For each query Q, BAM-PQ uses N_Q active dims (varies per query).
This script computes:
  - BAM-PQ retrieval: normalize(bam_q_masked) . normalize(bam_corpus_full)
  - MRL @ matched (per-query) retrieval: for each query, truncate MRL to N_Q
    prefix dims and retrieve at that budget
  - MRL @ full retrieval (reference): for context

Reports R@10 and R@50 for each.

This is the fairest comparison: per-query identical dim budget. No averaging
or interpolation. BAM-PQ wins by routing if and only if scattered dim
selection beats prefix selection at the same per-query budget.

Usage:
  python3 scripts/eval_zero_shot_matched.py \
      --mrl_config        configs/mrl_e5large.yaml \
      --mrl_checkpoint    /scratch/.../educational/mrl_e5large/best \
      --bam_pq_config     configs/bam_pq.yaml \
      --bam_pq_checkpoint /scratch/.../educational/bam_pq_e5large/best_bsr \
      --datasets scifact nfcorpus fiqa \
      --beir_data_root    /scratch/.../bampq-data/beir \
      --output_dir        results/zero_shot_matched/e5large/
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed                              # noqa: E402
from models.bam import BloomAlignedMRL                                     # noqa: E402
from models.encoder import MRLEncoder                                      # noqa: E402


def _load_state(model, ckpt_path, device, label):
    f = os.path.join(ckpt_path, "checkpoint.pt")
    if not os.path.exists(f):
        raise FileNotFoundError(f"{label} checkpoint not found: {f}")
    ckpt = torch.load(f, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"  Loaded {label} from {f}")


def load_bam(config, ckpt_path, device):
    config["training"]["loss"].setdefault("bloom_frequencies", [1 / 6] * 6)
    model = BloomAlignedMRL(config)
    _load_state(model, ckpt_path, device, "BAM-PQ")
    return model.to(device).eval()


def load_mrl(config, ckpt_path, device):
    mc = config["model"]
    model = MRLEncoder(
        model_name=mc["backbone"],
        embedding_dim=mc["embedding_dim"],
        mrl_dims=mc["mrl_dims"],
        pooling=mc.get("pooling", "cls"),
        backbone_type=mc.get("backbone_type", "standard"),
        query_instruction=mc.get("query_instruction", None),
        peft_model_name=mc.get("peft_model_name", None),
    )
    _load_state(model, ckpt_path, device, "MRL")
    return model.to(device).eval()


@torch.no_grad()
def encode_corpus(model, texts, tokenizer, device, batch_size=128, is_bam=False):
    """Encode a corpus, returning full (unmasked) doc embeddings."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        if is_bam:
            out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
            all_embs.append(out["masked_embedding"].cpu())   # full doc emb
        else:
            out = model(enc["input_ids"], enc["attention_mask"])
            all_embs.append(out["full"].cpu())
    return torch.cat(all_embs)


@torch.no_grad()
def encode_queries_bam(model, samples, tokenizer, device, batch_size=64):
    """Encode queries with BAM-PQ. Returns masked_q [N,D] and per-query active_dims [N]."""
    all_q = []
    all_active = []
    instr = getattr(model.encoder, "query_instruction", None)
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        texts = [s["query"] for s in batch]
        if instr:
            texts = [instr + t for t in texts]
        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        bloom_labels = torch.tensor(
            [s["bloom_level"] - 1 for s in batch],
            dtype=torch.long, device=device,
        )
        out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                   bloom_labels=bloom_labels)
        all_q.append(out["masked_embedding"].cpu())
        # Hard active dim count per query
        hard = (out["mask"] > 0.5).float()
        all_active.append(hard.sum(dim=-1).cpu())
    return torch.cat(all_q), torch.cat(all_active).long()


@torch.no_grad()
def encode_queries_mrl(model, samples, tokenizer, device, batch_size=64):
    """Encode queries with MRL. Returns full query embeddings."""
    all_q = []
    instr = getattr(model, "query_instruction", None)
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        texts = [s["query"] for s in batch]
        if instr:
            texts = [instr + t for t in texts]
        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(enc["input_ids"], enc["attention_mask"])
        all_q.append(out["full"].cpu())
    return torch.cat(all_q)


def retrieve_bam(q_masked, c_full, k, device, chunk=256):
    """BAM-PQ retrieval: masked_q . full_c."""
    N = len(q_masked)
    rankings = np.zeros((N, k), dtype=np.int64)
    c_dev = c_full.float().to(device)
    c_norm = F.normalize(c_dev, p=2, dim=-1)
    for i in range(0, N, chunk):
        q = q_masked[i:i + chunk].float().to(device)
        sim = torch.mm(q, c_norm.t())
        rankings[i:i + chunk] = sim.topk(k, dim=-1).indices.cpu().numpy()
        del sim
    del c_dev, c_norm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rankings


def retrieve_mrl_full(q_full, c_full, k, device, chunk=256):
    """MRL retrieval at full dim."""
    N = len(q_full)
    rankings = np.zeros((N, k), dtype=np.int64)
    c_norm = F.normalize(c_full.float().to(device), p=2, dim=-1)
    for i in range(0, N, chunk):
        q = F.normalize(q_full[i:i + chunk].float().to(device), p=2, dim=-1)
        sim = torch.mm(q, c_norm.t())
        rankings[i:i + chunk] = sim.topk(k, dim=-1).indices.cpu().numpy()
        del sim
    del c_norm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rankings


def retrieve_mrl_matched(q_full, c_full, active_dims, k, device):
    """
    Per-query matched MRL retrieval:
    for each query Q with active_dims[Q]=d, truncate to d prefix dims.

    Groups queries by dim count to amortise corpus-normalisation cost.
    """
    N = len(q_full)
    D = c_full.shape[1]
    rankings = np.zeros((N, k), dtype=np.int64)
    c_dev_full = c_full.float().to(device)
    q_dev_full = q_full.float().to(device)

    # Group queries by their dim budget
    groups = defaultdict(list)
    for i, d in enumerate(active_dims.tolist()):
        d_eff = max(1, min(int(d), D))
        groups[d_eff].append(i)

    for d, q_indices in groups.items():
        c_d = F.normalize(c_dev_full[:, :d], p=2, dim=-1)
        q_d = F.normalize(q_dev_full[q_indices, :d], p=2, dim=-1)
        sim = torch.mm(q_d, c_d.t())
        topk = sim.topk(k, dim=-1).indices.cpu().numpy()
        for j, qi in enumerate(q_indices):
            rankings[qi] = topk[j]
        del c_d, q_d, sim
    del c_dev_full, q_dev_full
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rankings


def recall_at_k(rankings, gt, k):
    """rankings: [N, K] top-K indices. gt: [N] positive doc indices."""
    hits = []
    for i, gt_i in enumerate(gt):
        hits.append(int(gt_i in rankings[i, :k]))
    return float(np.mean(hits))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mrl_config",        required=True)
    parser.add_argument("--mrl_checkpoint",    required=True)
    parser.add_argument("--bam_pq_config",     required=True)
    parser.add_argument("--bam_pq_checkpoint", required=True)
    parser.add_argument("--datasets",          nargs="+",
                        default=["scifact", "nfcorpus", "fiqa"])
    parser.add_argument("--beir_data_root",    default="/tmp/data/beir")
    parser.add_argument("--output_dir",        default="results/zero_shot_matched/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    bam_config = load_config(args.bam_pq_config)
    mrl_config = load_config(args.mrl_config)
    set_seed(bam_config["training"]["seed"])
    tokenizer = AutoTokenizer.from_pretrained(bam_config["model"]["backbone"])

    print(f"Backbone: {bam_config['model']['backbone']}")
    print(f"Datasets: {args.datasets}\n")

    print("Loading models...")
    bam_model = load_bam(bam_config, args.bam_pq_checkpoint, device)
    mrl_model = load_mrl(mrl_config, args.mrl_checkpoint, device)

    all_results = {}
    for ds in args.datasets:
        test_path   = os.path.join(args.beir_data_root, ds, "test.jsonl")
        corpus_path = os.path.join(args.beir_data_root, ds, "corpus.jsonl")
        if not (os.path.exists(test_path) and os.path.exists(corpus_path)):
            print(f"  SKIP {ds}: data not found at {args.beir_data_root}/{ds}/")
            continue

        print(f"\n=== Zero-shot matched: {ds} ===")

        # Load data
        corpus = [json.loads(l) for l in open(corpus_path)]
        corpus_id_to_idx = {p["id"]: i for i, p in enumerate(corpus)}
        samples_all = [json.loads(l) for l in open(test_path)]
        samples = [s for s in samples_all if s.get("positive_id", "") in corpus_id_to_idx]
        print(f"  Corpus: {len(corpus)}, test queries: {len(samples)}")

        gt = np.array([corpus_id_to_idx[s["positive_id"]] for s in samples])
        corpus_texts = [p["text"] for p in corpus]

        # Encode corpora
        print("  Encoding BAM-PQ corpus ...")
        bam_corpus = encode_corpus(bam_model, corpus_texts, tokenizer, device,
                                    batch_size=128, is_bam=True)
        print("  Encoding MRL corpus ...")
        mrl_corpus = encode_corpus(mrl_model, corpus_texts, tokenizer, device,
                                    batch_size=128, is_bam=False)

        # Encode queries
        print("  Encoding BAM-PQ queries ...")
        bam_q_masked, active_dims = encode_queries_bam(bam_model, samples, tokenizer, device)
        print(f"    avg active dims: {active_dims.float().mean().item():.1f}  "
              f"(min={active_dims.min().item()}, max={active_dims.max().item()})")
        print("  Encoding MRL queries ...")
        mrl_q_full = encode_queries_mrl(mrl_model, samples, tokenizer, device)

        # Retrieve (top-50)
        K = 50
        print("  BAM-PQ retrieval ...")
        bam_top = retrieve_bam(bam_q_masked, bam_corpus, K, device)
        print("  MRL-full retrieval ...")
        mrl_full_top = retrieve_mrl_full(mrl_q_full, mrl_corpus, K, device)
        print("  MRL-matched (per-query) retrieval ...")
        mrl_matched_top = retrieve_mrl_matched(mrl_q_full, mrl_corpus,
                                                active_dims, K, device)

        # Metrics
        bam_r10  = recall_at_k(bam_top,         gt, 10)
        bam_r50  = recall_at_k(bam_top,         gt, 50)
        mrl_f10  = recall_at_k(mrl_full_top,    gt, 10)
        mrl_f50  = recall_at_k(mrl_full_top,    gt, 50)
        mrl_m10  = recall_at_k(mrl_matched_top, gt, 10)
        mrl_m50  = recall_at_k(mrl_matched_top, gt, 50)
        avg_dims = float(active_dims.float().mean().item())

        all_results[ds] = {
            "n_queries":            len(samples),
            "avg_active_dims":      avg_dims,
            "MRL_full_R@10":        mrl_f10,
            "MRL_full_R@50":        mrl_f50,
            "MRL_matched_R@10":     mrl_m10,
            "MRL_matched_R@50":     mrl_m50,
            "BAM_PQ_R@10":          bam_r10,
            "BAM_PQ_R@50":          bam_r50,
            "delta_R@10_vs_matched": bam_r10 - mrl_m10,
            "delta_R@50_vs_matched": bam_r50 - mrl_m50,
        }

        print(f"  MRL @ full ({mrl_corpus.shape[1]} dims): R@10={mrl_f10:.4f}  R@50={mrl_f50:.4f}")
        print(f"  MRL @ matched (~{avg_dims:.0f}):          R@10={mrl_m10:.4f}  R@50={mrl_m50:.4f}")
        print(f"  BAM-PQ (~{avg_dims:.0f}):                 R@10={bam_r10:.4f}  R@50={bam_r50:.4f}")
        print(f"  Delta vs matched:                         R@10={bam_r10 - mrl_m10:+.4f}  "
              f"R@50={bam_r50 - mrl_m50:+.4f}")

        # Free memory
        del bam_corpus, mrl_corpus, bam_q_masked, mrl_q_full
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save
    out_path = os.path.join(args.output_dir, "matched_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Summary table ───────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("ZERO-SHOT BEIR — PER-QUERY MATCHED BUDGET")
    print("=" * 90)
    hdr = (f"{'Dataset':<12} {'avg dims':>9} "
           f"{'MRL@full R@10':>14} {'MRL@match R@10':>15} {'BAM-PQ R@10':>12} {'Δ vs match':>11}")
    print(hdr)
    print("-" * len(hdr))
    for ds, r in all_results.items():
        d = f"+{r['delta_R@10_vs_matched']:.4f}" if r['delta_R@10_vs_matched'] >= 0 \
            else f"{r['delta_R@10_vs_matched']:.4f}"
        print(f"{ds:<12} {r['avg_active_dims']:>9.1f} "
              f"{r['MRL_full_R@10']:>14.4f} {r['MRL_matched_R@10']:>15.4f} "
              f"{r['BAM_PQ_R@10']:>12.4f} {d:>11}")

    # LaTeX
    latex_path = os.path.join(args.output_dir, "matched_table.tex")
    with open(latex_path, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\\small\n")
        f.write("\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n")
        f.write("Dataset & Avg dims & MRL@full R@10 & MRL@matched R@10 "
                "& BAM-PQ R@10 & $\\Delta$ R@10 \\\\\n\\midrule\n")
        for ds, r in all_results.items():
            delta = r['delta_R@10_vs_matched']
            delta_s = (f"$+{delta:.4f}$" if delta >= 0 else f"${delta:.4f}$")
            bam_s = f"\\textbf{{{r['BAM_PQ_R@10']:.4f}}}" if delta >= 0 \
                    else f"{r['BAM_PQ_R@10']:.4f}"
            f.write(f"{ds} & {r['avg_active_dims']:.0f} & "
                    f"{r['MRL_full_R@10']:.4f} & {r['MRL_matched_R@10']:.4f} & "
                    f"{bam_s} & {delta_s} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Zero-shot BEIR at per-query matched dimension budgets. "
                "For each query, MRL is truncated to the same number of prefix "
                "dimensions BAM-PQ uses for that query. Bold $=$ BAM-PQ wins at "
                "matched budget.}\n")
        f.write("\\label{tab:zero-shot-matched}\n")
        f.write("\\end{table}\n")
    print(f"LaTeX table -> {latex_path}")


if __name__ == "__main__":
    main()
