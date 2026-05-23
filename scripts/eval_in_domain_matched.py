"""
In-domain BEIR evaluation with per-query matched dimension budget.

Mirror of eval_zero_shot_matched.py, but loads PER-DATASET trained
checkpoints (not the educational-trained checkpoints used in zero-shot).

For each (backbone, dataset):
  - Loads $CKPT_ROOT/<dataset>/mrl_<backbone>/best/checkpoint.pt
  - Loads $CKPT_ROOT/<dataset>/bam_pq_<backbone>/best_bsr/checkpoint.pt
  - Loads the pipeline-generated configs at
    $RESULTS_ROOT/<dataset>/configs/{mrl,bam_pq}_<backbone>.yaml
  - Computes R@10, R@50 for:
      * MRL @ full dim
      * MRL @ matched (per-query, truncated to BAM-PQ's per-query dim count)
      * BAM-PQ (routed)

This verifies the pipeline's in-domain numbers using the same retrieval
methodology as eval_zero_shot_matched.py (masked-query vs full-corpus,
fp32 cast, per-query exact MRL truncation).

Usage:
  python3 scripts/eval_in_domain_matched.py \
      --backbone     e5large \
      --ckpt_root    /scratch/.../bampq-checkpoints \
      --results_root results/multi_domain \
      --beir_root    /scratch/.../bampq-data/beir \
      --datasets scifact nfcorpus \
      --output_dir   results/in_domain_matched/e5large/
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
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        if is_bam:
            out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
            all_embs.append(out["masked_embedding"].cpu())
        else:
            out = model(enc["input_ids"], enc["attention_mask"])
            all_embs.append(out["full"].cpu())
    return torch.cat(all_embs)


@torch.no_grad()
def encode_queries_bam(model, samples, tokenizer, device, batch_size=64):
    all_q, all_active = [], []
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
        hard = (out["mask"] > 0.5).float()
        all_active.append(hard.sum(dim=-1).cpu())
    return torch.cat(all_q), torch.cat(all_active).long()


@torch.no_grad()
def encode_queries_mrl(model, samples, tokenizer, device, batch_size=64):
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
    N = len(q_masked)
    rankings = np.zeros((N, k), dtype=np.int64)
    c_norm = F.normalize(c_full.float().to(device), p=2, dim=-1)
    for i in range(0, N, chunk):
        q = q_masked[i:i + chunk].float().to(device)
        sim = torch.mm(q, c_norm.t())
        rankings[i:i + chunk] = sim.topk(k, dim=-1).indices.cpu().numpy()
        del sim
    del c_norm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rankings


def retrieve_mrl_full(q_full, c_full, k, device, chunk=256):
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
    """Per-query MRL prefix truncated to active_dims[i] dims."""
    N = len(q_full)
    D = c_full.shape[1]
    rankings = np.zeros((N, k), dtype=np.int64)
    c_dev = c_full.float().to(device)
    q_dev = q_full.float().to(device)
    groups = defaultdict(list)
    for i, d in enumerate(active_dims.tolist()):
        groups[max(1, min(int(d), D))].append(i)
    for d, q_indices in groups.items():
        c_d = F.normalize(c_dev[:, :d], p=2, dim=-1)
        q_d = F.normalize(q_dev[q_indices, :d], p=2, dim=-1)
        sim = torch.mm(q_d, c_d.t())
        topk = sim.topk(k, dim=-1).indices.cpu().numpy()
        for j, qi in enumerate(q_indices):
            rankings[qi] = topk[j]
        del c_d, q_d, sim
    del c_dev, q_dev
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rankings


def recall_at_k(rankings, gt, k):
    return float(np.mean([int(gt[i] in rankings[i, :k]) for i in range(len(gt))]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",     required=True,
                        help="Pipeline backbone key (e5large, bge, bge_base, arctic, mxbai, ...)")
    parser.add_argument("--ckpt_root",    required=True,
                        help="$CKPT_ROOT — contains <dataset>/{mrl_<bk>,bam_pq_<bk>} subdirs")
    parser.add_argument("--results_root", required=True,
                        help="Pipeline results root — contains <dataset>/configs/")
    parser.add_argument("--beir_root",    required=True,
                        help="BEIR data root — contains <dataset>/{test.jsonl,corpus.jsonl}")
    parser.add_argument("--datasets",     nargs="+",
                        default=["scifact", "nfcorpus"])
    parser.add_argument("--output_dir",   default="results/in_domain_matched/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    for ds in args.datasets:
        mrl_ckpt = os.path.join(args.ckpt_root, ds, f"mrl_{args.backbone}", "best")
        bam_ckpt = os.path.join(args.ckpt_root, ds, f"bam_pq_{args.backbone}", "best_bsr")
        mrl_cfg_p = os.path.join(args.results_root, ds, "configs",
                                  f"mrl_{args.backbone}.yaml")
        bam_cfg_p = os.path.join(args.results_root, ds, "configs",
                                  f"bam_pq_{args.backbone}.yaml")
        test_path   = os.path.join(args.beir_root, ds, "test.jsonl")
        corpus_path = os.path.join(args.beir_root, ds, "corpus.jsonl")

        missing = [p for p in [os.path.join(mrl_ckpt, "checkpoint.pt"),
                                os.path.join(bam_ckpt, "checkpoint.pt"),
                                mrl_cfg_p, bam_cfg_p,
                                test_path, corpus_path]
                   if not os.path.exists(p)]
        if missing:
            print(f"\n=== {ds} === SKIP — missing:")
            for p in missing:
                print(f"    {p}")
            continue

        print(f"\n=== In-domain matched: {args.backbone} on {ds} ===")
        bam_config = load_config(bam_cfg_p)
        mrl_config = load_config(mrl_cfg_p)
        set_seed(bam_config["training"]["seed"])
        tokenizer = AutoTokenizer.from_pretrained(bam_config["model"]["backbone"])

        bam_model = load_bam(bam_config, bam_ckpt, device)
        mrl_model = load_mrl(mrl_config, mrl_ckpt, device)

        # Load data
        corpus = [json.loads(l) for l in open(corpus_path)]
        corpus_id_to_idx = {p["id"]: i for i, p in enumerate(corpus)}
        samples_all = [json.loads(l) for l in open(test_path)]
        samples = [s for s in samples_all if s.get("positive_id", "") in corpus_id_to_idx]
        print(f"  Corpus: {len(corpus)}, test queries: {len(samples)}")

        gt = np.array([corpus_id_to_idx[s["positive_id"]] for s in samples])
        corpus_texts = [p["text"] for p in corpus]

        # Encode
        print("  Encoding BAM-PQ corpus ...")
        bam_corpus = encode_corpus(bam_model, corpus_texts, tokenizer, device,
                                    batch_size=128, is_bam=True)
        print("  Encoding MRL corpus ...")
        mrl_corpus = encode_corpus(mrl_model, corpus_texts, tokenizer, device,
                                    batch_size=128, is_bam=False)
        print("  Encoding BAM-PQ queries ...")
        bam_q_masked, active_dims = encode_queries_bam(bam_model, samples,
                                                       tokenizer, device)
        avg_d = active_dims.float().mean().item()
        print(f"    avg active dims: {avg_d:.1f}  "
              f"(min={active_dims.min().item()}, max={active_dims.max().item()})")
        print("  Encoding MRL queries ...")
        mrl_q_full = encode_queries_mrl(mrl_model, samples, tokenizer, device)

        K = 50
        print("  BAM-PQ retrieval ...")
        bam_top = retrieve_bam(bam_q_masked, bam_corpus, K, device)
        print("  MRL-full retrieval ...")
        mrl_full_top = retrieve_mrl_full(mrl_q_full, mrl_corpus, K, device)
        print("  MRL-matched (per-query) retrieval ...")
        mrl_matched_top = retrieve_mrl_matched(mrl_q_full, mrl_corpus,
                                                active_dims, K, device)

        r = {
            "n_queries":          len(samples),
            "avg_active_dims":    float(avg_d),
            "MRL_full_R@10":      recall_at_k(mrl_full_top, gt, 10),
            "MRL_full_R@50":      recall_at_k(mrl_full_top, gt, 50),
            "MRL_matched_R@10":   recall_at_k(mrl_matched_top, gt, 10),
            "MRL_matched_R@50":   recall_at_k(mrl_matched_top, gt, 50),
            "BAM_PQ_R@10":        recall_at_k(bam_top, gt, 10),
            "BAM_PQ_R@50":        recall_at_k(bam_top, gt, 50),
        }
        r["delta_R@10_vs_matched"] = r["BAM_PQ_R@10"] - r["MRL_matched_R@10"]
        r["delta_R@50_vs_matched"] = r["BAM_PQ_R@50"] - r["MRL_matched_R@50"]
        r["delta_R@10_vs_full"]    = r["BAM_PQ_R@10"] - r["MRL_full_R@10"]
        all_results[ds] = r

        print(f"  MRL @ full ({mrl_corpus.shape[1]} dims):  "
              f"R@10={r['MRL_full_R@10']:.4f}  R@50={r['MRL_full_R@50']:.4f}")
        print(f"  MRL @ matched (~{avg_d:.0f}):           "
              f"R@10={r['MRL_matched_R@10']:.4f}  R@50={r['MRL_matched_R@50']:.4f}")
        print(f"  BAM-PQ (~{avg_d:.0f}):                  "
              f"R@10={r['BAM_PQ_R@10']:.4f}  R@50={r['BAM_PQ_R@50']:.4f}")
        print(f"  Delta vs matched:                       "
              f"R@10={r['delta_R@10_vs_matched']:+.4f}  "
              f"R@50={r['delta_R@50_vs_matched']:+.4f}")

        # Cleanup before next dataset
        del bam_model, mrl_model, bam_corpus, mrl_corpus, bam_q_masked, mrl_q_full
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save
    out_path = os.path.join(args.output_dir, f"{args.backbone}_in_domain.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print("\n" + "=" * 90)
    print(f"IN-DOMAIN MATCHED — {args.backbone}")
    print("=" * 90)
    hdr = (f"{'Dataset':<12} {'avg dims':>9} "
           f"{'MRL@full R@10':>14} {'MRL@match R@10':>15} "
           f"{'BAM-PQ R@10':>12} {'Δ vs match':>11}")
    print(hdr)
    print("-" * len(hdr))
    for ds, r in all_results.items():
        d = (f"+{r['delta_R@10_vs_matched']:.4f}"
             if r['delta_R@10_vs_matched'] >= 0
             else f"{r['delta_R@10_vs_matched']:.4f}")
        print(f"{ds:<12} {r['avg_active_dims']:>9.1f} "
              f"{r['MRL_full_R@10']:>14.4f} {r['MRL_matched_R@10']:>15.4f} "
              f"{r['BAM_PQ_R@10']:>12.4f} {d:>11}")


if __name__ == "__main__":
    main()
