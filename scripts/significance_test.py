"""
Significance testing: BAM-PQ vs MRL baseline on per-query R@10.

Tests used:
  - Wilcoxon signed-rank test (one-sided, greater)
  - McNemar's test (exact binomial on discordant pairs)

Usage:
    python scripts/significance_test.py
    python scripts/significance_test.py --backbones arctic bge
    python scripts/significance_test.py --output_dir results/significance/
"""

import argparse, json, os, sys
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import wilcoxon, binom

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.misc import load_config, set_seed
from models.bam import BloomAlignedMRL
from models.encoder import MRLEncoder
from transformers import AutoTokenizer


BACKBONE_MAP = {
    "e5large": {
        "bam_config": "configs/bam_pq.yaml",
        "bam_ckpt":   "/tmp/uday/multi-domain/educational/bam_pq_e5large/best_bsr",
        "mrl_config": "configs/mrl_e5large.yaml",
        "mrl_ckpt":   "/tmp/uday/multi-domain/educational/mrl_e5large/best",
    },
    "bge": {
        "bam_config": "configs/bam_pq_bge_large.yaml",
        "bam_ckpt":   "/tmp/uday/multi-domain/educational/bam_pq_bge_large/best_bsr",
        "mrl_config": "configs/mrl_bge_large.yaml",
        "mrl_ckpt":   "/tmp/uday/multi-domain/educational/mrl_bge/best",
    },
    "arctic": {
        "bam_config": "configs/bam_pq_arctic.yaml",
        "bam_ckpt":   "/tmp/uday/multi-domain/educational/bam_pq_arctic/best_bsr",
        "mrl_config": "configs/mrl_arctic.yaml",
        "mrl_ckpt":   "/tmp/uday/multi-domain/educational/mrl_arctic/best",
    },
    "roberta": {
        "bam_config": "configs/bam_pq_roberta.yaml",
        "bam_ckpt":   "/tmp/uday/multi-domain/educational/bam_pq_roberta/best_bsr",
        "mrl_config": "configs/mrl_roberta.yaml",
        "mrl_ckpt":   "/tmp/uday/multi-domain/educational/mrl_roberta/best",
    },
    "qwen06b": {
        "bam_config": "configs/bam_pq_qwen06b.yaml",
        "bam_ckpt":   "/tmp/uday/multi-domain/educational/bam_pq_qwen06b/best_bsr",
        "mrl_config": "configs/mrl_qwen06b.yaml",
        "mrl_ckpt":   "/tmp/uday/multi-domain/educational/mrl_qwen06b/best",
    },
}


def load_data(config, split="test"):
    data_cfg = config["data"]
    corpus, queries = [], []
    with open(data_cfg["corpus_path"]) as f:
        for line in f:
            corpus.append(json.loads(line))
    if split == "test":
        query_path = data_cfg.get("test_path") or data_cfg.get("val_path")
    else:
        query_path = data_cfg.get("val_path") or data_cfg.get("test_path")
    with open(query_path) as f:
        for line in f:
            queries.append(json.loads(line))
    corpus_id_to_idx = {c["id"]: i for i, c in enumerate(corpus)}
    return corpus, queries, corpus_id_to_idx


@torch.no_grad()
def encode_corpus_mrl(model, tokenizer, corpus, device, batch_size=128):
    embs = []
    for i in tqdm(range(0, len(corpus), batch_size), desc="  MRL corpus", leave=False):
        batch = [c["text"] for c in corpus[i:i+batch_size]]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(enc["input_ids"], enc["attention_mask"])
        embs.append(out["full"].float().cpu())
    return torch.cat(embs)


@torch.no_grad()
def encode_queries_mrl(model, tokenizer, queries, device, batch_size=64):
    embs = []
    for i in tqdm(range(0, len(queries), batch_size), desc="  MRL queries", leave=False):
        batch = [q["query"] for q in queries[i:i+batch_size]]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(enc["input_ids"], enc["attention_mask"])
        embs.append(out["full"].float().cpu())
    return torch.cat(embs)


@torch.no_grad()
def encode_corpus_bam(model, tokenizer, corpus, device, batch_size=128):
    embs = []
    for i in tqdm(range(0, len(corpus), batch_size), desc="  BAM corpus", leave=False):
        batch = [c["text"] for c in corpus[i:i+batch_size]]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
        embs.append(out["full_embedding"].float().cpu())
    return torch.cat(embs)


@torch.no_grad()
def encode_queries_bam(model, tokenizer, queries, device, batch_size=64):
    embs = []
    for i in tqdm(range(0, len(queries), batch_size), desc="  BAM queries", leave=False):
        batch_q = queries[i:i+batch_size]
        enc = tokenizer([q["query"] for q in batch_q], padding=True,
                        truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        bloom_labels = torch.tensor(
            [q["bloom_level"] - 1 for q in batch_q], dtype=torch.long, device=device
        )
        out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                   bloom_labels=bloom_labels)
        embs.append(out["masked_embedding"].float().cpu())
    return torch.cat(embs)


def per_query_hits(query_embs, corpus_embs, gt_indices, k=10, device="cpu"):
    query_embs  = query_embs.float()
    corpus_embs = corpus_embs.float()
    hits = []
    for i in range(0, len(query_embs), 256):
        sim  = torch.mm(query_embs[i:i+256].to(device), corpus_embs.t().to(device))
        topk = sim.topk(k, dim=-1).indices.cpu().numpy()
        for j, gt in enumerate(gt_indices[i:i+256]):
            hits.append(int(gt in topk[j]))
    return np.array(hits)


def run_tests(hits_bam, hits_mrl, backbone):
    diff    = hits_bam.astype(float) - hits_mrl.astype(float)
    r10_bam = hits_bam.mean()
    r10_mrl = hits_mrl.mean()
    delta   = r10_bam - r10_mrl

    # Wilcoxon signed-rank (one-sided: BAM > MRL)
    if diff.sum() == 0:
        wilcox_stat, wilcox_p = float("nan"), 1.0
    else:
        wilcox_stat, wilcox_p = wilcoxon(diff, zero_method="wilcox", alternative="greater")

    # McNemar's test (one-sided: BAM wins more than MRL wins)
    b = int(((hits_mrl == 1) & (hits_bam == 0)).sum())  # MRL wins, BAM misses
    c = int(((hits_bam == 1) & (hits_mrl == 0)).sum())  # BAM wins, MRL misses
    n_disc = b + c
    if n_disc == 0:
        mcn_p = 1.0
    elif n_disc < 25:
        mcn_p = float(binom.sf(c - 1, n_disc, 0.5))
    else:
        from scipy.stats import chi2
        stat  = (abs(c - b) - 1) ** 2 / (b + c)
        mcn_p = float(chi2.sf(stat, df=1) / 2)

    def sig(p):
        return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

    return {
        "backbone":      backbone,
        "n_queries":     len(hits_bam),
        "r10_bam":       round(float(r10_bam), 4),
        "r10_mrl":       round(float(r10_mrl), 4),
        "delta_r10":     round(float(delta), 4),
        "bam_wins":      c,
        "mrl_wins":      b,
        "wilcoxon_stat": round(wilcox_stat, 2) if not np.isnan(wilcox_stat) else None,
        "wilcoxon_p":    round(wilcox_p, 6),
        "wilcoxon_sig":  sig(wilcox_p),
        "mcnemar_p":     round(mcn_p, 6),
        "mcnemar_sig":   sig(mcn_p),
    }


def run_backbone(name, paths, device, output_dir, split="test"):
    print(f"\n{'═'*60}")
    print(f"  {name}")
    print(f"{'═'*60}")

    bam_cfg = load_config(paths["bam_config"])
    mrl_cfg = load_config(paths["mrl_config"])
    set_seed(42)

    corpus, queries, corpus_id_to_idx = load_data(bam_cfg, split=split)
    gt_indices = np.array([corpus_id_to_idx[q["positive_id"]] for q in queries])
    print(f"  Corpus: {len(corpus)}, Queries ({split}): {len(queries)}")

    # BAM-PQ
    print("  Loading BAM-PQ ...")
    bam_model = BloomAlignedMRL(bam_cfg)
    ckpt = os.path.join(paths["bam_ckpt"], "checkpoint.pt")
    if os.path.exists(ckpt):
        bam_model.load_state_dict(
            torch.load(ckpt, map_location=device)["model_state_dict"], strict=False)
    else:
        print(f"  WARNING: BAM ckpt not found at {ckpt}")
    bam_model.to(device).eval()
    bam_tok = AutoTokenizer.from_pretrained(bam_cfg["model"]["backbone"])

    bam_corpus  = encode_corpus_bam(bam_model, bam_tok, corpus, device)
    bam_queries = encode_queries_bam(bam_model, bam_tok, queries, device)
    del bam_model; torch.cuda.empty_cache()

    # MRL baseline
    print("  Loading MRL baseline ...")
    mc = mrl_cfg["model"]
    mrl_model = MRLEncoder(
        model_name=mc["backbone"],
        embedding_dim=mc["embedding_dim"],
        mrl_dims=mc["mrl_dims"],
        pooling=mc.get("pooling", "cls"),
        backbone_type=mc.get("backbone_type", "standard"),
        query_instruction=mc.get("query_instruction", None),
        peft_model_name=mc.get("peft_model_name", None),
    )
    ckpt = os.path.join(paths["mrl_ckpt"], "checkpoint.pt")
    if os.path.exists(ckpt):
        mrl_model.load_state_dict(
            torch.load(ckpt, map_location=device)["model_state_dict"], strict=False)
    else:
        print(f"  WARNING: MRL ckpt not found at {ckpt}")
    mrl_model.to(device).eval()
    mrl_tok = AutoTokenizer.from_pretrained(mc["backbone"])

    mrl_corpus  = encode_corpus_mrl(mrl_model, mrl_tok, corpus, device)
    mrl_queries = encode_queries_mrl(mrl_model, mrl_tok, queries, device)
    del mrl_model; torch.cuda.empty_cache()

    print("  Computing per-query hits@10 ...")
    hits_bam = per_query_hits(bam_queries, bam_corpus, gt_indices, k=10, device=device)
    hits_mrl = per_query_hits(mrl_queries, mrl_corpus, gt_indices, k=10, device=device)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{name}_hits_bam.npy"), hits_bam)
    np.save(os.path.join(output_dir, f"{name}_hits_mrl.npy"), hits_mrl)

    return run_tests(hits_bam, hits_mrl, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", nargs="+",
                        default=["e5large", "bge", "arctic", "roberta", "qwen06b"])
    parser.add_argument("--split", default="test", choices=["val", "test"],
                        help="Which split to evaluate on (default: test)")
    parser.add_argument("--output_dir", default="results/significance/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for name in args.backbones:
        if name not in BACKBONE_MAP:
            print(f"  Unknown backbone: {name}")
            continue
        try:
            all_results.append(run_backbone(name, BACKBONE_MAP[name], device, args.output_dir, split=args.split))
        except Exception as e:
            print(f"  ERROR on {name}: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'═'*95}")
    print("  SIGNIFICANCE — BAM-PQ vs MRL Baseline (R@10, one-sided)")
    print(f"{'═'*95}")
    print(f"  {'Backbone':<10}  {'N':>5}  {'MRL R@10':>9}  {'BAM R@10':>9}  {'Δ':>7}  "
          f"{'BAM↑':>6}  {'MRL↑':>6}  {'Wilcoxon':>12}  {'McNemar':>11}")
    print("  " + "─" * 91)
    for r in all_results:
        print(f"  {r['backbone']:<10}  {r['n_queries']:>5}  {r['r10_mrl']:>9.4f}  "
              f"{r['r10_bam']:>9.4f}  {r['delta_r10']:>+7.4f}  "
              f"{r['bam_wins']:>6}  {r['mrl_wins']:>6}  "
              f"p={r['wilcoxon_p']:.5f} {r['wilcoxon_sig']:<3}  "
              f"p={r['mcnemar_p']:.5f} {r['mcnemar_sig']:<3}")
    print()
    print("  *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
    print("  Wilcoxon: one-sided (greater), zero_method=wilcox")
    print("  McNemar: one-sided exact binomial (discordant pairs)")
    print()

    out = os.path.join(args.output_dir, "significance_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved → {out}")


if __name__ == "__main__":
    main()
