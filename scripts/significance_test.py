"""
Significance testing: BAM-PQ vs MRL baseline on per-query R@10.

For each backbone, loads both BAM-PQ and MRL checkpoints, runs inference
on the full validation set, and tests whether BAM-PQ's R@10 improvement
is statistically significant.

Tests used:
  - Wilcoxon signed-rank test on per-query hits@10 differences
  - McNemar's test (exact, for binary outcomes)

Usage:
    python scripts/significance_test.py                          # all 5 backbones
    python scripts/significance_test.py --backbones arctic bge  # subset
    python scripts/significance_test.py --output_dir results/significance/

Output:
    results/significance/significance_results.json
    prints a summary table
"""

import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import wilcoxon, binom

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.misc import load_config, set_seed
from models.bam import BloomAlignedMRL
from models.encoder import MRLEncoder
from transformers import AutoTokenizer
from data.dataset import load_corpus, load_queries


# ── Backbone config map ────────────────────────────────────────────────────────
BACKBONE_MAP = {
    "e5large": {
        "bam_config":    "configs/bam_pq.yaml",
        "bam_ckpt":      "/tmp/uday/multi-domain/educational/bam_pq_e5large/best",
        "mrl_config":    "configs/standard_ft_e5large.yaml",
        "mrl_ckpt":      "/tmp/uday/multi-domain/educational/standard_ft_e5large/best",
    },
    "bge": {
        "bam_config":    "configs/bam_pq_bge_large.yaml",
        "bam_ckpt":      "/tmp/uday/multi-domain/educational/bam_pq_bge_large/best",
        "mrl_config":    "configs/standard_ft_bge.yaml",
        "mrl_ckpt":      "/tmp/uday/multi-domain/educational/standard_ft_bge/best",
    },
    "arctic": {
        "bam_config":    "configs/bam_pq_arctic.yaml",
        "bam_ckpt":      "/tmp/uday/multi-domain/educational/bam_pq_arctic/best",
        "mrl_config":    "configs/standard_ft_arctic.yaml",
        "mrl_ckpt":      "/tmp/uday/multi-domain/educational/standard_ft_arctic/best",
    },
    "roberta": {
        "bam_config":    "configs/bam_pq_roberta.yaml",
        "bam_ckpt":      "/tmp/uday/multi-domain/educational/bam_pq_roberta/best",
        "mrl_config":    "configs/standard_ft_roberta.yaml",
        "mrl_ckpt":      "/tmp/uday/multi-domain/educational/standard_ft_roberta/best",
    },
    "qwen06b": {
        "bam_config":    "configs/bam_pq_qwen06b.yaml",
        "bam_ckpt":      "/tmp/uday/multi-domain/educational/bam_pq_qwen06b/best",
        "mrl_config":    "configs/standard_ft_qwen06b.yaml",
        "mrl_ckpt":      "/tmp/uday/multi-domain/educational/standard_ft_qwen06b/best",
    },
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(config):
    data_cfg = config["data"]
    corpus, queries = [], []

    with open(data_cfg["corpus_path"]) as f:
        for line in f:
            corpus.append(json.loads(line))

    with open(data_cfg.get("val_path", data_cfg.get("test_path", ""))) as f:
        for line in f:
            queries.append(json.loads(line))

    corpus_id_to_idx = {c["id"]: i for i, c in enumerate(corpus)}
    return corpus, queries, corpus_id_to_idx


# ── Encoding helpers ───────────────────────────────────────────────────────────

@torch.no_grad()
def encode_corpus_mrl(model, tokenizer, corpus, device, batch_size=128):
    embs = []
    for i in tqdm(range(0, len(corpus), batch_size), desc="  MRL corpus", leave=False):
        batch = [c["text"] for c in corpus[i:i+batch_size]]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(enc["input_ids"], enc["attention_mask"])
        embs.append(out["full"].float().cpu())
    return torch.cat(embs)


@torch.no_grad()
def encode_queries_mrl(model, tokenizer, queries, device, batch_size=64):
    embs = []
    for i in tqdm(range(0, len(queries), batch_size), desc="  MRL queries", leave=False):
        batch = [q["query"] for q in queries[i:i+batch_size]]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(enc["input_ids"], enc["attention_mask"])
        embs.append(out["full"].float().cpu())
    return torch.cat(embs)


@torch.no_grad()
def encode_corpus_bam(model, tokenizer, corpus, device, batch_size=128):
    embs = []
    for i in tqdm(range(0, len(corpus), batch_size), desc="  BAM corpus", leave=False):
        batch = [c["text"] for c in corpus[i:i+batch_size]]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=256, return_tensors="pt")
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


# ── Per-query hits@k ───────────────────────────────────────────────────────────

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


# ── Wilcoxon + McNemar ────────────────────────────────────────────────────────

def run_tests(hits_bam, hits_mrl, backbone):
    n = len(hits_bam)
    diff = hits_bam.astype(float) - hits_mrl.astype(float)

    r10_bam = hits_bam.mean()
    r10_mrl = hits_mrl.mean()
    delta   = r10_bam - r10_mrl

    # Wilcoxon signed-rank (two-sided; zero_method='wilcox' drops ties)
    if diff.sum() == 0:
        wilcox_stat, wilcox_p = float("nan"), 1.0
    else:
        wilcox_stat, wilcox_p = wilcoxon(diff, zero_method="wilcox", alternative="greater")

    # McNemar's test (exact binomial on discordant pairs)
    b = int(((hits_mrl == 1) & (hits_bam == 0)).sum())  # MRL wins, BAM misses
    c = int(((hits_bam == 1) & (hits_mrl == 0)).sum())  # BAM wins, MRL misses
    # One-sided: P(X >= c) under H0: X ~ Binomial(b+c, 0.5)
    n_disc = b + c
    if n_disc == 0:
        mcn_p = 1.0
    elif n_disc < 25:
        mcn_p = float(binom.sf(c - 1, n_disc, 0.5))  # exact
    else:
        # chi2 approximation
        from scipy.stats import chi2
        stat = (abs(c - b) - 1) ** 2 / (b + c)
        mcn_p = float(chi2.sf(stat, df=1) / 2)  # one-sided

    sig_wilcox = "***" if wilcox_p < 0.001 else ("**" if wilcox_p < 0.01
                 else ("*" if wilcox_p < 0.05 else "ns"))
    sig_mcn    = "***" if mcn_p < 0.001 else ("**" if mcn_p < 0.01
                 else ("*" if mcn_p < 0.05 else "ns"))

    return {
        "backbone": backbone,
        "n_queries": n,
        "r10_bam": round(r10_bam, 4),
        "r10_mrl": round(r10_mrl, 4),
        "delta_r10": round(delta, 4),
        "wilcoxon_stat": round(wilcox_stat, 2) if not np.isnan(wilcox_stat) else None,
        "wilcoxon_p": round(wilcox_p, 5),
        "wilcoxon_sig": sig_wilcox,
        "mcnemar_p": round(mcn_p, 5),
        "mcnemar_sig": sig_mcn,
        "bam_wins": c,   # queries where BAM hit, MRL missed
        "mrl_wins": b,   # queries where MRL hit, BAM missed
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run_backbone(name, paths, device, output_dir):
    print(f"\n{'═'*60}")
    print(f"  {name}")
    print(f"{'═'*60}")

    bam_cfg = load_config(paths["bam_config"])
    mrl_cfg = load_config(paths["mrl_config"])
    set_seed(42)

    # Load data (use BAM config paths)
    corpus, queries, corpus_id_to_idx = load_data(bam_cfg)
    gt_indices = np.array([corpus_id_to_idx[q["positive_id"]] for q in queries])
    print(f"  Corpus: {len(corpus)}, Queries: {len(queries)}")

    # ── BAM-PQ ────────────────────────────────────────────────────────────────
    print("  Loading BAM-PQ ...")
    bam_model = BloomAlignedMRL(bam_cfg)
    ckpt_path = os.path.join(paths["bam_ckpt"], "checkpoint.pt")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        bam_model.load_state_dict(state["model_state_dict"], strict=False)
    bam_model.to(device).eval()
    bam_tok = AutoTokenizer.from_pretrained(bam_cfg["model"]["backbone"])

    bam_corpus = encode_corpus_bam(bam_model, bam_tok, corpus, device)
    bam_queries = encode_queries_bam(bam_model, bam_tok, queries, device)
    del bam_model; torch.cuda.empty_cache()

    # ── MRL baseline ──────────────────────────────────────────────────────────
    print("  Loading MRL baseline ...")
    mc = mrl_cfg["model"]
    mrl_model = MRLEncoder(model_name=mc["backbone"],
                           embedding_dim=mc["embedding_dim"],
                           mrl_dims=mc["mrl_dims"])
    mrl_ckpt = os.path.join(paths["mrl_ckpt"], "checkpoint.pt")
    if os.path.exists(mrl_ckpt):
        state = torch.load(mrl_ckpt, map_location=device)
        mrl_model.load_state_dict(state["model_state_dict"], strict=False)
    mrl_model.to(device).eval()
    mrl_tok = AutoTokenizer.from_pretrained(mc["backbone"])

    mrl_corpus = encode_corpus_mrl(mrl_model, mrl_tok, corpus, device)
    mrl_queries = encode_queries_mrl(mrl_model, mrl_tok, queries, device)
    del mrl_model; torch.cuda.empty_cache()

    # ── Per-query hits ─────────────────────────────────────────────────────────
    print("  Computing per-query hits@10 ...")
    hits_bam = per_query_hits(bam_queries, bam_corpus, gt_indices, k=10, device=device)
    hits_mrl = per_query_hits(mrl_queries, mrl_corpus, gt_indices, k=10, device=device)

    result = run_tests(hits_bam, hits_mrl, name)

    # Save per-query hits for reproducibility
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{name}_hits_bam.npy"), hits_bam)
    np.save(os.path.join(output_dir, f"{name}_hits_mrl.npy"), hits_mrl)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", nargs="+",
                        default=["e5large", "bge", "arctic", "roberta", "qwen06b"])
    parser.add_argument("--output_dir", default="results/significance/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for name in args.backbones:
        if name not in BACKBONE_MAP:
            print(f"  Unknown backbone {name}, skipping")
            continue
        try:
            r = run_backbone(name, BACKBONE_MAP[name], device, args.output_dir)
            all_results.append(r)
        except Exception as e:
            print(f"  ERROR on {name}: {e}")
            import traceback; traceback.print_exc()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'═'*90}")
    print("  SIGNIFICANCE TEST RESULTS — BAM-PQ vs MRL Baseline (R@10)")
    print(f"{'═'*90}")
    hdr = f"  {'Backbone':<10}  {'N':>5}  {'MRL R@10':>9}  {'BAM R@10':>9}  {'Δ':>7}  {'BAM↑':>6}  {'MRL↑':>6}  {'Wilcoxon':>10}  {'McNemar':>9}"
    print(hdr)
    print("  " + "─" * 86)
    for r in all_results:
        print(
            f"  {r['backbone']:<10}  {r['n_queries']:>5}  {r['r10_mrl']:>9.4f}  "
            f"{r['r10_bam']:>9.4f}  {r['delta_r10']:>+7.4f}  "
            f"{r['bam_wins']:>6}  {r['mrl_wins']:>6}  "
            f"p={r['wilcoxon_p']:.4f}{r['wilcoxon_sig']:>3}  "
            f"p={r['mcnemar_p']:.4f}{r['mcnemar_sig']:>3}"
        )
    print()
    print("  Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
    print("  Wilcoxon: one-sided (greater), zero_method=wilcox")
    print("  McNemar: one-sided, exact binomial if discordant <25, else chi2 approx")
    print()

    out_path = os.path.join(args.output_dir, "significance_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
