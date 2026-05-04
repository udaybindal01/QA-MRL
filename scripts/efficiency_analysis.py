"""
End-to-end efficiency analysis for BAM-PQ.

Measures latency of each pipeline stage and shows Bloom annotation
is negligible relative to encoding + retrieval.

Stages timed:
  1. Bloom council annotation        (DeBERTa + RoBERTa + BERT + SVM)
  2. Query encoding (BAM-PQ masked)
  3. FAISS retrieval (inner product over corpus)
  4. Total end-to-end

Comparisons:
  - BAM-PQ vs MRL baseline (same encoding + retrieval, no annotation)
  - Batch sizes: 1, 8, 32, 64 queries

Output:
  results/efficiency/efficiency_results.json
  prints summary table

Usage:
    python scripts/efficiency_analysis.py --backbone arctic
    python scripts/efficiency_analysis.py --backbone bge --n_timing_runs 50
"""

import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

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


def load_data(config):
    data_cfg = config["data"]
    corpus, queries = [], []
    with open(data_cfg["corpus_path"]) as f:
        for line in f:
            corpus.append(json.loads(line))
    val_path = data_cfg.get("val_path") or data_cfg.get("test_path")
    with open(val_path) as f:
        for line in f:
            queries.append(json.loads(line))
    return corpus, queries


def timed(fn, n_runs=10, warmup=3):
    """Run fn() n_runs times after warmup, return (mean_ms, std_ms, min_ms)."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times)), float(np.std(times)), float(np.min(times))


def build_faiss_index(corpus_embs):
    """Build a flat inner product index."""
    try:
        import faiss
        d = corpus_embs.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(corpus_embs.numpy().astype(np.float32))
        return index, "faiss"
    except ImportError:
        return corpus_embs, "torch"


def faiss_search(index, query_embs, k=10, index_type="faiss"):
    if index_type == "faiss":
        _, _ = index.search(query_embs.numpy().astype(np.float32), k)
    else:
        torch.mm(query_embs, index.t()).topk(k, dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="arctic",
                        choices=list(BACKBONE_MAP.keys()))
    parser.add_argument("--n_timing_runs", type=int, default=30,
                        help="Number of timing runs per stage (after warmup)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 8, 32, 64])
    parser.add_argument("--output_dir", default="results/efficiency/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    paths  = BACKBONE_MAP[args.backbone]
    bam_cfg = load_config(paths["bam_config"])
    mrl_cfg = load_config(paths["mrl_config"])

    print(f"\n{'═'*65}")
    print(f"  Efficiency Analysis — {args.backbone}")
    print(f"  Device: {device}")
    print(f"{'═'*65}")

    corpus, queries = load_data(bam_cfg)
    query_texts = [q["query"] for q in queries]
    bloom_labels_all = [q["bloom_level"] - 1 for q in queries]

    # ── Load BAM-PQ ───────────────────────────────────────────────────────────
    print("\n  Loading BAM-PQ ...")
    bam_model = BloomAlignedMRL(bam_cfg)
    ckpt = os.path.join(paths["bam_ckpt"], "checkpoint.pt")
    if os.path.exists(ckpt):
        bam_model.load_state_dict(
            torch.load(ckpt, map_location=device)["model_state_dict"], strict=False)
    bam_model.to(device).eval()
    bam_tok = AutoTokenizer.from_pretrained(bam_cfg["model"]["backbone"])

    # ── Load MRL ──────────────────────────────────────────────────────────────
    print("  Loading MRL baseline ...")
    mc = mrl_cfg["model"]
    mrl_model = MRLEncoder(model_name=mc["backbone"],
                           embedding_dim=mc["embedding_dim"],
                           mrl_dims=mc["mrl_dims"])
    ckpt = os.path.join(paths["mrl_ckpt"], "checkpoint.pt")
    if os.path.exists(ckpt):
        mrl_model.load_state_dict(
            torch.load(ckpt, map_location=device)["model_state_dict"], strict=False)
    mrl_model.to(device).eval()
    mrl_tok = AutoTokenizer.from_pretrained(mc["backbone"])

    # ── Build corpus index (once) ─────────────────────────────────────────────
    print("  Building corpus index ...")
    corpus_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(corpus), 128), desc="  corpus", leave=False):
            batch = [c["text"] for c in corpus[i:i+128]]
            enc = bam_tok(batch, padding=True, truncation=True,
                          max_length=256, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = bam_model.encode_documents(enc["input_ids"], enc["attention_mask"])
            corpus_embs.append(out["full_embedding"].float().cpu())
    corpus_embs = torch.cat(corpus_embs)
    index, index_type = build_faiss_index(corpus_embs)
    print(f"  Corpus: {len(corpus)} docs, {corpus_embs.shape[1]} dims, index={index_type}")

    # ── Try loading Bloom council ─────────────────────────────────────────────
    print("\n  Loading Bloom council ...")
    try:
        from data.bloom_classifier import classify_bloom_batch, _load_council
        _load_council()
        has_council = True
        print("  ✓ Council loaded")
    except Exception as e:
        has_council = False
        print(f"  ✗ Council not available: {e}")

    # ── Timing per batch size ─────────────────────────────────────────────────
    results_by_batch = {}

    for bs in args.batch_sizes:
        print(f"\n  ── Batch size = {bs} ─────────────────────────────────────")
        sample_texts  = query_texts[:bs]
        sample_blooms = torch.tensor(bloom_labels_all[:bs], dtype=torch.long, device=device)

        # 1. Bloom annotation
        if has_council:
            def _bloom():
                classify_bloom_batch(sample_texts, batch_size=bs)
            t_bloom_mean, t_bloom_std, t_bloom_min = timed(
                _bloom, n_runs=args.n_timing_runs, warmup=args.warmup)
        else:
            t_bloom_mean = t_bloom_std = t_bloom_min = 0.0

        # 2. BAM-PQ query encoding
        enc_cache = bam_tok(sample_texts, padding=True, truncation=True,
                            max_length=128, return_tensors="pt")
        enc_cache = {k: v.to(device) for k, v in enc_cache.items()}

        @torch.no_grad()
        def _bam_encode():
            bam_model.encode_queries(
                enc_cache["input_ids"], enc_cache["attention_mask"],
                bloom_labels=sample_blooms)

        t_bam_enc_mean, t_bam_enc_std, t_bam_enc_min = timed(
            _bam_encode, n_runs=args.n_timing_runs, warmup=args.warmup)

        # 3. MRL query encoding
        enc_mrl = mrl_tok(sample_texts, padding=True, truncation=True,
                          max_length=128, return_tensors="pt")
        enc_mrl = {k: v.to(device) for k, v in enc_mrl.items()}

        @torch.no_grad()
        def _mrl_encode():
            mrl_model(enc_mrl["input_ids"], enc_mrl["attention_mask"])

        t_mrl_enc_mean, t_mrl_enc_std, t_mrl_enc_min = timed(
            _mrl_encode, n_runs=args.n_timing_runs, warmup=args.warmup)

        # 4. FAISS retrieval (use precomputed BAM query emb)
        with torch.no_grad():
            q_emb = bam_model.encode_queries(
                enc_cache["input_ids"], enc_cache["attention_mask"],
                bloom_labels=sample_blooms)["masked_embedding"].float().cpu()

        def _retrieve():
            faiss_search(index, q_emb, k=10, index_type=index_type)

        t_retr_mean, t_retr_std, t_retr_min = timed(
            _retrieve, n_runs=args.n_timing_runs, warmup=args.warmup)

        # 5. Totals
        t_bam_total  = t_bloom_mean + t_bam_enc_mean + t_retr_mean
        t_mrl_total  = t_mrl_enc_mean + t_retr_mean
        bloom_frac   = (t_bloom_mean / t_bam_total * 100) if t_bam_total > 0 else 0.0

        # Per-query (divide by batch size)
        def pq(ms): return ms / bs

        print(f"  {'Stage':<30} {'Mean (ms)':>10} {'Std':>8} {'Per-query':>12}")
        print(f"  {'─'*62}")
        if has_council:
            print(f"  {'Bloom annotation':<30} {t_bloom_mean:>10.2f} {t_bloom_std:>8.2f} {pq(t_bloom_mean):>12.3f}")
        print(f"  {'BAM-PQ query encode':<30} {t_bam_enc_mean:>10.2f} {t_bam_enc_std:>8.2f} {pq(t_bam_enc_mean):>12.3f}")
        print(f"  {'MRL query encode':<30} {t_mrl_enc_mean:>10.2f} {t_mrl_enc_std:>8.2f} {pq(t_mrl_enc_mean):>12.3f}")
        print(f"  {'FAISS retrieval':<30} {t_retr_mean:>10.2f} {t_retr_std:>8.2f} {pq(t_retr_mean):>12.3f}")
        print(f"  {'─'*62}")
        print(f"  {'BAM-PQ total':<30} {t_bam_total:>10.2f} {'':>8} {pq(t_bam_total):>12.3f}")
        print(f"  {'MRL total (no annotation)':<30} {t_mrl_total:>10.2f} {'':>8} {pq(t_mrl_total):>12.3f}")
        if has_council:
            print(f"\n  Bloom annotation = {bloom_frac:.1f}% of BAM-PQ total latency")
            print(f"  Encoding overhead vs MRL = {((t_bam_enc_mean/t_mrl_enc_mean)-1)*100:+.1f}%")

        results_by_batch[bs] = {
            "batch_size": bs,
            "bloom_annotation_ms":   round(t_bloom_mean, 3),
            "bloom_annotation_std":  round(t_bloom_std, 3),
            "bam_encode_ms":         round(t_bam_enc_mean, 3),
            "bam_encode_std":        round(t_bam_enc_std, 3),
            "mrl_encode_ms":         round(t_mrl_enc_mean, 3),
            "mrl_encode_std":        round(t_mrl_enc_std, 3),
            "retrieval_ms":          round(t_retr_mean, 3),
            "retrieval_std":         round(t_retr_std, 3),
            "bam_total_ms":          round(t_bam_total, 3),
            "mrl_total_ms":          round(t_mrl_total, 3),
            "bloom_pct_of_total":    round(bloom_frac, 2),
            "per_query": {
                "bloom_ms":    round(pq(t_bloom_mean), 4),
                "bam_enc_ms":  round(pq(t_bam_enc_mean), 4),
                "mrl_enc_ms":  round(pq(t_mrl_enc_mean), 4),
                "retrieval_ms":round(pq(t_retr_mean), 4),
                "bam_total_ms":round(pq(t_bam_total), 4),
                "mrl_total_ms":round(pq(t_mrl_total), 4),
            }
        }

    # ── Summary across batch sizes ─────────────────────────────────────────────
    print(f"\n{'═'*75}")
    print(f"  SUMMARY — per-query latency (ms)  |  backbone={args.backbone}")
    print(f"{'═'*75}")
    print(f"  {'BS':>4}  {'Bloom':>8}  {'BAM enc':>9}  {'MRL enc':>9}  "
          f"{'Retrieval':>11}  {'BAM total':>11}  {'Bloom%':>7}")
    print(f"  {'─'*73}")
    for bs, r in results_by_batch.items():
        pq = r["per_query"]
        print(f"  {bs:>4}  {pq['bloom_ms']:>8.3f}  {pq['bam_enc_ms']:>9.3f}  "
              f"{pq['mrl_enc_ms']:>9.3f}  {pq['retrieval_ms']:>11.3f}  "
              f"{pq['bam_total_ms']:>11.3f}  {r['bloom_pct_of_total']:>6.1f}%")
    print()
    if has_council:
        # Average bloom% across batch sizes
        avg_bloom_pct = np.mean([r["bloom_pct_of_total"] for r in results_by_batch.values()])
        print(f"  Average Bloom annotation overhead: {avg_bloom_pct:.1f}% of end-to-end latency")
    print()

    out = {
        "backbone": args.backbone,
        "device": str(device),
        "corpus_size": len(corpus),
        "embedding_dim": int(corpus_embs.shape[1]),
        "n_timing_runs": args.n_timing_runs,
        "has_council": has_council,
        "results": results_by_batch,
    }
    out_path = os.path.join(args.output_dir, f"efficiency_{args.backbone}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
