"""
Bloom-level FAISS sub-indices for efficient retrieval (Challenge 5).

Builds 6 separate FAISS flat indices, one per Bloom level, each containing
corpus embeddings at that level's dimension budget. At retrieval time, queries
only against the sub-index matching their Bloom level.

Key insight: if Remember queries need only 192 dims, searching a 192-dim index
is ~4x faster (FLOPS) and ~4x smaller (memory) than searching the full 768-dim index.

BloomSubindexRetriever:
  - build(corpus_embs, dim_table): build 6 sub-indices
  - retrieve(query_emb, bloom_level, k): search the appropriate sub-index
  - benchmark(query_embs, bloom_labels, corpus_embs, k): measure latency + FLOPS

Usage:
    from evaluation.bloom_subindex import BloomSubindexRetriever, benchmark_subindex_efficiency

    retriever = BloomSubindexRetriever(dim_table={0: 192, 1: 256, 2: 320, 3: 384, 4: 512, 5: 640})
    retriever.build(corpus_embs)

    # Or run the script directly:
    python -m evaluation.bloom_subindex \
        --config configs/bam.yaml \
        --checkpoint /tmp/bam-ckpts/best/ \
        --output_dir results/efficiency/
"""

import time
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("WARNING: faiss not available. BloomSubindexRetriever will use torch fallback.")


class BloomSubindexRetriever:
    """
    6 FAISS flat IP sub-indices, one per Bloom level.

    Each sub-index stores corpus embeddings truncated to dim_table[bloom_level] dims
    and normalized to unit norm. At retrieval time, the query is also truncated and
    normalized before searching the sub-index.

    For scattered masks (Option B), truncation is replaced by mask-wise selection.
    In that case, all sub-indices are built at full 768 dims (same FLOPS as standard).
    The efficiency benefit of Option B comes from sparser activations, not truncation.
    """

    def __init__(self, dim_table: Dict[int, int]):
        """
        dim_table: {bloom_level (0-indexed): active_dims (int)}
                   e.g. {0: 192, 1: 256, 2: 320, 3: 384, 4: 512, 5: 640}
        """
        self.dim_table = dim_table
        self.indices = {}      # bloom_level → faiss.Index or torch.Tensor (fallback)
        self.corpus_size = 0
        self.use_faiss = FAISS_AVAILABLE

    def build(self, corpus_embs: torch.Tensor, verbose: bool = True):
        """
        Build 6 sub-indices from corpus embeddings.

        corpus_embs: [C, 768] full-dim corpus embeddings (should be normalized)
        """
        self.corpus_size = corpus_embs.shape[0]
        sizes = {}

        for bloom_level, dim in self.dim_table.items():
            dim = max(1, min(dim, corpus_embs.shape[1]))
            trunc = F.normalize(corpus_embs[:, :dim], p=2, dim=-1).numpy().astype(np.float32)

            if self.use_faiss:
                index = faiss.IndexFlatIP(dim)
                index.add(trunc)
                self.indices[bloom_level] = index
            else:
                # Torch tensor fallback
                self.indices[bloom_level] = torch.from_numpy(trunc)

            sizes[bloom_level] = dim
            if verbose:
                mem_mb = trunc.nbytes / 1024 / 1024
                print(f"  Bloom {bloom_level}: dim={dim}, size={trunc.shape}, mem={mem_mb:.1f}MB")

        if verbose:
            full_mem = corpus_embs.numpy().astype(np.float32).nbytes / 1024 / 1024
            sub_mem = sum(
                F.normalize(corpus_embs[:, :d], p=2, dim=-1).numpy().astype(np.float32).nbytes
                for d in self.dim_table.values()
            ) / 1024 / 1024
            print(f"\n  Full 768-dim index memory: {full_mem:.1f} MB")
            print(f"  6 sub-indices total memory: {sub_mem:.1f} MB")
            print(f"  Memory ratio: {sub_mem/full_mem:.2f}x")

    def retrieve(
        self,
        query_emb: torch.Tensor,
        bloom_level: int,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve top-k from the sub-index for the given Bloom level.

        query_emb: [D] or [1, D] query embedding
        Returns: (scores, indices) arrays of shape [k]
        """
        dim = self.dim_table.get(bloom_level, query_emb.shape[-1])
        q = F.normalize(query_emb.flatten()[:dim].unsqueeze(0), p=2, dim=-1)
        q_np = q.numpy().astype(np.float32)

        if self.use_faiss and bloom_level in self.indices:
            scores, indices = self.indices[bloom_level].search(q_np, k)
            return scores[0], indices[0]
        elif bloom_level in self.indices:
            # Torch fallback
            c = self.indices[bloom_level]
            sims = torch.mm(q, c.t())[0]
            topk = sims.topk(min(k, len(c)))
            return topk.values.numpy(), topk.indices.numpy()
        else:
            raise ValueError(f"Sub-index for bloom_level={bloom_level} not built. Call build() first.")

    def batch_retrieve(
        self,
        query_embs: torch.Tensor,
        bloom_labels: torch.Tensor,
        k: int = 10,
    ) -> np.ndarray:
        """
        Batch retrieval: each query routes to its Bloom sub-index.
        Returns rankings: [N, k]
        """
        N = len(query_embs)
        rankings = np.zeros((N, k), dtype=np.int64)

        for bloom_level in self.dim_table:
            mask = (bloom_labels == bloom_level).nonzero(as_tuple=True)[0]
            if len(mask) == 0:
                continue
            dim = self.dim_table[bloom_level]
            dim = max(1, min(dim, query_embs.shape[1]))
            q_group = F.normalize(query_embs[mask, :dim], p=2, dim=-1).numpy().astype(np.float32)

            if self.use_faiss and bloom_level in self.indices:
                _, topk_indices = self.indices[bloom_level].search(q_group, k)
            else:
                c = self.indices[bloom_level]
                q_t = torch.from_numpy(q_group)
                sims = torch.mm(q_t, c.t())
                topk_indices = sims.topk(min(k, c.shape[0]), dim=-1).indices.numpy()

            for j, idx in enumerate(mask.tolist()):
                rankings[idx] = topk_indices[j]

        return rankings


def benchmark_subindex_efficiency(
    model,
    corpus_embs: torch.Tensor,
    query_embs: torch.Tensor,
    bloom_labels: torch.Tensor,
    dim_table: Dict[int, int],
    k: int = 10,
    n_warmup: int = 5,
    n_repeats: int = 20,
) -> Dict:
    """
    Benchmark retrieval latency and FLOPS for:
      1. Full 768-dim index
      2. Bloom sub-indices (one per Bloom level)

    Returns dict with latency and FLOPS comparisons.
    """
    results = {}
    N = len(query_embs)

    # --- Build sub-indices ---
    print("Building sub-indices...")
    retriever = BloomSubindexRetriever(dim_table)
    retriever.build(corpus_embs, verbose=True)

    # --- Full 768-dim baseline ---
    if FAISS_AVAILABLE:
        full_index = faiss.IndexFlatIP(corpus_embs.shape[1])
        full_index.add(F.normalize(corpus_embs, p=2, dim=-1).numpy().astype(np.float32))
    else:
        full_index = F.normalize(corpus_embs, p=2, dim=-1)

    q_full = F.normalize(query_embs, p=2, dim=-1).numpy().astype(np.float32)

    # Warmup
    for _ in range(n_warmup):
        if FAISS_AVAILABLE:
            full_index.search(q_full[:16], k)
        else:
            torch.mm(torch.from_numpy(q_full[:16]), full_index.t())

    # Measure full index latency
    times_full = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        if FAISS_AVAILABLE:
            full_index.search(q_full, k)
        else:
            torch.mm(torch.from_numpy(q_full), full_index.t())
        times_full.append(time.perf_counter() - t0)

    # Measure sub-index latency
    times_sub = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        retriever.batch_retrieve(query_embs, bloom_labels, k=k)
        times_sub.append(time.perf_counter() - t0)

    # FLOPS calculation: 2 * N * D * C (multiply-accumulate ops for inner products)
    C = corpus_embs.shape[0]
    D_full = 768
    flops_full = 2 * N * D_full * C

    # Sub-index FLOPS: per Bloom level, proportional to its dim
    flops_sub = 0
    for level, dim in dim_table.items():
        n_level = int((bloom_labels == level).sum().item())
        flops_sub += 2 * n_level * dim * C
    flops_ratio = flops_sub / max(flops_full, 1)

    results = {
        "n_queries": N,
        "corpus_size": C,
        "full_index_latency_ms": {
            "mean": float(np.mean(times_full) * 1000),
            "std": float(np.std(times_full) * 1000),
        },
        "sub_index_latency_ms": {
            "mean": float(np.mean(times_sub) * 1000),
            "std": float(np.std(times_sub) * 1000),
        },
        "latency_speedup": float(np.mean(times_full) / max(np.mean(times_sub), 1e-9)),
        "flops": {
            "full_768dim": flops_full,
            "sub_indices": flops_sub,
            "ratio": flops_ratio,
            "reduction_pct": float((1 - flops_ratio) * 100),
        },
        "dim_table": {str(k): v for k, v in dim_table.items()},
        "avg_active_dims": float(
            np.mean([dim_table.get(int(l.item()), 768) for l in bloom_labels])
        ),
    }

    print(f"\n  Latency:  full={results['full_index_latency_ms']['mean']:.1f}ms, "
          f"sub={results['sub_index_latency_ms']['mean']:.1f}ms, "
          f"speedup={results['latency_speedup']:.2f}x")
    print(f"  FLOPS:    full={flops_full:.2e}, sub={flops_sub:.2e}, "
          f"reduction={results['flops']['reduction_pct']:.1f}%")

    return results


if __name__ == "__main__":
    import argparse
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from utils.misc import load_config, set_seed
    from models.bam import BloomAlignedMRL
    from transformers import AutoTokenizer
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bam.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="results/efficiency/")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    device = torch.device("cpu")  # benchmark on CPU for fair comparison
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])
    os.makedirs(args.output_dir, exist_ok=True)

    config["training"]["loss"]["bloom_frequencies"] = [1/6] * 6
    model = BloomAlignedMRL(config)
    ckpt = os.path.join(args.checkpoint, "checkpoint.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(
            torch.load(ckpt, map_location=device)["model_state_dict"], strict=False
        )
    model.eval()

    # Load and encode data
    corpus = []
    with open(config["data"]["corpus_path"]) as f:
        for line in f:
            corpus.append(json.loads(line.strip()))

    test_samples = []
    with open(config["data"]["test_path"]) as f:
        for line in f:
            test_samples.append(json.loads(line.strip()))

    print("Encoding corpus...")
    corpus_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(corpus), 128)):
            batch = [c["text"] for c in corpus[i:i + 128]]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=256, return_tensors="pt")
            out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
            corpus_embs.append(out["full_embedding"].cpu())
    corpus_embs = torch.cat(corpus_embs)

    print("Encoding queries...")
    query_embs, bloom_labels_list = [], []
    with torch.no_grad():
        for i in tqdm(range(0, min(len(test_samples), 1000), 64)):
            batch = test_samples[i:i + 64]
            enc = tokenizer([s["query"] for s in batch], padding=True, truncation=True,
                            max_length=128, return_tensors="pt")
            bloom_lbl = torch.tensor([s["bloom_level"] - 1 for s in batch],
                                     dtype=torch.long)
            out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                       bloom_labels=bloom_lbl)
            query_embs.append(out["full_embedding"].cpu())
            bloom_labels_list.append(bloom_lbl)
    query_embs = torch.cat(query_embs)
    bloom_labels = torch.cat(bloom_labels_list)

    # Get dim table from router
    dim_table = model.bloom_router.get_dim_table()
    print(f"\nLearned dim table: {dim_table}")

    print("\nBenchmarking...")
    bench = benchmark_subindex_efficiency(
        model, corpus_embs, query_embs, bloom_labels, dim_table, k=10
    )

    out_path = os.path.join(args.output_dir, "bloom_subindex_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(bench, f, indent=2)
    print(f"\nSaved benchmark to {out_path}")
