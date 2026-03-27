"""
Efficient Sparse Retrieval for QA-MRL.

Implements ACTUAL compute savings by only computing dot products over
active dimensions. Three strategies:

1. Per-query sparse dot product (correct, simple, O(n_active * n_corpus))
2. Group-sharded indices (precompute sub-indices per group combo)
3. Dimension-packed retrieval (repack active dims into dense vectors)

Strategy 3 is the most practical: for each query, extract active dims,
build/use a FAISS index over only those dims. Since many queries share
the same routing pattern, we cache indices per pattern.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class SparseRetriever:
    """
    Efficient retrieval that exploits QA-MRL's dimension routing.

    Key idea: group queries by their active dimension pattern,
    build a FAISS sub-index per pattern, search in reduced dimensionality.
    """

    def __init__(self, corpus_embs: np.ndarray, group_size: int = 96):
        """
        Args:
            corpus_embs: [N, D] full corpus embeddings (L2-normalized)
            group_size: dimensions per group
        """
        self.corpus_embs = np.ascontiguousarray(corpus_embs.astype(np.float32))
        self.N, self.D = corpus_embs.shape
        self.group_size = group_size
        self.num_groups = self.D // group_size
        self.index_cache = {}  # Cache sub-indices by active group pattern

    def search(self, query_embs: np.ndarray, query_masks: np.ndarray,
               k: int = 100) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Efficient search using only active dimensions per query.

        Args:
            query_embs: [N_q, D] query embeddings
            query_masks: [N_q, D] binary masks (from router)
            k: top-k to retrieve

        Returns:
            scores: [N_q, k]
            indices: [N_q, k]
            stats: timing and dimension statistics
        """
        N_q = query_embs.shape[0]
        all_scores = np.zeros((N_q, k), dtype=np.float32)
        all_indices = np.zeros((N_q, k), dtype=np.int64)

        # Group queries by their active group pattern
        patterns = defaultdict(list)
        for i in range(N_q):
            # Determine which groups are active
            active_groups = []
            for g in range(self.num_groups):
                start = g * self.group_size
                end = start + self.group_size
                if query_masks[i, start:end].mean() > 0.5:
                    active_groups.append(g)
            pattern = tuple(active_groups)
            patterns[pattern].append(i)

        stats = {
            "num_patterns": len(patterns),
            "pattern_sizes": {str(p): len(ids) for p, ids in patterns.items()},
            "total_queries": N_q,
            "full_dim_flops": N_q * self.D * self.N,  # Full retrieval FLOPS
            "sparse_flops": 0,
        }

        t0 = time.time()
        for active_groups, query_indices in patterns.items():
            if not active_groups:
                continue

            # Get active dimension indices
            active_dims = []
            for g in active_groups:
                active_dims.extend(range(g * self.group_size, (g + 1) * self.group_size))
            d_active = len(active_dims)

            # Extract active dims from queries and corpus
            q_active = query_embs[query_indices][:, active_dims].copy()
            c_active = self._get_or_build_subindex(active_groups, active_dims)

            # L2 normalize the sub-vectors
            q_norms = np.linalg.norm(q_active, axis=1, keepdims=True)
            q_active = q_active / (q_norms + 1e-9)

            # Search in reduced dimensionality
            q_active = np.ascontiguousarray(q_active.astype(np.float32))

            if HAS_FAISS:
                index = faiss.IndexFlatIP(d_active)
                index.add(c_active)
                scores, indices = index.search(q_active, min(k, self.N))
            else:
                import torch
                q_t = torch.from_numpy(q_active)
                c_t = torch.from_numpy(c_active)
                sim = torch.mm(q_t, c_t.t())
                scores_t, indices_t = sim.topk(min(k, self.N), dim=-1)
                scores = scores_t.numpy()
                indices = indices_t.numpy()

            for j, qi in enumerate(query_indices):
                all_scores[qi] = scores[j]
                all_indices[qi] = indices[j]

            # Track FLOPS
            stats["sparse_flops"] += len(query_indices) * d_active * self.N

        stats["search_time_s"] = time.time() - t0
        stats["ms_per_query"] = stats["search_time_s"] / N_q * 1000
        stats["flops_reduction"] = 1.0 - stats["sparse_flops"] / stats["full_dim_flops"]

        return all_scores, all_indices, stats

    def _get_or_build_subindex(self, active_groups: tuple,
                                active_dims: list) -> np.ndarray:
        """Get cached corpus sub-vectors or build them."""
        if active_groups not in self.index_cache:
            c_sub = self.corpus_embs[:, active_dims].copy()
            c_norms = np.linalg.norm(c_sub, axis=1, keepdims=True)
            c_sub = c_sub / (c_norms + 1e-9)
            self.index_cache[active_groups] = np.ascontiguousarray(c_sub.astype(np.float32))
        return self.index_cache[active_groups]

    def print_stats(self, stats: Dict):
        """Print retrieval statistics."""
        print(f"\n  Sparse Retrieval Statistics:")
        print(f"    Queries: {stats['total_queries']}")
        print(f"    Unique routing patterns: {stats['num_patterns']}")
        print(f"    Full-dim FLOPS: {stats['full_dim_flops']:.2e}")
        print(f"    Sparse FLOPS:   {stats['sparse_flops']:.2e}")
        print(f"    FLOPS reduction: {stats['flops_reduction']:.1%}")
        print(f"    Search time: {stats['search_time_s']:.2f}s "
              f"({stats['ms_per_query']:.2f}ms/query)")

        # Pattern distribution
        print(f"\n    Top routing patterns:")
        sorted_patterns = sorted(stats["pattern_sizes"].items(),
                                key=lambda x: x[1], reverse=True)[:5]
        for pattern, count in sorted_patterns:
            groups = eval(pattern)
            n_active = len(groups) * (self.D // self.num_groups)
            print(f"      Groups {pattern}: {count} queries "
                  f"({n_active}/{self.D} dims, {n_active/self.D:.0%})")


class EfficiencyBenchmark:
    """
    Benchmark comparing full vs sparse retrieval efficiency.
    Produces the efficiency table for the paper.
    """

    def __init__(self, corpus_embs: np.ndarray, group_size: int = 96):
        self.corpus_embs = corpus_embs
        self.group_size = group_size
        self.D = corpus_embs.shape[1]

    def benchmark(self, query_embs: np.ndarray,
                   query_masks: Optional[np.ndarray] = None,
                   k: int = 100, n_runs: int = 3) -> Dict:
        """Run full benchmark comparing different retrieval strategies."""
        results = {}

        N_q = query_embs.shape[0]
        N_c = self.corpus_embs.shape[0]

        # 1. Full-dimensional retrieval
        print("  Benchmarking full-dim retrieval...")
        times = []
        for _ in range(n_runs):
            t0 = time.time()
            q = np.ascontiguousarray(query_embs.astype(np.float32))
            c = np.ascontiguousarray(self.corpus_embs.astype(np.float32))
            if HAS_FAISS:
                index = faiss.IndexFlatIP(self.D)
                index.add(c)
                index.search(q, k)
            times.append(time.time() - t0)
        results["full_768"] = {
            "dims": self.D,
            "time_s": np.mean(times),
            "ms_per_query": np.mean(times) / N_q * 1000,
            "flops": N_q * self.D * N_c,
        }

        # 2. MRL truncation at various points
        for d in [64, 128, 256, 384, 512]:
            print(f"  Benchmarking MRL d={d}...")
            q_t = query_embs[:, :d].copy()
            c_t = self.corpus_embs[:, :d].copy()
            q_t /= (np.linalg.norm(q_t, axis=1, keepdims=True) + 1e-9)
            c_t /= (np.linalg.norm(c_t, axis=1, keepdims=True) + 1e-9)

            times = []
            for _ in range(n_runs):
                t0 = time.time()
                q = np.ascontiguousarray(q_t.astype(np.float32))
                c = np.ascontiguousarray(c_t.astype(np.float32))
                if HAS_FAISS:
                    index = faiss.IndexFlatIP(d)
                    index.add(c)
                    index.search(q, k)
                times.append(time.time() - t0)

            results[f"mrl_{d}"] = {
                "dims": d,
                "time_s": np.mean(times),
                "ms_per_query": np.mean(times) / N_q * 1000,
                "flops": N_q * d * N_c,
            }

        # 3. QA-MRL sparse retrieval
        if query_masks is not None:
            print("  Benchmarking QA-MRL sparse retrieval...")
            retriever = SparseRetriever(self.corpus_embs, self.group_size)
            _, _, sparse_stats = retriever.search(query_embs, query_masks, k=k)
            avg_active = (query_masks > 0.5).sum(axis=1).mean()

            results["qa_mrl_sparse"] = {
                "dims": float(avg_active),
                "time_s": sparse_stats["search_time_s"],
                "ms_per_query": sparse_stats["ms_per_query"],
                "flops": sparse_stats["sparse_flops"],
                "flops_reduction": sparse_stats["flops_reduction"],
                "num_patterns": sparse_stats["num_patterns"],
            }

        return results

    def print_benchmark(self, results: Dict):
        """Print efficiency benchmark table."""
        print(f"\n{'='*70}")
        print("EFFICIENCY BENCHMARK")
        print(f"{'='*70}")
        print(f"{'Method':20s} {'Dims':>8s} {'ms/query':>10s} {'FLOPS':>12s} {'Speedup':>10s}")
        print("-" * 70)

        base_flops = results.get("full_768", {}).get("flops", 1)
        base_time = results.get("full_768", {}).get("ms_per_query", 1)

        for key in ["full_768", "mrl_512", "mrl_384", "mrl_256", "mrl_128",
                     "mrl_64", "qa_mrl_sparse"]:
            if key not in results:
                continue
            r = results[key]
            name = key.replace("_", " ").replace("mrl ", "MRL d=").replace("full 768", "Full d=768")
            if key == "qa_mrl_sparse":
                name = "QA-MRL (sparse)"

            dims = r["dims"]
            ms = r["ms_per_query"]
            flops = r["flops"]
            speedup = base_time / ms if ms > 0 else 0

            if isinstance(dims, float):
                print(f"{name:20s} {dims:>8.0f} {ms:>10.2f} {flops:>12.2e} {speedup:>10.1f}x")
            else:
                print(f"{name:20s} {dims:>8d} {ms:>10.2f} {flops:>12.2e} {speedup:>10.1f}x")