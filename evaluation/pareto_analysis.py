"""
Pareto Efficiency Analysis for EMNLP.

Generates the key Recall@10 vs. Active Dimensions Pareto curve comparing:
  1. Fixed MRL truncations (d=64, 128, 256, 384, 512, 768)
  2. QA-MRL with learned routing
  3. BAM with Bloom-adaptive truncation

This is Figure 1 of the paper: shows QA-MRL/BAM achieves better
accuracy-efficiency tradeoffs than any fixed truncation.

Also produces per-Bloom Pareto curves (Figure 2): different Bloom levels
have different optimal operating points, validating adaptive routing.
"""

import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


BLOOM_NAMES = {0: "Remember", 1: "Understand", 2: "Apply",
               3: "Analyze", 4: "Evaluate", 5: "Create"}


class ParetoAnalyzer:
    """Generate Pareto efficiency data for the paper."""

    def __init__(self, mrl_dims: List[int] = None, ks: List[int] = None):
        self.mrl_dims = mrl_dims or [64, 128, 256, 384, 512, 768]
        self.ks = ks or [1, 5, 10]

    @torch.no_grad()
    def compute_pareto_data(
        self,
        query_embs: torch.Tensor,      # [N_q, D] full query embeddings
        corpus_embs: torch.Tensor,      # [N_c, D] full corpus embeddings
        gt_indices: np.ndarray,         # [N_q] ground-truth doc index
        query_blooms: np.ndarray,       # [N_q] Bloom levels (0-indexed)
        query_masks: Optional[torch.Tensor] = None,  # [N_q, D] from router
        model_name: str = "model",
    ) -> Dict:
        """
        Compute retrieval metrics at each efficiency level.

        Returns a list of (active_dims, recall@k, ...) points for Pareto plotting.
        """
        device = query_embs.device
        N_q = query_embs.shape[0]
        results = {"points": [], "bloom_points": {}}

        # 1. Fixed MRL truncations
        for d in self.mrl_dims:
            q_trunc = F.normalize(query_embs[:, :d], p=2, dim=-1)
            c_trunc = F.normalize(corpus_embs[:, :d], p=2, dim=-1)

            metrics = self._compute_metrics(q_trunc, c_trunc, gt_indices, device)
            point = {"method": f"MRL d={d}", "active_dims": d, **metrics}
            results["points"].append(point)

            # Per-Bloom metrics at this truncation
            bloom_metrics = self._compute_bloom_metrics(
                q_trunc, c_trunc, gt_indices, query_blooms, device
            )
            for bl, bm in bloom_metrics.items():
                if bl not in results["bloom_points"]:
                    results["bloom_points"][bl] = []
                results["bloom_points"][bl].append(
                    {"method": f"MRL d={d}", "active_dims": d, **bm}
                )

        # 2. Adaptive (QA-MRL / BAM) if masks provided
        if query_masks is not None:
            avg_active = float((query_masks > 0.5).float().sum(dim=-1).mean())
            q_masked = F.normalize(query_embs * query_masks, p=2, dim=-1)

            metrics = self._compute_metrics(q_masked, corpus_embs, gt_indices, device)
            point = {
                "method": model_name,
                "active_dims": avg_active,
                **metrics,
            }
            results["points"].append(point)

            bloom_metrics = self._compute_bloom_metrics(
                q_masked, corpus_embs, gt_indices, query_blooms, device
            )
            for bl, bm in bloom_metrics.items():
                per_bloom_mask = query_masks[query_blooms == bl]
                bl_active = float((per_bloom_mask > 0.5).float().sum(dim=-1).mean())
                if bl not in results["bloom_points"]:
                    results["bloom_points"][bl] = []
                results["bloom_points"][bl].append(
                    {"method": model_name, "active_dims": bl_active, **bm}
                )

        # 3. Compute Pareto frontier
        results["pareto_frontier"] = self._compute_pareto_frontier(
            results["points"], x_key="active_dims", y_key="recall@10"
        )

        return results

    def _compute_metrics(self, q_embs, c_embs, gt_indices, device,
                         chunk_size: int = 256) -> Dict[str, float]:
        """Compute recall@k and MRR from query/corpus embeddings."""
        all_rankings = []
        max_k = max(self.ks)

        for i in range(0, len(q_embs), chunk_size):
            q_chunk = q_embs[i:i+chunk_size].to(device)
            sim = torch.mm(q_chunk, c_embs.to(device).t())
            topk = sim.topk(max_k, dim=-1).indices.cpu().numpy()
            all_rankings.append(topk)

        rankings = np.concatenate(all_rankings, axis=0)
        N = len(gt_indices)
        metrics = {}

        for k in self.ks:
            topk = rankings[:, :k]
            hits = np.array([gt_indices[i] in topk[i] for i in range(N)])
            metrics[f"recall@{k}"] = float(hits.mean())

        # MRR
        mrrs = []
        for i in range(N):
            found = np.where(rankings[i] == gt_indices[i])[0]
            mrrs.append(1.0 / (found[0] + 1) if len(found) > 0 else 0.0)
        metrics["mrr"] = float(np.mean(mrrs))

        return metrics

    def _compute_bloom_metrics(self, q_embs, c_embs, gt_indices, query_blooms,
                                device) -> Dict[int, Dict[str, float]]:
        """Compute per-Bloom-level metrics."""
        bloom_metrics = {}
        for bl in range(6):
            mask = query_blooms == bl
            n_bl = int(mask.sum())
            if n_bl == 0:
                continue
            bl_q = q_embs[mask]
            bl_gt = gt_indices[mask]
            metrics = self._compute_metrics(bl_q, c_embs, bl_gt, device)
            metrics["n"] = n_bl
            bloom_metrics[bl] = metrics
        return bloom_metrics

    def _compute_pareto_frontier(self, points: List[Dict],
                                  x_key: str, y_key: str) -> List[Dict]:
        """Identify Pareto-optimal points (minimize x, maximize y)."""
        sorted_pts = sorted(points, key=lambda p: p[x_key])
        frontier = []
        best_y = -float("inf")
        for p in sorted_pts:
            if p[y_key] > best_y:
                frontier.append(p)
                best_y = p[y_key]
        return frontier

    def print_pareto_table(self, results: Dict):
        """Print Pareto analysis table."""
        print(f"\n{'='*80}")
        print("PARETO EFFICIENCY ANALYSIS")
        print(f"{'='*80}")
        print(f"{'Method':20s} {'Active Dims':>12s} {'R@1':>8s} {'R@5':>8s} {'R@10':>8s} {'MRR':>8s} {'Pareto?':>8s}")
        print("-" * 80)

        frontier_methods = {p["method"] for p in results.get("pareto_frontier", [])}

        for p in sorted(results["points"], key=lambda x: x["active_dims"]):
            name = p["method"]
            dims = p["active_dims"]
            r1 = p.get("recall@1", 0)
            r5 = p.get("recall@5", 0)
            r10 = p.get("recall@10", 0)
            mrr = p.get("mrr", 0)
            pareto = "Yes" if name in frontier_methods else ""

            if isinstance(dims, float):
                print(f"{name:20s} {dims:>12.1f} {r1:>8.4f} {r5:>8.4f} {r10:>8.4f} {mrr:>8.4f} {pareto:>8s}")
            else:
                print(f"{name:20s} {dims:>12d} {r1:>8.4f} {r5:>8.4f} {r10:>8.4f} {mrr:>8.4f} {pareto:>8s}")

        # Per-Bloom breakdown
        bloom_pts = results.get("bloom_points", {})
        if bloom_pts:
            print(f"\n  Per-Bloom Optimal Dimensions:")
            print(f"  {'Bloom Level':15s} {'Best Fixed d':>12s} {'Adaptive d':>12s} {'Fixed R@10':>12s} {'Adaptive R@10':>14s}")
            print(f"  {'-'*70}")
            for bl in range(6):
                if bl not in bloom_pts:
                    continue
                pts = bloom_pts[bl]
                fixed_pts = [p for p in pts if p["method"].startswith("MRL")]
                adaptive_pts = [p for p in pts if not p["method"].startswith("MRL")]

                if fixed_pts:
                    best_fixed = max(fixed_pts, key=lambda p: p.get("recall@10", 0))
                    bf_d = best_fixed["active_dims"]
                    bf_r = best_fixed.get("recall@10", 0)
                else:
                    bf_d, bf_r = 0, 0

                if adaptive_pts:
                    ad = adaptive_pts[0]
                    ad_d = ad["active_dims"]
                    ad_r = ad.get("recall@10", 0)
                else:
                    ad_d, ad_r = 0, 0

                name = BLOOM_NAMES.get(bl, str(bl))
                print(f"  {name:15s} {bf_d:>12.0f} {ad_d:>12.1f} {bf_r:>12.4f} {ad_r:>14.4f}")

    def save_results(self, results: Dict, path: str):
        """Save Pareto data for plotting."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        def convert(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, dict):
                return {str(k): convert(v) for k, v in o.items()}
            if isinstance(o, list):
                return [convert(v) for v in o]
            return o

        with open(path, "w") as f:
            json.dump(convert(results), f, indent=2)
