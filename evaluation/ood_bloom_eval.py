"""
Cross-Bloom Out-of-Distribution Generalization Evaluation.

Key EMNLP experiment: Does the model learn generalizable cognitive
complexity routing, or does it just memorize Bloom level → dim mappings?

Setup:
  1. Hold out Bloom levels {Evaluate, Create} from training
  2. Train QA-MRL/BAM on {Remember, Understand, Apply, Analyze} only
  3. Evaluate on held-out {Evaluate, Create} queries

If routing generalizes:
  - The model should still route higher-complexity queries to more dims
  - Performance gap between in-distribution and OOD should be small

If routing memorizes:
  - OOD queries will be routed randomly / to default dims
  - Large performance gap

This is Table 5 in the paper: ID vs OOD performance by method.
"""

import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


BLOOM_NAMES = {0: "Remember", 1: "Understand", 2: "Apply",
               3: "Analyze", 4: "Evaluate", 5: "Create"}


class OODBloomEvaluator:
    """
    Evaluate generalization to held-out Bloom levels.

    Supports two OOD protocols:
      1. leave_out_high: Train on {0,1,2,3}, evaluate on {4,5}
         Tests: does the model generalize to higher cognitive complexity?
      2. leave_out_alternating: Train on {0,2,4}, evaluate on {1,3,5}
         Tests: does the model interpolate between seen levels?
    """

    def __init__(self, ks: List[int] = None):
        self.ks = ks or [1, 5, 10]

    def split_by_bloom(
        self,
        test_samples: List[Dict],
        protocol: str = "leave_out_high",
    ) -> Tuple[List[Dict], List[Dict], set, set]:
        """
        Split test queries into ID and OOD sets based on Bloom level.

        Args:
            test_samples: list of test query dicts with 'bloom_level' (1-indexed)
            protocol: splitting strategy

        Returns:
            id_samples, ood_samples, id_levels, ood_levels
        """
        if protocol == "leave_out_high":
            # Train on Remember(1)..Analyze(4), test on Evaluate(5)+Create(6)
            id_levels = {1, 2, 3, 4}
            ood_levels = {5, 6}
        elif protocol == "leave_out_alternating":
            # Train on Remember(1), Apply(3), Evaluate(5)
            # Test on Understand(2), Analyze(4), Create(6)
            id_levels = {1, 3, 5}
            ood_levels = {2, 4, 6}
        elif protocol == "leave_out_low":
            # Train on Apply(3)..Create(6), test on Remember(1)+Understand(2)
            id_levels = {3, 4, 5, 6}
            ood_levels = {1, 2}
        else:
            raise ValueError(f"Unknown OOD protocol: {protocol}")

        id_samples = [s for s in test_samples if s["bloom_level"] in id_levels]
        ood_samples = [s for s in test_samples if s["bloom_level"] in ood_levels]

        return id_samples, ood_samples, id_levels, ood_levels

    @torch.no_grad()
    def evaluate_ood(
        self,
        model,
        test_samples: List[Dict],
        corpus: List[Dict],
        corpus_embs: torch.Tensor,
        corpus_id_to_idx: Dict[str, int],
        tokenizer,
        device: torch.device,
        protocol: str = "leave_out_high",
    ) -> Dict:
        """
        Run full OOD evaluation.

        Returns metrics for ID queries, OOD queries, and the gap.
        """
        model.eval()

        id_samples, ood_samples, id_levels, ood_levels = self.split_by_bloom(
            test_samples, protocol
        )

        results = {
            "protocol": protocol,
            "id_levels": sorted(id_levels),
            "ood_levels": sorted(ood_levels),
            "n_id": len(id_samples),
            "n_ood": len(ood_samples),
        }

        # Evaluate ID queries
        if id_samples:
            id_metrics, id_routing = self._evaluate_subset(
                model, id_samples, corpus_embs, corpus_id_to_idx,
                tokenizer, device, label="ID"
            )
            results["id_metrics"] = id_metrics
            results["id_routing"] = id_routing

        # Evaluate OOD queries
        if ood_samples:
            ood_metrics, ood_routing = self._evaluate_subset(
                model, ood_samples, corpus_embs, corpus_id_to_idx,
                tokenizer, device, label="OOD"
            )
            results["ood_metrics"] = ood_metrics
            results["ood_routing"] = ood_routing

        # Compute gap
        if id_samples and ood_samples:
            results["gap"] = {}
            for k in self.ks:
                id_r = id_metrics.get(f"recall@{k}", 0)
                ood_r = ood_metrics.get(f"recall@{k}", 0)
                results["gap"][f"recall@{k}_gap"] = id_r - ood_r
                results["gap"][f"recall@{k}_relative_gap"] = (
                    (id_r - ood_r) / (id_r + 1e-10)
                )

            # Routing behavior comparison
            if id_routing and ood_routing:
                results["routing_gap"] = {
                    "id_avg_dims": id_routing.get("avg_active_dims", 0),
                    "ood_avg_dims": ood_routing.get("avg_active_dims", 0),
                    "dim_increase_for_ood": (
                        ood_routing.get("avg_active_dims", 0)
                        - id_routing.get("avg_active_dims", 0)
                    ),
                }

        return results

    def _evaluate_subset(
        self,
        model,
        samples: List[Dict],
        corpus_embs: torch.Tensor,
        corpus_id_to_idx: Dict[str, int],
        tokenizer,
        device: torch.device,
        label: str = "",
    ) -> Tuple[Dict, Dict]:
        """Evaluate a subset of queries."""
        # Filter to valid samples
        valid = [s for s in samples if s.get("positive_id", "") in corpus_id_to_idx]
        if not valid:
            return {}, {}

        # Encode queries
        batch_size = 64
        all_embs, all_masks = [], []

        for i in range(0, len(valid), batch_size):
            batch_texts = [s["query"] for s in valid[i:i+batch_size]]
            enc = tokenizer(batch_texts, padding=True, truncation=True,
                           max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}

            if hasattr(model, "encode_queries"):
                out = model.encode_queries(enc["input_ids"], enc["attention_mask"])
                all_embs.append(out["masked_embedding"].cpu())
                if "mask" in out:
                    all_masks.append(out["mask"].cpu())
            else:
                out = model(enc["input_ids"], enc["attention_mask"])
                all_embs.append(out["full"].cpu())

        query_embs = torch.cat(all_embs)
        query_masks = torch.cat(all_masks) if all_masks else None

        # Ground truth
        gt_indices = np.array([corpus_id_to_idx[s["positive_id"]] for s in valid])

        # Compute similarities
        max_k = max(self.ks)
        all_rankings = []
        chunk_size = 256
        for i in range(0, len(query_embs), chunk_size):
            q_chunk = query_embs[i:i+chunk_size].to(device)
            sim = torch.mm(q_chunk, corpus_embs.to(device).t())
            topk = sim.topk(max_k, dim=-1).indices.cpu().numpy()
            all_rankings.append(topk)
        rankings = np.concatenate(all_rankings)

        # Metrics
        N = len(valid)
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

        # Per-Bloom breakdown within this subset
        blooms = np.array([s["bloom_level"] for s in valid])
        for bl in np.unique(blooms):
            mask = blooms == bl
            n_bl = int(mask.sum())
            if n_bl == 0:
                continue
            bl_rankings = rankings[mask]
            bl_gt = gt_indices[mask]
            for k in self.ks:
                topk = bl_rankings[:, :k]
                hits = np.array([bl_gt[j] in topk[j] for j in range(n_bl)])
                name = BLOOM_NAMES.get(bl - 1, str(bl))  # 1-indexed to 0-indexed
                metrics[f"bloom_{name}_recall@{k}"] = float(hits.mean())

        # Routing stats
        routing = {}
        if query_masks is not None:
            active = (query_masks > 0.5).float().sum(dim=-1)
            routing["avg_active_dims"] = float(active.mean())
            routing["std_active_dims"] = float(active.std())

            # Per-Bloom routing
            for bl in np.unique(blooms):
                mask = blooms == bl
                bl_active = active[torch.from_numpy(mask)]
                name = BLOOM_NAMES.get(bl - 1, str(bl))
                routing[f"bloom_{name}_avg_dims"] = float(bl_active.mean())

        return metrics, routing

    def print_results(self, results: Dict):
        """Print OOD evaluation results."""
        print(f"\n{'='*70}")
        print(f"CROSS-BLOOM OOD GENERALIZATION ({results['protocol']})")
        print(f"{'='*70}")
        print(f"  ID levels:  {[BLOOM_NAMES.get(l-1, l) for l in results['id_levels']]}")
        print(f"  OOD levels: {[BLOOM_NAMES.get(l-1, l) for l in results['ood_levels']]}")
        print(f"  N_id: {results['n_id']}, N_ood: {results['n_ood']}")

        id_m = results.get("id_metrics", {})
        ood_m = results.get("ood_metrics", {})
        gap = results.get("gap", {})

        print(f"\n  {'Metric':25s} {'ID':>10s} {'OOD':>10s} {'Gap':>10s} {'Rel Gap':>10s}")
        print(f"  {'-'*65}")
        for k in self.ks:
            key = f"recall@{k}"
            id_v = id_m.get(key, 0)
            ood_v = ood_m.get(key, 0)
            g = gap.get(f"{key}_gap", 0)
            rg = gap.get(f"{key}_relative_gap", 0)
            print(f"  {key:25s} {id_v:>10.4f} {ood_v:>10.4f} {g:>+10.4f} {rg:>10.1%}")

        id_v = id_m.get("mrr", 0)
        ood_v = ood_m.get("mrr", 0)
        print(f"  {'mrr':25s} {id_v:>10.4f} {ood_v:>10.4f}")

        rg = results.get("routing_gap", {})
        if rg:
            print(f"\n  Routing Behavior:")
            print(f"    ID avg dims:  {rg.get('id_avg_dims', 0):.1f}")
            print(f"    OOD avg dims: {rg.get('ood_avg_dims', 0):.1f}")
            print(f"    Dim increase for OOD: {rg.get('dim_increase_for_ood', 0):+.1f}")

            r10_gap = gap.get("recall@10_relative_gap", 0)
            if abs(r10_gap) < 0.1:
                print("  -> GENERALIZES WELL: <10% relative gap on OOD queries")
            elif abs(r10_gap) < 0.25:
                print("  -> MODERATE generalization: 10-25% relative gap")
            else:
                print("  -> POOR generalization: >25% relative gap on OOD queries")

    def save_results(self, results: Dict, path: str):
        """Save OOD results."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        def convert(o):
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, set):
                return sorted(o)
            if isinstance(o, dict):
                return {str(k): convert(v) for k, v in o.items()}
            if isinstance(o, list):
                return [convert(v) for v in o]
            return o

        with open(path, "w") as f:
            json.dump(convert(results), f, indent=2)
