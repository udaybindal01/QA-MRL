"""
Dimension Diagnostic Analysis - the GO/NO-GO experiment.

Validates that different queries activate different dimensions.
Three methods: gradient attribution, leave-one-group-out, mutual information.
"""

import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm

BLOOM_NAMES = {0: "Remember", 1: "Understand", 2: "Apply",
               3: "Analyze", 4: "Evaluate", 5: "Create"}


class DimensionDiagnostics:

    def __init__(self, embedding_dim=768, num_groups=8):
        self.embedding_dim = embedding_dim
        self.num_groups = num_groups
        self.group_size = embedding_dim // num_groups

    def gradient_attribution(self, model, dataloader, device, num_queries=1000):
        """
        Per-dimension importance via d(sim)/d(query_embedding).
        Returns importance arrays grouped by Bloom level.
        """
        model.eval()
        by_bloom = defaultdict(list)
        by_subject = defaultdict(list)
        count = 0

        for batch in tqdm(dataloader, desc="Gradient attribution"):
            if count >= num_queries:
                break

            q_ids = batch["query_input_ids"].to(device)
            q_mask = batch["query_attention_mask"].to(device)
            p_ids = batch["positive_input_ids"].to(device)
            p_mask = batch["positive_attention_mask"].to(device)
            blooms = batch["bloom_label"].numpy()
            subjects = batch["subject_label"].numpy()
            B = q_ids.size(0)

            q_out = model(q_ids, q_mask)
            q_emb = q_out["full"]
            q_emb.retain_grad()

            with torch.no_grad():
                p_out = model(p_ids, p_mask)
                p_emb = p_out["full"]

            sim = (q_emb * p_emb).sum(dim=-1)
            sim.sum().backward()

            importance = q_emb.grad.abs().detach().cpu().numpy()
            for i in range(min(B, num_queries - count)):
                by_bloom[int(blooms[i])].append(importance[i])
                by_subject[int(subjects[i])].append(importance[i])
                count += 1
            model.zero_grad()

        avg_bloom = {k: np.mean(v, axis=0) for k, v in by_bloom.items()}
        avg_subject = {k: np.mean(v, axis=0) for k, v in by_subject.items()}
        stats = self._compute_stats(avg_bloom, avg_subject)

        return {
            "importance_by_bloom": avg_bloom,
            "importance_by_subject": avg_subject,
            "stats": stats,
        }

    def _compute_stats(self, by_bloom, by_subject):
        stats = {}
        if not by_bloom:
            return stats

        bloom_imp = np.stack(list(by_bloom.values()))
        cross_var = bloom_imp.var(axis=0).mean()
        within_var = np.mean([v.var() for v in by_bloom.values()])
        stats["bloom_cross_category_variance"] = float(cross_var)
        stats["bloom_within_category_variance"] = float(within_var)
        stats["bloom_variance_ratio"] = float(cross_var / (within_var + 1e-10))

        # Per-level top group
        for level, imp in by_bloom.items():
            gi = imp.reshape(self.num_groups, self.group_size).mean(axis=1)
            stats[f"bloom_{level}_top_group"] = int(np.argmax(gi))
            stats[f"bloom_{level}_group_variance"] = float(gi.var())

        # Top-100 dim overlap
        top_k = 100
        levels = sorted(by_bloom.keys())
        overlaps = []
        for i, l1 in enumerate(levels):
            t1 = set(np.argsort(by_bloom[l1])[-top_k:])
            for l2 in levels[i+1:]:
                t2 = set(np.argsort(by_bloom[l2])[-top_k:])
                overlaps.append(len(t1 & t2) / top_k)
        if overlaps:
            stats["avg_bloom_dim_overlap"] = float(np.mean(overlaps))
            stats["min_bloom_dim_overlap"] = float(np.min(overlaps))

        return stats

    @torch.no_grad()
    def leave_one_group_out(self, model, dataloader, device):
        """Zero out each group and measure per-Bloom degradation."""
        model.eval()
        baseline = self._eval_with_mask(model, dataloader, device, mask=None)

        group_results = {}
        for g in range(self.num_groups):
            mask = torch.ones(self.embedding_dim, device=device)
            mask[g * self.group_size:(g+1) * self.group_size] = 0.0
            scores = self._eval_with_mask(model, dataloader, device, mask=mask)
            degradation = {k: baseline[k] - scores[k] for k in baseline}
            group_results[f"group_{g}"] = {
                "scores": scores, "degradation": degradation,
                "dims": f"{g*self.group_size}-{(g+1)*self.group_size}",
            }

        return {"baseline": baseline, "group_results": group_results}

    def _eval_with_mask(self, model, dataloader, device, mask=None):
        all_q, all_p, all_blooms = [], [], []
        for batch in dataloader:
            q_out = model(batch["query_input_ids"].to(device),
                          batch["query_attention_mask"].to(device))
            p_out = model(batch["positive_input_ids"].to(device),
                          batch["positive_attention_mask"].to(device))
            qe, pe = q_out["full"], p_out["full"]
            if mask is not None:
                qe = F.normalize(qe * mask, p=2, dim=-1)
                pe = F.normalize(pe * mask, p=2, dim=-1)
            all_q.append(qe.cpu())
            all_p.append(pe.cpu())
            all_blooms.append(batch["bloom_label"].numpy())

        q = torch.cat(all_q)
        p = torch.cat(all_p)
        blooms = np.concatenate(all_blooms)
        sim = torch.mm(q, p.t())
        n = sim.size(0)

        topk = sim.topk(10, dim=-1).indices
        correct = (topk == torch.arange(n).unsqueeze(-1)).any(dim=-1).float()
        scores = {"overall_recall@10": float(correct.mean())}

        for bl in range(6):
            m = blooms == bl
            if m.sum() > 0:
                scores[f"bloom_{BLOOM_NAMES[bl]}_recall@10"] = float(correct.numpy()[m].mean())
        return scores

    def mutual_information_analysis(self, embeddings, bloom_labels, subject_labels, num_bins=20):
        """Estimate MI between each dim group and labels."""
        from sklearn.metrics import mutual_info_score

        mi_bloom = np.zeros(self.num_groups)
        mi_subject = np.zeros(self.num_groups)

        for g in range(self.num_groups):
            ge = embeddings[:, g*self.group_size:(g+1)*self.group_size]
            U, S, Vt = np.linalg.svd(ge - ge.mean(axis=0), full_matrices=False)
            pc1 = U[:, 0]
            binned = np.digitize(pc1, np.linspace(pc1.min(), pc1.max(), num_bins))
            mi_bloom[g] = mutual_info_score(binned, bloom_labels)
            mi_subject[g] = mutual_info_score(binned, subject_labels)

        return {"mi_bloom": mi_bloom, "mi_subject": mi_subject,
                "specialization_index": np.abs(mi_bloom - mi_subject)}

    def save_diagnostics(self, results, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        def convert(o):
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, dict): return {k: convert(v) for k, v in o.items()}
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.integer,)): return int(o)
            return o
        with open(path, "w") as f:
            json.dump(convert(results), f, indent=2)

    def print_summary(self, results):
        stats = results.get("stats", {})
        vr = stats.get("bloom_variance_ratio", 0)
        ol = stats.get("avg_bloom_dim_overlap", 1.0)

        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)
        print(f"  Variance ratio (cross/within): {vr:.4f}")
        if vr > 0.1:
            print("  -> STRONG: Different Bloom levels use different dimensions")
        elif vr > 0.01:
            print("  -> MODERATE: Some differentiation")
        else:
            print("  -> WEAK: Dimensions used similarly across levels")
        print(f"  Top-100 dim overlap: {ol:.2%}")
        for lv in range(6):
            tg = stats.get(f"bloom_{lv}_top_group", "?")
            gv = stats.get(f"bloom_{lv}_group_variance", 0)
            print(f"  Bloom {lv} ({BLOOM_NAMES[lv]}): top_group={tg}, group_var={gv:.6f}")
