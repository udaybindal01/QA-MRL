"""
Full evaluation pipeline v3.

v3 changes:
- Removed bloom_aligned_recall@10 (invalid: docs have no Bloom level)
- Added bootstrap confidence intervals for Bloom-stratified metrics
- Bloom-stratified metrics use query Bloom labels only
"""

import time
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional
from collections import defaultdict
from tqdm import tqdm


BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply",
               4: "Analyze", 5: "Evaluate", 6: "Create"}


def bootstrap_ci(hits, n_bootstrap=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for a binary metric."""
    rng = np.random.RandomState(seed)
    n = len(hits)
    if n == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(hits, size=n, replace=True)
        means.append(sample.mean())
    means = np.sort(means)
    lo = means[int((1 - ci) / 2 * n_bootstrap)]
    hi = means[int((1 + ci) / 2 * n_bootstrap)]
    return float(hits.mean()), float(lo), float(hi)


class FullEvaluator:

    def __init__(self, config: dict):
        self.config = config
        self.ks = config["evaluation"]["retrieval_ks"]

    @torch.no_grad()
    def evaluate_model(self, model, test_data_path: str, corpus_path: str,
                       tokenizer, device: torch.device,
                       mrl_truncation_dims: List[int] = None,
                       compute_bootstrap: bool = True) -> Dict[str, float]:
        """
        Evaluate by:
        1. Encode the full corpus
        2. For each test query, find its positive_id in the corpus
        3. Retrieve top-K from corpus and check if positive_id is retrieved
        """
        model.eval()

        # 1. Load and encode corpus
        print("  Loading corpus...")
        corpus = []
        with open(corpus_path) as f:
            for line in f:
                corpus.append(json.loads(line.strip()))

        corpus_id_to_idx = {p["id"]: i for i, p in enumerate(corpus)}
        print(f"  Corpus: {len(corpus)} passages")

        print("  Encoding corpus...")
        corpus_embs = self._encode_texts(
            model, [p["text"] for p in corpus], tokenizer, device,
            is_query=False, batch_size=128,
        )
        print(f"  Corpus embeddings: {corpus_embs.shape}")

        # 2. Load test queries with their positive_id mappings
        print("  Loading test queries...")
        test_samples = []
        with open(test_data_path) as f:
            for line in f:
                test_samples.append(json.loads(line.strip()))
        print(f"  Test queries: {len(test_samples)}")

        # Filter to queries whose positive_id exists in corpus
        valid_samples = []
        for s in test_samples:
            pid = s.get("positive_id", "")
            if pid in corpus_id_to_idx:
                valid_samples.append(s)
        print(f"  Valid queries (positive in corpus): {len(valid_samples)}")

        if len(valid_samples) == 0:
            print("  ERROR: No valid query-passage pairs found!")
            return {}

        # 3. Encode queries
        print("  Encoding queries...")
        query_texts = [s["query"] for s in valid_samples]
        query_embs, query_masks, latencies, query_dims = self._encode_queries(
            model, query_texts, tokenizer, device,
            learner_blooms=[s["bloom_level"] for s in valid_samples],
        )

        # 4. Get ground truth indices
        gt_indices = np.array([corpus_id_to_idx[s["positive_id"]] for s in valid_samples])
        query_blooms = np.array([s["bloom_level"] for s in valid_samples])
        query_subjects = [s.get("subject", "") for s in valid_samples]
        query_topics = [s.get("topic", "") for s in valid_samples]

        # 5. Compute similarity and retrieve
        print("  Computing similarities...")
        all_rankings = []
        chunk_size = 256
        for i in range(0, len(query_embs), chunk_size):
            q_chunk = query_embs[i:i+chunk_size].to(device)
            c_embs = corpus_embs.to(device)
            sim = torch.mm(q_chunk, c_embs.t())
            topk_indices = sim.topk(max(self.ks), dim=-1).indices.cpu().numpy()
            all_rankings.append(topk_indices)
            del sim
            torch.cuda.empty_cache() if device.type == "cuda" else None

        rankings = np.concatenate(all_rankings, axis=0)

        # 6. Compute metrics
        metrics = {}
        N = len(valid_samples)

        # Standard metrics
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

        # NDCG@10
        ndcgs = []
        for i in range(N):
            for j, idx in enumerate(rankings[i, :10]):
                if idx == gt_indices[i]:
                    ndcgs.append(1.0 / np.log2(j + 2))
                    break
            else:
                ndcgs.append(0.0)
        metrics["ndcg@10"] = float(np.mean(ndcgs))

        # 7. Bloom-stratified metrics (query Bloom only)
        for level in range(1, 7):
            mask = query_blooms == level
            if mask.sum() == 0:
                continue
            name = BLOOM_NAMES[level]
            level_rankings = rankings[mask]
            level_gt = gt_indices[mask]
            n_level = int(mask.sum())

            for k in self.ks:
                topk = level_rankings[:, :k]
                hits = np.array([level_gt[i] in topk[i] for i in range(n_level)])
                metrics[f"bloom_{name}_recall@{k}"] = float(hits.mean())

                # Bootstrap CI for R@10
                if k == 10 and compute_bootstrap:
                    mean, lo, hi = bootstrap_ci(hits.astype(float))
                    metrics[f"bloom_{name}_recall@10_ci_lo"] = lo
                    metrics[f"bloom_{name}_recall@10_ci_hi"] = hi
                    metrics[f"bloom_{name}_n"] = n_level

        # v3: NO bloom_aligned_recall@10 — documents don't have Bloom levels.
        # A retrieved document is "good" if it's the right passage, regardless
        # of any source-assigned cognitive label.

        # 8. Efficiency — per-query dims and per-Bloom averages
        if query_dims is not None:
            metrics["avg_active_dims"] = float(query_dims.float().mean().item())
            for level in range(1, 7):
                mask = query_blooms == level
                if mask.sum() > 0:
                    name = BLOOM_NAMES[level]
                    metrics[f"bloom_{name}_avg_dim"] = float(query_dims[mask].float().mean().item())
        elif query_masks is not None:
            metrics["avg_active_dims"] = float((query_masks > 0.5).float().sum(dim=-1).mean().item())
        if latencies:
            metrics["avg_latency_ms"] = float(np.mean(latencies))

        # 9. MRL dimension comparison (baseline only)
        if mrl_truncation_dims and not hasattr(model, "query_router"):
            print("  Computing MRL truncation comparisons...")
            for d in mrl_truncation_dims:
                q_trunc = F.normalize(query_embs[:, :d], p=2, dim=-1)
                c_trunc = F.normalize(corpus_embs[:, :d], p=2, dim=-1)

                trunc_rankings = []
                for i in range(0, len(q_trunc), chunk_size):
                    chunk = q_trunc[i:i+chunk_size].to(device)
                    sim = torch.mm(chunk, c_trunc.to(device).t())
                    topk = sim.topk(max(self.ks), dim=-1).indices.cpu().numpy()
                    trunc_rankings.append(topk)
                trunc_rankings = np.concatenate(trunc_rankings)

                for k in self.ks:
                    topk = trunc_rankings[:, :k]
                    hits = np.array([gt_indices[i] in topk[i] for i in range(N)])
                    metrics[f"mrl_d{d}_recall@{k}"] = float(hits.mean())

        # Print summary
        self._print_summary(metrics, N, query_blooms, compute_bootstrap)
        return metrics

    def _encode_texts(self, model, texts, tokenizer, device,
                       is_query=False, batch_size=128) -> torch.Tensor:
        """Encode a list of texts."""
        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="    encoding", leave=False):
            batch_texts = texts[i:i+batch_size]
            enc = tokenizer(batch_texts, padding=True, truncation=True,
                           max_length=256, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}

            if is_query and hasattr(model, "encode_queries"):
                out = model.encode_queries(enc["input_ids"], enc["attention_mask"])
                all_embs.append(out["masked_embedding"].cpu())
            elif not is_query and hasattr(model, "encode_documents"):
                out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
                all_embs.append(out["masked_embedding"].cpu())
            else:
                out = model(enc["input_ids"], enc["attention_mask"])
                all_embs.append(out["full"].cpu())

        return torch.cat(all_embs)

    def _encode_queries(self, model, query_texts, tokenizer, device,
                         learner_blooms=None):
        """Encode queries with optional learner features.

        Returns:
            embs: [N, D] query embeddings (masked if BAM)
            masks: [N, D] binary masks or None
            latencies: list of per-query latency in ms
            discrete_dims: [N] active dim count per query or None
        """
        all_embs, all_masks, all_dims = [], [], []
        latencies = []
        batch_size = 64

        for i in range(0, len(query_texts), batch_size):
            batch = query_texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                           max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}

            # Build learner features (one-hot [B, 6]) for BAM
            lf = None
            if learner_blooms and hasattr(model, "query_router"):
                blooms = learner_blooms[i:i+len(batch)]
                lf = torch.zeros(len(batch), 6)
                for j, bl in enumerate(blooms):
                    assert 1 <= bl <= 6, (
                        f"Bloom level must be 1-6 (1-indexed), got {bl}."
                    )
                    lf[j, bl - 1] = 1.0
                lf = lf.to(device)

            t0 = time.time()
            if hasattr(model, "encode_queries"):
                out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                           learner_features=lf)
                all_embs.append(out["masked_embedding"].cpu())
                all_masks.append(out["mask"].cpu())
                # discrete_dim: number of active dims per query
                if "discrete_dim" in out:
                    all_dims.append(out["discrete_dim"].detach().cpu())
                elif "mask" in out:
                    # Fallback: count active dims from mask
                    all_dims.append(out["mask"].sum(dim=-1).cpu())
            else:
                out = model(enc["input_ids"], enc["attention_mask"])
                all_embs.append(out["full"].cpu())
            latencies.append((time.time() - t0) * 1000 / len(batch))

        embs = torch.cat(all_embs)
        masks = torch.cat(all_masks) if all_masks else None
        discrete_dims = torch.cat(all_dims) if all_dims else None
        return embs, masks, latencies, discrete_dims

    def _print_summary(self, metrics, N, query_blooms, show_ci=True):
        print(f"\n  === Results (N={N}) ===")
        print(f"  R@1:    {metrics.get('recall@1', 0):.4f}")
        print(f"  R@5:    {metrics.get('recall@5', 0):.4f}")
        print(f"  R@10:   {metrics.get('recall@10', 0):.4f}")
        print(f"  R@50:   {metrics.get('recall@50', 0):.4f}")
        print(f"  MRR:    {metrics.get('mrr', 0):.4f}")
        print(f"  NDCG@10:{metrics.get('ndcg@10', 0):.4f}")
        if "avg_active_dims" in metrics:
            print(f"  Active dims: {metrics['avg_active_dims']:.0f}")

        print(f"\n  Bloom-Stratified R@10 (query Bloom only):")
        for level in range(1, 7):
            name = BLOOM_NAMES[level]
            n = int((query_blooms == level).sum())
            r10 = metrics.get(f"bloom_{name}_recall@10", 0)
            if n > 0:
                ci_str = ""
                if show_ci:
                    lo = metrics.get(f"bloom_{name}_recall@10_ci_lo", 0)
                    hi = metrics.get(f"bloom_{name}_recall@10_ci_hi", 0)
                    ci_str = f"  95% CI=[{lo:.3f}, {hi:.3f}]"
                print(f"    {name:12s} (n={n:4d}): R@10={r10:.4f}{ci_str}")

    def save_results(self, metrics, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in metrics.items()}, f, indent=2)
