"""
Full evaluation pipeline v4.

v4 changes:
- Mask-aware eval: Option B (scattered mask) skips prefix slicing, uses masked dot product
- avg_active_dims_per_bloom logged from hard mask (active_dims) when discrete_dim unavailable
- bloom_routing_entropy per query logged when bloom_probs returned by model
- _encode_queries returns extended tuple: (..., active_dims, bloom_probs)
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


def permutation_test(hits_a: np.ndarray, hits_b: np.ndarray,
                     n_permutations: int = 5000, seed: int = 42) -> float:
    """
    Two-sample permutation test for difference in R@k between two Bloom levels.

    Tests H0: mean(hits_a) == mean(hits_b) under random group assignment.
    Returns p-value (two-tailed). Use alpha=0.05 for significance.

    Handles unequal sample sizes. If either group has <5 samples, returns
    p=1.0 (underpowered — report CI instead).
    """
    if len(hits_a) < 5 or len(hits_b) < 5:
        return 1.0
    observed_diff = abs(hits_a.mean() - hits_b.mean())
    combined = np.concatenate([hits_a, hits_b])
    n_a = len(hits_a)
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(combined)
        diff = abs(perm[:n_a].mean() - perm[n_a:].mean())
        if diff >= observed_diff:
            count += 1
    return count / n_permutations


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

        Routing modes (auto-detected from model output):
          Option A (prefix mask, discrete_dim returned): grouped sliced similarity
          Option B (scattered mask, active_dims returned): masked dot product
          MRL baseline (no router): full-dim dot product
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

        # 2. Load test queries
        print("  Loading test queries...")
        test_samples = []
        with open(test_data_path) as f:
            for line in f:
                test_samples.append(json.loads(line.strip()))
        print(f"  Test queries: {len(test_samples)}")

        # Load cached predicted Bloom labels if available (same labels used at training).
        # Fall back to ground-truth bloom_level (1-indexed → 0-indexed) if no cache.
        bloom_cache_path = test_data_path + ".bloom_cache.json"
        bloom_cache = None
        if os.path.exists(bloom_cache_path):
            import json as _json
            with open(bloom_cache_path) as _f:
                _cache = _json.load(_f)
            if len(_cache) == len(test_samples):
                bloom_cache = _cache
                print(f"  Using predicted Bloom labels from cache: {bloom_cache_path}")
            else:
                print(f"  WARNING: bloom cache size {len(_cache)} != {len(test_samples)}, "
                      f"falling back to ground-truth bloom_level.")

        valid_samples = [
            s for s in test_samples
            if s.get("positive_id", "") in corpus_id_to_idx
        ]
        print(f"  Valid queries (positive in corpus): {len(valid_samples)}")

        if len(valid_samples) == 0:
            print("  ERROR: No valid query-passage pairs found!")
            return {}

        # Build per-sample Bloom labels (0-indexed) for routing.
        # Prefer cached predicted labels (matches training); fall back to ground truth.
        if bloom_cache is not None:
            # cache is indexed over all test_samples; we need indices for valid_samples
            sample_to_cache_idx = {id(s): i for i, s in enumerate(test_samples)}
            learner_blooms_0idx = [
                bloom_cache[sample_to_cache_idx[id(s)]] for s in valid_samples
            ]
        else:
            learner_blooms_0idx = [s["bloom_level"] - 1 for s in valid_samples]

        # 3. Encode queries (extended return with active_dims and bloom_probs)
        print("  Encoding queries...")
        query_texts = [s["query"] for s in valid_samples]
        (query_embs, query_masks, latencies,
         query_dims, query_full_embs,
         query_active_dims, query_bloom_probs) = self._encode_queries(
            model, query_texts, tokenizer, device,
            learner_blooms_0idx=learner_blooms_0idx,
        )

        gt_indices = np.array([corpus_id_to_idx[s["positive_id"]] for s in valid_samples])
        # query_blooms is 1-indexed for display/stratification (BLOOM_NAMES keys are 1-6).
        # Use ground-truth bloom_level for stratification reporting regardless of routing source,
        # so Bloom-stratified metrics always reflect the true query taxonomy.
        query_blooms = np.array([s["bloom_level"] for s in valid_samples])

        # 4. Similarity and retrieval
        print("  Computing similarities...")
        N = len(valid_samples)
        chunk_size = 256
        rankings = np.zeros((N, max(self.ks)), dtype=np.int64)

        use_prefix_slicing = (query_dims is not None and query_full_embs is not None)
        # Option B: static scattered mask — corpus pre-masked per Bloom level.
        # Both training and eval use normalize(q * mask_b) · normalize(corpus * mask_b),
        # matching the masked-both-sides training objective.
        use_static_mask = (
            query_full_embs is not None and
            hasattr(model, "bloom_mask_head") and
            hasattr(model.bloom_mask_head, "bloom_logit")
        )

        if use_prefix_slicing:
            # Option A: normalize(q[:k]) · normalize(c[:k]) — prefix slicing.
            # Matches training: masked_q · masked_doc (both sides masked with prefix mask).
            k_values = query_dims.round().long()
            for k_val in k_values.unique():
                k = int(k_val.item())
                k = max(1, min(k, corpus_embs.size(1)))
                group_idx = (k_values == k_val).nonzero(as_tuple=True)[0]
                c_k = F.normalize(corpus_embs[:, :k], p=2, dim=-1).to(device)
                for i in range(0, len(group_idx), chunk_size):
                    chunk = group_idx[i:i + chunk_size]
                    q_k = F.normalize(query_full_embs[chunk, :k], p=2, dim=-1).to(device)
                    sim = torch.mm(q_k, c_k.t())
                    topk = sim.topk(max(self.ks), dim=-1).indices.cpu().numpy()
                    rankings[chunk.numpy()] = topk
                del c_k
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        elif use_static_mask:
            # Option B: per-level masked corpus.
            # normalize(q * mask_b) · normalize(corpus * mask_b) for each Bloom level b.
            bloom_logit_w = model.bloom_mask_head.bloom_logit.weight.detach().cpu()  # [6, 768]
            bloom_masks_bin = (torch.sigmoid(bloom_logit_w) > 0.5).float()           # [6, 768]
            routing_labels = torch.tensor(learner_blooms_0idx, dtype=torch.long)      # [N]
            for b in range(6):
                level_idx = (routing_labels == b).nonzero(as_tuple=True)[0]
                if len(level_idx) == 0:
                    continue
                mask_b = bloom_masks_bin[b].to(device)                                # [768]
                c_b = F.normalize(corpus_embs.to(device) * mask_b, p=2, dim=-1)      # [C, 768]
                for i in range(0, len(level_idx), chunk_size):
                    chunk = level_idx[i:i + chunk_size]
                    q_b = F.normalize(
                        query_full_embs[chunk].to(device) * mask_b, p=2, dim=-1
                    )
                    sim = torch.mm(q_b, c_b.t())
                    topk = sim.topk(max(self.ks), dim=-1).indices.cpu().numpy()
                    rankings[chunk.numpy()] = topk
                del c_b
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        else:
            # MRL baseline: full-dim dot product.
            for i in range(0, N, chunk_size):
                q_chunk = query_embs[i:i + chunk_size].to(device)
                sim = torch.mm(q_chunk, corpus_embs.to(device).t())
                topk = sim.topk(max(self.ks), dim=-1).indices.cpu().numpy()
                rankings[i:i + chunk_size] = topk
                del sim
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        # 5. Compute metrics
        metrics = {}

        for k in self.ks:
            topk = rankings[:, :k]
            hits = np.array([gt_indices[i] in topk[i] for i in range(N)])
            metrics[f"recall@{k}"] = float(hits.mean())

        mrrs = []
        for i in range(N):
            found = np.where(rankings[i] == gt_indices[i])[0]
            mrrs.append(1.0 / (found[0] + 1) if len(found) > 0 else 0.0)
        metrics["mrr"] = float(np.mean(mrrs))

        ndcgs = []
        for i in range(N):
            for j, idx in enumerate(rankings[i, :10]):
                if idx == gt_indices[i]:
                    ndcgs.append(1.0 / np.log2(j + 2))
                    break
            else:
                ndcgs.append(0.0)
        metrics["ndcg@10"] = float(np.mean(ndcgs))

        # 6. Bloom-stratified metrics
        bloom_entropies = None
        if query_bloom_probs is not None:
            # Compute per-query routing entropy H(bloom_probs)
            p = query_bloom_probs.clamp(min=1e-9)
            bloom_entropies = -(p * p.log()).sum(dim=-1).numpy()

        # Build corpus bloom_level lookup for bloom_consistent_recall
        # Counts a hit if any top-K doc has the same Bloom level as the query's positive.
        # Addresses single-positive evaluation noise (reviewer D): a query may retrieve
        # equally valid documents that happen not to be the labeled positive.
        corpus_bloom = np.array([c.get("bloom_level", 0) for c in corpus])
        query_positive_blooms = np.array(
            [corpus[corpus_id_to_idx[s["positive_id"]]].get("bloom_level", 0)
             for s in valid_samples]
        )
        has_bloom_in_corpus = (corpus_bloom > 0).any()

        # Collect per-level hits arrays for significance testing (reviewer J)
        level_hits_r10 = {}

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

                if k == 10:
                    level_hits_r10[level] = hits.astype(float)
                    if compute_bootstrap:
                        mean, lo, hi = bootstrap_ci(hits.astype(float))
                        metrics[f"bloom_{name}_recall@10_ci_lo"] = lo
                        metrics[f"bloom_{name}_recall@10_ci_hi"] = hi
                        metrics[f"bloom_{name}_n"] = n_level

                    # bloom_consistent_recall@10: counts hit if any top-10 doc has same
                    # Bloom level as the query's positive (addresses single-positive noise).
                    if has_bloom_in_corpus:
                        pos_blooms_level = query_positive_blooms[mask]
                        consistent_hits = np.array([
                            any(corpus_bloom[level_rankings[i, j]] == pos_blooms_level[i]
                                for j in range(10))
                            for i in range(n_level)
                        ])
                        metrics[f"bloom_{name}_consistent_recall@10"] = float(consistent_hits.mean())

            if bloom_entropies is not None:
                metrics[f"bloom_{name}_routing_entropy"] = float(bloom_entropies[mask].mean())

        # Pairwise significance testing across consecutive Bloom levels (reviewer J)
        # Tests H0: adjacent Bloom levels have the same R@10 under random group assignment.
        if compute_bootstrap and len(level_hits_r10) >= 2:
            levels_sorted = sorted(level_hits_r10.keys())
            for i in range(len(levels_sorted) - 1):
                la, lb = levels_sorted[i], levels_sorted[i + 1]
                p_val = permutation_test(level_hits_r10[la], level_hits_r10[lb])
                key = f"bloom_{BLOOM_NAMES[la]}_vs_{BLOOM_NAMES[lb]}_pvalue"
                metrics[key] = float(p_val)

        # 7. Efficiency metrics
        # Option A (prefix mask): avg_active_dims = contiguous prefix length.
        #   FAISS can index these as compact subvectors of size avg_active_dims.
        #   Retrieval speedup: linear in dim reduction (768→427 ≈ 1.8× faster).
        # Option B (scattered mask): avg_active_dims_scattered = non-contiguous active dims.
        #   IMPORTANT: scattered dims CANNOT be indexed as FAISS sub-vectors.
        #   Full 768-dim corpus index is required; savings come from sparse dot product only.
        #   Compute saving at query time = fraction of zero dims (768-active)/768.
        if query_dims is not None:
            # Option A: prefix mask
            metrics["avg_active_dims"] = float(query_dims.float().mean().item())
            metrics["efficiency_mode"] = "prefix_contiguous"
            metrics["sparse_ratio"] = float(
                1.0 - query_dims.float().mean().item() / 768.0
            )
            for level in range(1, 7):
                mask = query_blooms == level
                if mask.sum() > 0:
                    name = BLOOM_NAMES[level]
                    metrics[f"bloom_{name}_avg_dim"] = float(
                        query_dims[mask].float().mean().item()
                    )
        elif query_active_dims is not None:
            # Option B: scattered mask — note that this is NOT FAISS-compatible sub-indexing
            metrics["avg_active_dims"] = float(query_active_dims.float().mean().item())
            metrics["avg_active_dims_scattered"] = metrics["avg_active_dims"]
            metrics["efficiency_mode"] = "scattered_non_contiguous"
            metrics["sparse_ratio"] = float(
                1.0 - query_active_dims.float().mean().item() / 768.0
            )
            metrics["faiss_subindex_compatible"] = False
            for level in range(1, 7):
                mask = query_blooms == level
                if mask.sum() > 0:
                    name = BLOOM_NAMES[level]
                    metrics[f"bloom_{name}_avg_dim"] = float(
                        query_active_dims[mask].float().mean().item()
                    )
        elif query_masks is not None:
            # Soft mask fallback
            metrics["avg_active_dims"] = float(
                (query_masks > 0.5).float().sum(dim=-1).mean().item()
            )

        if latencies:
            metrics["avg_latency_ms"] = float(np.mean(latencies))

        # 8. MRL dimension comparison (baseline only)
        if mrl_truncation_dims and not hasattr(model, "query_router"):
            print("  Computing MRL truncation comparisons...")
            for d in mrl_truncation_dims:
                q_trunc = F.normalize(query_embs[:, :d], p=2, dim=-1)
                c_trunc = F.normalize(corpus_embs[:, :d], p=2, dim=-1)

                trunc_rankings = []
                for i in range(0, len(q_trunc), chunk_size):
                    chunk = q_trunc[i:i + chunk_size].to(device)
                    sim = torch.mm(chunk, c_trunc.to(device).t())
                    topk = sim.topk(max(self.ks), dim=-1).indices.cpu().numpy()
                    trunc_rankings.append(topk)
                trunc_rankings = np.concatenate(trunc_rankings)

                for k in self.ks:
                    topk = trunc_rankings[:, :k]
                    hits = np.array([gt_indices[i] in topk[i] for i in range(N)])
                    metrics[f"mrl_d{d}_recall@{k}"] = float(hits.mean())

        self._print_summary(metrics, N, query_blooms, compute_bootstrap)
        return metrics

    def _encode_texts(self, model, texts, tokenizer, device,
                      is_query=False, batch_size=128) -> torch.Tensor:
        """Encode a list of texts."""
        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="    encoding", leave=False):
            batch_texts = texts[i:i + batch_size]
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
                        learner_blooms_0idx=None):
        """
        Encode queries with optional Bloom routing.

        learner_blooms_0idx: list of int, 0-indexed Bloom levels (0=Remember … 5=Create).
            These should be the PREDICTED bloom labels (same source as training),
            not ground-truth labels, to avoid train/eval mismatch.

        Returns:
            embs:          [N, D] masked embeddings (BAM) or full embeddings (MRL)
            masks:         [N, D] masks or None
            latencies:     list of per-query latency in ms
            discrete_dims: [N] prefix dim per query (Option A) or None
            full_embs:     [N, D] full encoder outputs (Option A only) or None
            active_dims:   [N] count of active dims (Option B) or None
            bloom_probs:   [N, 6] softmax routing distribution or None
        """
        all_embs, all_masks, all_dims, all_full_embs = [], [], [], []
        all_active_dims, all_bloom_probs = [], []
        latencies = []
        batch_size = 64

        for i in range(0, len(query_texts), batch_size):
            batch = query_texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}

            lf = None
            if learner_blooms_0idx and hasattr(model, "query_router"):
                blooms_0 = learner_blooms_0idx[i:i + len(batch)]
                lf = torch.zeros(len(batch), 6)
                for j, bl in enumerate(blooms_0):
                    assert 0 <= bl <= 5, f"Bloom level must be 0-5 (0-indexed), got {bl}."
                    lf[j, bl] = 1.0
                lf = lf.to(device)

            t0 = time.time()
            if hasattr(model, "encode_queries"):
                out = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                           learner_features=lf)
                all_embs.append(out["masked_embedding"].cpu())
                all_masks.append(out["mask"].cpu())
                if "full_embedding" in out:
                    all_full_embs.append(out["full_embedding"].detach().cpu())
                if "discrete_dim" in out:
                    all_dims.append(out["discrete_dim"].detach().cpu())
                if "active_dims" in out:
                    all_active_dims.append(out["active_dims"].detach().cpu())
                if "bloom_probs" in out and out["bloom_probs"] is not None:
                    all_bloom_probs.append(out["bloom_probs"].detach().cpu())
            else:
                out = model(enc["input_ids"], enc["attention_mask"])
                all_embs.append(out["full"].cpu())
            latencies.append((time.time() - t0) * 1000 / len(batch))

        embs = torch.cat(all_embs)
        masks = torch.cat(all_masks) if all_masks else None
        discrete_dims = torch.cat(all_dims) if all_dims else None
        full_embs = torch.cat(all_full_embs) if all_full_embs else None
        active_dims = torch.cat(all_active_dims) if all_active_dims else None
        bloom_probs = torch.cat(all_bloom_probs) if all_bloom_probs else None
        return embs, masks, latencies, discrete_dims, full_embs, active_dims, bloom_probs

    def _print_summary(self, metrics, N, query_blooms, show_ci=True):
        print(f"\n  === Results (N={N}) ===")
        print(f"  R@1:    {metrics.get('recall@1', 0):.4f}")
        print(f"  R@5:    {metrics.get('recall@5', 0):.4f}")
        print(f"  R@10:   {metrics.get('recall@10', 0):.4f}")
        print(f"  R@50:   {metrics.get('recall@50', 0):.4f}")
        print(f"  MRR:    {metrics.get('mrr', 0):.4f}")
        print(f"  NDCG@10:{metrics.get('ndcg@10', 0):.4f}")
        if "avg_active_dims" in metrics:
            mode = metrics.get("efficiency_mode", "")
            sparse = metrics.get("sparse_ratio", 0.0)
            faiss_ok = metrics.get("faiss_subindex_compatible", True)
            faiss_note = "" if faiss_ok else " [scattered — NOT FAISS sub-index]"
            print(f"  Active dims: {metrics['avg_active_dims']:.0f} / 768  "
                  f"(sparse_ratio={sparse:.2f}{faiss_note})")

        print(f"\n  Bloom-Stratified R@10 (query Bloom only):")
        levels_printed = []
        for level in range(1, 7):
            name = BLOOM_NAMES[level]
            n = int((query_blooms == level).sum())
            r10 = metrics.get(f"bloom_{name}_recall@10", 0)
            avg_dim = metrics.get(f"bloom_{name}_avg_dim", None)
            entropy = metrics.get(f"bloom_{name}_routing_entropy", None)
            consistent = metrics.get(f"bloom_{name}_consistent_recall@10", None)
            if n > 0:
                ci_str = ""
                if show_ci:
                    lo = metrics.get(f"bloom_{name}_recall@10_ci_lo", 0)
                    hi = metrics.get(f"bloom_{name}_recall@10_ci_hi", 0)
                    ci_str = f"  95% CI=[{lo:.3f}, {hi:.3f}]"
                dim_str = f"  dim={avg_dim:.0f}" if avg_dim is not None else ""
                ent_str = f"  H={entropy:.3f}" if entropy is not None else ""
                con_str = f"  cons_R@10={consistent:.3f}" if consistent is not None else ""
                print(f"    {name:12s} (n={n:4d}): R@10={r10:.4f}{ci_str}{dim_str}{ent_str}{con_str}")
                levels_printed.append(level)
        # Significance: adjacent Bloom level pairs
        for i in range(len(levels_printed) - 1):
            la, lb = levels_printed[i], levels_printed[i + 1]
            pkey = f"bloom_{BLOOM_NAMES[la]}_vs_{BLOOM_NAMES[lb]}_pvalue"
            if pkey in metrics:
                p = metrics[pkey]
                sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
                print(f"      {BLOOM_NAMES[la]} vs {BLOOM_NAMES[lb]}: p={p:.3f} [{sig}]")

    def save_results(self, metrics, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in metrics.items()
                },
                f, indent=2,
            )
