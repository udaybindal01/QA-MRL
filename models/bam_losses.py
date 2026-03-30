"""
BAM Training Losses v3 — Query-Only Bloom.

Key conceptual fix from v2:
  Bloom level is a property of the QUERY's cognitive intent, not the
  document's content. A passage about photosynthesis has no inherent Bloom
  level — the same passage can answer "What is photosynthesis?" (Remember)
  and "Design an experiment to test photosynthesis rate" (Create).

  v2 weighted negatives by |bloom(query) - bloom(doc)|, which is wrong:
  it punishes the model for retrieving topically-correct documents whose
  *source-assigned* Bloom label differs from the query's.

  v3 removes all document Bloom labels from the loss. The Bloom signal
  enters ONLY through per-truncation query Bloom classifiers, which force
  the encoder to organize embedding dimensions by cognitive complexity.

Components:
  1. Standard contrastive loss (no Bloom weighting on negatives)
  2. BloomAlignedMRLLoss with D/sqrt(d_k) reweighting + per-truncation
     query Bloom classifiers
  3. CurriculumScheduledLoss: temperature annealing over training
  4. TruncationPolicyLoss: contrastive + efficiency + entropy bonus

The D/sqrt(d_k) reweighting:
  Standard MRL weights all truncation equally: w_k = 1/K
  Our reweighting: w_k = D / sqrt(d_k) / Sigma(D / sqrt(d_j))
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ContrastiveLoss(nn.Module):
    """
    Standard InfoNCE contrastive loss — no Bloom weighting on negatives.

    Bloom is a query property, not a document property. Negatives are
    penalized only by their semantic similarity, not by a spurious
    document-level Bloom label.
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_emb, positive_emb,
                negative_embs=None, **kwargs):
        # kwargs accepts and ignores legacy fields like query_bloom,
        # negative_blooms for backward compatibility

        pos_sim = (query_emb * positive_emb).sum(dim=-1) / self.temperature

        if negative_embs is not None:
            neg_sim = torch.bmm(
                negative_embs, query_emb.unsqueeze(-1)
            ).squeeze(-1) / self.temperature

            denom = pos_sim.exp() + neg_sim.exp().sum(dim=-1)
            loss = (-pos_sim + denom.log()).mean()
        else:
            sim = torch.mm(query_emb, positive_emb.t()) / self.temperature
            labels = torch.arange(sim.size(0), device=sim.device)
            loss = F.cross_entropy(sim, labels)

        return loss, {"contrastive": loss.item()}


class BloomAlignedMRLLoss(nn.Module):
    """
    MRL loss with two components:

    1. D/sqrt(d_k) reweighting: higher weight on smaller truncations
       forces discriminative info into lower dimensions

    2. Per-truncation query Bloom classifiers: each truncation level
       trained to predict the QUERY's Bloom level from the truncated
       query embedding. This forces the encoder to organize dimensions
       so cognitive complexity is decodable at every truncation level.

       Lower truncation dims capture simpler Bloom levels;
       full dims capture all levels.
    """

    def __init__(self, mrl_dims: List[int], embedding_dim: int = 768,
                 num_bloom_levels: int = 6, temperature: float = 0.05,
                 bloom_cls_weight: float = 0.3):
        super().__init__()
        self.mrl_dims = mrl_dims
        self.D = embedding_dim
        self.temperature = temperature
        self.bloom_cls_weight = bloom_cls_weight

        # D/sqrt(d_k) weights
        raw_weights = [embedding_dim / math.sqrt(d) for d in mrl_dims]
        total = sum(raw_weights)
        self.dim_weights = [w / total for w in raw_weights]

        print(f"  MRL dim weights (D/sqrt(d_k) normalized):")
        for d, w in zip(mrl_dims, self.dim_weights):
            print(f"    d={d:3d}: weight={w:.4f} (raw={embedding_dim/math.sqrt(d):.1f})")

        # Per-truncation QUERY Bloom classifiers
        self.bloom_classifiers = nn.ModuleDict()
        for d in mrl_dims:
            self.bloom_classifiers[str(d)] = nn.Sequential(
                nn.Linear(d, min(128, d)),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(min(128, d), num_bloom_levels),
            )

        # Which Bloom levels each truncation is expected to handle
        self.bloom_targets = {}
        for i, d in enumerate(mrl_dims):
            max_level = min(num_bloom_levels,
                           max(2, (i + 1) * num_bloom_levels // len(mrl_dims)))
            self.bloom_targets[d] = max_level

    def forward(self, query_truncated, positive_truncated, bloom_labels=None):
        """
        Args:
            query_truncated: {dim: [B, d]} truncated query embeddings
            positive_truncated: {dim: [B, d]} truncated positive doc embeddings
            bloom_labels: [B] query Bloom levels (0-indexed). Query-only.
        """
        total_loss = 0.0
        stats = {}

        for d, w in zip(self.mrl_dims, self.dim_weights):
            q = query_truncated[d]
            p = positive_truncated[d]

            # Contrastive at this truncation, reweighted by D/sqrt(d)
            sim = torch.mm(q, p.t()) / self.temperature
            labels = torch.arange(sim.size(0), device=sim.device)
            contrastive = F.cross_entropy(sim, labels)

            # Query Bloom classification at this truncation
            bloom_loss = torch.tensor(0.0, device=q.device)
            bloom_acc = 0.0
            if bloom_labels is not None:
                logits = self.bloom_classifiers[str(d)](q)  # Query only!
                max_level = self.bloom_targets[d]
                mask = bloom_labels < max_level
                if mask.sum() > 0:
                    bloom_loss = F.cross_entropy(logits[mask], bloom_labels[mask])
                    bloom_acc = (logits[mask].argmax(-1) == bloom_labels[mask]).float().mean().item()

            total_loss += w * (contrastive + self.bloom_cls_weight * bloom_loss)

            stats[f"mrl_d{d}_loss"] = contrastive.item()
            stats[f"mrl_d{d}_bloom_acc"] = bloom_acc
            stats[f"mrl_d{d}_weight"] = w

        stats["bloom_aligned_mrl"] = total_loss.item()
        return total_loss, stats


class CurriculumScheduledLoss(nn.Module):
    """
    Wraps contrastive loss with curriculum temperature annealing.

    v3 change: no longer ramps Bloom-distance weight on negatives.
    Instead, temperature annealing: warm start -> target temperature.
    """

    def __init__(self, temperature: float = 0.05, max_temperature: float = 0.1):
        super().__init__()
        self.target_temperature = temperature
        self.max_temperature = max_temperature
        self.base_loss = ContrastiveLoss(temperature=temperature)

    def forward(self, query_emb, positive_emb,
                negative_embs=None,
                curriculum_progress: float = 0.0,
                **kwargs):
        current_temp = self.max_temperature - (self.max_temperature - self.target_temperature) * curriculum_progress
        self.base_loss.temperature = current_temp

        loss, stats = self.base_loss(
            query_emb, positive_emb,
            negative_embs,
        )
        stats["curriculum_temperature"] = current_temp
        stats["curriculum_progress"] = curriculum_progress
        return loss, stats


class TruncationPolicyLoss(nn.Module):
    """
    Reward signal for adaptive truncation policy.

    v3 changes:
    - Removed bloom_align term that used doc Bloom labels
    - Added entropy bonus to prevent policy collapse to single dim
    """

    def __init__(self, mrl_dims: List[int], temperature: float = 0.05,
                 efficiency_weight: float = 0.3):
        super().__init__()
        self.mrl_dims = mrl_dims
        self.temperature = temperature
        self.efficiency_weight = efficiency_weight
        self.max_dim = max(mrl_dims)

    def forward(self, query_adaptive, positive_emb, policy_output,
                negative_embs=None, bloom_labels=None, **kwargs):
        selected_dim = policy_output["selected_dim"]

        # Contrastive with adaptive embedding
        if negative_embs is not None:
            pos_sim = (query_adaptive * positive_emb).sum(-1) / self.temperature
            neg_sim = torch.bmm(negative_embs, query_adaptive.unsqueeze(-1)).squeeze(-1) / self.temperature
            logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            contrastive = F.cross_entropy(logits, labels)
        else:
            sim = torch.mm(query_adaptive, positive_emb.t()) / self.temperature
            labels = torch.arange(sim.size(0), device=sim.device)
            contrastive = F.cross_entropy(sim, labels)

        # Per-Bloom efficiency: lower Bloom levels get stronger pressure to use fewer dims
        # Bloom 1 (Remember) → scale 0.8, Bloom 6 (Create) → scale 0.0
        if bloom_labels is not None:
            bloom_scale = 1.0 - (bloom_labels.float() / 5.0)
            efficiency = (bloom_scale * selected_dim / self.max_dim).mean()
        else:
            efficiency = (selected_dim / self.max_dim).mean()

        total = contrastive + self.efficiency_weight * efficiency

        return total, {
            "policy_contrastive": contrastive.item(),
            "policy_efficiency": efficiency.item(),
            "policy_avg_dim": selected_dim.mean().item(),
        }


class BAMCombinedLoss(nn.Module):
    """
    Full BAM loss v3.

    L = w1 * L_contrastive          (standard, no Bloom weighting)
      + w2 * L_bloom_aligned_mrl     (D/sqrt(d) + query Bloom classifiers)
      + w3 * L_truncation_policy     (contrastive + efficiency + entropy)
      + w4 * L_bloom_classifier      (query Bloom level prediction)

    The Bloom classifier loss trains the inference-time Bloom predictor.
    """

    def __init__(self, config: dict):
        super().__init__()
        mc = config["model"]
        lc = config["training"]["loss"]

        self.weights = {
            "contrastive": lc.get("bloom_contrastive_weight", 1.0),
            "mrl": lc.get("bloom_mrl_weight", 0.5),
            "policy": lc.get("policy_weight", 0.3),
        }

        self.curriculum_contrastive = CurriculumScheduledLoss(
            temperature=0.05,
            max_temperature=0.1,
        )
        self.bloom_mrl = BloomAlignedMRLLoss(
            mrl_dims=mc["mrl_dims"],
            embedding_dim=mc["embedding_dim"],
            bloom_cls_weight=lc.get("bloom_cls_weight", 0.3),
        )
        self.policy_loss = TruncationPolicyLoss(
            mrl_dims=mc["mrl_dims"],
            efficiency_weight=lc.get("efficiency_weight", 0.3),
        )

    def forward(self, query_emb, positive_emb, query_adaptive,
                query_truncated, positive_truncated, policy_output,
                bloom_labels=None, negative_embs=None,
                phase="joint", curriculum_progress=0.0,
                **kwargs):
        """
        Args:
            bloom_labels: query Bloom levels only (0-indexed).
            bloom_classifier_output: output of BloomClassifier (logits, probs).
            kwargs: accepts and ignores legacy fields (negative_blooms, etc.)
        """
        losses = {}
        total = torch.tensor(0.0, device=query_emb.device, requires_grad=True)

        # 1. Curriculum-scheduled contrastive (no Bloom weighting)
        l_cc, cc_stats = self.curriculum_contrastive(
            query_adaptive, positive_emb,
            negative_embs=negative_embs,
            curriculum_progress=curriculum_progress,
        )
        losses.update(cc_stats)
        total = total + self.weights["contrastive"] * l_cc

        # 2. Bloom-aligned MRL with D/sqrt(d) + query Bloom classifiers
        if bloom_labels is not None:
            l_mrl, mrl_stats = self.bloom_mrl(
                query_truncated, positive_truncated, bloom_labels,
            )
            losses.update(mrl_stats)
            total = total + self.weights["mrl"] * l_mrl

        # 3. Truncation policy (Bloom → dim mapping)
        if phase != "mrl_warmup":
            l_pol, pol_stats = self.policy_loss(
                query_adaptive, positive_emb, policy_output,
                negative_embs=negative_embs,
                bloom_labels=bloom_labels,
            )
            losses.update(pol_stats)
            total = total + self.weights["policy"] * l_pol

        # Log the learned Bloom → dim table
        if "bloom_dim_table" in policy_output:
            table = policy_output["bloom_dim_table"]
            bloom_names = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
            for i, name in enumerate(bloom_names):
                if i < len(table):
                    losses[f"dim_{name}"] = table[i].item()

        losses["total"] = total.item()
        return total, losses
