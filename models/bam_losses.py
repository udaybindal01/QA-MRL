"""
BAM Training Losses v2.

Novel components:
  1. BloomConditionedContrastiveLoss: negatives weighted by Bloom distance
  2. BloomAlignedMRLLoss with D/sqrt(d_k) reweighting per truncation slice
  3. CurriculumScheduledLoss: progressive hard negative difficulty
  4. TruncationPolicyLoss: reward signal for adaptive truncation

The D/sqrt(d_k) reweighting:
  Standard MRL weights all truncation equally: w_k = 1/K
  Our reweighting: w_k = D / sqrt(d_k) / Σ(D / sqrt(d_j))

  This gives HIGHER weight to smaller truncation dims, forcing the
  model to pack discriminative information into lower dimensions.
  For D=768:
    d=64:  w = 768/8   = 96    (highest weight)
    d=128: w = 768/11.3 = 67.9
    d=256: w = 768/16  = 48
    d=384: w = 768/19.6 = 39.2
    d=512: w = 768/22.6 = 34.0
    d=768: w = 768/27.7 = 27.7  (lowest weight)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class BloomConditionedContrastiveLoss(nn.Module):
    """
    Negatives weighted by Bloom-level distance from query.

    w_i = 1 + lambda * |bloom(query) - bloom(neg_i)|

    Same-topic-wrong-Bloom negatives get penalized more heavily,
    forcing the encoder to distinguish cognitive complexity.
    """

    def __init__(self, temperature: float = 0.05, bloom_weight: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.bloom_weight = bloom_weight

    def forward(self, query_emb, positive_emb, query_bloom=None,
                negative_embs=None, negative_blooms=None):

        pos_sim = (query_emb * positive_emb).sum(dim=-1) / self.temperature

        if negative_embs is not None:
            neg_sim = torch.bmm(
                negative_embs, query_emb.unsqueeze(-1)
            ).squeeze(-1) / self.temperature

            if negative_blooms is not None and query_bloom is not None and self.bloom_weight > 0:
                bloom_dist = (query_bloom.unsqueeze(-1).float() -
                             negative_blooms.float()).abs()
                weights = 1.0 + self.bloom_weight * bloom_dist
                weighted_neg = weights * neg_sim.exp()
                denom = pos_sim.exp() + weighted_neg.sum(dim=-1)
            else:
                denom = pos_sim.exp() + neg_sim.exp().sum(dim=-1)

            loss = (-pos_sim + denom.log()).mean()
        else:
            sim = torch.mm(query_emb, positive_emb.t()) / self.temperature
            labels = torch.arange(sim.size(0), device=sim.device)
            loss = F.cross_entropy(sim, labels)

        return loss, {"bloom_contrastive": loss.item()}


class BloomAlignedMRLLoss(nn.Module):
    """
    MRL loss with two novel components:

    1. D/sqrt(d_k) reweighting: higher weight on smaller truncations
       forces discriminative info into lower dimensions

    2. Per-level Bloom classifiers: each truncation trained to capture
       specific cognitive levels (lower dims -> simpler concepts)
    """

    def __init__(self, mrl_dims: List[int], embedding_dim: int = 768,
                 num_bloom_levels: int = 6, temperature: float = 0.05):
        super().__init__()
        self.mrl_dims = mrl_dims
        self.D = embedding_dim
        self.temperature = temperature

        # D/sqrt(d_k) weights
        raw_weights = [embedding_dim / math.sqrt(d) for d in mrl_dims]
        total = sum(raw_weights)
        self.dim_weights = [w / total for w in raw_weights]

        print(f"  MRL dim weights (D/sqrt(d_k) normalized):")
        for d, w in zip(mrl_dims, self.dim_weights):
            print(f"    d={d:3d}: weight={w:.4f} (raw={embedding_dim/math.sqrt(d):.1f})")

        # Per-truncation Bloom classifiers
        self.bloom_classifiers = nn.ModuleDict()
        for d in mrl_dims:
            self.bloom_classifiers[str(d)] = nn.Sequential(
                nn.Linear(d, min(128, d)),
                nn.GELU(),
                nn.Linear(min(128, d), num_bloom_levels),
            )

        # Bloom level target per truncation:
        # d=64  -> should capture levels 0-1 (Remember/Understand)
        # d=768 -> should capture all levels
        self.bloom_targets = {}
        for i, d in enumerate(mrl_dims):
            max_level = min(num_bloom_levels,
                           max(2, (i + 1) * num_bloom_levels // len(mrl_dims)))
            self.bloom_targets[d] = max_level

    def forward(self, query_truncated, positive_truncated, bloom_labels=None):
        total_loss = 0.0
        stats = {}

        for d, w in zip(self.mrl_dims, self.dim_weights):
            q = query_truncated[d]
            p = positive_truncated[d]

            # Contrastive at this truncation, reweighted by D/sqrt(d)
            sim = torch.mm(q, p.t()) / self.temperature
            labels = torch.arange(sim.size(0), device=sim.device)
            contrastive = F.cross_entropy(sim, labels)

            # Bloom classification
            bloom_loss = torch.tensor(0.0, device=q.device)
            bloom_acc = 0.0
            if bloom_labels is not None:
                logits = self.bloom_classifiers[str(d)](q)
                max_level = self.bloom_targets[d]
                mask = bloom_labels < max_level
                if mask.sum() > 0:
                    bloom_loss = F.cross_entropy(logits[mask], bloom_labels[mask])
                    bloom_acc = (logits[mask].argmax(-1) == bloom_labels[mask]).float().mean().item()

            # Weighted loss for this truncation
            total_loss += w * (contrastive + 0.3 * bloom_loss)

            stats[f"mrl_d{d}_loss"] = contrastive.item()
            stats[f"mrl_d{d}_bloom_acc"] = bloom_acc
            stats[f"mrl_d{d}_weight"] = w

        stats["bloom_aligned_mrl"] = total_loss.item()
        return total_loss, stats


class CurriculumScheduledLoss(nn.Module):
    """
    Wraps the Bloom-conditioned contrastive loss with curriculum scheduling.

    The bloom_weight parameter increases over training:
    - Early: bloom_weight ≈ 0 (standard contrastive, easy negatives)
    - Late:  bloom_weight = max_weight (heavy Bloom penalty, hard negatives)

    This curriculum prevents the model from being overwhelmed by hard
    negatives early in training when representations are still forming.
    """

    def __init__(self, temperature: float = 0.05, max_bloom_weight: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.max_bloom_weight = max_bloom_weight
        self.base_loss = BloomConditionedContrastiveLoss(temperature, bloom_weight=0.0)

    def forward(self, query_emb, positive_emb, query_bloom=None,
                negative_embs=None, negative_blooms=None,
                curriculum_progress: float = 0.0):
        """
        curriculum_progress: 0.0 (start) to 1.0 (end of training)
        """
        # Ramp bloom weight with curriculum
        current_weight = self.max_bloom_weight * curriculum_progress
        self.base_loss.bloom_weight = current_weight

        loss, stats = self.base_loss(
            query_emb, positive_emb, query_bloom,
            negative_embs, negative_blooms,
        )
        stats["curriculum_bloom_weight"] = current_weight
        stats["curriculum_progress"] = curriculum_progress
        return loss, stats


class TruncationPolicyLoss(nn.Module):
    """Reward signal for adaptive truncation policy."""

    def __init__(self, mrl_dims: List[int], temperature: float = 0.05,
                 efficiency_weight: float = 0.1):
        super().__init__()
        self.mrl_dims = mrl_dims
        self.temperature = temperature
        self.efficiency_weight = efficiency_weight
        self.max_dim = max(mrl_dims)

    def forward(self, query_adaptive, positive_emb, policy_output,
                bloom_labels=None, negative_embs=None):
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

        # Efficiency: prefer fewer dims
        efficiency = (selected_dim / self.max_dim).mean()

        # Bloom alignment: complex queries should use more dims
        bloom_align = torch.tensor(0.0, device=query_adaptive.device)
        if bloom_labels is not None:
            bloom_norm = bloom_labels.float() / 5.0
            dim_norm = selected_dim / self.max_dim
            bloom_align = F.mse_loss(dim_norm, bloom_norm)

        total = contrastive + self.efficiency_weight * efficiency + 0.2 * bloom_align

        return total, {
            "policy_contrastive": contrastive.item(),
            "policy_efficiency": efficiency.item(),
            "policy_bloom_align": bloom_align.item(),
            "policy_avg_dim": selected_dim.mean().item(),
            "policy_entropy": policy_output["entropy"].item(),
        }


class BAMCombinedLoss(nn.Module):
    """
    Full BAM loss combining all novel components.

    L = w1 * L_curriculum_contrastive  (Bloom-weighted, curriculum-scheduled)
      + w2 * L_bloom_aligned_mrl       (D/sqrt(d) reweighted, per-level classifiers)
      + w3 * L_truncation_policy       (adaptive truncation reward)
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
            max_bloom_weight=lc.get("bloom_negative_weight", 1.0),
        )
        self.bloom_mrl = BloomAlignedMRLLoss(
            mrl_dims=mc["mrl_dims"],
            embedding_dim=mc["embedding_dim"],
        )
        self.policy_loss = TruncationPolicyLoss(
            mrl_dims=mc["mrl_dims"],
            efficiency_weight=lc.get("efficiency_weight", 0.1),
        )

    def forward(self, query_emb, positive_emb, query_adaptive,
                query_truncated, positive_truncated, policy_output,
                bloom_labels=None, negative_embs=None, negative_blooms=None,
                phase="joint", curriculum_progress=0.0):

        losses = {}
        total = torch.tensor(0.0, device=query_emb.device, requires_grad=True)

        # 1. Curriculum-scheduled Bloom contrastive
        l_cc, cc_stats = self.curriculum_contrastive(
            query_adaptive, positive_emb, bloom_labels,
            negative_embs, negative_blooms,
            curriculum_progress=curriculum_progress,
        )
        losses.update(cc_stats)
        total = total + self.weights["contrastive"] * l_cc

        # 2. Bloom-aligned MRL with D/sqrt(d) reweighting
        if bloom_labels is not None:
            l_mrl, mrl_stats = self.bloom_mrl(
                query_truncated, positive_truncated, bloom_labels,
            )
            losses.update(mrl_stats)
            total = total + self.weights["mrl"] * l_mrl

        # 3. Truncation policy
        if phase != "mrl_warmup":
            l_pol, pol_stats = self.policy_loss(
                query_adaptive, positive_emb, policy_output,
                bloom_labels, negative_embs,
            )
            losses.update(pol_stats)
            total = total + self.weights["policy"] * l_pol

        losses["total"] = total.item()
        return total, losses