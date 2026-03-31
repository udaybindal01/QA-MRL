"""
BAM Training Losses v4 — Two-loss design.

1. BloomMaskedContrastiveLoss
   InfoNCE where both query and positive are masked by the query's binary mask.
   Documents are masked to the query's active dims so similarity is computed
   only in the subspace the router selected for this query.

2. BloomEfficiencyLoss
   Penalizes use of too many dimensions, weighted by Bloom level:
   Remember (bloom=0) → full penalty, Create (bloom=5) → no penalty.
   This pushes simple queries toward compact representations.

BAMCombinedLoss weights both with contrastive_weight and efficiency_weight.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class BloomMaskedContrastiveLoss(nn.Module):
    """
    InfoNCE loss computed in the query-selected subspace.

    Both query and positive (and negatives if provided) are masked by the
    query's binary mask before computing dot products. This forces the model
    to pack retrieval-relevant information into the active dims.
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_emb: torch.Tensor,       # [B, D] full query embedding
        positive_emb: torch.Tensor,    # [B, D] full positive doc embedding
        query_mask: torch.Tensor,      # [B, D] binary mask from router
        negative_embs: Optional[torch.Tensor] = None,  # [B, N, D]
    ):
        masked_q = F.normalize(query_emb * query_mask, p=2, dim=-1)        # [B, D]
        masked_p = F.normalize(positive_emb * query_mask, p=2, dim=-1)     # [B, D]

        if negative_embs is not None:
            # [B, N, D] → mask each negative with the query's mask
            mask_exp = query_mask.unsqueeze(1).expand_as(negative_embs)    # [B, N, D]
            masked_n = F.normalize(negative_embs * mask_exp, p=2, dim=-1)  # [B, N, D]

            pos_sim = (masked_q * masked_p).sum(dim=-1) / self.temperature  # [B]
            neg_sim = torch.bmm(masked_n, masked_q.unsqueeze(-1)).squeeze(-1) / self.temperature  # [B, N]
            logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)    # [B, N+1]
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, labels)
        else:
            # In-batch InfoNCE
            sim = torch.mm(masked_q, masked_p.t()) / self.temperature       # [B, B]
            labels = torch.arange(sim.size(0), device=sim.device)
            loss = F.cross_entropy(sim, labels)

        return loss, {"contrastive": loss.item()}


class BloomEfficiencyLoss(nn.Module):
    """
    Penalizes high-dimensional representations for low-complexity queries.

    efficiency = (continuous_dim / 768) * (1 - bloom_label / 5)

    Remember (bloom=0): weight 1.0 → full pressure to reduce dims
    Create   (bloom=5): weight 0.0 → no pressure (allowed full dims)
    Intermediate: linear interpolation
    """

    def forward(
        self,
        continuous_dim: torch.Tensor,  # [B] float, output of sigmoid * 768
        bloom_labels: torch.Tensor,    # [B] int, 0-indexed
    ):
        bloom_weight = 1.0 - bloom_labels.float() / 5.0    # [B], range [0, 1]
        efficiency = (continuous_dim / 768.0) * bloom_weight  # [B]
        loss = efficiency.mean()
        return loss, {
            "efficiency": loss.item(),
            "avg_dim": continuous_dim.mean().item(),
        }


class BAMCombinedLoss(nn.Module):
    """
    BAM v4 combined loss.

    total = contrastive_weight * L_contrastive
          + efficiency_weight  * L_efficiency
    """

    def __init__(self, config: dict):
        super().__init__()
        lc = config["training"]["loss"]
        self.contrastive_weight = lc.get("contrastive_weight", 1.0)
        self.efficiency_weight = lc.get("efficiency_weight", 0.3)

        self.contrastive = BloomMaskedContrastiveLoss(temperature=0.05)
        self.efficiency = BloomEfficiencyLoss()

    def forward(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        query_mask: torch.Tensor,
        continuous_dim: torch.Tensor,
        bloom_labels: torch.Tensor,
        negative_embs: Optional[torch.Tensor] = None,
    ):
        l_c, c_stats = self.contrastive(query_emb, positive_emb, query_mask, negative_embs)
        l_e, e_stats = self.efficiency(continuous_dim, bloom_labels)

        total = self.contrastive_weight * l_c + self.efficiency_weight * l_e

        stats = {}
        stats.update(c_stats)
        stats.update(e_stats)
        stats["total"] = total.item()

        # Per-Bloom avg dims for logging
        bloom_names = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
        for b, name in enumerate(bloom_names):
            mask_b = bloom_labels == b
            if mask_b.sum() > 0:
                stats[f"dim_{name}"] = continuous_dim[mask_b].mean().item()

        return total, stats
