"""
BAM Training Losses v5.

Two improvements over v4:

1. BloomTargetEfficiencyLoss (replaces BloomEfficiencyLoss)
   v4 formula: weight = 1 - bloom/5
   Problem: Remember gets weight 1.0 (crushed to 0 dims), Create gets 0.0 (no pressure).
   Fix: bilateral smooth-L1 loss around per-Bloom target dimensions.
   - Remember target ~150 dims: penalized for going above OR below target
   - Create target ~690 dims: penalized for going below target too
   - All 6 levels have pressure; none are ignored

2. Class-weighted BloomMaskedContrastiveLoss
   Dataset is ~64% Remember. Without reweighting, the loss gradient is dominated
   by Remember, suppressing learning for rare levels (Understand n=28, Evaluate n=72).
   Fix: per-sample loss weighted by inverse-sqrt of Bloom class frequency.
   Sqrt smoothing (not full inverse) avoids 40x weight explosion for Understand.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class BloomMaskedContrastiveLoss(nn.Module):
    """
    InfoNCE loss in query-selected subspace, with inverse-sqrt class weighting.

    Class weights compensate for Bloom level imbalance (~64% Remember).
    Each sample's loss is multiplied by its Bloom class weight before averaging.

    Default weights (inverse-sqrt of approximate dataset frequencies):
      Remember(64%) → 0.30,  Understand(1.5%) → 2.00,  Apply(15.6%) → 0.62,
      Analyze(5.9%) → 1.01,  Evaluate(3.9%) → 1.24,   Create(8.6%) → 0.83
    Normalized to mean=1 so total loss scale is unchanged.
    """

    def __init__(self, temperature: float = 0.05,
                 class_weights: Optional[list] = None):
        super().__init__()
        self.temperature = temperature
        # Default: inverse-sqrt of dataset Bloom frequencies, normalized to mean=1
        default_weights = [0.30, 2.00, 0.62, 1.01, 1.24, 0.83]
        w = torch.tensor(class_weights if class_weights else default_weights,
                         dtype=torch.float)
        self.register_buffer("class_weights", w)

    def forward(
        self,
        query_emb: torch.Tensor,       # [B, D]
        positive_emb: torch.Tensor,    # [B, D]
        query_mask: torch.Tensor,      # [B, D] binary
        negative_embs: Optional[torch.Tensor] = None,  # [B, N, D]
        bloom_labels: Optional[torch.Tensor] = None,   # [B] 0-indexed
    ):
        masked_q = F.normalize(query_emb * query_mask, p=2, dim=-1)
        masked_p = F.normalize(positive_emb * query_mask, p=2, dim=-1)

        if negative_embs is not None:
            mask_exp = query_mask.unsqueeze(1).expand_as(negative_embs)
            masked_n = F.normalize(negative_embs * mask_exp, p=2, dim=-1)
            pos_sim = (masked_q * masked_p).sum(dim=-1) / self.temperature    # [B]
            neg_sim = torch.bmm(masked_n, masked_q.unsqueeze(-1)).squeeze(-1) / self.temperature  # [B, N]
            logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)      # [B, N+1]
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            per_sample_loss = F.cross_entropy(logits, labels, reduction="none")  # [B]
        else:
            sim = torch.mm(masked_q, masked_p.t()) / self.temperature         # [B, B]
            labels = torch.arange(sim.size(0), device=sim.device)
            per_sample_loss = F.cross_entropy(sim, labels, reduction="none")  # [B]

        # Apply class weights if bloom_labels provided
        if bloom_labels is not None:
            w = self.class_weights.to(bloom_labels.device)[bloom_labels]      # [B]
            # Renormalize so weighted mean ≈ unweighted mean in expectation
            w = w * (w.size(0) / w.sum())
            loss = (per_sample_loss * w).mean()
        else:
            loss = per_sample_loss.mean()

        return loss, {"contrastive": loss.item()}


class BloomTargetEfficiencyLoss(nn.Module):
    """
    Bilateral smooth-L1 loss around per-Bloom target dimensions.

    v4 was unidirectional: Remember crushed to near-0, Create ignored.
    v5 is bilateral: every Bloom level is pulled toward its target.
      - Remember below target → penalized (use more dims)
      - Remember above target → penalized (use fewer dims)
      - Create below target  → penalized (use more dims)
      - Create above target  → penalized (use fewer dims)

    Default targets (configurable in bam.yaml → loss.bloom_target_ratios):
      Remember:   0.20 × 768 ≈ 154 dims
      Understand: 0.30 × 768 ≈ 230 dims
      Apply:      0.42 × 768 ≈ 323 dims
      Analyze:    0.56 × 768 ≈ 430 dims
      Evaluate:   0.70 × 768 ≈ 538 dims
      Create:     0.85 × 768 ≈ 653 dims

    smooth_l1_loss with beta=0.1 is used (linear beyond 0.1, quadratic within)
    so large deviations don't dominate.
    """

    EMBEDDING_DIM = 768
    # Default target ratios per Bloom level (0-indexed)
    DEFAULT_TARGETS = [0.20, 0.30, 0.42, 0.56, 0.70, 0.85]

    def __init__(self, bloom_target_ratios: Optional[list] = None):
        super().__init__()
        ratios = bloom_target_ratios if bloom_target_ratios else self.DEFAULT_TARGETS
        targets = torch.tensor(ratios, dtype=torch.float) * self.EMBEDDING_DIM
        self.register_buffer("bloom_targets", targets)  # [6]

    def forward(
        self,
        continuous_dim: torch.Tensor,  # [B] float
        bloom_labels: torch.Tensor,    # [B] int 0-indexed
    ):
        target_dim = self.bloom_targets.to(bloom_labels.device)[bloom_labels]  # [B]
        # smooth_l1 with beta=0.1 in ratio space to keep loss scale ~O(1)
        loss = F.smooth_l1_loss(
            continuous_dim / self.EMBEDDING_DIM,
            target_dim / self.EMBEDDING_DIM,
            beta=0.1,
        )
        return loss, {
            "efficiency": loss.item(),
            "avg_dim": continuous_dim.mean().item(),
        }


class MRLAnchorRegularizationLoss(nn.Module):
    """
    Keeps the encoder sharp at MRL anchor dims during BAM fine-tuning.

    The MRL baseline was trained at exactly [64, 128, 256, 384, 512, 768].
    BAM routes queries to arbitrary dims like 152 or 318 — dimensions the
    encoder was never explicitly trained on. This loss adds contrastive
    regularization at the anchor dims so the encoder maintains strong
    representations at the known-good truncation points.

    Applied to the FULL (unmasked) query and positive embeddings.
    Weighted by D/sqrt(d) so smaller truncations get more emphasis
    (they carry disproportionately less information).
    """

    def __init__(self, mrl_dims: List[int], temperature: float = 0.05):
        super().__init__()
        self.mrl_dims = mrl_dims
        self.temperature = temperature
        # D/sqrt(d_k) weights, normalized to sum=1
        D = max(mrl_dims)
        raw = [D / math.sqrt(d) for d in mrl_dims]
        total = sum(raw)
        self.dim_weights = [w / total for w in raw]

    def forward(
        self,
        query_full_emb: torch.Tensor,     # [B, 768] unmasked encoder output
        positive_full_emb: torch.Tensor,  # [B, 768] unmasked encoder output
    ):
        total = torch.tensor(0.0, device=query_full_emb.device)
        stats = {}

        for d, w in zip(self.mrl_dims, self.dim_weights):
            q = F.normalize(query_full_emb[:, :d], p=2, dim=-1)
            p = F.normalize(positive_full_emb[:, :d], p=2, dim=-1)
            sim = torch.mm(q, p.t()) / self.temperature
            labels = torch.arange(sim.size(0), device=sim.device)
            loss = F.cross_entropy(sim, labels)
            total = total + w * loss
            stats[f"mrl_anchor_d{d}"] = loss.item()

        stats["mrl_anchor"] = total.item()
        return total, stats


class BAMCombinedLoss(nn.Module):
    """
    BAM v5 combined loss.

    total = contrastive_weight * L_contrastive   (class-weighted masked InfoNCE)
          + efficiency_weight  * L_efficiency    (bilateral target loss)
          + mrl_anchor_weight  * L_mrl_anchor    (keep encoder sharp at anchor dims)
    """

    def __init__(self, config: dict):
        super().__init__()
        mc = config["model"]
        lc = config["training"]["loss"]
        self.contrastive_weight = lc.get("contrastive_weight", 1.0)
        self.efficiency_weight = lc.get("efficiency_weight", 0.3)
        self.mrl_anchor_weight = lc.get("mrl_anchor_weight", 0.2)

        self.contrastive = BloomMaskedContrastiveLoss(
            temperature=0.05,
            class_weights=lc.get("class_weights", None),
        )
        self.efficiency = BloomTargetEfficiencyLoss(
            bloom_target_ratios=lc.get("bloom_target_ratios", None),
        )
        self.mrl_anchor = MRLAnchorRegularizationLoss(
            mrl_dims=mc["mrl_dims"],
            temperature=0.05,
        )

    def forward(
        self,
        query_emb: torch.Tensor,          # [B, 768] full encoder output
        positive_emb: torch.Tensor,       # [B, 768] full encoder output
        query_mask: torch.Tensor,         # [B, 768] binary router mask
        continuous_dim: torch.Tensor,     # [B] float from router
        bloom_labels: torch.Tensor,       # [B] int 0-indexed
        negative_embs: Optional[torch.Tensor] = None,  # [B, N, 768]
    ):
        l_c, c_stats = self.contrastive(
            query_emb, positive_emb, query_mask, negative_embs, bloom_labels
        )
        l_e, e_stats = self.efficiency(continuous_dim, bloom_labels)
        l_a, a_stats = self.mrl_anchor(query_emb, positive_emb)

        total = (self.contrastive_weight * l_c
                 + self.efficiency_weight * l_e
                 + self.mrl_anchor_weight * l_a)

        stats = {}
        stats.update(c_stats)
        stats.update(e_stats)
        stats.update(a_stats)
        stats["total"] = total.item()

        # Per-Bloom avg dims for logging
        bloom_names = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
        for b, name in enumerate(bloom_names):
            mask_b = bloom_labels == b
            if mask_b.sum() > 0:
                stats[f"dim_{name}"] = continuous_dim[mask_b].mean().item()

        return total, stats
