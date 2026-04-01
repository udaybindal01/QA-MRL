"""
BAM Training Losses v7.

Efficiency loss — per-class averaged, cognitive-only weights:

  L_efficiency = (1/C) Σ_b [ cognitive(b) * mean_{i: bloom_i=b}(dim_i / D) ]

  where cognitive(b) = (1 - b/6):
    Remember=1.0, Understand=0.833, ..., Create=0.167

  Key change from v6: per-class averaging (not per-sample).
  In v6, the per-sample mean was dominated by Remember (64% of batches),
  causing the optimizer to aggressively compress Remember purely to satisfy
  the overall loss gradient — "frequency gaming."
  Per-class averaging gives each Bloom level an equal gradient contribution
  regardless of its frequency. The cognitive weight then independently controls
  how much compression pressure each level faces based on its cognitive complexity.

  Frequency is no longer in the per-sample weight. It was redundant once
  per-class averaging is applied at the aggregation level.

Class weights for contrastive loss:
  weight(b) = 1 / sqrt(freq(b)), normalized to mean=1
  Computed from training data — not hardcoded.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class BloomMaskedContrastiveLoss(nn.Module):
    """
    InfoNCE in query-masked subspace with inverse-sqrt class weighting.

    class_weights: [6] tensor, one weight per Bloom level (0-indexed).
    Computed from training data as 1/sqrt(freq), normalized to mean=1.

    temperature is set dynamically via set_temperature() during training.
    Typical schedule: 0.1 (epoch 0) → 0.02 (epoch 14) via cosine annealing.
    High T early = smooth gradients while encoder is still learning.
    Low T late = sharp ranking distribution that specifically improves R@1.
    """

    def __init__(self, temperature: float = 0.05,
                 class_weights: Optional[List[float]] = None):
        super().__init__()
        self.temperature = temperature
        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float)
        else:
            w = torch.ones(6)  # uniform fallback
        self.register_buffer("class_weights", w)

    def forward(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        query_mask: torch.Tensor,
        negative_embs: Optional[torch.Tensor] = None,
        bloom_labels: Optional[torch.Tensor] = None,
    ):
        masked_q = F.normalize(query_emb * query_mask, p=2, dim=-1)
        masked_p = F.normalize(positive_emb * query_mask, p=2, dim=-1)

        if negative_embs is not None:
            mask_exp = query_mask.unsqueeze(1).expand_as(negative_embs)
            masked_n = F.normalize(negative_embs * mask_exp, p=2, dim=-1)
            pos_sim = (masked_q * masked_p).sum(dim=-1) / self.temperature
            neg_sim = torch.bmm(masked_n, masked_q.unsqueeze(-1)).squeeze(-1) / self.temperature
            logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
        else:
            sim = torch.mm(masked_q, masked_p.t()) / self.temperature
            labels = torch.arange(sim.size(0), device=sim.device)
            per_sample_loss = F.cross_entropy(sim, labels, reduction="none")

        # Difficulty weighting: harder samples (higher loss) get more weight.
        # This signals to the router which Bloom levels need more dims —
        # levels with consistently high per-sample loss need more capacity.
        # Detach so difficulty weights don't create a second-order gradient loop.
        difficulty_w = (per_sample_loss.detach() + 1e-6)
        difficulty_w = difficulty_w / difficulty_w.mean()  # normalize to mean=1

        if bloom_labels is not None:
            class_w = self.class_weights.to(bloom_labels.device)[bloom_labels]
            class_w = class_w * (class_w.size(0) / class_w.sum())
            w = class_w * difficulty_w
            loss = (per_sample_loss * w).mean()
        else:
            loss = (per_sample_loss * difficulty_w).mean()

        return loss, {"contrastive": loss.item(), "temperature": self.temperature}

    def set_temperature(self, t: float):
        self.temperature = t


class BloomTwoFactorEfficiencyLoss(nn.Module):
    """
    Efficiency penalty with per-class averaging to prevent frequency gaming.

    L = (1/C) Σ_b [ cognitive(b) * mean_{i: bloom_i=b}(dim_i / D) ]

    cognitive(b) = (1 - b/6): compression pressure decreases with cognitive level.
      Remember=1.0, Understand=0.833, Apply=0.667, Analyze=0.500,
      Evaluate=0.333, Create=0.167 (floor, not zero).

    Per-class averaging ensures each Bloom level contributes equal gradient weight
    to the efficiency loss, regardless of its frequency in the batch.
    This prevents the optimizer from gaming the loss by compressing the majority
    class (Remember=64%) well beyond the semantically optimal point.

    bloom_frequencies is accepted for API compatibility but no longer used in the
    per-sample weights — frequency imbalance is handled by per-class averaging.
    """

    EMBEDDING_DIM = 768.0

    def __init__(self, bloom_frequencies: List[float]):
        super().__init__()
        # Cognitive-only weights: simplicity of each Bloom level
        cognitive = torch.tensor([1.0 - b / 6.0 for b in range(6)], dtype=torch.float)
        self.register_buffer("cognitive_weights", cognitive)

    def forward(
        self,
        continuous_dim: torch.Tensor,   # [B] float
        bloom_labels: torch.Tensor,     # [B] int 0-indexed
    ):
        total = torch.tensor(0.0, device=continuous_dim.device)
        n_classes = 0
        per_class_dims = {}

        for b in range(6):
            mask = (bloom_labels == b)
            if mask.sum() == 0:
                continue
            cw = self.cognitive_weights[b]
            class_mean_dim = continuous_dim[mask].mean()
            total = total + cw * (class_mean_dim / self.EMBEDDING_DIM)
            n_classes += 1
            per_class_dims[b] = class_mean_dim.item()

        loss = total / max(n_classes, 1)
        stats = {"efficiency": loss.item(), "avg_dim": continuous_dim.mean().item()}
        stats.update({f"dim_b{b}": d for b, d in per_class_dims.items()})
        return loss, stats


class RouterDiversityLoss(nn.Module):
    """
    Penalizes all 6 Bloom levels converging to the same dimension.

    Without this, the router can satisfy both efficiency (push all down) and
    contrastive (push all up) losses by assigning the same dim to every level —
    the path of least resistance. This loss rewards differentiation.

    L_diversity = -Var(dim_0, ..., dim_5)

    Normalized by span² so the scale is invariant to [MIN_DIM, MAX_DIM].
    """

    EMBEDDING_DIM = 768.0
    MIN_DIM = 128.0

    def forward(self, all_dims: torch.Tensor) -> torch.Tensor:
        """all_dims: [6] continuous dims for all Bloom levels."""
        span = self.EMBEDDING_DIM - self.MIN_DIM
        normalized = (all_dims - self.MIN_DIM) / span   # [6] in [0, 1]
        var = normalized.var()
        return -var  # maximize variance = minimize negative variance


class MRLAnchorRegularizationLoss(nn.Module):
    """
    InfoNCE at MRL anchor dims to keep encoder sharp during BAM fine-tuning.
    Weighted by D/sqrt(d) — smaller truncations get more emphasis.
    Applied to full (unmasked) query and positive embeddings.
    """

    def __init__(self, mrl_dims: List[int], temperature: float = 0.05):
        super().__init__()
        self.mrl_dims = mrl_dims
        self.temperature = temperature
        D = max(mrl_dims)
        raw = [D / math.sqrt(d) for d in mrl_dims]
        total = sum(raw)
        self.dim_weights = [w / total for w in raw]

    def forward(self, query_full_emb: torch.Tensor, positive_full_emb: torch.Tensor):
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
    BAM v6 combined loss.

    total = contrastive_weight * L_contrastive   (class-weighted masked InfoNCE)
          + efficiency_weight  * L_efficiency    (two-factor: cognitive × frequency)
          + mrl_anchor_weight  * L_mrl_anchor    (keep encoder sharp at anchor dims)

    bloom_frequencies: list of 6 floats summing to ~1.0, computed from training data.
    Passed via config["training"]["loss"]["bloom_frequencies"].
    """

    def __init__(self, config: dict):
        super().__init__()
        mc = config["model"]
        lc = config["training"]["loss"]

        self.contrastive_weight = lc.get("contrastive_weight", 1.0)
        self.efficiency_weight = lc.get("efficiency_weight", 0.3)
        self.mrl_anchor_weight = lc.get("mrl_anchor_weight", 0.2)

        # Bloom frequencies must be injected before trainer is constructed
        freqs = lc.get("bloom_frequencies")
        if freqs is None:
            freqs = [1/6] * 6   # uniform fallback
            print("WARNING: bloom_frequencies not set — using uniform. "
                  "Run compute_bloom_frequencies() in train_bam.py.")

        # Class weights for contrastive: 1/sqrt(freq), normalized
        f = torch.tensor(freqs, dtype=torch.float).clamp(min=1e-6)
        cw = 1.0 / f.sqrt()
        cw = (cw / cw.mean()).tolist()

        ts = lc.get("temperature_schedule", {})
        self.temp_start = ts.get("start", 0.1)
        self.temp_end   = ts.get("end",   0.02)
        # encoder_warmup_epochs: efficiency loss is 0 for this many epochs so the
        # encoder builds quality before routing compression turns on.
        self.encoder_warmup_epochs = lc.get("encoder_warmup_epochs", 0)

        self.diversity_weight = lc.get("diversity_weight", 0.05)

        self.contrastive = BloomMaskedContrastiveLoss(
            temperature=self.temp_start,
            class_weights=cw,
        )
        self.efficiency = BloomTwoFactorEfficiencyLoss(bloom_frequencies=freqs)
        self.mrl_anchor = MRLAnchorRegularizationLoss(
            mrl_dims=mc["mrl_dims"],
            temperature=self.temp_start,
        )
        self.diversity = RouterDiversityLoss()

    def set_epoch(self, epoch: int, total_epochs: int):
        """
        Call at the start of each epoch to update:
          1. Temperature: cosine anneal from temp_start to temp_end.
          2. Efficiency weight: 0 for encoder_warmup_epochs, then config value.
        """
        # Cosine temperature annealing
        progress = epoch / max(total_epochs - 1, 1)
        t = self.temp_end + 0.5 * (self.temp_start - self.temp_end) * (1 + math.cos(math.pi * progress))
        self.contrastive.set_temperature(t)
        self.mrl_anchor.temperature = t

        # Efficiency gate: zero out during encoder warmup
        if epoch < self.encoder_warmup_epochs:
            self._active_efficiency_weight = 0.0
        else:
            self._active_efficiency_weight = self.efficiency_weight

    def forward(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        query_mask: torch.Tensor,
        continuous_dim: torch.Tensor,
        bloom_labels: torch.Tensor,
        negative_embs: Optional[torch.Tensor] = None,
        all_bloom_dims: Optional[torch.Tensor] = None,  # [6] from router._all_dims()
    ):
        l_c, c_stats = self.contrastive(
            query_emb, positive_emb, query_mask, negative_embs, bloom_labels
        )
        l_e, e_stats = self.efficiency(continuous_dim, bloom_labels)
        l_a, a_stats = self.mrl_anchor(query_emb, positive_emb)

        eff_w = getattr(self, "_active_efficiency_weight", self.efficiency_weight)
        total = (self.contrastive_weight * l_c
                 + eff_w * l_e
                 + self.mrl_anchor_weight * l_a)
        e_stats["efficiency_weight_active"] = eff_w

        # Diversity loss: penalize all Bloom levels collapsing to same dim
        d_stats = {}
        if all_bloom_dims is not None:
            l_d = self.diversity(all_bloom_dims)
            total = total + self.diversity_weight * l_d
            d_stats["diversity"] = l_d.item()
            d_stats["dim_variance"] = float(all_bloom_dims.var().item())

        stats = {}
        stats.update(c_stats)
        stats.update(e_stats)
        stats.update(a_stats)
        stats.update(d_stats)
        stats["total"] = total.item()

        bloom_names = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
        for b, name in enumerate(bloom_names):
            mask_b = bloom_labels == b
            if mask_b.sum() > 0:
                stats[f"dim_{name}"] = continuous_dim[mask_b].mean().item()

        return total, stats
