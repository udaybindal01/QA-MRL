"""
BAM Training Losses v6.

Two-factor efficiency loss replacing all previous designs:

  weight(b) = (1 - b/6) * sqrt(freq(b)),  normalized to mean=1

  - (1 - b/6): cognitive simplicity — Remember=1.0, Create=0.167 (floor, not zero)
  - sqrt(freq(b)): dataset frequency — common classes face more compression
  - Multiplicative: high pressure only when BOTH factors are high
  - No hardcoded targets, no forced monotonicity
  - Frequencies computed from actual training data at runtime

Class weights for contrastive loss:
  weight(b) = 1 / sqrt(freq(b)), normalized to mean=1
  Boosts rare classes (Understand, Evaluate) without 40x explosion.
  Also computed from training data — not hardcoded.
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

        if bloom_labels is not None:
            w = self.class_weights.to(bloom_labels.device)[bloom_labels]
            w = w * (w.size(0) / w.sum())  # renormalize batch weights
            loss = (per_sample_loss * w).mean()
        else:
            loss = per_sample_loss.mean()

        return loss, {"contrastive": loss.item(), "temperature": self.temperature}

    def set_temperature(self, t: float):
        self.temperature = t


class BloomTwoFactorEfficiencyLoss(nn.Module):
    """
    Efficiency penalty: weight(b) = (1 - b/6) * sqrt(freq(b)), normalized.

    Cognitive factor (1 - b/6):
      Remember=1.000, Understand=0.833, Apply=0.667,
      Analyze=0.500, Evaluate=0.333, Create=0.167
      Create gets 0.167 not 0.0 — has a floor, not free pass.

    Frequency factor sqrt(freq(b)):
      Softens the raw frequency so rare classes (Understand 1.5%) aren't
      completely ignored, and common classes (Remember 64%) aren't crushed.

    Multiplicative: a Bloom level only faces strong pressure if it is BOTH
    cognitively simple AND frequently occurring. Rare+complex levels (Evaluate)
    get very low pressure — the model has flexibility to use dims as needed.

    Frequencies computed from training data at runtime (passed in constructor).
    """

    def __init__(self, bloom_frequencies: List[float]):
        super().__init__()
        freqs = torch.tensor(bloom_frequencies, dtype=torch.float).clamp(min=1e-6)
        cognitive = torch.tensor([1.0 - b / 6.0 for b in range(6)], dtype=torch.float)
        weights = cognitive * freqs.sqrt()
        weights = weights / weights.mean()   # normalize so mean=1
        self.register_buffer("weights", weights)

    def forward(
        self,
        continuous_dim: torch.Tensor,   # [B] float
        bloom_labels: torch.Tensor,     # [B] int 0-indexed
    ):
        w = self.weights.to(bloom_labels.device)[bloom_labels]   # [B]
        loss = (continuous_dim / 768.0 * w).mean()
        return loss, {
            "efficiency": loss.item(),
            "avg_dim": continuous_dim.mean().item(),
        }


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

        self.contrastive = BloomMaskedContrastiveLoss(
            temperature=self.temp_start,
            class_weights=cw,
        )
        self.efficiency = BloomTwoFactorEfficiencyLoss(bloom_frequencies=freqs)
        self.mrl_anchor = MRLAnchorRegularizationLoss(
            mrl_dims=mc["mrl_dims"],
            temperature=self.temp_start,
        )

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

        stats = {}
        stats.update(c_stats)
        stats.update(e_stats)
        stats.update(a_stats)
        stats["total"] = total.item()

        bloom_names = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
        for b, name in enumerate(bloom_names):
            mask_b = bloom_labels == b
            if mask_b.sum() > 0:
                stats[f"dim_{name}"] = continuous_dim[mask_b].mean().item()

        return total, stats
