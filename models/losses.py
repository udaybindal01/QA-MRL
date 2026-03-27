"""
Loss functions for QA-MRL training.

  - InfoNCELoss:            Standard contrastive loss
  - MRLContrastiveLoss:     Multi-resolution loss at each truncation point
  - MaskedContrastiveLoss:  Contrastive loss with learned dimension masks (core)
  - GroupSpecializationLoss: Encourage each dim group to specialize
  - SparsityLoss:           Diverse dimension usage across queries
  - BloomClassificationLoss: Auxiliary Bloom level prediction
  - QAMRLLoss:              Combined loss aggregating all the above
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class InfoNCELoss(nn.Module):
    """Standard InfoNCE contrastive loss with in-batch + hard negatives."""

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_emb, positive_emb, negative_embs=None):
        pos_sim = (query_emb * positive_emb).sum(dim=-1) / self.temperature

        if negative_embs is not None:
            neg_sim = torch.bmm(
                negative_embs, query_emb.unsqueeze(-1)
            ).squeeze(-1) / self.temperature
            logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        else:
            logits = torch.mm(query_emb, positive_emb.t()) / self.temperature
            labels = torch.arange(logits.size(0), device=logits.device)

        return F.cross_entropy(logits, labels)


class MRLContrastiveLoss(nn.Module):
    """Multi-resolution contrastive loss: InfoNCE at each truncation point."""

    def __init__(self, mrl_dims: List[int], temperature: float = 0.05,
                 dim_weights: Optional[List[float]] = None):
        super().__init__()
        self.mrl_dims = mrl_dims
        self.infonce = InfoNCELoss(temperature)
        if dim_weights is None:
            dim_weights = [1.0 / len(mrl_dims)] * len(mrl_dims)
        self.dim_weights = dim_weights

    def forward(self, query_truncated, positive_truncated, negative_truncated=None):
        total_loss = 0.0
        per_dim_losses = {}

        for dim, weight in zip(self.mrl_dims, self.dim_weights):
            q = query_truncated[dim]
            p = positive_truncated[dim]
            n = negative_truncated[dim] if negative_truncated else None
            loss = self.infonce(q, p, n)
            total_loss += weight * loss
            per_dim_losses[f"mrl_loss_d{dim}"] = loss.item()

        return total_loss, per_dim_losses


class MaskedContrastiveLoss(nn.Module):
    """
    Contrastive loss with query-adaptive dimension masking.
    Core QA-MRL loss: applies learned mask before similarity computation.
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_emb, positive_emb, query_mask,
                doc_mask=None, negative_embs=None, negative_masks=None):
        masked_q = F.normalize(query_emb * query_mask, p=2, dim=-1)
        if doc_mask is not None:
            masked_p = F.normalize(positive_emb * doc_mask, p=2, dim=-1)
        else:
            masked_p = F.normalize(positive_emb * query_mask, p=2, dim=-1)

        pos_sim = (masked_q * masked_p).sum(dim=-1) / self.temperature

        if negative_embs is not None:
            if negative_masks is not None:
                masked_n = F.normalize(negative_embs * negative_masks, p=2, dim=-1)
            else:
                masked_n = F.normalize(negative_embs * query_mask.unsqueeze(1), p=2, dim=-1)
            neg_sim = torch.bmm(masked_n, masked_q.unsqueeze(-1)).squeeze(-1) / self.temperature
            logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        else:
            logits = torch.mm(masked_q, masked_p.t()) / self.temperature
            labels = torch.arange(logits.size(0), device=logits.device)

        return F.cross_entropy(logits, labels)


class GroupSpecializationLoss(nn.Module):
    """
    Encourage each dimension group to specialize on different attributes.
    Per-group auxiliary classifiers for Bloom level and subject.
    """

    def __init__(self, group_size=96, num_groups=8, num_bloom_levels=6, num_subjects=5):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size

        self.bloom_classifiers = nn.ModuleList([
            nn.Linear(group_size, num_bloom_levels) for _ in range(num_groups)
        ])
        self.subject_classifiers = nn.ModuleList([
            nn.Linear(group_size, num_subjects) for _ in range(num_groups)
        ])

    def forward(self, full_embedding, bloom_labels, subject_labels):
        B = full_embedding.size(0)
        groups = full_embedding.view(B, self.num_groups, self.group_size)

        bloom_losses, subject_losses = [], []
        bloom_accs, subject_accs = [], []

        for g in range(self.num_groups):
            ge = groups[:, g]

            bl = self.bloom_classifiers[g](ge)
            bloom_losses.append(F.cross_entropy(bl, bloom_labels))
            bloom_accs.append((bl.argmax(dim=-1) == bloom_labels).float().mean())

            sl = self.subject_classifiers[g](ge)
            subject_losses.append(F.cross_entropy(sl, subject_labels))
            subject_accs.append((sl.argmax(dim=-1) == subject_labels).float().mean())

        bloom_acc_t = torch.stack(bloom_accs)
        subject_acc_t = torch.stack(subject_accs)
        perf = torch.stack([bloom_acc_t, subject_acc_t], dim=-1)  # [G, 2]

        # Negative variance -> encourage specialization
        specialization_bonus = -perf.var(dim=0).mean()

        total = torch.stack(bloom_losses).mean() + torch.stack(subject_losses).mean() \
                + 0.5 * specialization_bonus

        stats = {
            "bloom_group_accs": bloom_acc_t.detach(),
            "subject_group_accs": subject_acc_t.detach(),
            "specialization_variance": perf.var(dim=0).mean().item(),
        }
        return total, stats


class SparsityLoss(nn.Module):
    """Encourage diverse dimension usage across queries."""

    def __init__(self, target_sparsity: float = 0.6, diversity_weight: float = 0.5):
        super().__init__()
        self.target_sparsity = target_sparsity
        self.diversity_weight = diversity_weight

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        actual = mask.mean(dim=-1)
        sparsity_loss = (actual - self.target_sparsity).pow(2).mean()

        mask_norm = F.normalize(mask, p=2, dim=-1)
        sim = torch.mm(mask_norm, mask_norm.t())
        B = mask.size(0)
        off_diag = sim * (1 - torch.eye(B, device=mask.device))
        diversity_loss = off_diag.mean()

        return sparsity_loss + self.diversity_weight * diversity_loss


class BloomClassificationLoss(nn.Module):
    """Auxiliary: predict Bloom level from embedding."""

    def __init__(self, embedding_dim=768, num_levels=6):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_levels),
        )

    def forward(self, embedding, bloom_labels):
        logits = self.classifier(embedding)
        loss = F.cross_entropy(logits, bloom_labels)
        acc = (logits.argmax(dim=-1) == bloom_labels).float().mean().item()
        return loss, acc


class QAMRLLoss(nn.Module):
    """Combined loss for QA-MRL training, aggregating all sub-losses."""

    def __init__(self, config: dict):
        super().__init__()
        lc = config["training"]["loss"]
        mc = config["model"]

        self.weights = {
            "contrastive": lc["contrastive_weight"],
            "mrl": lc["mrl_weight"],
            "specialization": lc["specialization_weight"],
            "sparsity": lc["sparsity_weight"],
            "bloom": lc["bloom_classification_weight"],
        }

        self.masked_contrastive = MaskedContrastiveLoss(temperature=0.05)
        self.mrl_contrastive = MRLContrastiveLoss(mrl_dims=mc["mrl_dims"], temperature=0.05)
        self.specialization = GroupSpecializationLoss(
            group_size=mc["router"]["group_size"],
            num_groups=mc["router"]["num_groups"],
        )
        self.sparsity = SparsityLoss(target_sparsity=0.6)
        self.bloom_cls = BloomClassificationLoss(embedding_dim=mc["embedding_dim"])

    def forward(self, query_emb, positive_emb, query_mask,
                query_truncated, positive_truncated,
                bloom_labels=None, subject_labels=None,
                doc_mask=None, negative_embs=None, phase="joint"):
        losses = {}
        total = torch.tensor(0.0, device=query_emb.device, requires_grad=True)

        # 1. Masked contrastive (always)
        l_c = self.masked_contrastive(query_emb, positive_emb, query_mask, doc_mask, negative_embs)
        losses["contrastive"] = l_c.item()
        total = total + self.weights["contrastive"] * l_c

        # 2. MRL multi-resolution
        if phase in ("mrl_warmup", "joint"):
            l_mrl, mrl_det = self.mrl_contrastive(query_truncated, positive_truncated)
            losses["mrl"] = l_mrl.item()
            losses.update(mrl_det)
            total = total + self.weights["mrl"] * l_mrl

        # 3. Group specialization
        if phase in ("router_warmup", "joint") and bloom_labels is not None and subject_labels is not None:
            l_spec, spec_stats = self.specialization(query_emb, bloom_labels, subject_labels)
            losses["specialization"] = l_spec.item()
            losses["spec_variance"] = spec_stats["specialization_variance"]
            total = total + self.weights["specialization"] * l_spec

        # 4. Sparsity
        if phase in ("router_warmup", "joint"):
            l_sp = self.sparsity(query_mask)
            losses["sparsity"] = l_sp.item()
            total = total + self.weights["sparsity"] * l_sp

        # 5. Bloom classification
        if bloom_labels is not None:
            l_bl, bl_acc = self.bloom_cls(query_emb, bloom_labels)
            losses["bloom_cls"] = l_bl.item()
            losses["bloom_acc"] = bl_acc
            total = total + self.weights["bloom"] * l_bl

        losses["total"] = total.item()
        return total, losses
