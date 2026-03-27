"""
Query-Adaptive Dimension Router v2.

Improvements over v1:
  1. Remove gradient detach — let encoder learn routing-friendly representations
  2. Per-dimension soft weighting within active groups (not just binary on/off)
  3. Temperature annealing — start soft, gradually harden selections
  4. Better initialization — start with all groups equally likely
  5. Contrastive routing loss — encourage different queries to route differently
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from einops import rearrange


class SoftRouter(nn.Module):
    """Continuous dimension weighting."""

    def __init__(self, embedding_dim=768, hidden_dim=256,
                 use_learner_features=False, learner_feature_dim=32):
        super().__init__()
        self.embedding_dim = embedding_dim
        input_dim = embedding_dim
        if use_learner_features:
            input_dim += learner_feature_dim
            self.learner_proj = nn.Linear(6, learner_feature_dim)

        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate[-2].bias, 2.0)

    def forward(self, query_embedding, learner_features=None, **kwargs):
        # NO detach — let gradients flow to encoder
        gate_input = query_embedding
        if learner_features is not None and hasattr(self, "learner_proj"):
            gate_input = torch.cat([gate_input, self.learner_proj(learner_features)], dim=-1)

        mask = self.gate(gate_input)
        return {
            "mask": mask,
            "sparsity": mask.mean(dim=-1),
            "diversity": mask.std(dim=0).mean(),
            "active_dims": (mask > 0.5).float().sum(dim=-1).mean(),
            "active_groups": torch.tensor(0.0),
            "group_usage": torch.zeros(1),
            "load_balance_loss": torch.tensor(0.0, device=mask.device),
        }


class GroupRouter(nn.Module):
    """
    Group routing v2: hard group selection + soft intra-group weighting.

    Key improvements:
    - No gradient detach (encoder co-adapts with router)
    - Soft per-dim weights within active groups
    - Temperature parameter for annealing
    """

    def __init__(self, embedding_dim=768, num_groups=8, hidden_dim=256,
                 temperature=1.0, min_active_groups=2, max_active_groups=6,
                 use_learner_features=False, learner_feature_dim=32):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_groups = num_groups
        self.group_size = embedding_dim // num_groups
        self.temperature = temperature
        self.min_active_groups = min_active_groups
        self.max_active_groups = max_active_groups

        assert embedding_dim % num_groups == 0

        input_dim = embedding_dim
        if use_learner_features:
            input_dim += learner_feature_dim
            self.learner_proj = nn.Linear(6, learner_feature_dim)

        # Group selection network
        self.router_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_groups),
        )

        # Per-dimension importance within each group
        # This lets the router do fine-grained weighting, not just binary group on/off
        self.dim_importance = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Sigmoid(),
        )
        # Initialize dim_importance to output ~1.0 (pass-through initially)
        nn.init.constant_(self.dim_importance[-2].bias, 2.0)

        self.group_importance = nn.Parameter(torch.zeros(num_groups))

    def forward(self, query_embedding, learner_features=None, hard=None):
        if hard is None:
            hard = not self.training

        # NO detach — gradients flow back to encoder
        gate_input = query_embedding
        if learner_features is not None and hasattr(self, "learner_proj"):
            gate_input = torch.cat([gate_input, self.learner_proj(learner_features)], dim=-1)

        # Group-level selection
        group_logits = self.router_net(gate_input) + self.group_importance

        if self.training:
            group_probs = self._gumbel_sigmoid(group_logits, self.temperature)
        else:
            group_probs = torch.sigmoid(group_logits)

        if hard:
            group_selection = self._constrained_selection(group_probs)
        else:
            group_selection = group_probs

        # Expand group selection to full mask
        group_mask = group_selection.unsqueeze(-1).expand(-1, -1, self.group_size)
        group_mask = rearrange(group_mask, "b g gs -> b (g gs)")

        # Per-dimension importance (fine-grained weighting within groups)
        dim_weights = self.dim_importance(gate_input)  # [B, D]

        # Combined mask: group selection * per-dim importance
        mask = group_mask * dim_weights

        active_groups = (group_probs > 0.5).float().sum(dim=-1).mean()
        group_usage = (group_probs > 0.5).float().mean(dim=0)
        load_balance = self._load_balance_loss(group_probs)

        return {
            "mask": mask,
            "group_mask": group_mask,
            "dim_weights": dim_weights,
            "group_probs": group_probs,
            "group_selection": group_selection,
            "active_groups": active_groups,
            "group_usage": group_usage,
            "load_balance_loss": load_balance,
            "active_dims": (mask > 0.5).float().sum(dim=-1).mean(),
        }

    def _gumbel_sigmoid(self, logits, temperature):
        if not self.training:
            return torch.sigmoid(logits)
        u = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
        gumbel = -torch.log(-torch.log(u))
        return torch.sigmoid((logits + gumbel) / temperature)

    def _constrained_selection(self, probs):
        sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
        hard_mask = torch.zeros_like(probs)
        for i in range(self.min_active_groups):
            hard_mask.scatter_(1, sorted_indices[:, i:i+1], 1.0)
        for i in range(self.min_active_groups, self.max_active_groups):
            activate = (sorted_probs[:, i] > 0.5).float().unsqueeze(-1)
            hard_mask.scatter_(1, sorted_indices[:, i:i+1], activate)
        return hard_mask - probs.detach() + probs  # STE

    def _load_balance_loss(self, group_probs):
        f = group_probs.mean(dim=0)
        target = torch.ones_like(f) / self.num_groups
        return self.num_groups * (f * target).sum()


class HybridRouter(nn.Module):
    """Hard group selection + soft intra-group weighting (wraps GroupRouter)."""

    def __init__(self, embedding_dim=768, num_groups=8, hidden_dim=256,
                 temperature=1.0, min_active_groups=2, max_active_groups=6,
                 use_learner_features=False, learner_feature_dim=32):
        super().__init__()
        # GroupRouter v2 already has per-dim weighting, so this is just an alias
        self.inner = GroupRouter(
            embedding_dim, num_groups, hidden_dim, temperature,
            min_active_groups, max_active_groups,
            use_learner_features, learner_feature_dim,
        )

    def forward(self, query_embedding, learner_features=None, hard=None):
        return self.inner(query_embedding, learner_features, hard)


class DocumentRouter(nn.Module):
    """Lightweight router for documents (applied at indexing time)."""

    def __init__(self, embedding_dim=768, num_groups=8, hidden_dim=128):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = embedding_dim // num_groups
        self.router = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_groups),
            nn.Sigmoid(),
        )

    def forward(self, doc_embedding):
        group_weights = self.router(doc_embedding)
        mask = group_weights.unsqueeze(-1).expand(-1, -1, self.group_size)
        mask = rearrange(mask, "b g gs -> b (g gs)")
        return {"mask": mask, "group_weights": group_weights}


def build_router(config: dict) -> nn.Module:
    """Factory to build router from config."""
    rc = config["model"]["router"]
    emb_dim = config["model"]["embedding_dim"]
    common = dict(
        embedding_dim=emb_dim,
        hidden_dim=rc["hidden_dim"],
        use_learner_features=rc.get("use_learner_features", False),
        learner_feature_dim=rc.get("learner_feature_dim", 32),
    )
    if rc["type"] == "soft":
        return SoftRouter(**common)
    elif rc["type"] == "group":
        return GroupRouter(
            **common,
            num_groups=rc["num_groups"],
            temperature=rc["temperature"],
            min_active_groups=rc["min_active_groups"],
            max_active_groups=rc["max_active_groups"],
        )
    elif rc["type"] == "hybrid":
        return HybridRouter(
            **common,
            num_groups=rc["num_groups"],
            temperature=rc["temperature"],
            min_active_groups=rc["min_active_groups"],
            max_active_groups=rc["max_active_groups"],
        )
    else:
        raise ValueError(f"Unknown router type: {rc['type']}")