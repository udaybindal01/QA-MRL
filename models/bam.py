"""
Bloom-Aligned Matryoshka (BAM) Model v4.

Routing via BloomDimRouter: 6 learnable scalar parameters (one per Bloom level).
Each scalar → sigmoid → continuous dim in [0, 768] → binary mask.

Inference pipeline:
  Query → Encoder → full 768-dim embedding
       → External Bloom Classifier → predicted Bloom level (1-6)
       → BloomDimRouter → binary mask of size 768
       → masked_emb = normalize(full_emb * mask)
       → FAISS search (only active dims matter for dot product)

Document embeddings always stay full 768-dim.
At retrieval time, dot product query·doc naturally ignores zero-masked dims.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from models.encoder import MRLEncoder


class BloomDimRouter(nn.Module):
    """
    Learns one truncation dimension per Bloom level with guaranteed monotonicity.

    Architecture: cumulative softplus — each level's dim = previous level's dim + Δ_b,
    where Δ_b = softplus(raw_b) * scale. This ensures dim(b) >= dim(b-1) for all b,
    which matches the cognitive hierarchy (higher Bloom → more complex → more dims).

    Base (Remember) starts at MIN_DIM=128 + softplus(raw_0) * scale.
    Total span = EMBEDDING_DIM - MIN_DIM = 640, split across 6 levels.

    No vanilla sigmoid → no independent scalars → no risk of non-monotonic collapse.
    Straight-through estimator for the binary mask.
    """

    EMBEDDING_DIM = 768
    MIN_DIM = 128

    def __init__(self):
        super().__init__()
        # 6 raw params, each controls the *incremental* step above the previous level.
        # Initialize so that dims ≈ [150, 230, 330, 440, 550, 660].
        # Increments: [22, 80, 100, 110, 110, 110] in [128, 768] range.
        # softplus(x) ≈ x for x >> 0; solve softplus(raw) * scale = target_increment.
        # Use scale=640/6 ≈ 107 so each raw≈1 contributes ~107 dims.
        self.scale = (self.EMBEDDING_DIM - self.MIN_DIM) / 6.0  # 106.67
        target_increments = [22.0, 80.0, 100.0, 110.0, 110.0, 110.0]
        self.raw_deltas = nn.Parameter(torch.zeros(6))
        with torch.no_grad():
            for b, inc in enumerate(target_increments):
                # softplus(x) ≈ x + log(1+exp(-x)), solve for x: inv_softplus(y/scale)
                y = inc / self.scale
                # inv_softplus(y) = log(exp(y) - 1) for y > 0
                self.raw_deltas[b] = math.log(math.exp(y) - 1.0 + 1e-8)

    def forward(self, bloom_levels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            bloom_levels: [B] integer Bloom levels (0-indexed, 0=Remember, 5=Create)

        Returns:
            mask: [B, 768] binary mask (1 for active dims, 0 elsewhere)
            continuous_dim: [B] float dim before rounding (used by efficiency loss)
            discrete_dim: [B] float, rounded value with STE for gradient flow
        """
        # cumulative sum of softplus deltas gives monotonically increasing dims
        deltas = F.softplus(self.raw_deltas) * self.scale   # [6], all positive
        cum_dims = self.MIN_DIM + deltas.cumsum(dim=0)      # [6], monotone increasing
        cum_dims = cum_dims.clamp(max=self.EMBEDDING_DIM)

        continuous_dim = cum_dims[bloom_levels]  # [B]

        # Straight-through estimator
        rounded = continuous_dim.round()
        discrete_dim = continuous_dim + (rounded - continuous_dim).detach()

        arange = torch.arange(self.EMBEDDING_DIM, device=bloom_levels.device).float()
        mask = (arange.unsqueeze(0) < rounded.unsqueeze(-1)).float()  # [B, 768]

        return {
            "mask": mask,
            "continuous_dim": continuous_dim,
            "discrete_dim": discrete_dim,
        }

    def get_dim_table(self) -> Dict[int, float]:
        """Returns {bloom_level: expected_dim} for logging. No gradient."""
        with torch.no_grad():
            deltas = F.softplus(self.raw_deltas) * self.scale
            cum_dims = self.MIN_DIM + deltas.cumsum(dim=0)
            cum_dims = cum_dims.clamp(max=self.EMBEDDING_DIM)
        return {b: round(cum_dims[b].item()) for b in range(6)}


class BloomAlignedMRL(nn.Module):
    """
    Bloom-Aligned Matryoshka model v4.

    Key design:
    - BloomDimRouter: 6 scalar params → per-query binary mask
    - Queries are masked; documents stay full 768 dims
    - Dot product query·doc ignores zero-masked query dims naturally
    - Bloom labels from external pre-trained classifier (not trained here)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        mc = config["model"]

        self.encoder = MRLEncoder(
            model_name=mc["backbone"],
            embedding_dim=mc["embedding_dim"],
            mrl_dims=mc["mrl_dims"],
            pooling=mc["pooling"],
            normalize=mc["normalize_embeddings"],
        )

        self.bloom_router = BloomDimRouter()

        self.embedding_dim = mc["embedding_dim"]
        self.mrl_dims = mc["mrl_dims"]

    def encode_queries(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        bloom_labels: Optional[torch.Tensor] = None,
        learner_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode query with Bloom-adaptive masking.

        bloom_labels: [B] int 0-indexed. Required.
        If not provided, falls back to learner_features (one-hot [B,6]).
        If neither provided, defaults to full 768 dims (bloom level 5).
        """
        if bloom_labels is None:
            if learner_features is not None:
                bloom_labels = learner_features.argmax(dim=-1)
            else:
                # Default to full dims (Create level)
                B = input_ids.size(0)
                bloom_labels = torch.full((B,), 5, dtype=torch.long,
                                          device=input_ids.device)

        enc = self.encoder(input_ids, attention_mask, token_type_ids)
        full_emb = enc["full"]  # [B, D] normalized

        router_out = self.bloom_router(bloom_labels)
        mask = router_out["mask"]              # [B, 768]
        continuous_dim = router_out["continuous_dim"]  # [B]
        discrete_dim = router_out["discrete_dim"]      # [B]

        masked_emb = F.normalize(full_emb * mask, p=2, dim=-1)  # [B, 768]

        return {
            "full_embedding": full_emb,
            "masked_embedding": masked_emb,
            "mask": mask,
            "continuous_dim": continuous_dim,
            "discrete_dim": discrete_dim,
        }

    def encode_documents(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Documents always use full 768 dims."""
        enc = self.encoder(input_ids, attention_mask, token_type_ids)
        return {
            "full_embedding": enc["full"],
            "masked_embedding": enc["full"],
            "mask": torch.ones(enc["full"].size(0), self.embedding_dim,
                               device=enc["full"].device),
        }

    def forward(
        self,
        query_input_ids, query_attention_mask,
        positive_input_ids, positive_attention_mask,
        negative_input_ids=None, negative_attention_mask=None,
        learner_features=None, bloom_labels=None, subject_labels=None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass for training."""
        q = self.encode_queries(query_input_ids, query_attention_mask,
                                bloom_labels=bloom_labels)
        p = self.encode_documents(positive_input_ids, positive_attention_mask)

        result = {
            "query_embedding": q["full_embedding"],
            "query_masked": q["masked_embedding"],
            "query_mask": q["mask"],
            "continuous_dim": q["continuous_dim"],
            "discrete_dim": q["discrete_dim"],
            "positive_embedding": p["full_embedding"],
        }

        if negative_input_ids is not None:
            B, N, L = negative_input_ids.shape
            neg = self.encode_documents(
                negative_input_ids.view(B * N, L),
                negative_attention_mask.view(B * N, L),
            )
            result["negative_embeddings"] = neg["full_embedding"].view(B, N, -1)

        return result

    def get_parameter_groups(self, config):
        oc = config["training"]["optimizer"]
        # Router has only 6 params — needs high LR to learn quickly
        fast_lr = oc.get("router_lr", oc["encoder_lr"] * 50)
        return [
            {"params": list(self.encoder.parameters()), "lr": oc["encoder_lr"]},
            {"params": list(self.bloom_router.parameters()), "lr": fast_lr},
        ]

    def get_bloom_dim_table(self) -> Dict[int, float]:
        return self.bloom_router.get_dim_table()

    # Compatibility
    def freeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad = True

    @property
    def query_router(self):
        """Compatibility: eval scripts check hasattr(model, 'query_router')."""
        return self.bloom_router
