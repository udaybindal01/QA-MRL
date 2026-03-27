"""Pooling strategies for transformer outputs."""

import torch
import torch.nn as nn
from typing import Optional


class Pooler(nn.Module):
    """
    Pooling layer that converts token-level representations to
    a single sentence embedding.

    Supports:
        - cls: Use [CLS] token representation
        - mean: Mean pooling over non-padding tokens
        - max: Max pooling over non-padding tokens
    """

    def __init__(self, strategy: str = "cls"):
        super().__init__()
        assert strategy in ("cls", "mean", "max"), f"Unknown pooling: {strategy}"
        self.strategy = strategy

    def forward(
        self,
        hidden_states: torch.Tensor,      # [B, L, D]
        attention_mask: torch.Tensor,      # [B, L]
    ) -> torch.Tensor:                     # [B, D]
        if self.strategy == "cls":
            return hidden_states[:, 0]

        elif self.strategy == "mean":
            mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
            summed = (hidden_states * mask).sum(dim=1)   # [B, D]
            counts = mask.sum(dim=1).clamp(min=1e-9)     # [B, 1]
            return summed / counts

        elif self.strategy == "max":
            mask = attention_mask.unsqueeze(-1).float()
            # Set padding positions to large negative value
            hidden_states = hidden_states.masked_fill(mask == 0, -1e9)
            return hidden_states.max(dim=1).values

        else:
            raise ValueError(f"Unknown pooling strategy: {self.strategy}")
