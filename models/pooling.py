"""Pooling strategies for transformer outputs."""

import torch
import torch.nn as nn
from typing import Optional


class Pooler(nn.Module):
    """
    Pooling layer that converts token-level representations to
    a single sentence embedding.

    Supports:
        - cls:        Use [CLS] token representation (BERT-style, index 0)
        - mean:       Mean pooling over non-padding tokens
        - max:        Max pooling over non-padding tokens
        - last_token: Use the last non-padding token (EOS) — Qwen/decoder-style LLMs
    """

    def __init__(self, strategy: str = "cls"):
        super().__init__()
        assert strategy in ("cls", "mean", "max", "last_token"), f"Unknown pooling: {strategy}"
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

        elif self.strategy == "last_token":
            # Last non-padding token (EOS) — used by Qwen-Embedding and decoder-style LLMs.
            # attention_mask.sum() - 1 gives the 0-indexed position of the last real token.
            seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_idx, seq_lengths]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.strategy}")
