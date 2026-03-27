"""
Bloom-Aligned Matryoshka (BAM) Model.

The core architectural novelty:
  1. Same MRL encoder backbone, but trained so that truncation levels
     align with Bloom's cognitive complexity levels
  2. Adaptive truncation policy: lightweight MLP predicts optimal
     truncation point d* per query
  3. At inference: truly efficient — search in d* dimensions via FAISS,
     not 768 dims with a mask

This preserves MRL's contiguous truncation property (FAISS-friendly)
while making truncation query-adaptive.

Architecture:
  Query → Encoder → [768-dim embedding]
                  → Truncation Policy → d* ∈ {64, 128, 256, 384, 512, 768}
                  → Truncate to d* → FAISS search in d* dims

  Document → Encoder → [768-dim embedding] → Store full 768 dims
                                            → At search time, truncate to match query's d*
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from models.encoder import MRLEncoder


class TruncationPolicy(nn.Module):
    """
    Predicts the optimal MRL truncation dimension per query.

    Input: query embedding [B, D]
    Output: probability distribution over truncation points [B, K]
            and selected truncation dimension d* per query

    Key design choices:
    - Uses Gumbel-Softmax for differentiable discrete selection during training
    - Predicts a DISTRIBUTION over dims, not just one — allows soft mixing
    - Reward signal comes from contrastive loss at the selected dimension
    """

    TRUNCATION_DIMS = [64, 128, 256, 384, 512, 768]

    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        temperature: float = 1.0,
        truncation_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.dims = truncation_dims or self.TRUNCATION_DIMS
        self.num_choices = len(self.dims)
        self.temperature = temperature
        self.embedding_dim = embedding_dim

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.num_choices),
        )

        # Initialize with slight bias toward higher dims (safe start)
        with torch.no_grad():
            bias = torch.linspace(-0.5, 0.5, self.num_choices)
            self.policy_net[-1].bias.copy_(bias)

    def forward(
        self,
        query_embedding: torch.Tensor,  # [B, D]
        hard: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        if hard is None:
            hard = not self.training

        logits = self.policy_net(query_embedding)  # [B, K]

        if self.training:
            # Gumbel-Softmax for differentiable selection
            selection = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
        else:
            selection = F.softmax(logits, dim=-1)

        if hard:
            # Hard selection: argmax with straight-through
            hard_selection = torch.zeros_like(selection)
            indices = selection.argmax(dim=-1)
            hard_selection.scatter_(1, indices.unsqueeze(-1), 1.0)
            selection = hard_selection - selection.detach() + selection

        # Compute selected dimension per query
        dim_tensor = torch.tensor(self.dims, dtype=torch.float, device=query_embedding.device)
        selected_dim = (selection * dim_tensor).sum(dim=-1)  # [B]

        # Per-query probability of each truncation level
        probs = F.softmax(logits, dim=-1)

        # Statistics
        avg_dim = selected_dim.mean()
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()

        return {
            "selection": selection,         # [B, K] soft/hard selection weights
            "logits": logits,               # [B, K] raw logits
            "probs": probs,                 # [B, K] probabilities
            "selected_dim": selected_dim,   # [B] selected dimension per query
            "avg_dim": avg_dim,             # Scalar: mean selected dim
            "entropy": entropy,             # Scalar: selection entropy
            "indices": probs.argmax(dim=-1),  # [B] index of selected dim
        }


class BloomAlignedMRL(nn.Module):
    """
    Bloom-Aligned Matryoshka model with adaptive truncation.

    Key difference from QA-MRL:
    - QA-MRL: masks arbitrary dimensions (not FAISS-friendly, fake efficiency)
    - BAM: selects a truncation POINT (FAISS-friendly, real efficiency)

    The truncation policy says "for this query, use the first d* dims"
    where d* varies per query. This is truly efficient because:
    - You can build FAISS indices at each d ∈ {64,128,256,384,512,768}
    - Each query searches in the appropriate index
    - Simpler queries search in smaller indices = real speedup
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        mc = config["model"]

        # Shared encoder
        self.encoder = MRLEncoder(
            model_name=mc["backbone"],
            embedding_dim=mc["embedding_dim"],
            mrl_dims=mc["mrl_dims"],
            pooling=mc["pooling"],
            normalize=mc["normalize_embeddings"],
        )

        # Truncation policy
        self.policy = TruncationPolicy(
            embedding_dim=mc["embedding_dim"],
            hidden_dim=mc.get("policy", {}).get("hidden_dim", 256),
            temperature=mc.get("policy", {}).get("temperature", 1.0),
            truncation_dims=mc["mrl_dims"],
        )

        self.embedding_dim = mc["embedding_dim"]
        self.mrl_dims = mc["mrl_dims"]

    def encode_queries(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        learner_features: Optional[torch.Tensor] = None,  # Kept for API compat
    ) -> Dict[str, torch.Tensor]:
        """
        Encode query and determine optimal truncation.

        Returns:
            full_embedding: [B, D]
            truncated: {dim: [B, d]} at each MRL point
            adaptive_embedding: [B, D] soft mixture of truncated embeddings
            policy_output: truncation policy decisions
        """
        enc = self.encoder(input_ids, attention_mask, token_type_ids)
        full_emb = enc["full"]         # [B, D]
        truncated = enc["truncated"]   # {dim: [B, d]}

        # Get truncation decision
        policy_out = self.policy(full_emb)
        selection = policy_out["selection"]  # [B, K]

        # Compute adaptive embedding: weighted mixture of truncated embeddings
        # Each truncated embedding is padded to full dim with zeros
        B = full_emb.size(0)
        adaptive_emb = torch.zeros_like(full_emb)  # [B, D]

        for k, d in enumerate(self.mrl_dims):
            # Weight for this truncation level
            w = selection[:, k].unsqueeze(-1)  # [B, 1]
            # Padded truncated embedding
            padded = torch.zeros_like(full_emb)
            padded[:, :d] = truncated[d]
            adaptive_emb = adaptive_emb + w * padded

        adaptive_emb = F.normalize(adaptive_emb, p=2, dim=-1)

        return {
            "full_embedding": full_emb,
            "truncated": truncated,
            "adaptive_embedding": adaptive_emb,
            "masked_embedding": adaptive_emb,  # Alias for eval compatibility
            "mask": self._build_effective_mask(selection, B, full_emb.device),
            "policy_output": policy_out,
            "router_stats": {  # Compatibility with existing eval code
                "active_dims": policy_out["avg_dim"],
                "active_groups": policy_out["avg_dim"] / (self.embedding_dim // 8),
            },
        }

    def encode_documents(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode documents — always produce full embedding.
        Truncation happens at search time based on query's d*.
        """
        enc = self.encoder(input_ids, attention_mask, token_type_ids)
        return {
            "full_embedding": enc["full"],
            "truncated": enc["truncated"],
            "masked_embedding": enc["full"],  # Documents are never masked
            "mask": torch.ones_like(enc["full"]),
        }

    def _build_effective_mask(self, selection, B, device):
        """Build a soft mask for logging/analysis (not used in retrieval)."""
        mask = torch.zeros(B, self.embedding_dim, device=device)
        for k, d in enumerate(self.mrl_dims):
            w = selection[:, k].unsqueeze(-1)
            mask[:, :d] = mask[:, :d] + w
        return mask.clamp(0, 1)

    def forward(
        self,
        query_input_ids, query_attention_mask,
        positive_input_ids, positive_attention_mask,
        negative_input_ids=None, negative_attention_mask=None,
        learner_features=None, bloom_labels=None, subject_labels=None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward for training."""
        q = self.encode_queries(query_input_ids, query_attention_mask)
        p = self.encode_documents(positive_input_ids, positive_attention_mask)

        result = {
            "query_embedding": q["full_embedding"],
            "query_adaptive": q["adaptive_embedding"],
            "query_masked": q["adaptive_embedding"],
            "query_mask": q["mask"],
            "query_truncated": q["truncated"],
            "policy_output": q["policy_output"],
            "positive_embedding": p["full_embedding"],
            "positive_masked": p["full_embedding"],
            "positive_mask": p["mask"],
            "positive_truncated": p["truncated"],
            "router_stats": q["router_stats"],
        }

        if negative_input_ids is not None:
            B, N, L = negative_input_ids.shape
            neg = self.encode_documents(
                negative_input_ids.view(B * N, L),
                negative_attention_mask.view(B * N, L),
            )
            result["negative_embeddings"] = neg["full_embedding"].view(B, N, -1)
            result["negative_masked"] = neg["full_embedding"].view(B, N, -1)

        return result

    def get_parameter_groups(self, config):
        oc = config["training"]["optimizer"]
        return [
            {"params": list(self.encoder.parameters()), "lr": oc["encoder_lr"]},
            {"params": list(self.policy.parameters()), "lr": oc.get("router_lr", oc["encoder_lr"] * 5)},
        ]

    # Compatibility methods
    def freeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad = False
    def unfreeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad = True
    def freeze_router(self):
        for p in self.policy.parameters(): p.requires_grad = False
    def unfreeze_router(self):
        for p in self.policy.parameters(): p.requires_grad = True

    @property
    def query_router(self):
        """Compatibility: eval scripts check hasattr(model, 'query_router')."""
        return self.policy
