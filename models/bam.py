"""
Bloom-Aligned Matryoshka (BAM) Model v3.

Key design change from v2:
  Instead of a per-query MLP policy that predicts truncation dims (which
  collapsed to a fixed 384 for all queries), v3 learns a GLOBAL mapping:

    Bloom level → optimal truncation dimension

  This is both more interpretable ("Remember queries need 128 dims,
  Create queries need 512 dims") and avoids the policy collapse problem.

Inference pipeline:
  Query → Encoder → full 768-dim embedding
       → Bloom Classifier → predicted Bloom level (1-6)
       → Bloom-to-Dim table lookup → d*
       → Truncate to d* → FAISS search in d* dims

The Bloom classifier is a lightweight head on the encoder, trained
jointly. At inference, no ground-truth Bloom labels are needed.

Architecture:
  Query → Encoder → [768-dim embedding]
                  → Bloom Classifier → predicted level ∈ {1..6}
                  → Bloom-Dim Mapping → d* ∈ {64,128,256,384,512,768}
                  → Truncate to d* → FAISS search

  Document → Encoder → [768-dim embedding] → Store full 768 dims
                                            → Truncate at search time
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from models.encoder import MRLEncoder


class BloomClassifier(nn.Module):
    """
    Predicts the Bloom cognitive level of a query from its embedding.

    This runs at inference so the system doesn't need ground-truth
    Bloom labels. Trained jointly with the encoder using the query
    Bloom labels available in training data.
    """

    def __init__(self, embedding_dim: int = 768, num_bloom_levels: int = 6,
                 hidden_dim: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_bloom_levels),
        )

    def forward(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            embedding: [B, D] query embedding
        Returns:
            logits: [B, num_bloom_levels]
            predicted_level: [B] argmax prediction (0-indexed)
            probs: [B, num_bloom_levels] softmax probabilities
        """
        logits = self.classifier(embedding)
        probs = F.softmax(logits, dim=-1)
        predicted = logits.argmax(dim=-1)
        return {
            "logits": logits,
            "predicted_level": predicted,
            "probs": probs,
        }


class BloomDimMapping(nn.Module):
    """
    Learns a global mapping: Bloom level → optimal truncation dimension.

    Instead of a per-query MLP (which collapsed to fixed 384), this
    learns ONE dimension per Bloom level. The learned mapping is
    interpretable: simpler queries → fewer dims, complex queries → more.

    Implementation: 6 learnable logit vectors (one per Bloom level),
    each a distribution over the available truncation dims. During
    training, Gumbel-Softmax selects differentiably. After training,
    take argmax to get a fixed lookup table.
    """

    def __init__(self, num_bloom_levels: int = 6,
                 truncation_dims: Optional[List[int]] = None,
                 temperature: float = 1.0):
        super().__init__()
        self.dims = truncation_dims or [64, 128, 256, 384, 512, 768]
        self.num_dims = len(self.dims)
        self.num_blooms = num_bloom_levels
        self.temperature = temperature

        # One logit vector per Bloom level, over the available dims
        # Initialize with bias: lower Bloom → prefer lower dims
        self.bloom_dim_logits = nn.Parameter(torch.zeros(num_bloom_levels, self.num_dims))
        with torch.no_grad():
            for b in range(num_bloom_levels):
                # Bloom 0 (Remember) biased toward lower dims
                # Bloom 5 (Create) biased toward higher dims
                center = b * (self.num_dims - 1) / (num_bloom_levels - 1)
                for d in range(self.num_dims):
                    self.bloom_dim_logits[b, d] = -0.5 * (d - center) ** 2

    def forward(self, bloom_levels: torch.Tensor,
                hard: Optional[bool] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            bloom_levels: [B] integer Bloom levels (0-indexed)
            hard: if True, use hard selection (inference). If None, auto-detect.
        Returns:
            selection: [B, K] soft/hard selection weights
            selected_dim: [B] selected dimension per query
            bloom_dim_table: [num_blooms] the learned dim per Bloom level
        """
        if hard is None:
            hard = not self.training

        # Gather the logit vector for each query's Bloom level
        logits = self.bloom_dim_logits[bloom_levels]  # [B, K]

        if self.training:
            selection = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
        else:
            selection = F.softmax(logits, dim=-1)

        if hard:
            hard_sel = torch.zeros_like(selection)
            indices = selection.argmax(dim=-1)
            hard_sel.scatter_(1, indices.unsqueeze(-1), 1.0)
            selection = hard_sel - selection.detach() + selection

        dim_tensor = torch.tensor(self.dims, dtype=torch.float, device=bloom_levels.device)
        selected_dim = (selection * dim_tensor).sum(dim=-1)  # [B]

        # Compute the full table (for logging)
        with torch.no_grad():
            full_probs = F.softmax(self.bloom_dim_logits, dim=-1)  # [num_blooms, K]
            table_dims = (full_probs * dim_tensor.unsqueeze(0)).sum(dim=-1)  # [num_blooms]

        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()

        return {
            "selection": selection,
            "selected_dim": selected_dim,
            "avg_dim": selected_dim.mean(),
            "entropy": entropy,
            "bloom_dim_table": table_dims,
            "logits": logits,
            "probs": probs,
            "indices": probs.argmax(dim=-1),
        }

    def get_dim_table(self) -> Dict[int, int]:
        """Get the learned Bloom → dim mapping as a dict (for logging/inference)."""
        dim_tensor = torch.tensor(self.dims, dtype=torch.float)
        probs = F.softmax(self.bloom_dim_logits.detach(), dim=-1)
        table = {}
        for b in range(self.num_blooms):
            idx = probs[b].argmax().item()
            table[b] = self.dims[idx]
        return table


class BloomAlignedMRL(nn.Module):
    """
    Bloom-Aligned Matryoshka model v3.

    Key differences from v2:
    - No per-query MLP policy (collapsed to fixed dim)
    - Instead: learned Bloom → dim mapping (interpretable, doesn't collapse)
    - Integrated Bloom classifier (no ground-truth needed at inference)
    - Inference: query → classify Bloom → lookup dim → truncate → search
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

        # Bloom classifier: predicts query Bloom level from embedding
        self.bloom_classifier = BloomClassifier(
            embedding_dim=mc["embedding_dim"],
            num_bloom_levels=6,
            hidden_dim=mc.get("bloom_classifier", {}).get("hidden_dim", 256),
        )

        # Bloom → Dim mapping: learns optimal truncation per Bloom level
        self.bloom_dim_map = BloomDimMapping(
            num_bloom_levels=6,
            truncation_dims=mc["mrl_dims"],
            temperature=mc.get("policy", {}).get("temperature", 1.0),
        )

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
        Encode query and determine optimal truncation via Bloom level.

        If bloom_labels are provided (training), use them directly.
        If not (inference), predict them with the Bloom classifier.
        """
        enc = self.encoder(input_ids, attention_mask, token_type_ids)
        full_emb = enc["full"]
        truncated = enc["truncated"]

        # Classify Bloom level
        # detach: Bloom classifier trains on a frozen copy of the embedding,
        # so its gradients don't alter the encoder's representation space.
        # The classifier IS still trained (via bloom_cls_loss), just independently.
        bloom_out = self.bloom_classifier(full_emb.detach())

        # Determine which Bloom levels to use for dim mapping
        if bloom_labels is not None:
            # Training: use ground-truth Bloom labels
            bloom_for_dim = bloom_labels
        else:
            # Inference: use predicted Bloom labels
            bloom_for_dim = bloom_out["predicted_level"]

        # Get truncation via Bloom → dim mapping
        dim_out = self.bloom_dim_map(bloom_for_dim)
        selection = dim_out["selection"]  # [B, K]

        # Compute adaptive embedding: weighted mixture of truncated
        B = full_emb.size(0)
        adaptive_emb = torch.zeros_like(full_emb)

        for k, d in enumerate(self.mrl_dims):
            w = selection[:, k].unsqueeze(-1)
            padded = torch.zeros_like(full_emb)
            padded[:, :d] = truncated[d]
            adaptive_emb = adaptive_emb + w * padded

        adaptive_emb = F.normalize(adaptive_emb, p=2, dim=-1)

        return {
            "full_embedding": full_emb,
            "truncated": truncated,
            "adaptive_embedding": adaptive_emb,
            "masked_embedding": adaptive_emb,
            "mask": self._build_effective_mask(selection, B, full_emb.device),
            "bloom_classifier_output": bloom_out,
            "policy_output": {  # Compatibility with trainer/eval
                "selection": selection,
                "selected_dim": dim_out["selected_dim"],
                "avg_dim": dim_out["avg_dim"],
                "entropy": dim_out["entropy"],
                "bloom_dim_table": dim_out["bloom_dim_table"],
                "logits": dim_out["logits"],
                "probs": dim_out["probs"],
                "indices": dim_out["indices"],
            },
            "router_stats": {
                "active_dims": dim_out["avg_dim"],
                "active_groups": dim_out["avg_dim"] / (self.embedding_dim // 8),
            },
        }

    def encode_documents(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode documents — always produce full embedding."""
        enc = self.encoder(input_ids, attention_mask, token_type_ids)
        return {
            "full_embedding": enc["full"],
            "truncated": enc["truncated"],
            "masked_embedding": enc["full"],
            "mask": torch.ones_like(enc["full"]),
        }

    def _build_effective_mask(self, selection, B, device):
        """Build a [B, D] mask from truncation selection weights.

        Each MRL dim d gets weight selection[:, k] applied uniformly
        across its first d dimensions.
        """
        mask = torch.zeros(B, self.embedding_dim, device=device)
        for k, d in enumerate(self.mrl_dims):
            w = selection[:, k].unsqueeze(-1).expand(-1, d)  # [B, d]
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
        q = self.encode_queries(query_input_ids, query_attention_mask,
                                bloom_labels=bloom_labels)
        p = self.encode_documents(positive_input_ids, positive_attention_mask)

        result = {
            "query_embedding": q["full_embedding"],
            "query_adaptive": q["adaptive_embedding"],
            "query_masked": q["adaptive_embedding"],
            "query_mask": q["mask"],
            "query_truncated": q["truncated"],
            "policy_output": q["policy_output"],
            "bloom_classifier_output": q["bloom_classifier_output"],
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
            {"params": list(self.encoder.parameters()),
             "lr": oc["encoder_lr"]},
            {"params": list(self.bloom_classifier.parameters()),
             "lr": oc.get("router_lr", oc["encoder_lr"] * 5)},
            {"params": list(self.bloom_dim_map.parameters()),
             "lr": oc.get("router_lr", oc["encoder_lr"] * 5)},
        ]

    def get_bloom_dim_table(self) -> Dict[int, int]:
        """Get the learned Bloom → dim mapping."""
        return self.bloom_dim_map.get_dim_table()

    # Compatibility methods
    def freeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad = False
    def unfreeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad = True
    def freeze_router(self):
        for p in self.bloom_dim_map.parameters(): p.requires_grad = False
        for p in self.bloom_classifier.parameters(): p.requires_grad = False
    def unfreeze_router(self):
        for p in self.bloom_dim_map.parameters(): p.requires_grad = True
        for p in self.bloom_classifier.parameters(): p.requires_grad = True

    @property
    def query_router(self):
        """Compatibility: eval scripts check hasattr(model, 'query_router')."""
        return self.bloom_dim_map
