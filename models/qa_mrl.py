"""
Full QA-MRL Model: encoder + query-adaptive dimension router.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from models.encoder import MRLEncoder
from models.router import build_router, DocumentRouter


class QAMRL(nn.Module):
    """
    Query-Adaptive Matryoshka Representation Learning model.

    1. Shared MRL encoder produces full embeddings
    2. Query router selects per-query dimension mask
    3. (Optional) Document router selects per-doc dimension mask
    4. Masked similarity for retrieval
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
        self.query_router = build_router(config)

        self.use_asymmetric = mc["asymmetric"]["enabled"]
        if self.use_asymmetric:
            self.doc_router = DocumentRouter(
                embedding_dim=mc["embedding_dim"],
                num_groups=mc["router"]["num_groups"],
                hidden_dim=mc["asymmetric"]["doc_router_hidden_dim"],
            )
        self.embedding_dim = mc["embedding_dim"]

    def encode_queries(self, input_ids, attention_mask,
                       learner_features=None, token_type_ids=None,
                       return_mask=True):
        enc = self.encoder(input_ids, attention_mask, token_type_ids)
        result = {"full_embedding": enc["full"], "truncated": enc["truncated"]}

        if return_mask:
            # Auto-create dummy learner features if router expects them
            if learner_features is None and hasattr(self.query_router, "learner_proj"):
                learner_features = torch.zeros(input_ids.size(0), 6, device=input_ids.device)
            router_out = self.query_router(enc["full"], learner_features)
            mask = router_out["mask"]
            masked = F.normalize(enc["full"] * mask, p=2, dim=-1)
            result["masked_embedding"] = masked
            result["mask"] = mask
            result["router_stats"] = router_out
        return result

    def encode_documents(self, input_ids, attention_mask, token_type_ids=None):
        enc = self.encoder(input_ids, attention_mask, token_type_ids)
        result = {"full_embedding": enc["full"], "truncated": enc["truncated"]}

        if self.use_asymmetric:
            dr = self.doc_router(enc["full"])
            masked = F.normalize(enc["full"] * dr["mask"], p=2, dim=-1)
            result["masked_embedding"] = masked
            result["mask"] = dr["mask"]
        else:
            result["masked_embedding"] = enc["full"]
            result["mask"] = torch.ones_like(enc["full"])
        return result

    def forward(self, query_input_ids, query_attention_mask,
                positive_input_ids, positive_attention_mask,
                negative_input_ids=None, negative_attention_mask=None,
                learner_features=None, bloom_labels=None, subject_labels=None):
        q = self.encode_queries(query_input_ids, query_attention_mask,
                                learner_features=learner_features)
        p = self.encode_documents(positive_input_ids, positive_attention_mask)

        result = {
            "query_embedding": q["full_embedding"],
            "query_masked": q["masked_embedding"],
            "query_mask": q["mask"],
            "query_truncated": q["truncated"],
            "positive_embedding": p["full_embedding"],
            "positive_masked": p["masked_embedding"],
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
            result["negative_masked"] = neg["masked_embedding"].view(B, N, -1)

        return result

    def get_parameter_groups(self, config):
        oc = config["training"]["optimizer"]
        encoder_params = list(self.encoder.parameters())
        router_params = list(self.query_router.parameters())
        if self.use_asymmetric:
            router_params += list(self.doc_router.parameters())
        return [
            {"params": encoder_params, "lr": oc["encoder_lr"]},
            {"params": router_params, "lr": oc["router_lr"]},
        ]

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def freeze_router(self):
        for p in self.query_router.parameters():
            p.requires_grad = False
        if self.use_asymmetric:
            for p in self.doc_router.parameters():
                p.requires_grad = False

    def unfreeze_router(self):
        for p in self.query_router.parameters():
            p.requires_grad = True
        if self.use_asymmetric:
            for p in self.doc_router.parameters():
                p.requires_grad = True