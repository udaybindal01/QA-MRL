"""
MRL Backbone Encoder.

Wraps a pretrained transformer and produces Matryoshka embeddings
valid at multiple truncation points.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional, Tuple

from .pooling import Pooler


class MRLEncoder(nn.Module):
    """
    Matryoshka Representation Learning encoder.

    The first d dimensions of the embedding form a valid representation
    for any d in `mrl_dims`.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        embedding_dim: int = 768,
        mrl_dims: Optional[List[int]] = None,
        pooling: str = "cls",
        normalize: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.mrl_dims = mrl_dims or [64, 128, 256, 384, 512, 768]
        self.normalize = normalize

        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pooler = Pooler(strategy=pooling)

        actual_dim = self.transformer.config.hidden_size
        assert actual_dim == embedding_dim, \
            f"Model hidden size {actual_dim} != configured {embedding_dim}"
        for d in self.mrl_dims:
            assert d <= embedding_dim, f"MRL dim {d} > embedding dim {embedding_dim}"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            full:         [B, D] full-dimensional embedding
            truncated:    {dim: [B, d]} truncated embeddings at each MRL dim
            hidden_states: [B, L, D] last hidden states (for probing)
        """
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.transformer(**kwargs)
        hidden_states = outputs.last_hidden_state
        embedding = self.pooler(hidden_states, attention_mask)

        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=-1)

        truncated = {}
        for d in self.mrl_dims:
            trunc = embedding[:, :d]
            if self.normalize:
                trunc = F.normalize(trunc, p=2, dim=-1)
            truncated[d] = trunc

        return {
            "full": embedding,
            "truncated": truncated,
            "hidden_states": hidden_states,
        }

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 256,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """Encode a list of texts into embeddings. Utility for evaluation."""
        self.eval()
        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Encoding")

        with torch.no_grad():
            for start in iterator:
                batch_texts = texts[start:start + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                device = next(self.parameters()).device
                encoded = {k: v.to(device) for k, v in encoded.items()}
                out = self.forward(**encoded)
                all_embeddings.append(out["full"].cpu())

        return torch.cat(all_embeddings, dim=0)

    def get_tokenizer(self):
        return self.tokenizer
