"""
MRL Backbone Encoder.

Wraps a pretrained transformer and produces Matryoshka embeddings
valid at multiple truncation points.

Supported backbone types:
  standard   — BERT/RoBERTa/E5/BGE with CLS or mean pooling (default)
  qwen       — Qwen-Embedding: last-token (EOS) pooling, optional instruction prefix
  llm2vec    — LLM2Vec (Mistral/LLaMA with bidirectional attention + PEFT adapter)
  gritlm     — GritLM-7B unified encoder-decoder
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
        torch_dtype: str = None,
        gradient_checkpointing: bool = False,
        backbone_type: str = "standard",   # "standard" | "qwen" | "llm2vec" | "gritlm"
        query_instruction: Optional[str] = None,  # instruction prefix prepended to queries
        peft_model_name: Optional[str] = None,    # LLM2Vec supervised PEFT adapter repo
    ):
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.mrl_dims = mrl_dims or [64, 128, 256, 384, 512, 768]
        self.normalize = normalize
        self.backbone_type = backbone_type
        self.query_instruction = query_instruction

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype = dtype_map.get(torch_dtype) if torch_dtype else None

        if backbone_type == "llm2vec":
            self._load_llm2vec(model_name, peft_model_name, dtype)
        elif backbone_type == "gritlm":
            self._load_gritlm(model_name, dtype)
        else:
            # standard or qwen — both load via AutoModel
            self.transformer = self._load_automodel(model_name, dtype)
            if gradient_checkpointing:
                self.transformer.gradient_checkpointing_enable()

        self.tokenizer = self._load_tokenizer(model_name)

        # Qwen tokenizer needs a pad token (it uses EOS by default, fine for inference)
        if backbone_type == "qwen" and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Infer pooling from backbone_type when not explicitly set
        if backbone_type == "qwen" and pooling == "cls":
            pooling = "last_token"
        elif backbone_type in ("llm2vec", "gritlm") and pooling == "cls":
            pooling = "mean"

        self.pooler = Pooler(strategy=pooling)

        actual_dim = self.transformer.config.hidden_size
        assert actual_dim == embedding_dim, \
            f"Model hidden size {actual_dim} != configured {embedding_dim}"
        for d in self.mrl_dims:
            assert d <= embedding_dim, f"MRL dim {d} > embedding dim {embedding_dim}"

    # ── Private loaders ───────────────────────────────────────────────────────

    @staticmethod
    def _load_automodel(model_name: str, dtype):
        """AutoModel.from_pretrained with automatic offline fallback."""
        try:
            return AutoModel.from_pretrained(model_name, torch_dtype=dtype)
        except Exception as e:
            if "connect" in str(e).lower() or "network" in str(e).lower() \
                    or "name or service" in str(e).lower() or "closed" in str(e).lower():
                print(f"  Network unavailable — loading {model_name} from cache (local_files_only).")
                return AutoModel.from_pretrained(
                    model_name, torch_dtype=dtype, local_files_only=True)
            raise

    @staticmethod
    def _load_tokenizer(model_name: str):
        """AutoTokenizer.from_pretrained with automatic offline fallback."""
        try:
            return AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            if "connect" in str(e).lower() or "network" in str(e).lower() \
                    or "name or service" in str(e).lower() or "closed" in str(e).lower():
                print(f"  Network unavailable — loading tokenizer from cache (local_files_only).")
                return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            raise

    def _load_llm2vec(self, base_model_name: str, peft_model_name: Optional[str], dtype):
        """Load LLM2Vec via llm2vec library (handles bidirectional attention patches + PEFT)."""
        try:
            from llm2vec import LLM2Vec
            effective_peft = peft_model_name or base_model_name
            llm2vec_obj = LLM2Vec.from_pretrained(
                base_model_name,
                peft_model_name_or_path=effective_peft,
                torch_dtype=dtype or torch.bfloat16,
                enable_bidirectional=True,
            )
            self.transformer = llm2vec_obj.model
        except ImportError:
            print(
                "WARNING: llm2vec not installed — falling back to AutoModel (no "
                "bidirectional attention patches; representation quality will be lower).\n"
                "Install: pip install llm2vec"
            )
            self.transformer = AutoModel.from_pretrained(
                base_model_name, torch_dtype=dtype or torch.bfloat16
            )

    def _load_gritlm(self, model_name: str, dtype):
        """Load GritLM via gritlm library (unified encoder-decoder in embedding mode)."""
        try:
            from gritlm import GritLM as _GritLM
            gritlm_obj = _GritLM(
                model_name,
                torch_dtype=dtype or torch.bfloat16,
                mode="embedding",
            )
            self.transformer = gritlm_obj.model
        except ImportError:
            print(
                "WARNING: gritlm not installed — falling back to AutoModel.\n"
                "Install: pip install gritlm"
            )
            self.transformer = AutoModel.from_pretrained(
                model_name, torch_dtype=dtype or torch.bfloat16
            )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            full:           [B, D] full-dimensional embedding
            truncated:      {dim: [B, d]} truncated embeddings at each MRL dim
            hidden_states:  [B, L, D] last hidden states (for probing)
            routing_hidden: [B, D] unnormalized hidden state for BAM-PQ query MLP
                            — CLS token for standard, last token for Qwen,
                              mean-pooled for LLM2Vec/GritLM
        """
        kwargs: Dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.transformer(**kwargs)
        hidden_states = outputs.last_hidden_state  # [B, L, D]

        # routing_hidden: the raw (unnormalized) hidden state used by BAM-PQ query MLP
        routing_hidden = self._routing_hidden(hidden_states, attention_mask)

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
            "routing_hidden": routing_hidden,
        }

    def _routing_hidden(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Unnormalized hidden state passed to the BAM-PQ query MLP."""
        strategy = self.pooler.strategy
        if strategy == "cls":
            return hidden_states[:, 0, :]
        elif strategy == "last_token":
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_idx, seq_lengths]
        else:
            # mean / max: use mean-pooled unnormalized hidden state
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    # ── Encode utility ────────────────────────────────────────────────────────

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 256,
        show_progress: bool = False,
        is_query: bool = False,
    ) -> torch.Tensor:
        """Encode a list of texts into full-dim embeddings. Utility for evaluation."""
        self.eval()
        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Encoding")

        with torch.no_grad():
            for start in iterator:
                batch_texts = texts[start:start + batch_size]
                # Prepend instruction prefix to queries (Qwen-Embedding style)
                if is_query and self.query_instruction:
                    batch_texts = [self.query_instruction + t for t in batch_texts]
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
