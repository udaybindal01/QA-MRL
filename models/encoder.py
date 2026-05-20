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


def _is_network_or_disk_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in (
        "connect", "network", "name or service", "closed",
        "not enough free disk", "disk space", "errno",
    ))


# Models that ship custom modeling code via `auto_map` in their HF config.
# When loaded via AutoModel.from_pretrained, these REQUIRE trust_remote_code=True.
# Auto-detected by prefix match so users don't have to set it in every config
# (and so the 8 different MRLEncoder() call sites in scripts/ also work).
_TRUST_REMOTE_CODE_PREFIXES = (
    "Alibaba-NLP/gte-",       # gte-v1.5 series — custom BERT++ (RoPE + GLU)
    "nomic-ai/nomic-embed",   # nomic-embed-text-v1.x — custom modeling file
    "jinaai/jina-embeddings", # jina-v3 — task-specific LoRA adapters
    "jinaai/jina-",           # other jina models
)


def _auto_trust_remote_code(model_name: str, explicit: bool) -> bool:
    """Auto-enable trust_remote_code for models known to require it."""
    if explicit:
        return True
    if any(model_name.startswith(p) for p in _TRUST_REMOTE_CODE_PREFIXES):
        return True
    return False


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
        trust_remote_code: bool = False,          # NEW: required for GTE-v1.5, Nomic, Jina-v3, EmbeddingGemma
    ):
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.mrl_dims = mrl_dims or [64, 128, 256, 384, 512, 768]
        self.normalize = normalize
        self.backbone_type = backbone_type
        self.query_instruction = query_instruction
        # Auto-enable trust_remote_code for known custom-code model families
        # (GTE-v1.5, Nomic, Jina-v3) even when not explicitly set in config.
        self.trust_remote_code = _auto_trust_remote_code(model_name, trust_remote_code)

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype = dtype_map.get(torch_dtype) if torch_dtype else None

        if backbone_type == "llm2vec":
            self._load_llm2vec(model_name, peft_model_name, dtype)
        elif backbone_type == "gritlm":
            self._load_gritlm(model_name, dtype)
        else:
            # standard or qwen — both load via AutoModel
            self.transformer = self._load_automodel(model_name, dtype, self.trust_remote_code)
            if gradient_checkpointing:
                self.transformer.config.use_cache = False
                self.transformer.gradient_checkpointing_enable()

        self.tokenizer = self._load_tokenizer(model_name, self.trust_remote_code)

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
    def _load_automodel(model_name: str, dtype, trust_remote_code: bool = False):
        """AutoModel.from_pretrained with automatic offline/disk-error fallback.

        trust_remote_code=True is required for models that ship custom modeling
        code via `auto_map` in their HF config (GTE-v1.5, Nomic-embed, Jina-v3, etc.).
        """
        try:
            return AutoModel.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        except Exception as e:
            if _is_network_or_disk_error(e):
                print(f"  Network/disk error — loading {model_name} from cache.")
                return AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    trust_remote_code=trust_remote_code,
                    local_files_only=True,
                )
            raise

    @staticmethod
    def _load_tokenizer(model_name: str, trust_remote_code: bool = False):
        """AutoTokenizer.from_pretrained with automatic offline/disk-error fallback."""
        try:
            tok = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )
        except Exception as e:
            if _is_network_or_disk_error(e):
                print(f"  Network/disk error — loading tokenizer from cache.")
                tok = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=trust_remote_code,
                    local_files_only=True,
                )
            else:
                raise
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    @staticmethod
    def _hf_kwargs() -> dict:
        """Return local_files_only=True when the node has no internet / HF offline mode set."""
        import os as _os
        offline = _os.environ.get("HF_HUB_OFFLINE", "0") == "1" \
                  or _os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
        return {"local_files_only": True} if offline else {}

    def _load_llm2vec(self, base_model_name: str, peft_model_name: Optional[str], dtype):
        """Load LLM2Vec; handles both old and new API conventions, retries on network errors.

        LLM2Vec ≥0.2.x changed from_pretrained to take the supervised PEFT model as the
        first arg (it infers the base model from the PEFT adapter config).  Older versions
        expected the base model first + peft_model_name_or_path as a kwarg.  We try the
        new convention first; if model_class comes back None (AttributeError) we retry with
        the old convention; if both fail we fall back to plain AutoModel.
        """
        def _try_load(first_arg: str, peft_kwarg: Optional[str], local_only: bool):
            kw = {"local_files_only": True} if local_only else {}
            try:
                from llm2vec import LLM2Vec
                load_kw = dict(
                    torch_dtype=dtype or torch.bfloat16,
                    enable_bidirectional=True,
                    **kw,
                )
                if peft_kwarg is not None:
                    load_kw["peft_model_name_or_path"] = peft_kwarg
                obj = LLM2Vec.from_pretrained(first_arg, **load_kw)
                self.transformer = obj.model
                return True
            except ImportError:
                print(
                    "WARNING: llm2vec not installed — falling back to AutoModel.\n"
                    "Install: pip install llm2vec"
                )
                self.transformer = self._load_automodel(base_model_name, dtype or torch.bfloat16)
                return True
            except AttributeError as e:
                # model_class lookup returned None — wrong API convention or unsupported arch.
                return e
            except Exception as e:
                return e

        # Try new API: supervised PEFT model as primary arg (llm2vec ≥0.2.x)
        peft = peft_model_name or base_model_name
        result = _try_load(peft, None, local_only=False)
        if result is True:
            return

        # Try old API: base model primary + peft_model_name_or_path kwarg (llm2vec <0.2)
        if isinstance(result, (AttributeError, Exception)):
            print(f"  LLM2Vec new API failed ({type(result).__name__}: {result}) "
                  f"— retrying with old API (base_model + peft kwarg).")
            result = _try_load(base_model_name, peft, local_only=False)
            if result is True:
                return

        err = result
        if _is_network_or_disk_error(err):
            print(f"  Network/disk error — retrying with local_files_only.")
            for first, pkw in [(peft, None), (base_model_name, peft)]:
                r = _try_load(first, pkw, local_only=True)
                if r is True:
                    return
            err = r

        # Both LLM2Vec API attempts failed — fall back to AutoModel (no bidirectional patch)
        print(f"  WARNING: all LLM2Vec load attempts failed ({type(err).__name__}: {err}).\n"
              f"  Falling back to AutoModel — bidirectional attention will NOT be applied.\n"
              f"  To fix: check llm2vec version compatibility with transformers.")
        self.transformer = self._load_automodel(base_model_name, dtype or torch.bfloat16)

    @staticmethod
    def _patch_mistral_config_rope_theta():
        """Compatibility shim: add rope_theta to MistralConfig for transformers < 4.35."""
        try:
            from transformers import MistralConfig
            if not hasattr(MistralConfig, 'rope_theta'):
                _orig_init = MistralConfig.__init__
                def _patched(self, *args, rope_theta=10000.0, **kwargs):
                    _orig_init(self, *args, **kwargs)
                    if not hasattr(self, 'rope_theta'):
                        self.rope_theta = rope_theta
                MistralConfig.__init__ = _patched
                print("  Applied MistralConfig.rope_theta shim (transformers < 4.35 compat).")
        except Exception:
            pass

    def _load_gritlm(self, model_name: str, dtype):
        """Load GritLM; retries with local_files_only on network / disk-space errors."""
        self._patch_mistral_config_rope_theta()

        def _try_load(local_only: bool):
            kw = {"local_files_only": True} if local_only else {}
            try:
                from gritlm import GritLM as _GritLM
                obj = _GritLM(
                    model_name,
                    torch_dtype=dtype or torch.bfloat16,
                    mode="embedding",
                    **kw,
                )
                self.transformer = obj.model
                return True
            except ImportError:
                print(
                    "WARNING: gritlm not installed — falling back to AutoModel.\n"
                    "Install: pip install gritlm"
                )
                self.transformer = self._load_automodel(
                    model_name, dtype or torch.bfloat16)
                return True
            except Exception as e:
                return e

        result = _try_load(local_only=False)
        if result is True:
            return
        err = result
        if _is_network_or_disk_error(err):
            print(f"  Network/disk error — retrying {model_name} with local_files_only.")
            result2 = _try_load(local_only=True)
            if result2 is True:
                return
            err = result2
        raise err

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
