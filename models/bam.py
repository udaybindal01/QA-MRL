"""
Bloom-Aligned Matryoshka (BAM) Model v4+.

Two routing modes (config: model.use_mask_routing):

  Option A — default (use_mask_routing=False):
    BloomDimRouter: 6 learned scalar params → prefix binary mask via STE.
    Active dims are prefix-contiguous → fast grouped sliced similarity at eval.
    Supports soft routing: bloom_probs @ all_dims instead of argmax lookup.
    (set use_soft_bloom_routing=True in config)

  Option B (use_mask_routing=True):
    BloomMaskHead: 2-layer MLP (CLS + Bloom one-hot → sigmoid) → scattered mask.
    Active dims are scattered (not prefix-contiguous).
    Retrieval: masked_query · full_doc dot product (zero dims are naturally ignored).
    Needs BloomMaskSparsityLoss + BloomMaskDiversityLoss to prevent all-ones collapse.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from models.encoder import MRLEncoder


class BloomMaskHead(nn.Module):
    """
    Option B: Static learned scattered mask — one mask per Bloom level.

    Architecture:
        bloom_logit: Embedding(6, 768) — one logit vector per Bloom level

        Training (soft mask, SMEC §3.3):
            soft_mask = sigmoid((logit + Gumbel_noise) / τ)   ← continuous [0,1]
            mask      = soft_mask                              ← NO hard mask, NO STE

        Eval (hard mask for corpus pre-masking):
            soft_mask = sigmoid(logit)
            mask      = (soft_mask > 0.5).float()              ← binary for efficiency

    Why soft mask during training (SMEC §3.3 insight):
        SMEC ADS uses z = softmax_τ(ẑ + G) — purely continuous weights throughout
        training. Paper: "rather than enforcing a deterministic selection of top-k
        dimensions". With hard binary mask + STE, dims with logit < 0 are zeroed out
        in the forward pass, completely blocking gradient flow to those dimensions.
        Soft mask ensures ALL dims receive gradient signal, allowing the model to
        continuously re-evaluate dimension importance.

        Gumbel noise adds stochastic exploration of the discrete space. As τ → 0
        the sigmoid sharpens toward binary; at τ = 1.0 it is close to Bernoulli(sigmoid).
        Temperature τ is annealed 1.0 → 0.1 over training (call set_temperature()).

    Initialization:
        Uniform init at Φ^{-1}(0.46) = -0.100 → 354 active dims per level.
        Why uniform (not cognitive): Option A learned Understand/Evaluate → 432 dims,
        others → 205-230. NOT monotone with cognitive level. Cognitive init enforced
        wrong ordering. Uniform start lets contrastive+distillation find the natural
        per-level equilibrium. Offset -0.100 (vs 0.0=384 dims) avoids frequency
        imbalance: Remember (n=1198) would otherwise float highest fastest.

        Why not N(0,1)? At μ=0, P(logit>0)=0.50 → 384 active dims. Contrastive gets
        7.5× more gradient updates for Remember (n=1198) vs Create (n=159), pushing
        Remember logits up fastest → dims invert (Remember highest, Create lowest).
        Cognitive-ordered init encodes the right ordering from epoch 0.
    """

    EMBEDDING_DIM = 768
    BLOOM_DIM = 6

    def __init__(self, sparsity_target: float = 0.44, gumbel_temperature: float = 1.0,
                 level_targets: Optional[dict] = None):
        super().__init__()
        self.sparsity_target = sparsity_target
        self.gumbel_temperature = gumbel_temperature
        self.bloom_logit = nn.Embedding(self.BLOOM_DIM, self.EMBEDDING_DIM)
        with torch.no_grad():
            # Per-level inference-calibrated init using the Gaussian quantile.
            #
            # At inference (eval mode), no Gumbel noise is added:
            #   soft_mask = sigmoid(logit)
            #   hard_mask = (sigmoid(logit) > 0.5) = (logit > 0)
            #
            # So P(hard_mask[d] = 1) = P(logit_d > 0) for logit_d ~ N(μ, σ²)
            #                        = Φ(μ/σ)                [Gaussian CDF]
            #
            # Solving for μ given inference target active fraction f and σ=1:
            #   Φ(μ) = f  →  μ = Φ⁻¹(f) = sqrt(2) × erfinv(2f − 1)
            #
            # The Gumbel formula is WRONG here — Gumbel calibrates training-time
            # P(logit+G > 0) under noise, but eval has no noise. Using the Gumbel
            # formula gives logits like -1.20 for 26% target, where nearly all dims
            # are inactive at inference (P(N(-1.2,1)>0)≈0.8%), collapsing quality.
            #
            # std=1.0 preserves good per-dim variance so the within-level mask is
            # a clean mix of strong-on and strong-off dims (cosine ≈ 0 across levels).
            if level_targets:
                for b in range(self.BLOOM_DIM):
                    f = level_targets.get(b, sparsity_target if sparsity_target else 0.46)
                    f = max(1e-4, min(1.0 - 1e-4, f))
                    # mu = Φ⁻¹(f) = sqrt(2) × erfinv(2f − 1)
                    mu = (torch.erfinv(torch.tensor(2.0 * f - 1.0)) * math.sqrt(2)).item()
                    nn.init.normal_(self.bloom_logit.weight[b], mean=mu, std=1.0)
            else:
                # Uniform init: Φ⁻¹(0.46) = -0.100 → 46% active at inference → 354 dims.
                # std=1.0 gives healthy variance so masks start uncorrelated (cosine ≈ 0).
                nn.init.normal_(self.bloom_logit.weight, mean=-0.100, std=1.0)

    def set_temperature(self, temperature: float):
        """Anneal Gumbel temperature each epoch (start=1.0 → end=0.1)."""
        self.gumbel_temperature = max(temperature, 1e-3)

    def forward(
        self,
        cls_token: torch.Tensor,       # [B, 768] — unused, kept for API compatibility
        bloom_labels: torch.Tensor,    # [B] int 0-indexed Bloom levels
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            mask:        [B, 768] soft sigmoid during training; hard binary at eval
            soft_mask:   [B, 768] Gumbel-perturbed sigmoid (used by sparsity/diversity losses)
            active_dims: [B] count of active dims per query (from hard_mask > 0.5)
        """
        logits = self.bloom_logit(bloom_labels)  # [B, 768]

        if self.training:
            # Gumbel noise: G = -log(-log(U)),  U ~ Uniform(0, 1)
            U = torch.rand_like(logits).clamp(min=1e-10, max=1.0 - 1e-10)
            gumbel_noise = -torch.log(-torch.log(U))
            soft_mask = torch.sigmoid(
                (logits + gumbel_noise) / self.gumbel_temperature
            )
            # SMEC ADS (§3.3): use soft continuous weights during training — no hard
            # binary mask, no STE. Paper: "rather than enforcing a deterministic
            # selection of top-k dimensions". Soft mask lets gradients flow to ALL
            # dims (even low-importance ones), avoiding blocked gradient paths.
            mask = soft_mask
            hard_mask = (soft_mask > 0.5).float()
        else:
            # Eval: hard binary mask for corpus pre-masking and dim counting.
            # No Gumbel noise at eval — use raw logits for deterministic selection.
            soft_mask = torch.sigmoid(logits)
            hard_mask = (soft_mask > 0.5).float()
            mask = hard_mask

        return {
            "mask": mask,
            "soft_mask": soft_mask,
            "active_dims": hard_mask.sum(dim=-1),            # [B]
        }


class BloomDimRouter(nn.Module):
    """
    Option A: Learns one truncation dimension per Bloom level via independent MLP heads.

    Each Bloom level b gets its own learned embedding → 2-layer MLP → sigmoid → dim.
    No ordinal inductive bias — the model freely learns that Evaluate needs 512 dims
    while Create needs only 256 if the data supports it.

    Architecture per level:
        Embedding(6, hidden_dim=32) → Linear(32, 16) → ReLU → Linear(16, 1) → sigmoid
        → continuous_dim ∈ [MIN_DIM, EMBEDDING_DIM] = [128, 768]

    Initialization: all levels start at ~448 dims (midpoint). Output layer zero-initialized;
    hidden layer uses Kaiming default (critical: zero-initializing it collapses all levels
    to identical gradient paths — they can never diverge).

    Supports soft routing: pass bloom_probs=[B, 6] to get continuous_dim = bloom_probs @ all_dims.
    Degrades more gracefully under Bloom classifier noise than hard argmax routing.
    """

    EMBEDDING_DIM = 768
    MIN_DIM = 128
    MASK_TEMPERATURE = 10.0  # sigmoid sharpness; meaningful gradients within ±46 dims

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self._span = self.EMBEDDING_DIM - self.MIN_DIM  # 640

        self.bloom_emb = nn.Embedding(6, hidden_dim)
        self.dim_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        with torch.no_grad():
            nn.init.normal_(self.bloom_emb.weight, mean=0.0, std=0.02)
            # dim_head[0]: leave PyTorch Kaiming default (do NOT zero-initialize)
            # dim_head[2]: zero weight + zero bias → logit=0 → sigmoid(0)=0.5 →
            #   all levels start at ~448 dims (midpoint). ✓
            #   With zero weight, ∂logit/∂hidden = 0, so bloom_emb gets no gradient
            #   through the weight — all 6 levels share the same gradient path via
            #   the bias. Efficiency cognitive weights scale that shared gradient,
            #   driving levels apart by compression rate. Diversity then spreads them
            #   spatially. This is the v8 condition that produced 283-dim spread.
            nn.init.zeros_(self.dim_head[2].weight)
            nn.init.zeros_(self.dim_head[2].bias)

    def _all_dims(self) -> torch.Tensor:
        """Continuous dim for all 6 levels. [6] — used in forward + get_dim_table."""
        idx = torch.arange(6, device=self.bloom_emb.weight.device)
        emb = self.bloom_emb(idx)
        logit = self.dim_head(emb).squeeze(-1)
        return torch.sigmoid(logit) * self._span + self.MIN_DIM  # [6] in [128, 768]

    def forward(
        self,
        bloom_levels: torch.Tensor,
        bloom_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            bloom_levels: [B] integer Bloom levels (0-indexed). Used when bloom_probs is None.
            bloom_probs:  [B, 6] softmax distribution over Bloom levels (soft routing).
                          When provided, continuous_dim = bloom_probs @ all_dims (weighted avg).
                          Stored in output for entropy logging.

        Returns:
            mask:           [B, 768] STE prefix mask
            continuous_dim: [B] float dim before rounding (used by efficiency loss)
            discrete_dim:   [B] float, rounded value with STE (used by eval for prefix slicing)
            bloom_probs:    [B, 6] or None, passed through for entropy logging
        """
        all_dims = self._all_dims()  # [6]

        if bloom_probs is not None:
            # Soft routing: weighted average dim across all Bloom levels
            continuous_dim = (bloom_probs * all_dims.unsqueeze(0)).sum(dim=-1)  # [B]
        else:
            continuous_dim = all_dims[bloom_levels]  # [B]

        arange = torch.arange(self.EMBEDDING_DIM, device=bloom_levels.device).float()

        # Soft sigmoid mask for gradient flow, hard prefix mask for forward pass
        soft_mask = torch.sigmoid(
            self.MASK_TEMPERATURE * (continuous_dim.unsqueeze(-1) - arange.unsqueeze(0))
        )  # [B, 768]
        rounded = continuous_dim.round()
        hard_mask = (arange.unsqueeze(0) < rounded.unsqueeze(-1)).float()  # [B, 768]

        # STE: hard forward, soft backward
        mask = hard_mask + (soft_mask - soft_mask.detach())
        discrete_dim = continuous_dim + (rounded - continuous_dim).detach()

        return {
            "mask": mask,
            "continuous_dim": continuous_dim,
            "discrete_dim": discrete_dim,
            "bloom_probs": bloom_probs,
        }

    def get_dim_table(self) -> Dict[int, float]:
        """Returns {bloom_level: expected_dim} for logging. No gradient."""
        with torch.no_grad():
            dims = self._all_dims()
        return {b: round(dims[b].item()) for b in range(6)}


class BloomAlignedMRL(nn.Module):
    """
    Bloom-Aligned Matryoshka model v4+.

    Routing mode selected by config model.use_mask_routing:
      False (default) → BloomDimRouter (prefix mask, Option A)
      True            → BloomMaskHead  (scattered mask, Option B)

    Queries are masked; documents stay at full 768 dims.
    Dot product query·doc naturally ignores zero-masked query dims.
    Bloom labels come from an external pre-trained classifier (not learned here).
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

        self.bloom_router = BloomDimRouter()  # always present (used for Option A, dims table)

        self.use_mask_routing = mc.get("use_mask_routing", False)
        self.use_soft_bloom_routing = mc.get("use_soft_bloom_routing", False)

        if self.use_mask_routing:
            lc = config.get("training", {}).get("loss", {})
            level_targets_cfg = lc.get("mask_level_targets", None)
            level_targets = (
                {int(k): float(v) for k, v in level_targets_cfg.items()}
                if level_targets_cfg else None
            )
            self.bloom_mask_head = BloomMaskHead(
                sparsity_target=mc.get("mask_sparsity_target", None),
                level_targets=level_targets,
            )

        self.embedding_dim = mc["embedding_dim"]
        self.mrl_dims = mc["mrl_dims"]

    def encode_queries(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        bloom_labels: Optional[torch.Tensor] = None,
        bloom_probs: Optional[torch.Tensor] = None,   # [B, 6] for soft routing (Option A)
        learner_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode query with Bloom-adaptive masking.

        bloom_labels: [B] int 0-indexed. Preferred.
        bloom_probs:  [B, 6] soft distribution for soft routing (Option A only).
        learner_features: [B, 6] one-hot fallback if bloom_labels not provided.
        If none provided, defaults to Bloom level 5 (Create = max dims).
        """
        if bloom_labels is None:
            if learner_features is not None:
                bloom_labels = learner_features.argmax(dim=-1)
            else:
                B = input_ids.size(0)
                bloom_labels = torch.full(
                    (B,), 5, dtype=torch.long, device=input_ids.device
                )

        enc = self.encoder(input_ids, attention_mask, token_type_ids)
        full_emb = enc["full"]  # [B, 768] normalized

        if self.use_mask_routing:
            # Option B: scattered soft mask from BloomMaskHead
            cls_token = enc["hidden_states"][:, 0, :]  # [B, 768] unnormalized CLS
            head_out = self.bloom_mask_head(cls_token, bloom_labels)
            mask = head_out["mask"]
            masked_emb = F.normalize(full_emb * mask, p=2, dim=-1)
            return {
                "full_embedding": full_emb,
                "masked_embedding": masked_emb,
                "mask": mask,
                "soft_mask": head_out["soft_mask"],
                "active_dims": head_out["active_dims"],
                # No continuous_dim/discrete_dim — scattered mask has no prefix dim
            }
        else:
            # Option A: prefix mask from BloomDimRouter
            if self.use_soft_bloom_routing and bloom_probs is None:
                # Degenerate soft probs = one-hot (no change in behavior, but enables the code path)
                bloom_probs = F.one_hot(bloom_labels, num_classes=6).float()

            router_out = self.bloom_router(bloom_labels, bloom_probs=bloom_probs)
            mask = router_out["mask"]
            masked_emb = F.normalize(full_emb * mask, p=2, dim=-1)
            return {
                "full_embedding": full_emb,
                "masked_embedding": masked_emb,
                "mask": mask,
                "continuous_dim": router_out["continuous_dim"],
                "discrete_dim": router_out["discrete_dim"],
                "bloom_probs": router_out["bloom_probs"],
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
            "mask": torch.ones(
                enc["full"].size(0), self.embedding_dim, device=enc["full"].device
            ),
        }

    def forward(
        self,
        query_input_ids, query_attention_mask,
        positive_input_ids, positive_attention_mask,
        negative_input_ids=None, negative_attention_mask=None,
        learner_features=None, bloom_labels=None, subject_labels=None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass for training."""
        q = self.encode_queries(
            query_input_ids, query_attention_mask,
            bloom_labels=bloom_labels,
        )
        p = self.encode_documents(positive_input_ids, positive_attention_mask)

        result = {
            "query_embedding": q["full_embedding"],
            "query_masked": q["masked_embedding"],
            "query_mask": q["mask"],
            "positive_embedding": p["full_embedding"],
        }

        # Option A outputs
        if "continuous_dim" in q:
            result["continuous_dim"] = q["continuous_dim"]
            result["discrete_dim"] = q["discrete_dim"]
        if "bloom_probs" in q:
            result["bloom_probs"] = q["bloom_probs"]

        # Option B outputs
        if "soft_mask" in q:
            result["soft_mask"] = q["soft_mask"]
        if "active_dims" in q:
            result["active_dims"] = q["active_dims"]

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
        freeze = config["training"].get("freeze_encoder", False)
        fast_lr = oc.get("router_lr", oc["encoder_lr"] * 10)
        # Only include params for the active routing mode — the other router is not
        # used in forward() and including it causes AdamW weight decay on dead params.
        if self.use_mask_routing:
            routing_params = list(self.bloom_mask_head.parameters())
        else:
            routing_params = list(self.bloom_router.parameters())
        groups = [{"params": routing_params, "lr": fast_lr}]
        if not freeze:
            groups.append(
                {"params": list(self.encoder.parameters()), "lr": oc["encoder_lr"]}
            )
        return groups

    def get_bloom_dim_table(self) -> Dict[int, float]:
        return self.bloom_router.get_dim_table()

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    @property
    def query_router(self):
        """Compatibility: eval scripts check hasattr(model, 'query_router')."""
        return self.bloom_router
