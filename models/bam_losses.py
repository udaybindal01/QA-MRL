"""
BAM Training Losses v9.

New in v9:
  BloomMaskVarianceLoss  — maximises per-dim variance of mean activation across Bloom levels;
                           768 gradient signals vs 15 pairs in DiversityLoss; weight 0.5 in config

New in v8:
  BloomMaskSparsityLoss  — keeps Option B mask active dims near target (default 339/768 ≈ 0.44)
  BloomMaskDiversityLoss — penalizes different Bloom levels learning identical scattered masks
  PCGradOptimizer        — gradient surgery wrapper: projects conflicting gradients to orthogonal

v7 losses unchanged:
  BloomMaskedContrastiveLoss  — class-weighted (1/√freq) InfoNCE in masked subspace
  BloomTwoFactorEfficiencyLoss — per-class averaged efficiency with cognitive weights (1 − b/6)
  RouterDiversityLoss         — pairwise distance between Bloom-level prefix dims (Option A)
  MRLAnchorRegularizationLoss — InfoNCE at MRL anchor dims, weighted D/√d
  BAMCombinedLoss             — orchestrates all losses; call set_epoch() each epoch
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ---------------------------------------------------------------------------
# Existing losses (v7, unchanged)
# ---------------------------------------------------------------------------

class BloomMaskedContrastiveLoss(nn.Module):
    """
    InfoNCE in query-masked subspace with inverse-sqrt class weighting.

    Masking strategy differs by routing mode:
        Option A (prefix mask, mask_documents=True):
            masked_q = normalize(query_emb * prefix_mask)
            masked_p = normalize(positive_emb * prefix_mask)
            eval: normalize(q[:k]) · normalize(c[:k]) — both sides prefix-sliced.
            Correct because MRL training packs max information in the first k dims,
            so prefix-masking the document is a meaningful (lossless) compression.

        Option B (scattered mask, mask_documents=False):
            masked_q = normalize(query_emb * scattered_mask)
            full_p   = normalize(positive_emb)                ← document NOT masked
            eval: normalize(q * mask_b) · normalize(corpus_full).
            Masking docs with a randomly-initialized scattered mask degrades the
            document representation unpredictably — the encoder was not trained to
            pack Bloom-level information into arbitrary scattered dims. Query-only
            masking gives a clean gradient: "which query dims help identify the
            correct document from the full corpus?"

    class_weights: [6] tensor, one weight per Bloom level (0-indexed).
    Computed from training data as 1/sqrt(freq), normalized to mean=1.
    """

    def __init__(self, temperature: float = 0.05,
                 class_weights: Optional[List[float]] = None,
                 mask_documents: bool = True):
        super().__init__()
        self.temperature = temperature
        self.mask_documents = mask_documents
        w = torch.tensor(class_weights, dtype=torch.float) if class_weights else torch.ones(6)
        self.register_buffer("class_weights", w)

    def forward(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        query_mask: torch.Tensor,
        negative_embs: Optional[torch.Tensor] = None,
        bloom_labels: Optional[torch.Tensor] = None,
    ):
        masked_q = F.normalize(query_emb * query_mask, p=2, dim=-1)

        if self.mask_documents:
            # Option A: mask both sides (consistent with prefix-sliced eval)
            doc_p = F.normalize(positive_emb * query_mask, p=2, dim=-1)
        else:
            # Option B: query only — docs stay at full 768 dims
            doc_p = F.normalize(positive_emb, p=2, dim=-1)

        if negative_embs is not None:
            if self.mask_documents:
                mask_exp = query_mask.unsqueeze(1).expand_as(negative_embs)
                doc_n = F.normalize(negative_embs * mask_exp, p=2, dim=-1)
            else:
                doc_n = F.normalize(negative_embs, p=2, dim=-1)
            pos_sim = (masked_q * doc_p).sum(dim=-1) / self.temperature
            neg_sim = torch.bmm(doc_n, masked_q.unsqueeze(-1)).squeeze(-1) / self.temperature
            logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
        else:
            sim = torch.mm(masked_q, doc_p.t()) / self.temperature
            labels = torch.arange(sim.size(0), device=sim.device)
            per_sample_loss = F.cross_entropy(sim, labels, reduction="none")

        difficulty_w = (per_sample_loss.detach() + 1e-6)
        difficulty_w = difficulty_w / difficulty_w.mean()

        if bloom_labels is not None:
            class_w = self.class_weights.to(bloom_labels.device)[bloom_labels]
            class_w = class_w * (class_w.size(0) / class_w.sum())
            w = class_w * difficulty_w
            loss = (per_sample_loss * w).mean()
        else:
            loss = (per_sample_loss * difficulty_w).mean()

        return loss, {"contrastive": loss.item(), "temperature": self.temperature}

    def set_temperature(self, t: float):
        self.temperature = t


class BloomTwoFactorEfficiencyLoss(nn.Module):
    """
    Cognitive-load efficiency penalty with per-class averaging.

    L = (1/C) Σ_b [ cognitive(b) * mean_{i: bloom_i=b}(dim_i / D) ]

    cognitive(b) = (1 - b/6): compression pressure decreases with cognitive level.
      Remember=1.0, Understand=0.833, Apply=0.667, Analyze=0.500,
      Evaluate=0.333, Create=0.167.

    Per-class averaging ensures each Bloom level contributes equal gradient weight
    regardless of batch frequency — rare levels (Understand=1.4%) get the same
    gradient update opportunity as common ones (Remember=63.7%).

    bloom_frequencies accepted for API compatibility but unused — frequency
    imbalance is handled by per-class averaging, not per-sample weighting.
    The two-factor (cognitive × freq) approach was tried but caused Understand
    to get too few updates (rare + low combined weight → stuck near init dims).
    """

    EMBEDDING_DIM = 768.0

    def __init__(self, bloom_frequencies: List[float]):
        super().__init__()
        cognitive = torch.tensor([1.0 - b / 6.0 for b in range(6)], dtype=torch.float)
        self.register_buffer("cognitive_weights", cognitive)

    def forward(
        self,
        continuous_dim: torch.Tensor,   # [B] float — dim or active_dim count
        bloom_labels: torch.Tensor,     # [B] int 0-indexed
    ):
        total = torch.tensor(0.0, device=continuous_dim.device)
        n_classes = 0
        per_class_dims = {}

        for b in range(6):
            mask = (bloom_labels == b)
            if mask.sum() == 0:
                continue
            cw = self.cognitive_weights[b]
            class_mean_dim = continuous_dim[mask].mean()
            total = total + cw * (class_mean_dim / self.EMBEDDING_DIM)
            n_classes += 1
            per_class_dims[b] = class_mean_dim.item()

        loss = total / max(n_classes, 1)
        stats = {"efficiency": loss.item(), "avg_dim": continuous_dim.mean().item()}
        stats.update({f"dim_b{b}": d for b, d in per_class_dims.items()})
        return loss, stats


class RouterDiversityLoss(nn.Module):
    """
    Penalizes all 6 Bloom levels converging to the same prefix dimension (Option A).

    L_diversity = -mean_{i<j}(|dim_i - dim_j|) / span

    Pairwise distance (not variance) has non-zero gradient even when all dims are
    identical, which allows it to break initial symmetry and push levels apart.
    """

    EMBEDDING_DIM = 768.0
    MIN_DIM = 128.0

    def forward(self, all_dims: torch.Tensor) -> torch.Tensor:
        span = self.EMBEDDING_DIM - self.MIN_DIM
        normalized = (all_dims - self.MIN_DIM) / span
        diff = normalized.unsqueeze(0) - normalized.unsqueeze(1)
        pairwise_dist = diff.abs()
        mask = torch.triu(torch.ones(6, 6, device=all_dims.device), diagonal=1)
        mean_dist = (pairwise_dist * mask).sum() / mask.sum()
        return -mean_dist  # maximize spread = minimize negative distance


class MRLAnchorRegularizationLoss(nn.Module):
    """
    InfoNCE at MRL anchor dims to keep encoder sharp during BAM fine-tuning.
    Weighted by D/sqrt(d) — smaller truncations get more emphasis.
    """

    def __init__(self, mrl_dims: List[int], temperature: float = 0.05):
        super().__init__()
        self.mrl_dims = mrl_dims
        self.temperature = temperature
        D = max(mrl_dims)
        raw = [D / math.sqrt(d) for d in mrl_dims]
        total = sum(raw)
        self.dim_weights = [w / total for w in raw]

    def forward(self, query_full_emb: torch.Tensor, positive_full_emb: torch.Tensor):
        total = torch.tensor(0.0, device=query_full_emb.device)
        stats = {}
        for d, w in zip(self.mrl_dims, self.dim_weights):
            q = F.normalize(query_full_emb[:, :d], p=2, dim=-1)
            p = F.normalize(positive_full_emb[:, :d], p=2, dim=-1)
            sim = torch.mm(q, p.t()) / self.temperature
            labels = torch.arange(sim.size(0), device=sim.device)
            loss = F.cross_entropy(sim, labels)
            total = total + w * loss
            stats[f"mrl_anchor_d{d}"] = loss.item()
        stats["mrl_anchor"] = total.item()
        return total, stats


# ---------------------------------------------------------------------------
# New losses (v8, Option B)
# ---------------------------------------------------------------------------

class BloomMaskSparsityLoss(nn.Module):
    """
    Penalizes deviation from target active dim count per Bloom level (Option B).

    L = mean_b [ |mean_{i: bloom_i=b}(mean_d(soft_mask[i])) - target_sparsity_b| ]

    Default per-level targets follow the cognitive load hypothesis:
    lower Bloom levels (Remember) need fewer dims; higher levels (Create) need more.

        Remember  (b=0): 0.40 → 307 dims
        Understand(b=1): 0.45 → 346 dims
        Apply     (b=2): 0.50 → 384 dims
        Analyze   (b=3): 0.55 → 422 dims
        Evaluate  (b=4): 0.60 → 461 dims
        Create    (b=5): 0.65 → 499 dims

    These targets force genuine Bloom-level differentiation in the mask AND
    encode the cognitive load hypothesis directly into the loss structure.
    Override with level_targets dict {0: target0, ...} or global_target for uniform.
    """

    # Cognitively-motivated defaults: more dims for higher Bloom levels
    _COGNITIVE_DEFAULTS = {0: 0.40, 1: 0.45, 2: 0.50, 3: 0.55, 4: 0.60, 5: 0.65}

    def __init__(
        self,
        global_target: Optional[float] = None,
        level_targets: Optional[dict] = None,
    ):
        super().__init__()
        # Per-level targets: explicit level_targets > global_target > cognitive defaults
        self.targets = {}
        for b in range(6):
            if level_targets and b in level_targets:
                self.targets[b] = level_targets[b]
            elif global_target is not None:
                self.targets[b] = global_target
            else:
                self.targets[b] = self._COGNITIVE_DEFAULTS[b]

    def forward(
        self,
        soft_mask: torch.Tensor,    # [B, 768] sigmoid outputs from BloomMaskHead
        bloom_labels: torch.Tensor, # [B] int 0-indexed
    ):
        total = torch.tensor(0.0, device=soft_mask.device)
        n_classes = 0
        stats = {}
        for b in range(6):
            mask = bloom_labels == b
            if mask.sum() == 0:
                continue
            mean_active_frac = soft_mask[mask].mean()  # mean over batch & dims
            target = torch.tensor(self.targets[b], device=soft_mask.device)
            level_loss = (mean_active_frac - target).abs()
            total = total + level_loss
            n_classes += 1
            stats[f"sparsity_b{b}"] = mean_active_frac.item()
        loss = total / max(n_classes, 1)
        stats["mask_sparsity"] = loss.item()
        return loss, stats


class BloomMaskDiversityLoss(nn.Module):
    """
    Penalizes different Bloom levels from learning identical scattered masks (Option B).

    For each pair of Bloom levels (b_i, b_j), compute mean mask over all queries at
    that level, then penalize high cosine similarity between level means.

    L = mean_{i<j} max(0, sim(mean_mask_i, mean_mask_j) - margin)

    margin: similarity threshold above which a penalty is incurred.
    Default margin=0.3 (tight) — with masks at 0.97 similarity, margin=0.7 only
    yields 0.27 penalty per pair; margin=0.3 yields 0.67, giving 2.5× stronger gradient
    to push Bloom levels toward genuinely different dimension subsets.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        soft_mask: torch.Tensor,    # [B, 768]
        bloom_labels: torch.Tensor, # [B] int 0-indexed
    ):
        # Compute mean mask per Bloom level
        level_means = []
        present_levels = []
        for b in range(6):
            mask = bloom_labels == b
            if mask.sum() > 0:
                level_means.append(soft_mask[mask].mean(dim=0))  # [768]
                present_levels.append(b)

        if len(level_means) < 2:
            return torch.tensor(0.0, device=soft_mask.device), {"mask_diversity": 0.0}

        level_means_t = torch.stack(level_means, dim=0)  # [L, 768]
        # Pairwise cosine similarity
        normed = F.normalize(level_means_t, p=2, dim=-1)   # [L, 768]
        sim_matrix = torch.mm(normed, normed.t())           # [L, L]

        # Upper triangle, penalize pairs above margin
        L = len(level_means)
        triu_mask = torch.triu(torch.ones(L, L, device=soft_mask.device), diagonal=1)
        penalties = F.relu(sim_matrix - self.margin) * triu_mask
        n_pairs = triu_mask.sum().clamp(min=1)
        loss = penalties.sum() / n_pairs

        return loss, {"mask_diversity": loss.item()}


class BloomMaskVarianceLoss(nn.Module):
    """
    Maximizes per-dimension activation variance across Bloom levels (Option B).

    For each of the 768 dimensions, computes the variance of mean activation
    probability across all Bloom levels present in the batch. Maximizing this
    rewards each dimension being "owned" by specific Bloom levels rather than
    uniformly active or inactive across all levels.

    Why this works better than pairwise cosine diversity:
      - Operates at the dimension level (768 signals) vs pair level (15 signals)
      - Directly rewards specialization: dim d has high variance if it's active
        for Remember queries but inactive for Create queries (or vice versa)
      - Gradient flows to individual soft_mask entries, not just their aggregate

    L = -mean_d( Var_b( mean_{i: bloom_i=b}(soft_mask[i, d]) ) )

    We negate because we minimise loss but want to maximise variance.
    Scale: soft_mask ∈ [0,1], so max variance per dim ≈ 0.25 (Bernoulli).
    A weight of 0.5–1.0 is appropriate relative to sparsity/diversity losses.
    """

    def forward(
        self,
        soft_mask: torch.Tensor,    # [B, 768] sigmoid outputs
        bloom_labels: torch.Tensor, # [B] int 0-indexed
    ) -> tuple:
        level_means = []
        for b in range(6):
            idx = bloom_labels == b
            if idx.sum() > 0:
                level_means.append(soft_mask[idx].mean(dim=0))  # [768]

        if len(level_means) < 2:
            return torch.tensor(0.0, device=soft_mask.device), {"mask_variance": 0.0}

        stacked = torch.stack(level_means, dim=0)   # [L, 768]
        # Variance across levels for each dim (unbiased=False: L can be small)
        col_var = stacked.var(dim=0, unbiased=False)  # [768]
        # Maximise variance → minimise its negative
        loss = -col_var.mean()

        return loss, {
            "mask_variance": col_var.mean().item(),
            "mask_var_max_dim": col_var.max().item(),
            "mask_var_min_dim": col_var.min().item(),
        }


# ---------------------------------------------------------------------------
# PCGrad optimizer wrapper (optional, Challenge 1 fix)
# ---------------------------------------------------------------------------

class PCGradOptimizer:
    """
    Gradient surgery wrapper for any PyTorch optimizer.

    For N losses, projects each loss's gradients to be orthogonal to any conflicting
    gradients (dot product < 0) from other losses on shared parameters.

    Usage:
        pcgrad = PCGradOptimizer(optimizer)
        pcgrad.pc_backward([contrastive_loss, routing_loss])
        pcgrad.step()
        pcgrad.zero_grad()

    Applied only to SHARED parameters (encoder). Router-only parameters have no
    conflicting gradients since only one loss touches them.
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def pc_backward(self, losses: List[torch.Tensor]):
        """
        Compute and project gradients for each loss, then accumulate.
        All losses must be scalar tensors sharing the same computation graph.
        """
        # Collect all parameters from optimizer
        all_params = [
            p for group in self.optimizer.param_groups
            for p in group["params"]
            if p.requires_grad
        ]

        # Compute gradients for each loss separately
        grads = []
        for i, loss in enumerate(losses):
            self.optimizer.zero_grad()
            loss.backward(retain_graph=(i < len(losses) - 1))
            grad_i = []
            for p in all_params:
                grad_i.append(p.grad.clone() if p.grad is not None else None)
            grads.append(grad_i)

        # Project conflicting gradients
        merged = [None] * len(all_params)
        for param_idx in range(len(all_params)):
            for task_idx in range(len(losses)):
                g_i = grads[task_idx][param_idx]
                if g_i is None:
                    continue
                projected = g_i.clone()
                for other_idx in range(len(losses)):
                    if other_idx == task_idx:
                        continue
                    g_j = grads[other_idx][param_idx]
                    if g_j is None:
                        continue
                    dot = (projected * g_j).sum()
                    if dot < 0:
                        # Project g_i to remove component along g_j
                        projected = projected - (dot / (g_j.norm() ** 2 + 1e-12)) * g_j
                if merged[param_idx] is None:
                    merged[param_idx] = projected
                else:
                    merged[param_idx] = merged[param_idx] + projected

        # Apply merged gradients
        self.optimizer.zero_grad()
        for p, g in zip(all_params, merged):
            if g is not None:
                p.grad = g

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)

    @property
    def param_groups(self):
        return self.optimizer.param_groups


# ---------------------------------------------------------------------------
# Combined loss (orchestrates everything)
# ---------------------------------------------------------------------------

class BAMCombinedLoss(nn.Module):
    """
    BAM combined loss v8.

    Option A (use_mask_routing=False):
      total = contrastive_weight * L_contrastive
            + efficiency_weight  * L_efficiency
            + mrl_anchor_weight  * L_mrl_anchor
            + diversity_weight   * L_router_diversity   (pairwise prefix-dim spread)

    Option B (use_mask_routing=True, when soft_mask provided):
      total = contrastive_weight    * L_contrastive
            + efficiency_weight     * L_efficiency       (using active_dims count)
            + mrl_anchor_weight     * L_mrl_anchor
            + mask_sparsity_weight  * L_mask_sparsity
            + mask_diversity_weight * L_mask_diversity

    Call set_epoch() at the start of each training epoch.
    bloom_frequencies: list of 6 floats, computed from training data by train_bam.py.
    """

    def __init__(self, config: dict):
        super().__init__()
        mc = config["model"]
        lc = config["training"]["loss"]

        self.contrastive_weight   = lc.get("contrastive_weight", 1.0)
        self.efficiency_weight    = lc.get("efficiency_weight", 0.3)
        self.mrl_anchor_weight    = lc.get("mrl_anchor_weight", 0.2)
        self.diversity_weight     = lc.get("diversity_weight", 0.05)
        self.mask_sparsity_weight  = lc.get("mask_sparsity_weight", 0.1)
        self.mask_diversity_weight = lc.get("mask_diversity_weight", 0.05)
        self.mask_variance_weight  = lc.get("mask_variance_weight", 0.0)

        freqs = lc.get("bloom_frequencies")
        if freqs is None:
            freqs = [1 / 6] * 6
            print("WARNING: bloom_frequencies not set — using uniform. "
                  "Run compute_bloom_frequencies() in train_bam.py.")

        f = torch.tensor(freqs, dtype=torch.float).clamp(min=1e-6)
        cw = 1.0 / f.sqrt()
        cw = (cw / cw.mean()).tolist()

        ts = lc.get("temperature_schedule", {})
        self.temp_start = ts.get("start", 0.1)
        self.temp_end   = ts.get("end", 0.02)
        self.encoder_warmup_epochs = lc.get("encoder_warmup_epochs", 0)

        # Option B uses query-only masking; Option A masks both sides.
        use_mask_routing = mc.get("use_mask_routing", False)
        self.contrastive = BloomMaskedContrastiveLoss(
            temperature=self.temp_start,
            class_weights=cw,
            mask_documents=not use_mask_routing,
        )
        self.efficiency = BloomTwoFactorEfficiencyLoss(bloom_frequencies=freqs)
        self.mrl_anchor = MRLAnchorRegularizationLoss(
            mrl_dims=mc["mrl_dims"],
            temperature=self.temp_start,
        )
        self.diversity = RouterDiversityLoss()

        # Option B losses — per-level sparsity targets and diversity margin from config
        global_sparsity = mc.get("mask_sparsity_target", None)
        level_targets_cfg = lc.get("mask_level_targets", None)
        if level_targets_cfg is not None:
            level_targets = {int(k): float(v) for k, v in level_targets_cfg.items()}
        else:
            level_targets = None
        self.mask_sparsity = BloomMaskSparsityLoss(
            global_target=global_sparsity,
            level_targets=level_targets,
        )
        diversity_margin = lc.get("mask_diversity_margin", 0.3)
        self.mask_diversity = BloomMaskDiversityLoss(margin=diversity_margin)
        self.mask_variance  = BloomMaskVarianceLoss()

    def set_epoch(self, epoch: int, total_epochs: int, freeze_encoder: bool = False):
        """
        Update temperature (cosine anneal) and loss gates at start of each epoch.

        freeze_encoder=True: use fixed temp_end (annealing with frozen encoder causes
        loss-scale instability that parks the router at a fixed equilibrium).

        All non-contrastive losses (efficiency, mask-sparsity/diversity/variance) are
        gated off for the first encoder_warmup_epochs epochs. This lets the encoder and
        mask co-train on pure contrastive + MRL-anchor signal first, building a stable
        representation before compression and differentiation pressure is applied.
        """
        if freeze_encoder:
            t = self.temp_end
        else:
            progress = epoch / max(total_epochs - 1, 1)
            t = self.temp_end + 0.5 * (self.temp_start - self.temp_end) * (
                1 + math.cos(math.pi * progress)
            )
        self.contrastive.set_temperature(t)
        self.mrl_anchor.temperature = t

        past_warmup = epoch >= self.encoder_warmup_epochs
        self._active_efficiency_weight = self.efficiency_weight if past_warmup else 0.0
        self._active_mask_sparsity_weight  = self.mask_sparsity_weight  if past_warmup else 0.0
        self._active_mask_diversity_weight = self.mask_diversity_weight if past_warmup else 0.0
        self._active_mask_variance_weight  = self.mask_variance_weight  if past_warmup else 0.0

    def forward(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        query_mask: torch.Tensor,
        bloom_labels: torch.Tensor,
        continuous_dim: Optional[torch.Tensor] = None,   # Option A: router continuous_dim [B]
        active_dims: Optional[torch.Tensor] = None,      # Option B: mask active dim count [B]
        negative_embs: Optional[torch.Tensor] = None,
        all_bloom_dims: Optional[torch.Tensor] = None,   # Option A: [6] from router._all_dims()
        soft_mask: Optional[torch.Tensor] = None,        # Option B: [B, 768] raw sigmoid
    ):
        l_c, c_stats = self.contrastive(
            query_emb, positive_emb, query_mask, negative_embs, bloom_labels
        )

        # Efficiency: use soft_mask mean (continuous, non-zero gradient even at collapse)
        # for Option B; continuous_dim for Option A.
        # Using hard active_dims for Option B causes zero-gradient when mask collapses to 0,
        # making recovery impossible. soft_mask.mean(dim=-1) * D gives an equivalent
        # expected active-dim count but is always differentiable.
        if soft_mask is not None:
            # Option B: convert soft_mask fraction → expected active dims [B]
            eff_input = soft_mask.mean(dim=-1) * BloomTwoFactorEfficiencyLoss.EMBEDDING_DIM
        elif continuous_dim is not None:
            eff_input = continuous_dim
        else:
            eff_input = None
        if eff_input is not None:
            l_e, e_stats = self.efficiency(eff_input, bloom_labels)
        else:
            l_e = torch.tensor(0.0, device=query_emb.device)
            e_stats = {}

        l_a, a_stats = self.mrl_anchor(query_emb, positive_emb)

        eff_w = getattr(self, "_active_efficiency_weight", self.efficiency_weight)
        total = (
            self.contrastive_weight * l_c
            + eff_w * l_e
            + self.mrl_anchor_weight * l_a
        )
        e_stats["efficiency_weight_active"] = eff_w

        d_stats = {}

        # Option A: router diversity loss (prefix dims spread)
        if all_bloom_dims is not None:
            l_d = self.diversity(all_bloom_dims)
            total = total + self.diversity_weight * l_d
            d_stats["diversity"] = l_d.item()
            d_stats["dim_variance"] = float(all_bloom_dims.var().item())

        # Option B: mask sparsity + diversity + variance (gated behind warmup)
        if soft_mask is not None:
            l_sp, sp_stats = self.mask_sparsity(soft_mask, bloom_labels)
            l_md, md_stats = self.mask_diversity(soft_mask, bloom_labels)
            l_mv, mv_stats = self.mask_variance(soft_mask, bloom_labels)
            sp_w = getattr(self, "_active_mask_sparsity_weight",  self.mask_sparsity_weight)
            md_w = getattr(self, "_active_mask_diversity_weight", self.mask_diversity_weight)
            mv_w = getattr(self, "_active_mask_variance_weight",  self.mask_variance_weight)
            total = total + sp_w * l_sp
            total = total + md_w * l_md
            total = total + mv_w * l_mv
            d_stats.update(sp_stats)
            d_stats.update(md_stats)
            d_stats.update(mv_stats)

        stats = {}
        stats.update(c_stats)
        stats.update(e_stats)
        stats.update(a_stats)
        stats.update(d_stats)
        stats["total"] = total.item()

        bloom_names = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
        dim_src = eff_input
        if dim_src is not None:
            for b, name in enumerate(bloom_names):
                mask_b = bloom_labels == b
                if mask_b.sum() > 0:
                    stats[f"dim_{name}"] = dim_src[mask_b].mean().item()

        return total, stats

    def forward_split(
        self,
        query_emb, positive_emb, query_mask, bloom_labels,
        continuous_dim=None, active_dims=None,
        negative_embs=None, all_bloom_dims=None, soft_mask=None,
    ):
        """
        Returns (contrastive_loss, routing_loss) as separate tensors for PCGrad.
        contrastive_loss: InfoNCE only
        routing_loss:     efficiency + MRL anchor + diversity + mask losses
        """
        l_c, _ = self.contrastive(
            query_emb, positive_emb, query_mask, negative_embs, bloom_labels
        )

        if soft_mask is not None:
            eff_input = soft_mask.mean(dim=-1) * BloomTwoFactorEfficiencyLoss.EMBEDDING_DIM
        elif continuous_dim is not None:
            eff_input = continuous_dim
        else:
            eff_input = None
        eff_w = getattr(self, "_active_efficiency_weight", self.efficiency_weight)

        routing = torch.tensor(0.0, device=query_emb.device)

        if eff_input is not None:
            l_e, _ = self.efficiency(eff_input, bloom_labels)
            routing = routing + eff_w * l_e

        l_a, _ = self.mrl_anchor(query_emb, positive_emb)
        routing = routing + self.mrl_anchor_weight * l_a

        if all_bloom_dims is not None:
            l_d = self.diversity(all_bloom_dims)
            routing = routing + self.diversity_weight * l_d

        if soft_mask is not None:
            l_sp, _ = self.mask_sparsity(soft_mask, bloom_labels)
            l_md, _ = self.mask_diversity(soft_mask, bloom_labels)
            l_mv, _ = self.mask_variance(soft_mask, bloom_labels)
            sp_w = getattr(self, "_active_mask_sparsity_weight",  self.mask_sparsity_weight)
            md_w = getattr(self, "_active_mask_diversity_weight", self.mask_diversity_weight)
            mv_w = getattr(self, "_active_mask_variance_weight",  self.mask_variance_weight)
            routing = routing + sp_w * l_sp
            routing = routing + md_w * l_md
            routing = routing + mv_w * l_mv

        return self.contrastive_weight * l_c, routing
