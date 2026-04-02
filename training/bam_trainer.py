"""
BAM Trainer v5.

New in v5:
  - Two-stage training: encoder_freeze_after_epochs triggers mid-training encoder freeze
    (distinct from static freeze_encoder=True). Optimizer rebuilt at stage transition.
  - PCGrad gradient surgery: optional, off by default. Wraps optimizer with PCGradOptimizer,
    requires two separate backward calls via criterion.forward_split().
  - Bloom noise injection: bloom_noise_rate randomly flips a fraction of Bloom labels
    per batch, forcing routing robustness to classifier errors.
  - Stage logging: logs "Stage 1 / Stage 2" at start of each epoch.

v4 features retained:
  - Resume support: skip already-completed epochs
  - Saves every epoch for post-hoc best-checkpoint selection
  - Temperature cosine annealing (frozen encoder → fixed temp_end)
"""

import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict
from tqdm import tqdm

from models.bam import BloomAlignedMRL
from models.bam_losses import BAMCombinedLoss, PCGradOptimizer
from utils.misc import (
    AverageMeter, TrainingState,
    move_to_device, set_seed, count_parameters,
)
from utils.logging_utils import setup_logger, WandbLogger


class BAMTrainer:

    def __init__(self, config, model: BloomAlignedMRL, train_loader, val_loader):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = setup_logger("bam-trainer")
        self.wandb = WandbLogger(config, enabled=config["logging"]["use_wandb"])

        tc = config["training"]
        self.num_epochs = tc["num_epochs"]
        self.grad_accum = tc["gradient_accumulation_steps"]
        self.max_grad_norm = tc["max_grad_norm"]
        self.use_fp16 = tc["fp16"]
        self.eval_every = tc["eval_every_n_steps"]
        self.save_every = tc["save_every_n_steps"]
        self.checkpoint_dir = tc["checkpoint_dir"]

        # v5 new params
        self.encoder_freeze_after = tc.get("encoder_freeze_after_epochs", None)
        self.use_pcgrad = tc.get("use_pcgrad", False)
        self.bloom_noise_rate = tc.get("bloom_noise_rate", 0.0)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = BAMCombinedLoss(config).to(self.device)

        self.optimizer = AdamW(
            model.get_parameter_groups(config),
            weight_decay=tc["optimizer"]["weight_decay"],
        )

        # Wrap with PCGrad if requested
        if self.use_pcgrad:
            self.pcgrad = PCGradOptimizer(self.optimizer)
            self.logger.info("PCGrad gradient surgery ENABLED.")
        else:
            self.pcgrad = None

        total_steps = len(train_loader) * self.num_epochs // self.grad_accum
        warmup_steps = int(tc["scheduler"]["warmup_ratio"] * total_steps)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[
            LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_steps),
            CosineAnnealingLR(self.optimizer, T_max=max(total_steps - warmup_steps, 1)),
        ], milestones=[warmup_steps])
        self.scaler = GradScaler(enabled=self.use_fp16)

        self.state = TrainingState()
        self._resumed = False
        self._stage2_active = False

        self.logger.info(f"BAM Parameters: {count_parameters(model)}")
        self.logger.info(f"Training {self.num_epochs} epochs.")
        if self.encoder_freeze_after is not None:
            self.logger.info(
                f"Two-stage: encoder freezes after epoch {self.encoder_freeze_after - 1}."
            )

    def _maybe_transition_to_stage2(self, epoch: int):
        """Freeze encoder and rebuild optimizer for stage 2 if the epoch threshold is hit."""
        if (
            self.encoder_freeze_after is not None
            and epoch >= self.encoder_freeze_after
            and not self._stage2_active
        ):
            self.logger.info(
                f"Epoch {epoch}: STAGE 2 — freezing encoder, router-only training."
            )
            self.model.freeze_encoder()

            # Rebuild optimizer for router/mask-head params only
            oc = self.config["training"]["optimizer"]
            fast_lr = oc.get("router_lr", oc["encoder_lr"] * 10)
            routing_params = list(self.model.bloom_router.parameters())
            if hasattr(self.model, "bloom_mask_head"):
                routing_params += list(self.model.bloom_mask_head.parameters())

            self.optimizer = AdamW(
                [{"params": routing_params, "lr": fast_lr}],
                weight_decay=oc["weight_decay"],
            )
            if self.use_pcgrad:
                self.pcgrad = PCGradOptimizer(self.optimizer)

            # Rebuild scheduler for remaining epochs
            remaining_steps = (
                len(self.train_loader) * (self.num_epochs - epoch) // self.grad_accum
            )
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=max(remaining_steps, 1)
            )
            self._stage2_active = True

    def _inject_bloom_noise(self, bloom_label: torch.Tensor) -> torch.Tensor:
        """Randomly flip a fraction of Bloom labels (0-5 range) for robustness."""
        if self.bloom_noise_rate <= 0.0 or bloom_label is None:
            return bloom_label
        noise_mask = torch.rand_like(bloom_label.float()) < self.bloom_noise_rate
        random_labels = torch.randint(0, 6, bloom_label.shape, device=bloom_label.device)
        return torch.where(noise_mask, random_labels, bloom_label)

    def train_step(self, batch):
        batch = move_to_device(batch, self.device)

        bloom_label = batch.get("bloom_label")
        bloom_label = self._inject_bloom_noise(bloom_label)

        if self.use_pcgrad and self.pcgrad is not None:
            return self._train_step_pcgrad(batch, bloom_label)
        else:
            return self._train_step_standard(batch, bloom_label)

    def _train_step_standard(self, batch, bloom_label):
        with autocast(enabled=self.use_fp16):
            outputs = self.model(
                query_input_ids=batch["query_input_ids"],
                query_attention_mask=batch["query_attention_mask"],
                positive_input_ids=batch["positive_input_ids"],
                positive_attention_mask=batch["positive_attention_mask"],
                negative_input_ids=batch.get("negative_input_ids"),
                negative_attention_mask=batch.get("negative_attention_mask"),
                bloom_labels=bloom_label,
            )

            # Pass all_bloom_dims only for Option A (prefix routing)
            all_bloom_dims = (
                None if self.model.use_mask_routing
                else self.model.bloom_router._all_dims()
            )

            loss, loss_dict = self.criterion(
                query_emb=outputs["query_embedding"],
                positive_emb=outputs["positive_embedding"],
                query_mask=outputs["query_mask"],
                bloom_labels=bloom_label,
                continuous_dim=outputs.get("continuous_dim"),
                active_dims=outputs.get("active_dims"),
                negative_embs=outputs.get("negative_embeddings"),
                all_bloom_dims=all_bloom_dims,
                soft_mask=outputs.get("soft_mask"),
            )
            loss = loss / self.grad_accum

        self.scaler.scale(loss).backward()
        return loss_dict

    def _train_step_pcgrad(self, batch, bloom_label):
        with autocast(enabled=self.use_fp16):
            outputs = self.model(
                query_input_ids=batch["query_input_ids"],
                query_attention_mask=batch["query_attention_mask"],
                positive_input_ids=batch["positive_input_ids"],
                positive_attention_mask=batch["positive_attention_mask"],
                negative_input_ids=batch.get("negative_input_ids"),
                negative_attention_mask=batch.get("negative_attention_mask"),
                bloom_labels=bloom_label,
            )

            all_bloom_dims = (
                None if self.model.use_mask_routing
                else self.model.bloom_router._all_dims()
            )

            l_c, l_r = self.criterion.forward_split(
                query_emb=outputs["query_embedding"],
                positive_emb=outputs["positive_embedding"],
                query_mask=outputs["query_mask"],
                bloom_labels=bloom_label,
                continuous_dim=outputs.get("continuous_dim"),
                active_dims=outputs.get("active_dims"),
                negative_embs=outputs.get("negative_embeddings"),
                all_bloom_dims=all_bloom_dims,
                soft_mask=outputs.get("soft_mask"),
            )
            l_c = l_c / self.grad_accum
            l_r = l_r / self.grad_accum

        self.pcgrad.pc_backward([l_c, l_r])
        total = (l_c + l_r).item() * self.grad_accum
        return {"total": total, "contrastive": l_c.item() * self.grad_accum,
                "routing": l_r.item() * self.grad_accum}

    def train_epoch(self, epoch):
        self.model.train()
        meters = {
            k: AverageMeter()
            for k in ["total", "contrastive", "efficiency", "mrl_anchor", "avg_dim"]
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            ld = self.train_step(batch)

            for k, v in ld.items():
                if k in meters:
                    meters[k].update(v)

            if (step + 1) % self.grad_accum == 0:
                if not self.use_pcgrad:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # PCGrad already applied gradients via pc_backward
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.pcgrad.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.state.global_step += 1

                if self.state.global_step % self.config["logging"]["log_every_n_steps"] == 0:
                    log = {f"train/{k}": m.avg for k, m in meters.items() if m.count > 0}
                    self.wandb.log(log, step=self.state.global_step)

                if self.state.global_step % self.eval_every == 0 and self.val_loader:
                    vm = self.validate()
                    self.wandb.log(
                        {f"val/{k}": v for k, v in vm.items()},
                        step=self.state.global_step,
                    )
                    if vm.get("ndcg_10", 0) > self.state.best_metric:
                        self.state.best_metric = vm["ndcg_10"]
                        self.state.best_epoch = epoch
                        self.save_checkpoint("inbatch_best")
                    self.model.train()

                if self.state.global_step % self.save_every == 0:
                    self.save_checkpoint(f"step_{self.state.global_step}")

            dim_str = f"{meters['avg_dim'].avg:.0f}" if meters["avg_dim"].count else "N/A"
            pbar.set_postfix(
                loss=f"{meters['total'].avg:.4f}",
                dim=dim_str,
            )

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        all_q, all_p = [], []

        for batch in tqdm(self.val_loader, desc="Val", leave=False):
            batch = move_to_device(batch, self.device)
            out = self.model(
                query_input_ids=batch["query_input_ids"],
                query_attention_mask=batch["query_attention_mask"],
                positive_input_ids=batch["positive_input_ids"],
                positive_attention_mask=batch["positive_attention_mask"],
                bloom_labels=batch.get("bloom_label"),
            )
            all_q.append(out["query_masked"].cpu())
            all_p.append(out["positive_embedding"].cpu())

        q, p = torch.cat(all_q), torch.cat(all_p)
        sim = torch.mm(q, p.t())
        n = sim.size(0)

        metrics = {}
        for k in [1, 5, 10]:
            topk = sim.topk(k, dim=-1).indices
            hits = (topk == torch.arange(n).unsqueeze(-1)).any(dim=-1).float()
            metrics[f"recall_{k}"] = hits.mean().item()

        ranks = (
            sim.argsort(dim=-1, descending=True) == torch.arange(n).unsqueeze(-1)
        ).nonzero()[:, 1].float()
        metrics["ndcg_10"] = (
            (1.0 / torch.log2(ranks[ranks < 10] + 2)).sum().item() / n
        )

        self.logger.info(
            f"Val: R@1={metrics['recall_1']:.4f} "
            f"R@10={metrics['recall_10']:.4f} "
            f"NDCG@10={metrics['ndcg_10']:.4f}"
        )
        return metrics

    def save_checkpoint(self, name):
        d = os.path.join(self.checkpoint_dir, name)
        os.makedirs(d, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "training_state": vars(self.state),
                "config": self.config,
            },
            os.path.join(d, "checkpoint.pt"),
        )

    def load_checkpoint(self, path):
        ckpt = torch.load(
            os.path.join(path, "checkpoint.pt"), map_location=self.device
        )
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "training_state" in ckpt:
            self.state = TrainingState(**ckpt["training_state"])
            self._resumed = True
        if "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                pass  # param groups differ (e.g. MRL→BAM), start fresh
        self.logger.info(
            f"Loaded checkpoint from {path} (epoch {self.state.epoch})"
        )

    def train(self):
        self.logger.info("Starting BAM v5 training...")
        set_seed(self.config["training"]["seed"])

        bloom_names = {
            0: "Remember", 1: "Understand", 2: "Apply",
            3: "Analyze",  4: "Evaluate",   5: "Create",
        }

        start_epoch = (self.state.epoch + 1) if self._resumed else 0
        if self._resumed:
            self.logger.info(
                f"Resuming from epoch {self.state.epoch}, "
                f"starting at epoch {start_epoch}"
            )
            # Restore stage 2 if the saved epoch is past the freeze threshold
            if (
                self.encoder_freeze_after is not None
                and start_epoch >= self.encoder_freeze_after
            ):
                self._maybe_transition_to_stage2(start_epoch)

        for epoch in range(start_epoch, self.num_epochs):
            self.state.epoch = epoch

            # Check two-stage transition
            self._maybe_transition_to_stage2(epoch)

            # Determine if encoder is frozen (static or dynamic stage 2)
            freeze_encoder = (
                self.config["training"].get("freeze_encoder", False)
                or self._stage2_active
            )

            self.criterion.set_epoch(epoch, self.num_epochs, freeze_encoder=freeze_encoder)
            t = self.criterion.contrastive.temperature
            ew = getattr(
                self.criterion, "_active_efficiency_weight",
                self.criterion.efficiency_weight,
            )
            stage = "Stage 2 (encoder frozen)" if freeze_encoder else "Stage 1 (full model)"
            if self.bloom_noise_rate > 0:
                stage += f" [noise={self.bloom_noise_rate:.0%}]"
            self.logger.info(
                f"Epoch {epoch}: {stage} | temperature={t:.4f}, efficiency_weight={ew:.3f}"
            )

            self.train_epoch(epoch)
            self.save_checkpoint(f"epoch_{epoch}")

            # Log Bloom → dim table (Option A) or skip (Option B)
            if not self.model.use_mask_routing:
                table = self.model.bloom_router.get_dim_table()
                table_str = ", ".join(
                    f"{bloom_names.get(b, b)}→{d}" for b, d in sorted(table.items())
                )
                self.logger.info(f"Bloom→Dim table: {table_str}")

            if self.val_loader:
                vm = self.validate()
                self.state.metrics_history.append(vm)
                self.logger.info(
                    f"Epoch {epoch}: val_ndcg={vm.get('ndcg_10', 0):.4f} "
                    f"(best={self.state.best_metric:.4f} @ epoch {self.state.best_epoch})"
                )

        self.save_checkpoint("final")
        self.logger.info(
            f"Done. Best val NDCG@10={self.state.best_metric:.4f} "
            f"@ epoch {self.state.best_epoch}"
        )
        self.logger.info(
            "NOTE: Val metric is in-batch only. Run eval_bam.py on each epoch checkpoint "
            "for true corpus-level metrics."
        )
        self.wandb.finish()
