"""
QA-MRL Trainer v2 - Gradual Router Integration.

The v1 hard freeze/unfreeze approach caused catastrophic performance drops
because the router learned masks for frozen embeddings that became invalid
when the encoder was unfrozen.

New strategy:
  Phase 1 (Warmup, epochs 0-2): Train encoder+router jointly from the start,
      but with router_alpha ramping from 0→1. At alpha=0, the mask is all-ones
      (equivalent to standard MRL). At alpha=1, the mask is fully from the router.
      This lets the router gradually take effect without disrupting the encoder.

  Phase 2 (Full, epochs 3+): Router fully active (alpha=1), all losses active,
      everything trainable end-to-end.

The key insight: instead of freezing components, we interpolate:
    effective_mask = (1 - alpha) * ones + alpha * router_mask
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict
from tqdm import tqdm

from models.qa_mrl import QAMRL
from models.losses import QAMRLLoss
from utils.misc import (
    AverageMeter, EarlyStopping, TrainingState,
    move_to_device, set_seed, count_parameters,
)
from utils.logging_utils import setup_logger, WandbLogger


class QAMRLTrainer:

    def __init__(self, config, model: QAMRL, train_loader, val_loader):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = setup_logger("qa-mrl-trainer")
        self.wandb = WandbLogger(config, enabled=config["logging"]["use_wandb"])

        tc = config["training"]
        self.num_epochs = tc["num_epochs"]
        self.grad_accum = tc["gradient_accumulation_steps"]
        self.max_grad_norm = tc["max_grad_norm"]
        self.use_fp16 = tc["fp16"]
        self.eval_every = tc["eval_every_n_steps"]
        self.save_every = tc["save_every_n_steps"]
        self.checkpoint_dir = tc["checkpoint_dir"]

        # Phase config
        phases = tc["phases"]
        self.warmup_epochs = phases["mrl_warmup_epochs"]  # Router ramp-up phase
        self.total_epochs = self.num_epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = QAMRLLoss(config).to(self.device)

        # Everything trainable from the start — no freezing
        self.optimizer = AdamW(
            model.get_parameter_groups(config),
            weight_decay=tc["optimizer"]["weight_decay"],
        )
        total_steps = len(train_loader) * self.num_epochs // self.grad_accum
        warmup_steps = int(tc["scheduler"]["warmup_ratio"] * total_steps)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[
            LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_steps),
            CosineAnnealingLR(self.optimizer, T_max=max(total_steps - warmup_steps, 1)),
        ], milestones=[warmup_steps])
        self.scaler = GradScaler(enabled=self.use_fp16)

        self.state = TrainingState()
        self.early_stopping = EarlyStopping(patience=5)

        self.logger.info(f"Parameters: {count_parameters(model)}")
        self.logger.info(f"Strategy: Gradual router integration over {self.warmup_epochs} epochs, "
                         f"then full routing for remaining {self.total_epochs - self.warmup_epochs} epochs")

    def get_router_alpha(self, epoch: int, step: int, steps_per_epoch: int) -> float:
        """
        Compute router interpolation weight.
        Ramps linearly from 0 to 1 over warmup_epochs.
        After warmup, stays at 1.0.
        """
        if epoch >= self.warmup_epochs:
            return 1.0
        # Linear ramp within warmup phase
        progress = (epoch * steps_per_epoch + step) / (self.warmup_epochs * steps_per_epoch)
        return min(1.0, progress)

    def apply_alpha_to_mask(self, mask: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Interpolate between all-ones mask and router mask.
        At alpha=0: returns ones (standard retrieval, no routing)
        At alpha=1: returns the router's mask fully
        """
        if alpha >= 1.0:
            return mask
        ones = torch.ones_like(mask)
        return (1.0 - alpha) * ones + alpha * mask

    def train_step(self, batch, alpha):
        batch = move_to_device(batch, self.device)

        with autocast(enabled=self.use_fp16):
            outputs = self.model(
                query_input_ids=batch["query_input_ids"],
                query_attention_mask=batch["query_attention_mask"],
                positive_input_ids=batch["positive_input_ids"],
                positive_attention_mask=batch["positive_attention_mask"],
                negative_input_ids=batch["negative_input_ids"],
                negative_attention_mask=batch["negative_attention_mask"],
                learner_features=batch.get("learner_features"),
            )

            # Apply alpha interpolation to the mask
            raw_mask = outputs["query_mask"]
            effective_mask = self.apply_alpha_to_mask(raw_mask, alpha)

            # Recompute masked embeddings with interpolated mask
            query_emb = outputs["query_embedding"]
            masked_query = F.normalize(query_emb * effective_mask, p=2, dim=-1)

            # Also handle document mask if asymmetric
            doc_mask = outputs.get("positive_mask")
            if doc_mask is not None:
                doc_mask = self.apply_alpha_to_mask(doc_mask, alpha)

            # Determine phase for loss computation
            phase = "joint" if alpha > 0.5 else "mrl_warmup"

            loss, loss_dict = self.criterion(
                query_emb=query_emb,
                positive_emb=outputs["positive_embedding"],
                query_mask=effective_mask,
                query_truncated=outputs["query_truncated"],
                positive_truncated=outputs["positive_truncated"],
                bloom_labels=batch.get("bloom_label"),
                subject_labels=batch.get("subject_label"),
                doc_mask=doc_mask,
                negative_embs=outputs.get("negative_embeddings"),
                phase=phase,
            )
            loss = loss / self.grad_accum

        self.scaler.scale(loss).backward()

        # Log router stats
        rs = outputs.get("router_stats", {})
        loss_dict["alpha"] = alpha
        if "active_dims" in rs:
            loss_dict["active_dims"] = rs["active_dims"].item()
        if "active_groups" in rs:
            loss_dict["active_groups"] = rs["active_groups"].item()
        # Effective active dims (after alpha interpolation)
        loss_dict["effective_active_dims"] = (effective_mask > 0.5).float().sum(dim=-1).mean().item()

        return loss_dict

    def train_epoch(self, epoch):
        self.model.train()
        steps_per_epoch = len(self.train_loader)

        meters = {k: AverageMeter() for k in
                  ["total", "contrastive", "mrl", "specialization", "sparsity",
                   "bloom_cls", "active_dims", "active_groups", "alpha",
                   "effective_active_dims"]}

        phase_name = "warmup" if epoch < self.warmup_epochs else "full"
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [{phase_name}]")

        for step, batch in enumerate(pbar):
            alpha = self.get_router_alpha(epoch, step, steps_per_epoch)
            ld = self.train_step(batch, alpha)

            for k, v in ld.items():
                if k in meters:
                    meters[k].update(v)

            if (step + 1) % self.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.state.global_step += 1

                if self.state.global_step % self.config["logging"]["log_every_n_steps"] == 0:
                    self.wandb.log(
                        {f"train/{k}": m.avg for k, m in meters.items() if m.count > 0},
                        step=self.state.global_step,
                    )

                if self.state.global_step % self.eval_every == 0 and self.val_loader:
                    vm = self.validate()
                    self.wandb.log({f"val/{k}": v for k, v in vm.items()},
                                   step=self.state.global_step)
                    if vm.get("ndcg_10", 0) > self.state.best_metric:
                        self.state.best_metric = vm["ndcg_10"]
                        self.state.best_epoch = epoch
                        self.save_checkpoint("best")
                    self.model.train()

                if self.state.global_step % self.save_every == 0:
                    self.save_checkpoint(f"step_{self.state.global_step}")

            pbar.set_postfix(
                loss=f"{meters['total'].avg:.4f}",
                alpha=f"{alpha:.2f}",
                dims=f"{meters['effective_active_dims'].avg:.0f}" if meters['effective_active_dims'].count else "N/A",
            )

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        all_q, all_p, all_m = [], [], []

        for batch in tqdm(self.val_loader, desc="Val", leave=False):
            batch = move_to_device(batch, self.device)
            out = self.model(
                query_input_ids=batch["query_input_ids"],
                query_attention_mask=batch["query_attention_mask"],
                positive_input_ids=batch["positive_input_ids"],
                positive_attention_mask=batch["positive_attention_mask"],
                learner_features=batch.get("learner_features"),
            )
            all_q.append(out["query_masked"].cpu())
            all_p.append(out["positive_masked"].cpu())
            all_m.append(out["query_mask"].cpu())

        q, p, m = torch.cat(all_q), torch.cat(all_p), torch.cat(all_m)
        sim = torch.mm(q, p.t())
        n = sim.size(0)

        metrics = {}
        for k in [1, 5, 10]:
            topk = sim.topk(k, dim=-1).indices
            hits = (topk == torch.arange(n).unsqueeze(-1)).any(dim=-1).float()
            metrics[f"recall_{k}"] = hits.mean().item()

        ranks = (sim.argsort(dim=-1, descending=True) == torch.arange(n).unsqueeze(-1)).nonzero()[:, 1].float()
        metrics["ndcg_10"] = (1.0 / torch.log2(ranks[ranks < 10] + 2)).sum().item() / n
        metrics["avg_active_dims"] = (m > 0.5).float().sum(dim=-1).mean().item()

        self.logger.info(f"Val: R@1={metrics['recall_1']:.4f} R@10={metrics['recall_10']:.4f} "
                         f"NDCG@10={metrics['ndcg_10']:.4f} dims={metrics['avg_active_dims']:.0f}")
        return metrics

    def save_checkpoint(self, name):
        d = os.path.join(self.checkpoint_dir, name)
        os.makedirs(d, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "training_state": vars(self.state),
            "config": self.config,
        }, os.path.join(d, "checkpoint.pt"))

    def load_checkpoint(self, path):
        ckpt = torch.load(os.path.join(path, "checkpoint.pt"), map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.state = TrainingState(**ckpt["training_state"])

    def train(self):
        self.logger.info("Starting QA-MRL training (gradual router integration)...")
        set_seed(self.config["training"]["seed"])

        for epoch in range(self.num_epochs):
            self.state.epoch = epoch
            self.train_epoch(epoch)

            if self.val_loader:
                vm = self.validate()
                self.state.metrics_history.append(vm)
                if self.early_stopping(vm.get("ndcg_10", 0)):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

        self.save_checkpoint("final")
        self.logger.info(f"Done. Best NDCG@10={self.state.best_metric:.4f} @ epoch {self.state.best_epoch}")
        self.wandb.finish()