"""
BAM Trainer v4 — simplified two-loss training.

No warmup phases, no alpha interpolation, no temperature annealing.
The mask IS the routing — every step trains both encoder and router jointly.
Saves every epoch for post-hoc best-checkpoint selection.
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
from models.bam_losses import BAMCombinedLoss
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = BAMCombinedLoss(config).to(self.device)

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

        self.logger.info(f"BAM Parameters: {count_parameters(model)}")
        self.logger.info(f"Training {self.num_epochs} epochs (no warmup, no early stopping).")
        self.logger.info(f"Saving every epoch for post-hoc eval.")

    def train_step(self, batch):
        batch = move_to_device(batch, self.device)

        with autocast(enabled=self.use_fp16):
            outputs = self.model(
                query_input_ids=batch["query_input_ids"],
                query_attention_mask=batch["query_attention_mask"],
                positive_input_ids=batch["positive_input_ids"],
                positive_attention_mask=batch["positive_attention_mask"],
                negative_input_ids=batch.get("negative_input_ids"),
                negative_attention_mask=batch.get("negative_attention_mask"),
                bloom_labels=batch.get("bloom_label"),
            )

            loss, loss_dict = self.criterion(
                query_emb=outputs["query_embedding"],
                positive_emb=outputs["positive_embedding"],
                query_mask=outputs["query_mask"],
                continuous_dim=outputs["continuous_dim"],
                bloom_labels=batch.get("bloom_label"),
                negative_embs=outputs.get("negative_embeddings"),
                all_bloom_dims=self.model.bloom_router._all_dims(),
            )
            loss = loss / self.grad_accum

        self.scaler.scale(loss).backward()
        return loss_dict

    def train_epoch(self, epoch):
        self.model.train()
        meters = {k: AverageMeter() for k in ["total", "contrastive", "efficiency", "mrl_anchor", "avg_dim"]}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            ld = self.train_step(batch)

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
                    log = {f"train/{k}": m.avg for k, m in meters.items() if m.count > 0}
                    self.wandb.log(log, step=self.state.global_step)

                if self.state.global_step % self.eval_every == 0 and self.val_loader:
                    vm = self.validate()
                    self.wandb.log({f"val/{k}": v for k, v in vm.items()},
                                   step=self.state.global_step)
                    if vm.get("ndcg_10", 0) > self.state.best_metric:
                        self.state.best_metric = vm["ndcg_10"]
                        self.state.best_epoch = epoch
                        # NOTE: "inbatch_best" is selected on val-set pairwise NDCG,
                        # NOT full corpus retrieval. This is an approximation only.
                        # Run scripts/find_best_epoch.py after training for the true best.
                        self.save_checkpoint("inbatch_best")
                    self.model.train()

                if self.state.global_step % self.save_every == 0:
                    self.save_checkpoint(f"step_{self.state.global_step}")

            dim_str = f"{meters['avg_dim'].avg:.0f}" if meters['avg_dim'].count else "N/A"
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

        ranks = (sim.argsort(dim=-1, descending=True) == torch.arange(n).unsqueeze(-1)).nonzero()[:, 1].float()
        metrics["ndcg_10"] = (1.0 / torch.log2(ranks[ranks < 10] + 2)).sum().item() / n

        self.logger.info(f"Val: R@1={metrics['recall_1']:.4f} R@10={metrics['recall_10']:.4f} "
                         f"NDCG@10={metrics['ndcg_10']:.4f}")
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
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        # training_state only exists in BAM checkpoints, not MRL checkpoints
        if "training_state" in ckpt:
            self.state = TrainingState(**ckpt["training_state"])
        if "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                pass  # param groups differ (e.g. MRL→BAM frozen), start fresh optimizer
        self.logger.info(f"Loaded checkpoint from {path}")

    def train(self):
        self.logger.info("Starting BAM v4 training...")
        set_seed(self.config["training"]["seed"])

        bloom_names = {0: "Remember", 1: "Understand", 2: "Apply",
                       3: "Analyze", 4: "Evaluate", 5: "Create"}

        for epoch in range(self.num_epochs):
            self.state.epoch = epoch
            # Update temperature (cosine anneal) and efficiency gate
            self.criterion.set_epoch(epoch, self.num_epochs)
            t = self.criterion.contrastive.temperature
            ew = getattr(self.criterion, "_active_efficiency_weight", self.criterion.efficiency_weight)
            self.logger.info(f"Epoch {epoch}: temperature={t:.4f}, efficiency_weight={ew:.3f}")
            self.train_epoch(epoch)
            self.save_checkpoint(f"epoch_{epoch}")

            # Log learned Bloom → dim table
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
            f"Done. Best val NDCG@10={self.state.best_metric:.4f} @ epoch {self.state.best_epoch}"
        )
        self.logger.info(
            "NOTE: Val metric is in-batch only. Run eval_bam.py on each epoch checkpoint "
            "for true corpus-level metrics."
        )
        self.wandb.finish()
