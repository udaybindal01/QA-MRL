"""
BAM Trainer v3 with curriculum progression.

v3 changes:
- Removed negative_blooms from training (docs have no Bloom labels)
- Removed early stopping on val NDCG (misleading — val is in-batch,
  real eval is full-corpus FAISS). Train all epochs, eval post-hoc.
- Save checkpoint every epoch for post-hoc best selection
- Curriculum progress controls temperature annealing, not Bloom weighting
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
        self.warmup_epochs = tc["phases"]["mrl_warmup_epochs"]

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
        self.logger.info(f"Strategy: {self.warmup_epochs} warmup + "
                         f"{self.num_epochs - self.warmup_epochs} full, "
                         f"curriculum-scheduled losses")
        self.logger.info(f"Training all {self.num_epochs} epochs (no early stopping). "
                         f"Saving every epoch for post-hoc eval.")

    def get_curriculum_progress(self, epoch, step, steps_per_epoch):
        """Overall training progress 0->1 for curriculum scheduling."""
        total_steps = self.num_epochs * steps_per_epoch
        current_step = epoch * steps_per_epoch + step
        return min(1.0, current_step / total_steps)

    def get_policy_alpha(self, epoch, step, steps_per_epoch):
        """Policy integration alpha: 0->1 over warmup epochs."""
        if epoch >= self.warmup_epochs:
            return 1.0
        progress = (epoch * steps_per_epoch + step) / (self.warmup_epochs * steps_per_epoch)
        return min(1.0, progress)

    def train_step(self, batch, alpha, curriculum_progress):
        batch = move_to_device(batch, self.device)

        with autocast(enabled=self.use_fp16):
            outputs = self.model(
                query_input_ids=batch["query_input_ids"],
                query_attention_mask=batch["query_attention_mask"],
                positive_input_ids=batch["positive_input_ids"],
                positive_attention_mask=batch["positive_attention_mask"],
                negative_input_ids=batch["negative_input_ids"],
                negative_attention_mask=batch["negative_attention_mask"],
                bloom_labels=batch.get("bloom_label"),
            )

            # Interpolate adaptive/full embedding
            full_emb = outputs["query_embedding"]
            adaptive_emb = outputs["query_adaptive"]
            effective_emb = (1 - alpha) * full_emb + alpha * adaptive_emb
            effective_emb = F.normalize(effective_emb, p=2, dim=-1)

            phase = "joint" if alpha > 0.5 else "mrl_warmup"

            # v3: No negative_blooms — docs have no Bloom labels
            loss, loss_dict = self.criterion(
                query_emb=outputs["query_embedding"],
                positive_emb=outputs["positive_embedding"],
                query_adaptive=effective_emb,
                query_truncated=outputs["query_truncated"],
                positive_truncated=outputs["positive_truncated"],
                policy_output=outputs["policy_output"],
                bloom_labels=batch.get("bloom_label"),
                negative_embs=outputs.get("negative_embeddings"),
                bloom_classifier_output=outputs.get("bloom_classifier_output"),
                phase=phase,
                curriculum_progress=curriculum_progress,
            )
            loss = loss / self.grad_accum

        self.scaler.scale(loss).backward()

        loss_dict["alpha"] = alpha
        loss_dict["curriculum"] = curriculum_progress
        po = outputs.get("policy_output", {})
        if "avg_dim" in po:
            loss_dict["avg_dim"] = po["avg_dim"].item()

        return loss_dict

    def train_epoch(self, epoch):
        self.model.train()
        spe = len(self.train_loader)
        phase_name = "warmup" if epoch < self.warmup_epochs else "full"

        meters = {k: AverageMeter() for k in [
            "total", "contrastive", "policy_contrastive",
            "policy_avg_dim", "policy_entropy", "alpha", "curriculum",
            "curriculum_temperature", "bloom_cls_acc",
        ]}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [{phase_name}]")
        for step, batch in enumerate(pbar):
            alpha = self.get_policy_alpha(epoch, step, spe)
            curriculum = self.get_curriculum_progress(epoch, step, spe)
            ld = self.train_step(batch, alpha, curriculum)

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
                    # Track best for logging, but do NOT early stop
                    if vm.get("ndcg_10", 0) > self.state.best_metric:
                        self.state.best_metric = vm["ndcg_10"]
                        self.state.best_epoch = epoch
                        self.save_checkpoint("best")
                    self.model.train()

                if self.state.global_step % self.save_every == 0:
                    self.save_checkpoint(f"step_{self.state.global_step}")

            dim_str = f"{meters['policy_avg_dim'].avg:.0f}" if meters['policy_avg_dim'].count else "N/A"
            ent_str = f"{meters['policy_entropy'].avg:.2f}" if meters['policy_entropy'].count else "N/A"
            bacc_str = f"{meters['bloom_cls_acc'].avg:.2f}" if meters['bloom_cls_acc'].count else "N/A"
            pbar.set_postfix(
                loss=f"{meters['total'].avg:.4f}",
                alpha=f"{alpha:.2f}",
                dim=dim_str,
                bloom_acc=bacc_str,
                cur=f"{curriculum:.2f}",
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
            all_q.append(out["query_adaptive"].cpu())
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
        self.state = TrainingState(**ckpt["training_state"])

    def train(self):
        self.logger.info("Starting BAM v3 training (query-only Bloom, no early stopping)...")
        set_seed(self.config["training"]["seed"])
        for epoch in range(self.num_epochs):
            self.state.epoch = epoch
            self.train_epoch(epoch)

            # Save every epoch for post-hoc evaluation
            self.save_checkpoint(f"epoch_{epoch}")

            # Log learned Bloom → dim table
            if hasattr(self.model, 'get_bloom_dim_table'):
                table = self.model.get_bloom_dim_table()
                bloom_names = {0: "Remember", 1: "Understand", 2: "Apply",
                               3: "Analyze", 4: "Evaluate", 5: "Create"}
                table_str = ", ".join(f"{bloom_names.get(b, b)}→{d}" for b, d in sorted(table.items()))
                self.logger.info(f"Bloom→Dim table: {table_str}")

            if self.val_loader:
                vm = self.validate()
                self.state.metrics_history.append(vm)
                self.logger.info(f"Epoch {epoch}: val_ndcg={vm.get('ndcg_10', 0):.4f} "
                                 f"(best={self.state.best_metric:.4f} @ epoch {self.state.best_epoch})")

        self.save_checkpoint("final")
        self.logger.info(f"Done. Best val NDCG@10={self.state.best_metric:.4f} @ epoch {self.state.best_epoch}")
        self.logger.info(f"NOTE: Val metric is in-batch only. Run eval_bam.py on each epoch checkpoint "
                         f"for true corpus-level metrics.")
        self.wandb.finish()
