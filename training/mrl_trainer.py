"""Standard MRL Baseline Trainer."""

import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict
from tqdm import tqdm

from models.encoder import MRLEncoder
from models.losses import MRLContrastiveLoss, InfoNCELoss
from utils.misc import AverageMeter, move_to_device, set_seed, count_parameters
from utils.logging_utils import setup_logger, WandbLogger


class MRLBaselineTrainer:

    def __init__(self, config, model: MRLEncoder, train_loader, val_loader):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = setup_logger("mrl-baseline")
        self.wandb = WandbLogger(config, enabled=config["logging"]["use_wandb"])

        tc = config["training"]
        self.num_epochs = tc["num_epochs"]
        self.grad_accum = tc["gradient_accumulation_steps"]
        self.max_grad_norm = tc["max_grad_norm"]
        self.use_fp16 = tc["fp16"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.mrl_loss = MRLContrastiveLoss(mrl_dims=config["model"]["mrl_dims"]).to(self.device)
        self.full_loss = InfoNCELoss().to(self.device)

        self.optimizer = AdamW(model.parameters(), lr=tc["optimizer"]["lr"],
                               weight_decay=tc["optimizer"]["weight_decay"])
        total_steps = len(train_loader) * self.num_epochs // self.grad_accum
        warmup = int(tc["scheduler"]["warmup_ratio"] * total_steps)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[
            LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup),
            CosineAnnealingLR(self.optimizer, T_max=max(total_steps - warmup, 1)),
        ], milestones=[warmup])
        self.scaler = GradScaler(enabled=self.use_fp16)
        self.best_metric = 0.0
        self.global_step = 0

        self.logger.info(f"Parameters: {count_parameters(model)}")

    def train(self):
        self.logger.info("Starting MRL baseline training...")
        set_seed(self.config["training"]["seed"])

        for epoch in range(self.num_epochs):
            self.model.train()
            meter = AverageMeter()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

            for step, batch in enumerate(pbar):
                batch = move_to_device(batch, self.device)
                with autocast(enabled=self.use_fp16):
                    q = self.model(batch["query_input_ids"], batch["query_attention_mask"])
                    p = self.model(batch["positive_input_ids"], batch["positive_attention_mask"])
                    loss = self.full_loss(q["full"], p["full"])
                    l_mrl, _ = self.mrl_loss(q["truncated"], p["truncated"])
                    loss = (loss + 0.5 * l_mrl) / self.grad_accum

                self.scaler.scale(loss).backward()
                meter.update(loss.item() * self.grad_accum)

                if (step + 1) % self.grad_accum == 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.global_step += 1

                pbar.set_postfix(loss=f"{meter.avg:.4f}")

            if self.val_loader:
                vm = self.validate()
                self.logger.info(f"Epoch {epoch}: {vm}")
                if vm.get("ndcg_10", 0) > self.best_metric:
                    self.best_metric = vm["ndcg_10"]
                    # NOTE: selected on val-set pairwise NDCG, not full corpus.
                    # Run scripts/find_best_epoch.py after training for true best.
                    self.save_checkpoint("inbatch_best")

        self.save_checkpoint("final")
        self.wandb.finish()

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        all_q, all_p = [], []
        for batch in self.val_loader:
            batch = move_to_device(batch, self.device)
            q = self.model(batch["query_input_ids"], batch["query_attention_mask"])
            p = self.model(batch["positive_input_ids"], batch["positive_attention_mask"])
            all_q.append(q["full"].cpu())
            all_p.append(p["full"].cpu())

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

        # Per MRL dim
        for d in self.config["model"]["mrl_dims"]:
            qt = torch.nn.functional.normalize(q[:, :d], p=2, dim=-1)
            pt = torch.nn.functional.normalize(p[:, :d], p=2, dim=-1)
            sim_d = torch.mm(qt, pt.t())
            topk = sim_d.topk(10, dim=-1).indices
            hits = (topk == torch.arange(n).unsqueeze(-1)).any(dim=-1).float()
            metrics[f"recall_10_d{d}"] = hits.mean().item()

        return metrics

    def save_checkpoint(self, name):
        d = os.path.join(self.config["training"]["checkpoint_dir"], f"mrl_baseline_{name}")
        os.makedirs(d, exist_ok=True)
        torch.save({"model_state_dict": self.model.state_dict(), "config": self.config},
                   os.path.join(d, "checkpoint.pt"))
        self.logger.info(f"Saved to {d}")
