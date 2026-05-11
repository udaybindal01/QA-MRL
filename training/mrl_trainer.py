"""Standard MRL Baseline Trainer."""

import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from tqdm import tqdm

import torch.nn.functional as F

from models.encoder import MRLEncoder
from utils.misc import AverageMeter, move_to_device, set_seed, count_parameters
from utils.logging_utils import setup_logger, WandbLogger


def _make_optimizer(params, lr, weight_decay, use_8bit=False):
    """AdamW8bit (bitsandbytes) if requested and available, else standard AdamW.
    8-bit Adam reduces optimizer state memory from 56 GB (FP32) to ~14 GB for 7B models.
    """
    if use_8bit:
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            print("WARNING: bitsandbytes not installed — falling back to standard AdamW. "
                  "Install with: pip install bitsandbytes")
    return AdamW(params, lr=lr, weight_decay=weight_decay)


def _freeze_except_last_n_layers(model: MRLEncoder, n: int) -> int:
    """Freeze all transformer layer blocks except the last n.

    Works for both BERT-style (.encoder.layer[i]) and LLaMA-style (.model.layers[i]).
    Returns the number of trainable parameters after freezing.
    """
    transformer = model.transformer

    # Collect all numbered transformer blocks
    # Try LLaMA/Qwen style first, then BERT style
    layers = None
    for attr_path in ("model.layers", "encoder.layer", "layers"):
        obj = transformer
        for part in attr_path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__"):
            layers = obj
            break

    if layers is None:
        print(f"WARNING: could not locate transformer blocks — skipping layer freeze.")
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    num_layers = len(layers)
    freeze_up_to = num_layers - n
    print(f"  Freezing layers 0–{freeze_up_to - 1}, keeping layers {freeze_up_to}–{num_layers - 1} trainable.")

    # Freeze embeddings
    for sub in ("embeddings", "embed_tokens", "wte", "wpe"):
        emb = getattr(transformer, sub, None)
        if emb is not None:
            for p in emb.parameters():
                p.requires_grad = False

    # Freeze transformer blocks
    for i, layer in enumerate(layers):
        requires_grad = i >= freeze_up_to
        for p in layer.parameters():
            p.requires_grad = requires_grad

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} params ({100 * trainable / total:.1f}%)")
    return trainable


class InfoNCELoss(nn.Module):
    """Standard InfoNCE contrastive loss with in-batch + hard negatives."""

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_emb, positive_emb, negative_embs=None):
        pos_sim = (query_emb * positive_emb).sum(dim=-1) / self.temperature

        if negative_embs is not None:
            neg_sim = torch.bmm(
                negative_embs, query_emb.unsqueeze(-1)
            ).squeeze(-1) / self.temperature
            logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        else:
            logits = torch.mm(query_emb, positive_emb.t()) / self.temperature
            labels = torch.arange(logits.size(0), device=logits.device)

        return F.cross_entropy(logits, labels)


class MRLContrastiveLoss(nn.Module):
    """Multi-resolution contrastive loss: InfoNCE at each truncation point."""

    def __init__(self, mrl_dims, temperature: float = 0.05, dim_weights=None):
        super().__init__()
        self.mrl_dims = mrl_dims
        self.infonce = InfoNCELoss(temperature)
        if dim_weights is None:
            dim_weights = [1.0 / len(mrl_dims)] * len(mrl_dims)
        self.dim_weights = dim_weights

    def forward(self, query_truncated, positive_truncated, negative_truncated=None):
        total_loss = 0.0
        per_dim_losses = {}
        for dim, weight in zip(self.mrl_dims, self.dim_weights):
            q = query_truncated[dim]
            p = positive_truncated[dim]
            n = negative_truncated[dim] if negative_truncated else None
            loss = self.infonce(q, p, n)
            total_loss += weight * loss
            per_dim_losses[f"mrl_loss_d{dim}"] = loss.item()
        return total_loss, per_dim_losses


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

        # Optional layer freezing — reduces trainable params / optimizer-state memory for 8B models
        freeze_n = tc.get("freeze_except_last_n_layers", None)
        if freeze_n is not None:
            _freeze_except_last_n_layers(model, int(freeze_n))

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        use_8bit = tc.get("optim_8bit", False)
        self.optimizer = _make_optimizer(
            trainable_params,
            lr=tc["optimizer"]["lr"],
            weight_decay=tc["optimizer"]["weight_decay"],
            use_8bit=use_8bit,
        )
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

            # Save every epoch for post-hoc best selection via find_best_epoch.py
            self.save_checkpoint(f"epoch_{epoch}")

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
        d = os.path.join(self.config["training"]["checkpoint_dir"], name)
        os.makedirs(d, exist_ok=True)
        torch.save({"model_state_dict": self.model.state_dict(), "config": self.config},
                   os.path.join(d, "checkpoint.pt"))
        self.logger.info(f"Saved to {d}")
