"""Train standard MRL baseline.

Usage:
    # Fresh training:
    python scripts/train_baseline_mrl.py --config configs/neurips.yaml

    # Continue from existing checkpoint (MRL-continued baseline for fair comparison with BAM):
    python scripts/train_baseline_mrl.py --config configs/neurips.yaml \
        --resume /tmp/mrl-ckpts/best/ \
        --checkpoint_dir /tmp/mrl-continued-ckpts/
"""
import argparse, sys, os
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from models.encoder import MRLEncoder
from data.dataset import build_dataloaders
from training.mrl_trainer import MRLBaselineTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint dir (for MRL-continued baseline)")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Override checkpoint_dir from config")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])

    if args.checkpoint_dir:
        config["training"]["checkpoint_dir"] = args.checkpoint_dir

    mc = config["model"]
    model = MRLEncoder(model_name=mc["backbone"], embedding_dim=mc["embedding_dim"],
                       mrl_dims=mc["mrl_dims"], pooling=mc["pooling"],
                       normalize=mc["normalize_embeddings"])

    if args.resume:
        ckpt_path = os.path.join(args.resume, "checkpoint.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            print(f"Resumed MRL weights from {args.resume}")
        else:
            print(f"WARNING: checkpoint not found at {ckpt_path}, starting from scratch")

    loaders = build_dataloaders(config, model.get_tokenizer())

    trainer = MRLBaselineTrainer(config, model, loaders.get("train"), loaders.get("val"))
    trainer.train()

if __name__ == "__main__":
    main()
