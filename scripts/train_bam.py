"""
Train Bloom-Aligned Matryoshka (BAM) model.

Usage:
    python scripts/train_bam.py --config configs/bam.yaml
    python scripts/train_bam.py --config configs/bam.yaml --init_encoder /tmp/qa-mrl-ckpts/mrl_baseline_best/
"""
import argparse, sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed, count_parameters
from models.bam import BloomAlignedMRL
from data.dataset import build_dataloaders
from training.bam_trainer import BAMTrainer
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bam.yaml")
    parser.add_argument("--init_encoder", default=None,
                        help="Initialize encoder from MRL baseline checkpoint")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])

    print("=" * 60)
    print("Training BAM: Bloom-Aligned Matryoshka")
    print("=" * 60)

    model = BloomAlignedMRL(config)

    # Initialize from baseline
    if args.init_encoder:
        ckpt_path = os.path.join(args.init_encoder, "checkpoint.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.encoder.load_state_dict(ckpt["model_state_dict"], strict=False)
            print(f"Loaded encoder from {args.init_encoder}")

    print(f"Parameters: {count_parameters(model)}")

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])
    loaders = build_dataloaders(config, tokenizer)
    print(f"Train: {len(loaders.get('train', []))} batches")
    print(f"Val: {len(loaders.get('val', []))} batches")

    trainer = BAMTrainer(config, model, loaders.get("train"), loaders.get("val"))
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
