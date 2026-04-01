"""
Train Bloom-Aligned Matryoshka (BAM) model.

Usage:
    python scripts/train_bam.py --config configs/bam.yaml
    python scripts/train_bam.py --config configs/bam.yaml --init_encoder /tmp/mrl-ckpts/mrl_baseline_final/
"""
import argparse, json, sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed, count_parameters
from models.bam import BloomAlignedMRL
from data.dataset import build_dataloaders
from training.bam_trainer import BAMTrainer
from transformers import AutoTokenizer


def compute_bloom_frequencies(train_path: str):
    """
    Scan training JSONL and compute Bloom level frequencies.
    Returns list of 6 floats (0-indexed: [Remember, ..., Create]).
    bloom_level in data is 1-indexed (1=Remember, 6=Create).
    """
    counts = [0] * 6
    total = 0
    with open(train_path) as f:
        for line in f:
            s = json.loads(line.strip())
            level = s.get("bloom_level", 1)
            if 1 <= level <= 6:
                counts[level - 1] += 1
                total += 1
    if total == 0:
        return [1/6] * 6
    freqs = [c / total for c in counts]
    print("Bloom level frequencies (0-indexed, from training data):")
    names = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
    for i, (n, f) in enumerate(zip(names, freqs)):
        print(f"  {n:12s} (bloom={i}): {f:.4f}  ({counts[i]} samples)")
    return freqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bam.yaml")
    parser.add_argument("--init_encoder", default=None,
                        help="Initialize encoder from MRL baseline checkpoint dir")
    parser.add_argument("--resume", default=None,
                        help="Resume from BAM checkpoint dir (e.g. for hard-negative fine-tuning)")
    parser.add_argument("--train_path", default=None,
                        help="Override data.train_path (e.g. train_hard.jsonl)")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Override training.checkpoint_dir")
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Override training.num_epochs (e.g. 5 for fine-tuning pass)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])

    # CLI overrides
    if args.train_path:
        config["data"]["train_path"] = args.train_path
    if args.checkpoint_dir:
        config["training"]["checkpoint_dir"] = args.checkpoint_dir
    if args.num_epochs is not None:
        config["training"]["num_epochs"] = args.num_epochs

    print("=" * 60)
    print("Training BAM: Bloom-Aligned Matryoshka")
    print("=" * 60)

    # Compute Bloom frequencies from training data and inject into loss config
    train_path = config["data"]["train_path"]
    freqs = compute_bloom_frequencies(train_path)
    config["training"]["loss"]["bloom_frequencies"] = freqs

    model = BloomAlignedMRL(config)

    # Initialize encoder from MRL baseline (warm start)
    if args.init_encoder:
        ckpt_path = os.path.join(args.init_encoder, "checkpoint.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.encoder.load_state_dict(ckpt["model_state_dict"], strict=False)
            print(f"Loaded encoder weights from {args.init_encoder}")
        else:
            print(f"WARNING: checkpoint not found at {ckpt_path}, starting from scratch")

    print(f"Parameters: {count_parameters(model)}")

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])
    loaders = build_dataloaders(config, tokenizer)
    print(f"Train: {len(loaders.get('train', []))} batches")
    print(f"Val:   {len(loaders.get('val', []))} batches")
    print(f"Checkpoints → {config['training']['checkpoint_dir']}")

    trainer = BAMTrainer(config, model, loaders.get("train"), loaders.get("val"))
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
