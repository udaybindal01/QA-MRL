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
    tc = config.get("training", {})
    model = MRLEncoder(
        model_name=mc["backbone"],
        embedding_dim=mc["embedding_dim"],
        mrl_dims=mc["mrl_dims"],
        pooling=mc["pooling"],
        normalize=mc["normalize_embeddings"],
        torch_dtype=mc.get("torch_dtype", None),
        gradient_checkpointing=tc.get("gradient_checkpointing", False),
        backbone_type=mc.get("backbone_type", "standard"),
        query_instruction=mc.get("query_instruction", None),
        peft_model_name=mc.get("peft_model_name", None),
    )

    if args.resume:
        ckpt_path = os.path.join(args.resume, "checkpoint.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            print(f"Resumed MRL weights from {args.resume}")
        else:
            print(f"WARNING: checkpoint not found at {ckpt_path}, starting from scratch")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        backbone = mc["backbone"]
        backbone_type = mc.get("backbone_type", "standard")
        raise RuntimeError(
            f"Model has 0 trainable parameters ({backbone}, type={backbone_type}).\n"
            f"For llm2vec/gritlm this usually means the required library is not installed\n"
            f"and the AutoModel fallback loaded a frozen PEFT checkpoint.\n"
            f"Fix: pip install llm2vec   (for LLM2Vec)\n"
            f"     pip install gritlm    (for GritLM)"
        )

    loaders = build_dataloaders(config, model.get_tokenizer())

    train_loader = loaders.get("train")
    if train_loader is None:
        raise RuntimeError(
            f"Training data not found at: {config['data']['train_path']}\n"
            f"Run the build step first to download/prepare the dataset."
        )

    trainer = MRLBaselineTrainer(config, model, train_loader, loaders.get("val"))
    trainer.train()

if __name__ == "__main__":
    main()
