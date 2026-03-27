"""Train QA-MRL model.
Usage: python scripts/train_qa_mrl.py --config configs/default.yaml
"""
import argparse, sys, os
import torch    
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed, count_parameters
from models.qa_mrl import QAMRL
from data.dataset import build_dataloaders
from training.qa_mrl_trainer import QAMRLTrainer
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_data.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init_encoder", default=None, 
                        help="Initialize encoder from MRL baseline checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])

    model = QAMRL(config)
    
    # Load baseline encoder weights if provided
    if args.init_encoder:
        ckpt_path = os.path.join(args.init_encoder, "checkpoint.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            # Load only encoder weights (not router)
            encoder_state = {k.replace("model.", "").replace("encoder.", ""): v 
                           for k, v in ckpt["model_state_dict"].items()}
            model.encoder.load_state_dict(encoder_state, strict=False)
            print(f"Loaded encoder from {args.init_encoder}")

    print(f"Parameters: {count_parameters(model)}")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])
    loaders = build_dataloaders(config, tokenizer)

    trainer = QAMRLTrainer(config, model, loaders.get("train"), loaders.get("val"))
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()

if __name__ == "__main__":
    main()
