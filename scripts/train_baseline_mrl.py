"""Train standard MRL baseline.
Usage: python scripts/train_baseline_mrl.py --config configs/default.yaml
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from models.encoder import MRLEncoder
from data.dataset import build_dataloaders
from training.mrl_trainer import MRLBaselineTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])

    mc = config["model"]
    model = MRLEncoder(model_name=mc["backbone"], embedding_dim=mc["embedding_dim"],
                       mrl_dims=mc["mrl_dims"], pooling=mc["pooling"],
                       normalize=mc["normalize_embeddings"])
    loaders = build_dataloaders(config, model.get_tokenizer())

    trainer = MRLBaselineTrainer(config, model, loaders.get("train"), loaders.get("val"))
    trainer.train()

if __name__ == "__main__":
    main()
