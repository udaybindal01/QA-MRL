"""Run probing experiments.
Usage: python scripts/run_probing.py --config configs/default.yaml --checkpoint checkpoints/qa_mrl/best/
"""
import argparse, json, sys, os
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from models.qa_mrl import QAMRL
from models.encoder import MRLEncoder
from data.dataset import build_dataloaders
from analysis.probing import DimensionGroupProber
from analysis.visualization import plot_group_specialization_matrix
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_type", default="qa_mrl", choices=["qa_mrl", "mrl"])
    parser.add_argument("--output_dir", default="results/probing/")
    parser.add_argument("--max_samples", type=int, default=5000)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mc = config["model"]
    if args.model_type == "qa_mrl":
        model = QAMRL(config)
    else:
        model = MRLEncoder(model_name=mc["backbone"], embedding_dim=mc["embedding_dim"],
                           mrl_dims=mc["mrl_dims"])

    f = os.path.join(args.checkpoint, "checkpoint.pt")
    if os.path.exists(f):
        model.load_state_dict(torch.load(f, map_location=device)["model_state_dict"])
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(mc["backbone"])
    loaders = build_dataloaders(config, tokenizer)
    loader = loaders.get("val") or loaders.get("test")

    prober = DimensionGroupProber(mc["embedding_dim"], mc["router"]["num_groups"],
                                  config["diagnostics"]["probing"]["probe_type"])
    data = prober.extract_embeddings(model, loader, device, args.max_samples)
    results = prober.probe_all_groups(data["embeddings"],
        {"bloom_level": data["bloom_labels"], "subject_area": data["subject_labels"]})

    prober.print_results(results)
    plot_group_specialization_matrix(results, mc["router"]["num_groups"],
        save_path=os.path.join(args.output_dir, "fig2_specialization.png"))
    with open(os.path.join(args.output_dir, "probing_results.json"), "w") as f_out:
        json.dump(results, f_out, indent=2)

if __name__ == "__main__":
    main()
