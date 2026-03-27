"""Run dimension diagnostic analysis (go/no-go experiment).
Usage: python scripts/run_diagnostics.py --config configs/default.yaml --checkpoint checkpoints/mrl_baseline_best/
"""
import argparse, sys, os
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from models.encoder import MRLEncoder
from data.dataset import build_dataloaders
from evaluation.diagnostic import DimensionDiagnostics
from analysis.visualization import plot_dimension_importance_heatmap, plot_leave_one_out_degradation
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="results/diagnostics/")
    parser.add_argument("--num_queries", type=int, default=1000)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mc = config["model"]
    model = MRLEncoder(model_name=mc["backbone"], embedding_dim=mc["embedding_dim"],
                       mrl_dims=mc["mrl_dims"])
    ckpt = os.path.join(args.checkpoint, "checkpoint.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(mc["backbone"])
    loaders = build_dataloaders(config, tokenizer)
    loader = loaders.get("val") or loaders.get("test")

    diag = DimensionDiagnostics(mc["embedding_dim"], mc["router"]["num_groups"])

    print("1. Gradient Attribution...")
    attr = diag.gradient_attribution(model, loader, device, args.num_queries)
    diag.print_summary(attr)
    if "importance_by_bloom" in attr:
        plot_dimension_importance_heatmap(attr["importance_by_bloom"],
            save_path=os.path.join(args.output_dir, "fig1_dim_importance.png"))

    print("\n2. Leave-One-Group-Out...")
    logo = diag.leave_one_group_out(model, loader, device)
    print(f"Baseline R@10: {logo['baseline']['overall_recall@10']:.4f}")
    for gn, gd in logo["group_results"].items():
        print(f"  {gn}: degradation = {gd['degradation']['overall_recall@10']:+.4f}")
    plot_leave_one_out_degradation(logo,
        save_path=os.path.join(args.output_dir, "fig4_logo.png"))

    diag.save_diagnostics({**attr, "logo": logo},
        os.path.join(args.output_dir, "diagnostics.json"))

    vr = attr.get("stats", {}).get("bloom_variance_ratio", 0)
    print(f"\nVerdict: variance_ratio={vr:.4f} -> ",
          "PROCEED" if vr > 0.01 else "RECONSIDER")

if __name__ == "__main__":
    main()
