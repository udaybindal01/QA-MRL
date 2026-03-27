"""
Full NeurIPS experiment pipeline.

1. Build MS MARCO training data
2. Train MRL baseline on MS MARCO
3. Train QA-MRL on MS MARCO (init from baseline)
4. Evaluate both on BEIR
5. Generate all figures and tables

Usage:
    python scripts/run_neurips_pipeline.py --config configs/msmarco.yaml
"""

import argparse
import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_cmd(cmd, desc):
    print(f"\n{'='*60}")
    print(f"STEP: {desc}")
    print(f"CMD:  {cmd}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\nERROR: {desc} failed with code {result.returncode}")
        print("Continuing to next step...")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/msmarco.yaml")
    parser.add_argument("--skip_data", action="store_true", help="Skip data building")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip baseline training")
    parser.add_argument("--skip_training", action="store_true", help="Skip QA-MRL training")
    parser.add_argument("--max_train", type=int, default=100000,
                        help="Max MS MARCO training pairs (reduce for quick test)")
    args = parser.parse_args()

    config_path = args.config

    # Step 1: Build MS MARCO data
    if not args.skip_data:
        run_cmd(
            f"python data/build_msmarco.py "
            f"--output_dir /tmp/data/msmarco "
            f"--max_train {args.max_train} "
            f"--num_hard_negatives 1",
            "Build MS MARCO training data"
        )

    # Step 2: Train MRL baseline
    if not args.skip_baseline:
        run_cmd(
            f"python scripts/train_baseline_mrl.py --config {config_path}",
            "Train MRL baseline on MS MARCO"
        )

    # Step 3: Train QA-MRL (init from baseline)
    if not args.skip_training:
        baseline_ckpt = "/tmp/qa-mrl-ckpts-msmarco/mrl_baseline_best"
        run_cmd(
            f"python scripts/train_qa_mrl.py --config {config_path} "
            f"--init_encoder {baseline_ckpt}",
            "Train QA-MRL on MS MARCO (initialized from baseline)"
        )

    # Step 4: Evaluate on BEIR
    qa_ckpt = "/tmp/qa-mrl-ckpts-msmarco/best"
    bl_ckpt = "/tmp/qa-mrl-ckpts-msmarco/mrl_baseline_best"

    run_cmd(
        f"python scripts/eval_beir.py --config {config_path} "
        f"--checkpoint {qa_ckpt} "
        f"--baseline {bl_ckpt} "
        f"--datasets scifact nfcorpus fiqa arguana scidocs "
        f"--sparse",
        "Evaluate on BEIR benchmarks"
    )

    # Step 5: Efficiency benchmark
    run_cmd(
        f"python scripts/run_efficiency.py --config {config_path} "
        f"--checkpoint {qa_ckpt}",
        "Run efficiency benchmark"
    )

    # Step 6: Generate figures
    run_cmd(
        f"python scripts/generate_figures.py "
        f"--results_dir results/ --output_dir figures/",
        "Generate paper figures"
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("\nResults in: results/")
    print("Figures in: figures/")
    print("\nTo evaluate baselines:")
    print(f"  python scripts/eval_baselines.py --config {config_path} "
          f"--datasets scifact nfcorpus fiqa --baselines bm25 contriever e5-base")


if __name__ == "__main__":
    main()
