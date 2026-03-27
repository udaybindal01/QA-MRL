"""
Generate all paper figures from experiment results.

Usage:
    python scripts/generate_figures.py --results_dir results/ --output_dir figures/

This reads from:
    results/diagnostics/diagnostics.json
    results/evaluation/results.json
    results/probing/probing_results.json
    results/beir/beir_results.json
    results/efficiency/efficiency_results.json

And produces figures/ with all PNGs and PDFs for the paper.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.visualization import generate_all_figures


def main():
    parser = argparse.ArgumentParser(description="Generate all paper figures")
    parser.add_argument("--results_dir", default="results/",
                        help="Directory containing all experiment results")
    parser.add_argument("--output_dir", default="figures/",
                        help="Directory to save figures")
    args = parser.parse_args()

    print("=" * 60)
    print("Generating Paper Figures")
    print("=" * 60)

    generate_all_figures(args.results_dir, args.output_dir)

    # List generated files
    if os.path.exists(args.output_dir):
        files = sorted(os.listdir(args.output_dir))
        print(f"\nGenerated {len(files)} files:")
        for f in files:
            size = os.path.getsize(os.path.join(args.output_dir, f))
            print(f"  {f} ({size/1024:.1f} KB)")


if __name__ == "__main__":
    main()