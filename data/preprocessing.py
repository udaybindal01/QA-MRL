"""
Data preprocessing pipeline.
Usage: python -m data.preprocessing --config configs/default.yaml
"""

import argparse
import json
import os
import random

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import EducationalCorpusBuilder
from utils.misc import load_config


def preprocess_data(config, output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("QA-MRL Data Preprocessing Pipeline")
    print("=" * 60)

    builder = EducationalCorpusBuilder(config)
    corpus = builder.build_synthetic_corpus(num_passages_per_topic_level=5)
    print(f"\nBuilt {len(corpus)} passages")

    pairs = builder.generate_training_pairs(
        num_pairs=15000, num_hard_negatives=config["data"]["num_hard_negatives"],
    )
    print(f"Generated {len(pairs)} training pairs")

    random.shuffle(pairs)
    n = len(pairs)
    splits = {
        "train": pairs[:int(0.8 * n)],
        "val": pairs[int(0.8 * n):int(0.9 * n)],
        "test": pairs[int(0.9 * n):],
    }
    for name, sp in splits.items():
        print(f"  {name}: {len(sp)} pairs")

    builder.save_corpus(os.path.join(output_dir, "corpus.jsonl"))
    for name, sp in splits.items():
        builder.save_pairs(sp, os.path.join(output_dir, f"{name}.jsonl"))

    meta = {
        "num_passages": len(corpus), "num_train": len(splits["train"]),
        "num_val": len(splits["val"]), "num_test": len(splits["test"]),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_dir", default="data/processed")
    args = parser.parse_args()
    preprocess_data(load_config(args.config), args.output_dir)
