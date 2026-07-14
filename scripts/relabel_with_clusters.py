"""
Relabel training/val/test data with k-means cluster IDs in place of Bloom
labels, so BAM-PQ's training loop can be reused verbatim to train a
"cluster-routed" model.

The trainer reads predicted routing labels from `<data_path>.bloom_cache.json`
(a JSON list of 0-indexed ints, one per query). This script:
  1. Loads an MRL-fine-tuned encoder.
  2. Encodes all training queries and fits k-means (k=6 by default).
  3. Predicts cluster IDs for train / val / test.
  4. Symlinks the jsonl files into --output_dir and writes new
     bloom_cache.json files whose values are cluster IDs (0..k-1).
  5. Copies the corpus (no relabelling needed).
  6. Saves the fitted KMeans model for reproducibility.

After running this, point a BAM-PQ config's `data.{train,val,test,corpus}_path`
at the paths under --output_dir and run scripts/train_bam.py normally;
the trainer sees "predicted_bloom_level" values that are actually cluster
IDs — no code changes required.

Usage:
    python scripts/relabel_with_clusters.py \\
        --config     configs/mrl_bge_base.yaml \\
        --checkpoint /scratch/ishaan.karan/cluster_baseline/mrl-ckpts/mrl_bge_base/best/ \\
        --input_dir  data/real \\
        --output_dir /scratch/ishaan.karan/cluster_baseline/data_bge_base_clustered \\
        --k 6
"""
import argparse
import json
import os
import shutil
import sys
import time

import joblib
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from scripts.eval_bam import load_mrl


# Splits the trainer expects. `train_curriculum` is the default educational
# training split (BM25 curriculum negatives).
SPLIT_FILES = {
    "train": "train_curriculum.jsonl",
    "val":   "val.jsonl",
    "test":  "test.jsonl",
}


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device,
                 is_query: bool, batch_size: int = 128):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=256 if not is_query else 128,
                        return_tensors="pt").to(device)
        if hasattr(model, "encode_queries") and is_query:
            out = model.encode_queries(enc["input_ids"], enc["attention_mask"])
            emb = out.get("masked_embedding", out.get("full")) \
                if isinstance(out, dict) else out
        else:
            out = model(enc["input_ids"], enc["attention_mask"])
            emb = out["full"] if isinstance(out, dict) else out
        all_embs.append(emb.float().cpu().numpy())
    return normalize(np.concatenate(all_embs, axis=0), norm="l2", axis=1)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",     required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="MRL fine-tuned checkpoint dir")
    ap.add_argument("--input_dir",  default="data/real",
                    help="Directory containing train_curriculum.jsonl / "
                         "val.jsonl / test.jsonl / corpus.jsonl")
    ap.add_argument("--output_dir", required=True,
                    help="Where the clustered mirror of --input_dir is written")
    ap.add_argument("--k",          type=int, default=6,
                    help="Number of clusters (default: 6, matching Bloom levels)")
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load MRL encoder ===
    print(f"Loading config: {args.config}")
    cfg = load_config(args.config)
    set_seed(cfg["training"].get("seed", args.seed))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])
    model = load_mrl(cfg, args.checkpoint, device)
    model.eval()

    # === Load splits ===
    splits = {}
    for name, fname in SPLIT_FILES.items():
        src = os.path.join(args.input_dir, fname)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing split: {src}")
        splits[name] = load_jsonl(src)
        print(f"  {name}: {len(splits[name])} queries from {src}")

    # === Encode all queries ===
    print("\nEncoding queries with MRL...")
    embs = {}
    for name, samples in splits.items():
        t0 = time.time()
        embs[name] = encode_texts(model, tokenizer,
                                  [s["query"] for s in samples],
                                  device, is_query=True)
        print(f"  {name}: {embs[name].shape}  ({time.time() - t0:.1f} s)")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # === Fit k-means on TRAIN embeddings only, then predict for all splits ===
    print(f"\nFitting KMeans(k={args.k}) on {len(embs['train'])} train queries...")
    t0 = time.time()
    km = KMeans(n_clusters=args.k, random_state=args.seed, n_init=10)
    km.fit(embs["train"])
    print(f"  Fitted in {time.time() - t0:.1f} s")

    cluster_ids = {name: km.predict(embs[name]).astype(int)
                   for name in splits}

    for name, cids in cluster_ids.items():
        _, counts = np.unique(cids, return_counts=True)
        dist = "  ".join(f"c{c}={n}" for c, n in enumerate(counts))
        print(f"  {name} cluster distribution: {dist}")

    # === Report Bloom-level agreement for diagnostics ===
    # Not used by the trainer; just a sanity signal on how much clusters
    # correlate with true Bloom levels.
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    for name, samples in splits.items():
        true_bloom = np.array([s["bloom_level"] - 1 for s in samples
                               if "bloom_level" in s])
        if len(true_bloom) == len(cluster_ids[name]):
            ari = adjusted_rand_score(true_bloom, cluster_ids[name])
            nmi = normalized_mutual_info_score(true_bloom, cluster_ids[name])
            print(f"  {name}: ARI vs Bloom = {ari:+.4f}   NMI = {nmi:.4f}")

    # === Materialise the clustered data directory ===
    print(f"\nWriting clustered mirror to {args.output_dir} ...")
    for name, fname in SPLIT_FILES.items():
        src = os.path.abspath(os.path.join(args.input_dir, fname))
        dst = os.path.join(args.output_dir, fname)
        # Prefer a symlink so we don't duplicate multi-MB jsonl files.
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        try:
            os.symlink(src, dst)
            link_kind = "symlink"
        except OSError:
            shutil.copy2(src, dst)
            link_kind = "copy"
        # Write cache with cluster IDs (0-indexed, matching the format the
        # trainer expects for predicted_bloom_level).
        cache_path = dst + ".bloom_cache.json"
        with open(cache_path, "w") as f:
            json.dump(cluster_ids[name].tolist(), f)
        print(f"  {name}: {link_kind} -> {src}")
        print(f"         cache -> {cache_path} ({len(cluster_ids[name])} labels)")

    # Corpus: symlink if present (BAM-PQ eval needs it).
    corpus_src = os.path.abspath(os.path.join(args.input_dir, "corpus.jsonl"))
    corpus_dst = os.path.join(args.output_dir, "corpus.jsonl")
    if os.path.exists(corpus_src):
        if os.path.exists(corpus_dst) or os.path.islink(corpus_dst):
            os.remove(corpus_dst)
        try:
            os.symlink(corpus_src, corpus_dst)
        except OSError:
            shutil.copy2(corpus_src, corpus_dst)
        print(f"  corpus: symlink -> {corpus_src}")

    # === Save the KMeans model for reproducibility ===
    km_path = os.path.join(args.output_dir, "kmeans.joblib")
    joblib.dump(km, km_path)
    print(f"  Saved KMeans model -> {km_path}")

    with open(os.path.join(args.output_dir, "cluster_metadata.json"), "w") as f:
        json.dump({
            "config":     args.config,
            "checkpoint": args.checkpoint,
            "k":          args.k,
            "seed":       args.seed,
            "input_dir":  os.path.abspath(args.input_dir),
            "cluster_counts": {name: cluster_ids[name].tolist()
                               for name in splits},
        }, f)

    print("\nDone. Point a BAM-PQ config's data.*_path at:")
    for name, fname in SPLIT_FILES.items():
        print(f"  {name}: {os.path.abspath(os.path.join(args.output_dir, fname))}")
    print(f"  corpus: {os.path.abspath(corpus_dst)}")


if __name__ == "__main__":
    main()
