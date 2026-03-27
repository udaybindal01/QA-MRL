"""
Error Analysis and Qualitative Case Studies.

Generates examples showing:
  1. Where QA-MRL succeeds and baseline fails (routing helps)
  2. Where baseline succeeds and QA-MRL fails (routing hurts)
  3. How routing patterns differ across Bloom levels
  4. Most common failure modes per Bloom level

Usage:
    python scripts/run_error_analysis.py --config configs/real_data.yaml \
        --checkpoint /tmp/qa-mrl-ckpts/best/ \
        --baseline /tmp/qa-mrl-ckpts/mrl_baseline_best/
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import load_config, set_seed
from models.qa_mrl import QAMRL
from models.encoder import MRLEncoder
from transformers import AutoTokenizer

BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply",
               4: "Analyze", 5: "Evaluate", 6: "Create"}


@torch.no_grad()
def get_retrieval_results(model, test_path, corpus_path, tokenizer, device, model_name="model"):
    """Get per-query retrieval results with rankings and scores."""
    model.eval()

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            corpus.append(json.loads(line.strip()))
    corpus_id_to_idx = {p["id"]: i for i, p in enumerate(corpus)}

    # Encode corpus
    corpus_embs = []
    for i in range(0, len(corpus), 128):
        batch = [c["text"] for c in corpus[i:i+128]]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        if hasattr(model, "encode_documents"):
            out = model.encode_documents(enc["input_ids"], enc["attention_mask"])
            corpus_embs.append(out["full_embedding"].cpu())
        else:
            out = model(enc["input_ids"], enc["attention_mask"])
            corpus_embs.append(out["full"].cpu())
    corpus_embs = torch.cat(corpus_embs)

    # Load queries
    samples = []
    with open(test_path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    valid = [s for s in samples if s.get("positive_id", "") in corpus_id_to_idx]

    # Encode and retrieve
    results = []
    for i in range(0, len(valid), 64):
        batch_samples = valid[i:i+64]
        batch_texts = [s["query"] for s in batch_samples]
        enc = tokenizer(batch_texts, padding=True, truncation=True,
                       max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        if hasattr(model, "encode_queries"):
            out = model.encode_queries(enc["input_ids"], enc["attention_mask"])
            q_embs = out["masked_embedding"]
            masks = out["mask"].cpu().numpy()
        else:
            out = model(enc["input_ids"], enc["attention_mask"])
            q_embs = out["full"]
            masks = np.ones((len(batch_texts), 768))

        sim = torch.mm(q_embs, corpus_embs.t())
        topk_scores, topk_indices = sim.topk(20, dim=-1)

        for j, sample in enumerate(batch_samples):
            gt_idx = corpus_id_to_idx[sample["positive_id"]]
            retrieved = topk_indices[j].cpu().numpy()
            scores = topk_scores[j].cpu().numpy()

            hit_at_10 = gt_idx in retrieved[:10]
            rank = None
            for r, idx in enumerate(retrieved):
                if idx == gt_idx:
                    rank = r + 1
                    break

            active_dims = int((masks[j] > 0.5).sum())

            # Get active groups
            group_size = 96
            active_groups = []
            for g in range(8):
                if masks[j][g*group_size:(g+1)*group_size].mean() > 0.5:
                    active_groups.append(g)

            results.append({
                "query": sample["query"],
                "bloom_level": sample["bloom_level"],
                "bloom_name": BLOOM_NAMES.get(sample["bloom_level"], "?"),
                "subject": sample.get("subject", ""),
                "topic": sample.get("topic", ""),
                "positive_text": sample["positive_text"][:200],
                "hit_at_10": hit_at_10,
                "rank": rank,
                "active_dims": active_dims,
                "active_groups": active_groups,
                "top_retrieved_texts": [
                    corpus[int(idx)]["text"][:150] for idx in retrieved[:5]
                    if int(idx) < len(corpus)
                ],
            })

    return results


def analyze_errors(qa_results, bl_results, output_dir):
    """Compare QA-MRL vs baseline and categorize differences."""
    os.makedirs(output_dir, exist_ok=True)

    assert len(qa_results) == len(bl_results)
    N = len(qa_results)

    # Categorize
    qa_wins = []    # QA-MRL hits, baseline misses
    bl_wins = []    # Baseline hits, QA-MRL misses
    both_hit = []   # Both hit
    both_miss = []  # Both miss

    for i in range(N):
        qa_hit = qa_results[i]["hit_at_10"]
        bl_hit = bl_results[i]["hit_at_10"]

        entry = {
            "query": qa_results[i]["query"],
            "bloom": qa_results[i]["bloom_name"],
            "bloom_level": qa_results[i]["bloom_level"],
            "subject": qa_results[i]["subject"],
            "qa_rank": qa_results[i]["rank"],
            "bl_rank": bl_results[i]["rank"],
            "active_dims": qa_results[i]["active_dims"],
            "active_groups": qa_results[i]["active_groups"],
            "positive_text": qa_results[i]["positive_text"],
        }

        if qa_hit and not bl_hit:
            qa_wins.append(entry)
        elif bl_hit and not qa_hit:
            bl_wins.append(entry)
        elif qa_hit and bl_hit:
            both_hit.append(entry)
        else:
            both_miss.append(entry)

    # Summary
    print(f"\n{'='*60}")
    print("ERROR ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"  Total queries: {N}")
    print(f"  QA-MRL wins (QA hits, BL misses): {len(qa_wins)} ({len(qa_wins)/N:.1%})")
    print(f"  Baseline wins (BL hits, QA misses): {len(bl_wins)} ({len(bl_wins)/N:.1%})")
    print(f"  Both hit: {len(both_hit)} ({len(both_hit)/N:.1%})")
    print(f"  Both miss: {len(both_miss)} ({len(both_miss)/N:.1%})")

    # Bloom-level breakdown of wins
    print(f"\n  QA-MRL wins by Bloom level:")
    qa_win_bloom = defaultdict(int)
    bl_win_bloom = defaultdict(int)
    total_bloom = defaultdict(int)
    for e in qa_wins:
        qa_win_bloom[e["bloom"]] += 1
    for e in bl_wins:
        bl_win_bloom[e["bloom"]] += 1
    for r in qa_results:
        total_bloom[r["bloom_name"]] += 1

    for bloom in sorted(total_bloom.keys()):
        qa_w = qa_win_bloom.get(bloom, 0)
        bl_w = bl_win_bloom.get(bloom, 0)
        total = total_bloom[bloom]
        print(f"    {bloom:12s}: QA wins {qa_w:3d}, BL wins {bl_w:3d}, total {total:4d}")

    # Routing pattern analysis for QA-MRL wins
    print(f"\n  Routing patterns in QA-MRL wins:")
    group_patterns = defaultdict(int)
    for e in qa_wins:
        pattern = tuple(sorted(e["active_groups"]))
        group_patterns[pattern] += 1
    for pattern, count in sorted(group_patterns.items(), key=lambda x: -x[1])[:5]:
        print(f"    Groups {pattern}: {count} wins")

    # Save qualitative examples
    examples = {
        "qa_mrl_wins": qa_wins[:20],
        "baseline_wins": bl_wins[:20],
        "both_miss": both_miss[:20],
        "summary": {
            "total": N,
            "qa_wins": len(qa_wins),
            "bl_wins": len(bl_wins),
            "both_hit": len(both_hit),
            "both_miss": len(both_miss),
            "qa_win_by_bloom": dict(qa_win_bloom),
            "bl_win_by_bloom": dict(bl_win_bloom),
        }
    }

    with open(os.path.join(output_dir, "error_analysis.json"), "w") as f:
        json.dump(examples, f, indent=2, default=str)

    # Generate LaTeX table for paper
    latex = generate_error_latex(examples["summary"], total_bloom)
    with open(os.path.join(output_dir, "error_table.tex"), "w") as f:
        f.write(latex)

    print(f"\n  Saved to {output_dir}/")
    return examples


def generate_error_latex(summary, total_bloom):
    """Generate LaTeX table for error analysis."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Error analysis: QA-MRL wins vs. Baseline wins by Bloom level.}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Bloom Level & QA-MRL Wins & Baseline Wins & Total \\\\",
        "\\midrule",
    ]
    qa_wins = summary.get("qa_win_by_bloom", {})
    bl_wins = summary.get("bl_win_by_bloom", {})
    for bloom in ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]:
        qw = qa_wins.get(bloom, 0)
        bw = bl_wins.get(bloom, 0)
        total = total_bloom.get(bloom, 0)
        lines.append(f"{bloom} & {qw} & {bw} & {total} \\\\")
    lines.extend([
        "\\midrule",
        f"Total & {summary['qa_wins']} & {summary['bl_wins']} & {summary['total']} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_data.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output_dir", default="results/error_analysis/")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])

    test_path = config["data"]["test_path"]
    corpus_path = config["data"]["corpus_path"]

    # Load QA-MRL
    print("Loading QA-MRL...")
    qa_model = QAMRL(config)
    ckpt = os.path.join(args.checkpoint, "checkpoint.pt")
    if os.path.exists(ckpt):
        qa_model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
    qa_model.to(device)

    # Load baseline
    print("Loading MRL Baseline...")
    mc = config["model"]
    bl_model = MRLEncoder(model_name=mc["backbone"], embedding_dim=mc["embedding_dim"],
                          mrl_dims=mc["mrl_dims"])
    ckpt = os.path.join(args.baseline, "checkpoint.pt")
    if os.path.exists(ckpt):
        bl_model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
    bl_model.to(device)

    # Get results
    print("\nRetrieving with QA-MRL...")
    qa_results = get_retrieval_results(qa_model, test_path, corpus_path, tokenizer, device)
    print(f"  QA-MRL R@10: {sum(r['hit_at_10'] for r in qa_results)/len(qa_results):.4f}")

    print("\nRetrieving with Baseline...")
    bl_results = get_retrieval_results(bl_model, test_path, corpus_path, tokenizer, device)
    print(f"  Baseline R@10: {sum(r['hit_at_10'] for r in bl_results)/len(bl_results):.4f}")

    # Analyze
    analyze_errors(qa_results, bl_results, args.output_dir)


if __name__ == "__main__":
    main()