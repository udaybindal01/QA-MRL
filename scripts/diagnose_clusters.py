"""
Diagnose what the k-means clusters (fitted by scripts/relabel_with_clusters.py)
actually capture. For each cluster, prints:
  - Top scientific subjects (biology / chemistry / physics / ...)
  - Bloom-level distribution
  - 3 example queries

If clusters are dominated by a single subject (with Bloom distributions
roughly matching the global distribution inside each cluster), the routing
signal is topical rather than cognitive — this is the empirical
confirmation of "k-means captures topic, not cognitive complexity".

Usage:
    python scripts/diagnose_clusters.py \\
        --data_dir  /scratch/$USER/bampq_cluster_baseline/data/bge_base_k6 \\
        --train_jsonl data/real/train_curriculum.jsonl
"""
import argparse
import json
import os
import sys
from collections import Counter


BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply",
               4: "Analyze",  5: "Evaluate",   6: "Create"}


def load_jsonl(path):
    return [json.loads(line) for line in open(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Clustered data mirror dir written by "
                         "relabel_with_clusters.py (contains cluster_metadata.json)")
    ap.add_argument("--train_jsonl", default="data/real/train_curriculum.jsonl",
                    help="Original training jsonl (before relabel), needed to read "
                         "subject + bloom_level fields per query")
    ap.add_argument("--top_k_subjects", type=int, default=4)
    ap.add_argument("--num_examples", type=int, default=3)
    args = ap.parse_args()

    meta_path = os.path.join(args.data_dir, "cluster_metadata.json")
    if not os.path.exists(meta_path):
        sys.exit(f"ERROR: {meta_path} does not exist. "
                 "Did you run scripts/relabel_with_clusters.py first?")

    meta = json.load(open(meta_path))
    cluster_ids_train = meta["cluster_counts"]["train"]
    k = meta["k"]

    if not os.path.exists(args.train_jsonl):
        sys.exit(f"ERROR: {args.train_jsonl} does not exist.")
    samples = load_jsonl(args.train_jsonl)
    if len(samples) != len(cluster_ids_train):
        sys.exit(f"ERROR: length mismatch — {len(samples)} samples vs "
                 f"{len(cluster_ids_train)} cluster IDs. Did you use the same "
                 f"training file when running relabel_with_clusters.py?")

    # Global Bloom distribution for reference
    global_bloom = Counter(s.get("bloom_level", 0) for s in samples)
    total = len(samples)
    print("=" * 78)
    print(f"Cluster composition diagnostic — {args.data_dir}")
    print(f"Total training samples: {total}, k={k}")
    print("=" * 78)
    print(f"\nGlobal Bloom distribution:")
    for b in sorted(global_bloom):
        pct = global_bloom[b] / total * 100
        name = BLOOM_NAMES.get(b, f"?{b}")
        print(f"  L{b} {name:<12}: {global_bloom[b]:>5}  ({pct:5.1f}%)")

    for c in range(k):
        member_samples = [s for s, cid in zip(samples, cluster_ids_train)
                          if cid == c]
        if not member_samples:
            continue
        print("\n" + "-" * 78)
        print(f"Cluster {c}  (N={len(member_samples)}, "
              f"{len(member_samples)/total*100:.1f}% of train)")
        print("-" * 78)

        subs = Counter(s.get("subject", "?") for s in member_samples)
        print(f"  Top subjects:")
        for sub, n in subs.most_common(args.top_k_subjects):
            print(f"    {sub:<20} {n:>5}  ({n/len(member_samples)*100:5.1f}%)")

        blooms = Counter(s.get("bloom_level", 0) for s in member_samples)
        print(f"  Bloom distribution (in-cluster vs global):")
        for b in sorted(blooms):
            in_pct = blooms[b] / len(member_samples) * 100
            gl_pct = global_bloom[b] / total * 100
            delta = in_pct - gl_pct
            marker = "  " if abs(delta) < 5 else ("^^" if delta > 0 else "vv")
            name = BLOOM_NAMES.get(b, f"?{b}")
            print(f"    L{b} {name:<12}: in-cluster {in_pct:5.1f}%  "
                  f"vs global {gl_pct:5.1f}%  (Δ{delta:+5.1f}) {marker}")

        print(f"  Example queries:")
        for q in [s["query"] for s in member_samples[:args.num_examples]]:
            print(f"    - {q[:110]}")

    print("\n" + "=" * 78)
    print("Interpretation:")
    print("  If each cluster is dominated by 1-2 subjects (biology / chemistry /")
    print("  physics / ...) and the in-cluster Bloom distribution roughly matches")
    print("  the global one (Δ within ±5pp), k-means found TOPIC clusters, not")
    print("  cognitive clusters. This is the empirical confirmation that the")
    print("  routing signal captured by clustering is topical, not Bloom-aligned.")
    print("=" * 78)


if __name__ == "__main__":
    main()
