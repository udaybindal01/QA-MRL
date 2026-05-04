"""
Extract alpha (per-query residual weight) from BAM-PQ checkpoints.

alpha = sigmoid(alpha_raw)  (starts near 0.047 from alpha_raw=-3.0)

Usage:
    python scripts/extract_alpha.py
    python scripts/extract_alpha.py --ckpt_root /tmp/my-ckpts/
"""
import argparse, os, glob
import torch

BACKBONES = {
    "e5large":  "/tmp/uday/multi-domain/educational/bam_pq_e5large/best/",
    "bge":      "/tmp/uday/multi-domain/educational/bam_pq_bge_large/best/",
    "arctic":   "/tmp/uday/multi-domain/educational/bam_pq_arctic/best/",
    "roberta":  "/tmp/uday/multi-domain/educational/bam_pq_roberta/best/",
    "qwen06b":  "/tmp/uday/multi-domain/educational/bam_pq_qwen06b/best/",
}

def read_alpha(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = state.get("model_state_dict", state)
    # Key can be bloom_mask_head.alpha_raw or model.bloom_mask_head.alpha_raw
    for key in sd:
        if "alpha_raw" in key:
            raw = sd[key].item()
            alpha = torch.sigmoid(torch.tensor(raw)).item()
            return raw, alpha
    return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_root", default=None,
                        help="Override root dir (scan epoch_*/checkpoint.pt inside it)")
    args = parser.parse_args()

    print(f"\n{'Backbone':<12}  {'Ckpt':<35}  {'alpha_raw':>10}  {'alpha=σ(raw)':>13}")
    print("-" * 76)

    backbones = BACKBONES
    if args.ckpt_root:
        backbones = {"custom": args.ckpt_root}

    for name, root in sorted(backbones.items()):
        if not os.path.isdir(root):
            print(f"  {name:<12}  {'[dir not found]':<35}  {'—':>10}  {'—':>13}")
            continue

        # root already points to best/ dir; also try epoch_* as fallback
        candidates = [os.path.join(root, "checkpoint.pt")]
        parent = os.path.dirname(root.rstrip("/"))
        epoch_ckpts = sorted(glob.glob(os.path.join(parent, "epoch_*", "checkpoint.pt")))
        candidates += epoch_ckpts[::-1]  # highest epoch first

        found = False
        for ckpt in candidates:
            if os.path.isfile(ckpt):
                try:
                    raw, alpha = read_alpha(ckpt)
                    tag = os.path.relpath(ckpt, root)
                    if raw is not None:
                        print(f"  {name:<12}  {tag:<35}  {raw:>10.4f}  {alpha:>13.4f}")
                    else:
                        print(f"  {name:<12}  {tag:<35}  {'no alpha_raw key':>10}")
                    found = True
                    break
                except Exception as e:
                    print(f"  {name:<12}  {ckpt}  ERROR: {e}")
                    found = True
                    break

        if not found:
            print(f"  {name:<12}  {'[no checkpoint.pt found]':<35}  {'—':>10}  {'—':>13}")

    print()
    print("  Note: alpha_raw init=-3.0 → alpha_init=0.047")
    print("        alpha grows toward 1 if per-query routing helps;")
    print("        stays near 0.05 if Bloom prior alone is sufficient.")
    print()

if __name__ == "__main__":
    main()
