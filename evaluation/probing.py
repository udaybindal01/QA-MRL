"""
Dimension-Range Probing Analysis for EMNLP.

Key experiment: Train linear probes on different MRL dimension ranges to
show that cognitive complexity information is encoded non-uniformly across
the embedding space. This provides mechanistic justification for the
query-adaptive routing approach.

Probing tasks:
  1. Bloom level classification (from query embeddings)
  2. Subject classification (from query/doc embeddings)
  3. Query type classification (factual/conceptual/procedural/metacognitive)

Analysis:
  - Per-MRL-truncation probing accuracy (dims 0:64, 0:128, ..., 0:768)
  - Per-group probing accuracy (dims 0:96, 96:192, ..., 672:768)
  - Information gain curves: how much does each additional dim group add?
  - Selectivity index: does each group specialize on a specific property?
"""

import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


BLOOM_NAMES = {0: "Remember", 1: "Understand", 2: "Apply",
               3: "Analyze", 4: "Evaluate", 5: "Create"}


class DimensionProber:
    """
    Train linear probes on different dimension ranges to characterize
    what information each range encodes.

    This produces Table 3 / Figure 4 of the paper: probing accuracy
    heatmap showing which dim groups encode which properties.
    """

    def __init__(self, mrl_dims: List[int] = None, num_groups: int = 8,
                 embedding_dim: int = 768, n_folds: int = 5):
        self.mrl_dims = mrl_dims or [64, 128, 256, 384, 512, 768]
        self.num_groups = num_groups
        self.embedding_dim = embedding_dim
        self.group_size = embedding_dim // num_groups
        self.n_folds = n_folds

    @torch.no_grad()
    def extract_embeddings(self, model, dataloader, device,
                           max_samples: int = 5000) -> Dict[str, np.ndarray]:
        """Extract query embeddings and labels for probing."""
        model.eval()
        all_embs, all_blooms, all_subjects, all_qtypes = [], [], [], []
        count = 0

        for batch in tqdm(dataloader, desc="Extracting embeddings for probing"):
            if count >= max_samples:
                break

            q_ids = batch["query_input_ids"].to(device)
            q_mask = batch["query_attention_mask"].to(device)

            if hasattr(model, "encoder"):
                out = model.encoder(q_ids, q_mask)
            else:
                out = model(q_ids, q_mask)
            emb = out["full"].cpu().numpy()

            all_embs.append(emb)
            all_blooms.append(batch["bloom_label"].numpy())
            all_subjects.append(batch["subject_label"].numpy())
            count += len(emb)

        return {
            "embeddings": np.concatenate(all_embs)[:max_samples],
            "bloom_labels": np.concatenate(all_blooms)[:max_samples],
            "subject_labels": np.concatenate(all_subjects)[:max_samples],
        }

    def probe_mrl_truncations(self, data: Dict[str, np.ndarray]) -> Dict:
        """
        Probe at each MRL truncation point (cumulative: first d dims).

        Returns accuracy for each (truncation, task) pair.
        This answers: "How much Bloom/subject info is captured by d dims?"
        """
        embs = data["embeddings"]
        results = {}

        for d in self.mrl_dims:
            emb_trunc = embs[:, :d]
            # Normalize truncated embeddings
            norms = np.linalg.norm(emb_trunc, axis=1, keepdims=True)
            emb_trunc = emb_trunc / (norms + 1e-9)

            for task_name, labels in [("bloom", data["bloom_labels"]),
                                      ("subject", data["subject_labels"])]:
                acc, f1, std = self._cross_val_probe(emb_trunc, labels)
                results[f"trunc_d{d}_{task_name}_acc"] = acc
                results[f"trunc_d{d}_{task_name}_f1"] = f1
                results[f"trunc_d{d}_{task_name}_std"] = std

        return results

    def probe_individual_groups(self, data: Dict[str, np.ndarray]) -> Dict:
        """
        Probe each dimension group INDEPENDENTLY (non-overlapping ranges).

        Returns accuracy for each (group, task) pair.
        This answers: "Which groups specialize on which properties?"
        """
        embs = data["embeddings"]
        results = {}

        for g in range(self.num_groups):
            start = g * self.group_size
            end = start + self.group_size
            emb_group = embs[:, start:end]
            norms = np.linalg.norm(emb_group, axis=1, keepdims=True)
            emb_group = emb_group / (norms + 1e-9)

            for task_name, labels in [("bloom", data["bloom_labels"]),
                                      ("subject", data["subject_labels"])]:
                acc, f1, std = self._cross_val_probe(emb_group, labels)
                results[f"group_{g}_{task_name}_acc"] = acc
                results[f"group_{g}_{task_name}_f1"] = f1
                results[f"group_{g}_{task_name}_std"] = std

        return results

    def compute_information_gain(self, data: Dict[str, np.ndarray]) -> Dict:
        """
        Information gain: how much does adding each group improve probing?

        For each group g, compare:
          - probe(dims 0..g*96)  vs  probe(dims 0..(g+1)*96)
        The marginal gain tells us where new information is encoded.
        """
        embs = data["embeddings"]
        results = {"bloom_gains": [], "subject_gains": []}

        prev_bloom_acc = 0.0
        prev_subject_acc = 0.0

        for g in range(self.num_groups):
            end = (g + 1) * self.group_size
            emb_cumulative = embs[:, :end]
            norms = np.linalg.norm(emb_cumulative, axis=1, keepdims=True)
            emb_cumulative = emb_cumulative / (norms + 1e-9)

            bloom_acc, _, _ = self._cross_val_probe(emb_cumulative, data["bloom_labels"])
            subject_acc, _, _ = self._cross_val_probe(emb_cumulative, data["subject_labels"])

            results["bloom_gains"].append(bloom_acc - prev_bloom_acc)
            results["subject_gains"].append(subject_acc - prev_subject_acc)
            results[f"cumulative_g{g}_bloom_acc"] = bloom_acc
            results[f"cumulative_g{g}_subject_acc"] = subject_acc

            prev_bloom_acc = bloom_acc
            prev_subject_acc = subject_acc

        return results

    def compute_selectivity_index(self, group_results: Dict) -> Dict:
        """
        Selectivity index: how specialized is each group?

        SI(g) = |bloom_acc(g) - subject_acc(g)| / max(bloom_acc(g), subject_acc(g))

        SI=0 means the group is equally informative for both tasks.
        SI=1 means the group is completely specialized for one task.
        """
        selectivity = {}
        for g in range(self.num_groups):
            bloom_acc = group_results.get(f"group_{g}_bloom_acc", 0)
            subject_acc = group_results.get(f"group_{g}_subject_acc", 0)
            max_acc = max(bloom_acc, subject_acc, 1e-10)
            si = abs(bloom_acc - subject_acc) / max_acc
            selectivity[f"group_{g}_selectivity"] = si
            selectivity[f"group_{g}_specializes_on"] = (
                "bloom" if bloom_acc > subject_acc else "subject"
            )
        selectivity["mean_selectivity"] = np.mean([
            selectivity[f"group_{g}_selectivity"] for g in range(self.num_groups)
        ])
        return selectivity

    def _cross_val_probe(self, features: np.ndarray, labels: np.ndarray,
                         ) -> Tuple[float, float, float]:
        """Train a linear probe with stratified cross-validation."""
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0, 0.0, 0.0

        # Ensure minimum samples per class for stratified CV
        min_per_class = min(np.bincount(labels))
        n_folds = min(self.n_folds, min_per_class)
        if n_folds < 2:
            return 0.0, 0.0, 0.0

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        accs, f1s = [], []

        for train_idx, test_idx in skf.split(features, labels):
            clf = LogisticRegression(
                max_iter=1000, solver="lbfgs", multi_class="multinomial",
                C=1.0, random_state=42,
            )
            clf.fit(features[train_idx], labels[train_idx])
            preds = clf.predict(features[test_idx])
            accs.append(accuracy_score(labels[test_idx], preds))
            f1s.append(f1_score(labels[test_idx], preds, average="macro"))

        return float(np.mean(accs)), float(np.mean(f1s)), float(np.std(accs))

    def run_full_analysis(self, model, dataloader, device,
                          max_samples: int = 5000) -> Dict:
        """Run all probing analyses and return combined results."""
        print("\n  [Probing] Extracting embeddings...")
        data = self.extract_embeddings(model, dataloader, device, max_samples)
        n = len(data["embeddings"])
        print(f"  [Probing] Extracted {n} embeddings, dim={data['embeddings'].shape[1]}")

        print("  [Probing] Probing MRL truncations...")
        trunc_results = self.probe_mrl_truncations(data)

        print("  [Probing] Probing individual groups...")
        group_results = self.probe_individual_groups(data)

        print("  [Probing] Computing information gain curves...")
        gain_results = self.compute_information_gain(data)

        print("  [Probing] Computing selectivity indices...")
        selectivity = self.compute_selectivity_index(group_results)

        combined = {
            "n_samples": n,
            "mrl_truncation_probing": trunc_results,
            "group_probing": group_results,
            "information_gain": gain_results,
            "selectivity": selectivity,
        }

        return combined

    def print_summary(self, results: Dict):
        """Print a formatted summary of probing results."""
        print(f"\n{'='*70}")
        print("DIMENSION PROBING ANALYSIS")
        print(f"{'='*70}")

        # MRL truncation probing
        print(f"\n  MRL Truncation Probing (cumulative dims):")
        print(f"  {'Dims':>6s}  {'Bloom Acc':>10s}  {'Subject Acc':>12s}  {'Bloom F1':>9s}  {'Subject F1':>11s}")
        print(f"  {'-'*55}")
        trunc = results.get("mrl_truncation_probing", {})
        for d in self.mrl_dims:
            ba = trunc.get(f"trunc_d{d}_bloom_acc", 0)
            sa = trunc.get(f"trunc_d{d}_subject_acc", 0)
            bf = trunc.get(f"trunc_d{d}_bloom_f1", 0)
            sf = trunc.get(f"trunc_d{d}_subject_f1", 0)
            print(f"  {d:>6d}  {ba:>10.4f}  {sa:>12.4f}  {bf:>9.4f}  {sf:>11.4f}")

        # Per-group probing
        print(f"\n  Per-Group Probing (non-overlapping dim ranges):")
        print(f"  {'Group':>5s}  {'Dims':>10s}  {'Bloom Acc':>10s}  {'Subject Acc':>12s}  {'Selectivity':>12s}  {'Specializes':>12s}")
        print(f"  {'-'*68}")
        grp = results.get("group_probing", {})
        sel = results.get("selectivity", {})
        for g in range(self.num_groups):
            start = g * self.group_size
            end = start + self.group_size
            ba = grp.get(f"group_{g}_bloom_acc", 0)
            sa = grp.get(f"group_{g}_subject_acc", 0)
            si = sel.get(f"group_{g}_selectivity", 0)
            sp = sel.get(f"group_{g}_specializes_on", "?")
            print(f"  {g:>5d}  {start:>3d}-{end:<3d}    {ba:>10.4f}  {sa:>12.4f}  {si:>12.4f}  {sp:>12s}")

        # Information gain
        print(f"\n  Information Gain (marginal accuracy from adding each group):")
        gain = results.get("information_gain", {})
        bg = gain.get("bloom_gains", [])
        sg = gain.get("subject_gains", [])
        for g in range(min(self.num_groups, len(bg))):
            print(f"    Group {g}: Bloom +{bg[g]:.4f}  Subject +{sg[g]:.4f}")

        ms = sel.get("mean_selectivity", 0)
        print(f"\n  Mean Selectivity Index: {ms:.4f}")
        if ms > 0.3:
            print("  -> STRONG specialization: groups encode different properties")
        elif ms > 0.1:
            print("  -> MODERATE specialization")
        else:
            print("  -> WEAK specialization: groups encode similar information")

    def save_results(self, results: Dict, path: str):
        """Save probing results to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        def convert(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, dict):
                return {k: convert(v) for k, v in o.items()}
            if isinstance(o, list):
                return [convert(v) for v in o]
            return o

        with open(path, "w") as f:
            json.dump(convert(results), f, indent=2)
