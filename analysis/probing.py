"""
Probing classifiers for dimension group analysis.
Train lightweight classifiers on each group to determine what it encodes.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


class DimensionGroupProber:

    def __init__(self, embedding_dim=768, num_groups=8, probe_type="linear"):
        self.embedding_dim = embedding_dim
        self.num_groups = num_groups
        self.group_size = embedding_dim // num_groups
        self.probe_type = probe_type

    @torch.no_grad()
    def extract_embeddings(self, model, dataloader, device, max_samples=5000):
        model.eval()
        all_embs, all_blooms, all_subjects = [], [], []
        count = 0

        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            if count >= max_samples:
                break
            ids = batch["query_input_ids"].to(device)
            mask = batch["query_attention_mask"].to(device)

            if hasattr(model, "encode_queries"):
                out = model.encode_queries(ids, mask, return_mask=False)
                emb = out["full_embedding"]
            else:
                out = model(ids, mask)
                emb = out["full"]

            all_embs.append(emb.cpu().numpy())
            all_blooms.append(batch["bloom_label"].numpy())
            all_subjects.append(batch["subject_label"].numpy())
            count += emb.size(0)

        return {
            "embeddings": np.concatenate(all_embs)[:max_samples],
            "bloom_labels": np.concatenate(all_blooms)[:max_samples],
            "subject_labels": np.concatenate(all_subjects)[:max_samples],
        }

    def probe_all_groups(self, embeddings, labels_dict, cv_folds=5):
        """Train probing classifier for each (group, task) combo."""
        results = {}
        configs = {}

        for g in range(self.num_groups):
            configs[f"group_{g}"] = (g * self.group_size, (g+1) * self.group_size)
        for d in [64, 128, 256, 512]:
            if d <= self.embedding_dim:
                configs[f"mrl_d{d}"] = (0, d)
        configs["full"] = (0, self.embedding_dim)

        for name, (start, end) in tqdm(configs.items(), desc="Probing"):
            scaler = StandardScaler()
            feats = scaler.fit_transform(embeddings[:, start:end])
            results[name] = {}
            for task, labels in labels_dict.items():
                results[name][task] = self._probe(feats, labels, cv_folds)

        return results

    def _probe(self, features, labels, cv_folds):
        if self.probe_type == "linear":
            clf = LogisticRegression(max_iter=1000, solver="lbfgs",
                                     multi_class="multinomial", C=1.0)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500,
                                early_stopping=True, validation_fraction=0.1)
        scores = cross_val_score(clf, features, labels, cv=cv_folds, scoring="accuracy")
        return float(scores.mean())

    def compute_specialization_matrix(self, results):
        group_names = [f"group_{g}" for g in range(self.num_groups)]
        task_names = list(next(iter(results.values())).keys())
        matrix = np.zeros((self.num_groups, len(task_names)))
        for g, gn in enumerate(group_names):
            for t, tn in enumerate(task_names):
                matrix[g, t] = results.get(gn, {}).get(tn, 0)
        return matrix, group_names, task_names

    def print_results(self, results):
        tasks = list(next(iter(results.values())).keys())
        print(f"\n{'':15s}" + "".join(f"{t:>15s}" for t in tasks))
        for g in range(self.num_groups):
            row = f"{'group_'+str(g):15s}"
            for t in tasks:
                row += f"{results.get(f'group_{g}', {}).get(t, 0):>15.3f}"
            print(row)
        print()
        for key in sorted(results.keys()):
            if key.startswith("mrl_") or key == "full":
                row = f"{key:15s}"
                for t in tasks:
                    row += f"{results[key].get(t, 0):>15.3f}"
                print(row)
