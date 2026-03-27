"""Case study generation for qualitative paper examples."""

import json
import os
import torch
import numpy as np
from typing import Dict, List
from collections import defaultdict

BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply",
               4: "Analyze", 5: "Evaluate", 6: "Create"}


class CaseStudyGenerator:

    def __init__(self, num_groups=8, embedding_dim=768):
        self.num_groups = num_groups
        self.group_size = embedding_dim // num_groups

    @torch.no_grad()
    def generate_routing_examples(self, model, dataloader, device, num_examples=20):
        model.eval()
        examples = []
        count = 0

        for batch in dataloader:
            if count >= num_examples:
                break
            B = batch["query_input_ids"].size(0)
            out = model.encode_queries(
                batch["query_input_ids"].to(device),
                batch["query_attention_mask"].to(device),
                learner_features=batch.get("learner_features", torch.zeros(B, 6)).to(device),
            )
            masks = out["mask"].cpu().numpy()

            for i in range(min(B, num_examples - count)):
                m = masks[i]
                ga = []
                for g in range(self.num_groups):
                    ga.append(float((m[g*self.group_size:(g+1)*self.group_size] > 0.5).mean()))

                bloom = batch["bloom_label"][i].item() + 1
                examples.append({
                    "bloom_level": bloom,
                    "bloom_name": BLOOM_NAMES.get(bloom, "?"),
                    "group_activations": ga,
                    "active_dims": int((m > 0.5).sum()),
                    "active_groups": sum(1 for a in ga if a > 0.5),
                })
                count += 1
        return examples

    def analyze_patterns(self, examples):
        by_bloom = defaultdict(list)
        for ex in examples:
            by_bloom[ex["bloom_level"]].append(ex)

        analysis = {}
        for lv, exs in sorted(by_bloom.items()):
            ga = np.array([e["group_activations"] for e in exs])
            analysis[f"bloom_{lv}_{BLOOM_NAMES[lv]}"] = {
                "avg_activations": ga.mean(axis=0).tolist(),
                "top_groups": np.argsort(ga.mean(axis=0))[::-1][:3].tolist(),
                "avg_active_dims": float(np.mean([e["active_dims"] for e in exs])),
                "count": len(exs),
            }
        return analysis

    def save(self, examples, analysis, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "routing_examples.json"), "w") as f:
            json.dump(examples, f, indent=2)
        with open(os.path.join(output_dir, "routing_analysis.json"), "w") as f:
            json.dump(analysis, f, indent=2)
