"""Visualization code for paper figures."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from collections import defaultdict

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 14, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply",
               4: "Analyze", 5: "Evaluate", 6: "Create"}
BLOOM_COLORS = {1: "#2196F3", 2: "#4CAF50", 3: "#FF9800",
                4: "#F44336", 5: "#9C27B0", 6: "#795548"}


def plot_dimension_importance_heatmap(importance_by_bloom, num_groups=8, save_path=None):
    """Fig 1: Which dim groups matter for each Bloom level."""
    levels = sorted(importance_by_bloom.keys())
    gs = len(list(importance_by_bloom.values())[0]) // num_groups

    mat = np.zeros((len(levels), num_groups))
    for i, lv in enumerate(levels):
        imp = importance_by_bloom[lv]
        for g in range(num_groups):
            mat[i, g] = imp[g*gs:(g+1)*gs].mean()
    mat = mat / (mat.sum(axis=1, keepdims=True) + 1e-10)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=[f"G{g}" for g in range(num_groups)],
                yticklabels=[f"L{l}: {BLOOM_NAMES.get(l, '?')}" for l in levels], ax=ax)
    ax.set_xlabel("Dimension Group")
    ax.set_ylabel("Bloom Level")
    ax.set_title("Dimension Group Importance by Bloom Level")
    if save_path:
        fig.savefig(save_path)
    plt.close()
    return fig


def plot_group_specialization_matrix(probing_results, num_groups=8, save_path=None):
    """Fig 2: Probing accuracy per (group, task)."""
    tasks = list(next(iter(probing_results.values())).keys())
    mat = np.zeros((num_groups, len(tasks)))
    for g in range(num_groups):
        for t, tn in enumerate(tasks):
            mat[g, t] = probing_results.get(f"group_{g}", {}).get(tn, 0)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1,
                xticklabels=tasks, yticklabels=[f"Group {g}" for g in range(num_groups)], ax=ax)
    ax.set_title("Group Specialization (Probing Accuracy)")
    if save_path:
        fig.savefig(save_path)
    plt.close()
    return fig


def plot_bloom_stratified_comparison(model_metrics, save_path=None):
    """Fig 3: Bar chart comparing models across Bloom levels."""
    levels = list(range(1, 7))
    models = list(model_metrics.keys())
    n = len(models)
    colors = plt.cm.Set2(np.linspace(0, 1, n))

    fig, ax = plt.subplots(figsize=(14, 6))
    w = 0.8 / n
    for i, (name, m) in enumerate(model_metrics.items()):
        vals = [m.get(f"bloom_{BLOOM_NAMES[l]}_recall@10", 0) for l in levels]
        ax.bar(np.arange(6) + i*w, vals, width=w, label=name, color=colors[i])

    ax.set_xticks(np.arange(6) + w*(n-1)/2)
    ax.set_xticklabels([f"L{l}: {BLOOM_NAMES[l]}" for l in levels], rotation=30, ha="right")
    ax.set_ylabel("Recall@10")
    ax.set_title("Retrieval by Bloom Level")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    if save_path:
        fig.savefig(save_path)
    plt.close()
    return fig


def plot_leave_one_out_degradation(logo_results, save_path=None):
    """Fig 4: LOGO degradation heatmap."""
    gr = logo_results["group_results"]
    ng = len(gr)
    bnames = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

    mat = np.zeros((ng, 6))
    for g in range(ng):
        for bl, bn in enumerate(bnames):
            mat[g, bl] = gr[f"group_{g}"]["degradation"].get(f"bloom_{bn}_recall@10", 0)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="RdYlGn_r", center=0,
                xticklabels=bnames, yticklabels=[f"Remove G{g}" for g in range(ng)], ax=ax)
    ax.set_title("R@10 Degradation per Removed Group")
    if save_path:
        fig.savefig(save_path)
    plt.close()
    return fig


def plot_router_behavior(masks, bloom_labels, num_groups=8, save_path=None):
    """Fig 5: Router group activation patterns per Bloom level."""
    gs = masks.shape[1] // num_groups
    levels = sorted(set(bloom_labels.tolist()))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, lv in enumerate(levels[:6]):
        lm = masks[bloom_labels == lv]
        if len(lm) == 0:
            continue
        ga = np.zeros((len(lm), num_groups))
        for g in range(num_groups):
            ga[:, g] = (lm[:, g*gs:(g+1)*gs] > 0.5).mean(axis=1)
        ax = axes[idx // 3][idx % 3]
        ax.bar(range(num_groups), ga.mean(axis=0), yerr=ga.std(axis=0),
               capsize=3, color=BLOOM_COLORS.get(lv+1, "#333"), alpha=0.8)
        ax.set_title(f"Bloom {lv+1}: {BLOOM_NAMES.get(lv+1, '?')}")
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Group")
        ax.set_ylabel("Activation Rate")

    plt.suptitle("Router Activation by Bloom Level", fontsize=16)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close()
    return fig
