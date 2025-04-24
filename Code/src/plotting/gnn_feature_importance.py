import random

import matplotlib.pyplot as plt
import numpy as np

from pipeline import ModelPipeline


def sample_and_plot_feature_importance(
    p: ModelPipeline,
    n_samples: int=1000,
    illicit_transactions_only: bool=False,
) -> None:
    n_edges = p.edge_index.size(1)

    # Determine eligible edges
    if illicit_transactions_only:
        eligible_edges = [i for i in range(n_edges) if p.y[i] == 1]
    else:
        eligible_edges = list(range(n_edges))
    if not eligible_edges:
        raise ValueError("No eligible edges found for sampling.")
    if n_samples > len(eligible_edges):
        raise ValueError(
            f"Too few edges {len(eligible_edges)} given # samples {n_samples}"
        )

    all_node_masks = []
    all_edge_masks = []
    sampled_edges = set()
    while len(sampled_edges) < n_samples and len(sampled_edges) < len(eligible_edges):
        target_edge = random.choice(eligible_edges)
        if target_edge in sampled_edges:
            continue
        sampled_edges.add(target_edge)

        node_mask, edge_mask = p.explain(target_edge=target_edge)
        all_node_masks.append(node_mask.cpu().numpy())
        all_edge_masks.append(edge_mask.cpu().numpy())

    avg_node_mask = np.mean(all_node_masks, axis=0)
    avg_edge_mask = np.mean(all_edge_masks, axis=0)

    # Sort node features by importance
    node_sorted_indices = np.argsort(avg_node_mask)[::-1]
    node_sorted_importances = avg_node_mask[node_sorted_indices]
    node_sorted_labels = [p.node_feature_labels[i] for i in node_sorted_indices]

    # Sort edge features by importance
    edge_sorted_indices = np.argsort(avg_edge_mask)[::-1]
    edge_sorted_importances = avg_edge_mask[edge_sorted_indices]
    edge_sorted_labels = [p.edge_feature_labels[i] for i in edge_sorted_indices]

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(11, 7))

    axs[0].barh(range(len(node_sorted_importances)), node_sorted_importances)
    axs[0].set_yticks(range(len(node_sorted_labels)))
    axs[0].set_yticklabels(node_sorted_labels)
    axs[0].invert_yaxis()  # Highest importance on top
    axs[0].set_title(f"Node Feature Importance ({n_samples} Samples)")

    axs[1].barh(range(len(edge_sorted_importances)), edge_sorted_importances)
    axs[1].set_yticks(range(len(edge_sorted_labels)))
    axs[1].set_yticklabels(edge_sorted_labels)
    axs[1].invert_yaxis()  # Highest importance on top
    axs[1].set_title(f"Edge Feature Importance ({n_samples} Samples)")

    plt.tight_layout()
    plt.show()
