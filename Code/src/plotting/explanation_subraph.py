import logging
from typing import TYPE_CHECKING

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.utils import to_networkx

if TYPE_CHECKING:
    from torch_geometric.explain import Explanation


def plot_explanation_subgraph(
    explanation: "Explanation",
    plot_node_labels: bool=False,
) -> None:
    """Given an explanation for a transaction, plot the subgraph with
    node and edge masking represented as node and edge shading

    Args:
        explanation (torch_geometric.explain.Explanation): The
            explanation for a given edge
        plot_node_labels (bool): Whether node labels should be shown
            on the plot. More input would be necessary to get the
            original account and bank information here

    Returns: None
    """
    index = explanation.index.item()

    # Preserve node indices from the original graph
    if plot_node_labels:
        node_indices = torch.nonzero(explanation.node_mask, as_tuple=True)[0]
        node_labels = {i: int(node_indices[i]) for i in range(len(node_indices))}
        with_labels = True
    else:
        node_labels=None
        with_labels = False

    # Preserve edge indices from the original graph
    edge_indices = torch.nonzero(explanation.edge_mask, as_tuple=True)[0]

    # Get the relevant (unmasked) portion of the explanation
    sub_explanation = explanation.get_explanation_subgraph()
    sub_G = to_networkx(sub_explanation)
    logging.info(f"Plotting the explanation subgraph: {sub_G}")

    # Node and edge colors
    node_color = [float(x) for x in sub_explanation.node_mask.cpu().numpy()]
    # Scale edge color between 0.2 and 1.0 to avoid white (hidden) edges
    edge_mask = sub_explanation.edge_mask.cpu().numpy()
    edge_colors = [
        0.2 + 0.8*(float(x) - edge_mask.min())/(edge_mask.max()-edge_mask.min())
        for x in edge_mask
    ]

    # Plot the subgraph
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    pos = nx.spring_layout(sub_G)
    nx.draw(
        sub_G,
        pos,
        ax=ax,
        # Nodes are colored based on their weight in the node mask
        node_color=node_color,
        cmap=plt.cm.Reds,
        # Nodes with a black outline for visibility
        edgecolors="black",
        linewidths=1.0,
        # Nodes can be labeled
        with_labels=with_labels,
        labels=node_labels,
        # Edges are weighted based on their weight in the edge mask
        width=edge_colors,
    )

    # Draw edge labels
    # Maps edge tuples to original edge indices
    edge_index = sub_explanation.edge_index.cpu().numpy()
    edge_tuples = [tuple(edge) for edge in edge_index.T]
    edge_label_dict = {edge: str(edge_indices[i].item()) for i, edge in enumerate(edge_tuples)}
    nx.draw_networkx_edge_labels(
        sub_G,
        pos,
        ax=ax,
        edge_labels=edge_label_dict,
        font_size=8,
    )

    # Node importance color bar
    node_sm = cm.ScalarMappable(cmap=plt.cm.Reds, norm=mcolors.Normalize(vmin=0, vmax=100))
    node_sm.set_array([])
    node_cbar = fig.colorbar(node_sm, ax=ax)
    node_cbar.set_label("Node Importance")

    # Edge importance color bar
    edge_sm = cm.ScalarMappable(cmap=plt.cm.Greys, norm=mcolors.Normalize(vmin=0, vmax=1))
    edge_sm.set_array([])
    edge_cbar = fig.colorbar(edge_sm, ax=ax, orientation="vertical", location="left", pad=0.02)
    edge_cbar.set_label("Edge Importance")

    # Show the plot
    plt.title(f"Explanation Subgraph for Transaction {index}")
    plt.show()
