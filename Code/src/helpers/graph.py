from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


def scale_graph_edge_attributes(
    graph_data: Data,
    columns_to_scale: list[int],
    scaling_algorithm: str="standardization"
) -> Data:
    """Scales the provided columns on the torch graph data object.
    Assumes invalid values on the column are `-1` and not `nan`. Scales
    the column data to have a mean of zero and standard deviation of
    one.

    Args:
        graph_data (torch_geometric.data.data.Data): Python data object
            representing a homogeneous graph
        columns_to_scale (list[int]): List of integer column
            identifiers to scale
        scaling_algorithm (str): The scaling algorithm to use to scale
            each column

    Returns:
        graph_data (torch_geometric.data.data.Data): The updated graph
            with the requested columns scaled
    """

    # Infer device from edge_attr tensor
    device = graph_data.edge_attr.device

    # Computation per column is done on the CPU
    edge_attr_cpu = graph_data.edge_attr.cpu().numpy()

    for column_id in columns_to_scale:
        column_data = deepcopy(edge_attr_cpu[:, column_id])

        # Mask to identify valid (non -1) values
        mask = column_data != -1
        if np.sum(mask) == 0:
            # Skip column if all are -1
            continue

        # Scale the column using the requested normalization technique
        if scaling_algorithm == "standardization":
            median = np.median(column_data[mask])
            column_data[~mask] = median
            edge_attr_cpu[:, column_id] = StandardScaler()\
                .fit_transform(column_data.reshape(-1, 1))\
                .flatten()
        else:
            raise ValueError(
                f"Invalid scaling algorithm '{scaling_algorithm}'."
            )

    # Update the graph. Necessary to push data onto the graph with the
    # same device the graph data used, which may not be (is not likely to 
    # be) the CPU
    graph_data.edge_attr = torch.from_numpy(edge_attr_cpu).float().to(device)
    graph_data.x = graph_data.x.to(device)
    graph_data.y = graph_data.y.to(device)

    return graph_data


def get_graph_edge(
    edge_index: torch.Tensor,
    edge: Optional[int]=None,
    random: bool=True,
    exclude_self_loops: bool=True,
) -> torch.Tensor:
    """Selects a random edge from the `edge_index` and returns the two
    nodes it connects.

    Args:
        edge_index (torch.Tensor): The edge index tensor of shape
            [2, n_edges].
        exclude_self_loops (bool): If True, filters out edges where
            source == target. This is necessary for our use-case,
            given that the graph can contain self-loops (assuming that
            reinvestment transactions are included)

    Returns:
        torch.Tensor: A tensor containing the source and target node
        indices: [src, dst]
    """

    # To be safe, ensure that excluding self loops does not modify the
    # original edge index on the model pipeline
    edge_index = deepcopy(edge_index)

    # Remove self-loops (buckles) from the edge index
    if exclude_self_loops:
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]

        if edge_index.size(1) == 0:
            raise ValueError(
                "No edges available after filtering out buckles."
            )

    # Define the edge-of-interest, which can be random or can be
    # specified as input
    if random and edge is not None:
        raise ValueError(
            f"Edge {edge} specified, but a random edge was requested. "
            "Please specify an edge, or ask for a random edge, but not "
            "both."
        )
    elif random:
        edge = torch.randint(0, edge_index.size(1), (1,)).item()

    # Get and return a tensor specifying the pair of nodes the edge is
    # connecting based on the provided edge index
    src_node = edge_index[0, edge]
    dst_node = edge_index[1, edge]
    return torch.tensor([src_node.item(), dst_node.item()])
