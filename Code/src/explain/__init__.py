import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.utils import k_hop_subgraph

from model import GNN


class GNNEdgeExplainer(torch.nn.Module):
    """An adaptation of the `GNNExplainer` from:
    
        - <https://arxiv.org/pdf/1903.03894>

    That includes both node and edge features. Given the following:

        - The input `model` is a GNN, and is trained
        - The `model` accepts node features `x` and edge features 
          `edge_attr` as input

    This model `GNNEdgeExplainer` learns a soft mask for both the node
    and edge features by:

        "maximizing the mutual information between the original
        prediction and the predictions of the perturbed graph"

    The weight for each feature in each mask is taken to represent
    feature importance. That is if the optimal mask sets a given
    feature to zero, it is unimportant. If it sets it to one, it is as
    important as possible
    """

    def __init__(
        self,
        model: GNN,
        n_node_feats: int,
        n_edge_feats: int,
        epochs: int=100,
        lr: float=0.01,
    ) -> None:
        """Initialize the explanation model

        Args:
            model (): The trained GNN model that uses both node and
                edge features
            n_node_feats (int): The number of node features
            n_edge_feats (int): The number of edge features
            epochs (int): The number of steps over which to train the
                explanation itself
            lr (float): The learning rate for the explanation model

        Returns:
            None
        """
        super().__init__()

        self.model = model
        self.epochs = epochs
        self.lr = lr

        # Assuming that this explanation algorithm is used after the
        # GNN itself is trained, some components of both models need to
        # be on the same device
        self.device = next(model.parameters()).device

        # Create masks as leaf tensors, then move to device
        self.node_mask = Parameter(torch.randn(n_node_feats))
        self.edge_mask = Parameter(torch.randn(n_edge_feats))

    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: torch.Tensor, 
        target_edge: int,
        num_hops: int=5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Provides a node and edge mask over the features for a given
        edge, defined by its index

        Args:
            x (torch.Tensor): The node feature matrix corresponding to
                the trained model
            edge_index (torch.Tensor): The edge indices corresponding
                to the graph the GNN is trained on
            edge_attr (torch.Tensor): The edge feature matrix
                corresponding to the trained GNN
            target_edge (int): The edge index to explain
            num_hops (int): The neighborhood surrounding the edge-of-
                interest over which to train the explanation

        Returns:
            node_mask (torch.Tensor): A mask for the node features
                associated with the edge-of-interest
            edge_mask (torch.Tensor): A mask for the edge features
                associated with the edge-of-interest
        """
        u = edge_index[0, target_edge].item()
        v = edge_index[1, target_edge].item()

        # Ensures that all computation occurs on the same device
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)
        self.node_mask.data = self.node_mask.data.to(self.device)
        self.edge_mask.data = self.edge_mask.data.to(self.device)

        # Extract a K-hop neighborhood subgraph
        node_idx = torch.tensor([u, v])
        subset, edge_index_k, edge_attr_k, edge_mask = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=num_hops,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=x.size(0),
            flow="source_to_target",
        )
        x_k = x[subset]
        edge_attr_k = edge_attr[edge_mask]

        # Masked predictions are compared to the original prediction
        # from the GNN
        with torch.no_grad():
            original_pred = self.model(x_k, edge_index_k, edge_attr_k)

        # Get explanation masks 
        optimizer = torch.optim.Adam([self.node_mask, self.edge_mask], lr=self.lr)
        for _ in range(self.epochs):
            optimizer.zero_grad()

            # Apply masks
            x_masked = x_k * self.node_mask.sigmoid()
            edge_attr_masked = edge_attr_k * self.edge_mask.sigmoid()

            pred = self.model(x_masked, edge_index_k, edge_attr_masked)

            loss = F.mse_loss(pred, original_pred)
            loss.backward()
            optimizer.step()

        return (
            self.node_mask.sigmoid().detach(),
            self.edge_mask.sigmoid().detach(),
        )
