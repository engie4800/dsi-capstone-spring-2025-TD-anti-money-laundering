import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, PNAConv


class GNN(nn.Module):

   def __init__(
        self,
        n_node_feats,
        n_edge_feats,
        num_gnn_layers=2,
        n_classes=1,
        n_hidden=100,
        edge_updates=True,
        residual=True,
        dropout=0.0,
        final_dropout=0.10527690625126304,
        deg=None,
        gnn_flavor="GINe",
    ):
        if gnn_flavor not in ["GINe", "PNA"]:
            raise ValueError(
                "Unsupported GNN flavor"
            )

        super().__init__()

        # Flavor-specific setup
        if gnn_flavor == "GINe":
            self.n_hidden = n_hidden
        elif gnn_flavor == "PNA":
            n_hidden = int((n_hidden // 5) * 5)
            self.n_hidden = n_hidden
            aggregators = ["mean", "min", "max", "std"]
            scalers = ["identity", "amplification", "attenuation"]

        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout

        self.node_emb = nn.Linear(n_node_feats, n_hidden)
        self.edge_emb = nn.Linear(n_edge_feats, n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(self.num_gnn_layers):
            # Flavor-specific convolution
            if gnn_flavor == "GINe":
                conv = GINEConv(nn.Sequential(
                    nn.Linear(self.n_hidden, self.n_hidden),
                    nn.ReLU(),
                    nn.Linear(self.n_hidden, self.n_hidden)
                ), edge_dim=self.n_hidden)
            elif gnn_flavor == "PNA":
                conv = PNAConv(
                    in_channels=n_hidden,
                    out_channels=n_hidden,
                    aggregators=aggregators,
                    scalers=scalers,
                    deg=deg,
                    edge_dim=n_hidden,
                    towers=5,
                    pre_layers=1,
                    post_layers=1,
                    divide_input=False,
                )

            if self.edge_updates: self.emlps.append(nn.Sequential(
                nn.Linear(3 * self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden),
            ))
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(n_hidden))

        self.mlp = nn.Sequential(
            nn.Linear(n_hidden*3, 50),
            nn.ReLU(),
            nn.Dropout(self.final_dropout),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Dropout(self.final_dropout),
            nn.Linear(25, n_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):

            # Node update
            x_res = self.convs[i](x, edge_index, edge_attr)
            x = (x + F.relu(self.batch_norms[i](x_res))) / 2

            # Edge update
            if self.edge_updates:
                edge_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
                edge_attr = edge_attr + self.emlps[i](edge_input) / 2

        # Final prediction
        x_edge = torch.cat([x[src], x[dst], edge_attr], dim=-1)

        return self.mlp(x_edge).squeeze(-1)
