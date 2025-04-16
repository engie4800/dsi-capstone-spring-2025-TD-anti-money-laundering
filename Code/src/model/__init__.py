import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv


class GINe(nn.Module):
    def __init__(self, n_node_feats, n_edge_feats, num_gnn_layers=2, n_classes=1,
                n_hidden=100, edge_updates=True, residual=True,
                dropout=0.0, final_dropout=0.10527690625126304):
        super().__init__()
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout

        self.node_emb = nn.Linear(n_node_feats, n_hidden)
        self.edge_emb = nn.Linear(n_edge_feats, n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(self.num_gnn_layers):
            conv = GINEConv(nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden)
                ), edge_dim=self.n_hidden)

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

    def forward(self, x, edge_index, edge_attr):
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
