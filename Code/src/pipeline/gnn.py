import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.explain import GNNExplainer
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkNeighborLoader

from explain import GNNEdgeExplainer
from model import GINe, GNNTrainer
from pipeline import BaseModelPipeline
from pipeline.checks import Checker
from pipeline.reverse_mp_utils import create_hetero_data
from model.features import add_ports

if TYPE_CHECKING:
    from torch_geometric.explain import Explanation


class GNNModelPipeline(BaseModelPipeline):
    def __init__(self, data_file):
        super().__init__(data_file)
    
    def should_keep_acct_idx(self):
        return False
    
    def split_train_test_val(self, X_cols=None, y_col="is_laundering", test_size=0.15, val_size=0.15, split_type="temporal_agg"):
            return super().split_train_test_val(X_cols, y_col, test_size, val_size, split_type)
           
    def split_train_test_val_graph(self, edge_features:list[str]=None, edge_features_to_scale:list[str]=None, reverse_mp:bool=False, ports:bool=False) -> None:
        """Creates graph objects for use in GNN.
        """
        logging.info("Splitting into train, test, validation graphs")
        Checker.train_val_test_node_features_added(self)
        
        # make sure sorted
        self.df.sort_values("edge_id", inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # Prereq: make edge_id first col, exclude some standard feats (if not already excluded)
        exclude_cols = {"edge_id", "from_bank", "to_bank", "from_account_idx", "to_account_idx"}
        cols = self.X_cols if edge_features is None else edge_features
        self.edge_features = ['edge_id'] + [col for col in cols if col not in exclude_cols]

        # Nodes
        tr_x = torch.tensor(self.train_nodes.drop(columns="node_id").values, dtype=torch.float)
        val_x = torch.tensor(self.val_nodes.drop(columns="node_id").values, dtype=torch.float)
        te_x = torch.tensor(self.test_nodes.drop(columns="node_id").values, dtype=torch.float)

        # Labels
        self.y = torch.LongTensor(self.df["is_laundering"].to_numpy())

        # Edge index
        self.edge_index = torch.LongTensor(self.df[["from_account_idx", "to_account_idx"]].to_numpy().T)

        # Edge attr
        edge_attr = torch.tensor(self.df[self.edge_features].to_numpy(), dtype=torch.float)
        
        # Edge timestamps
        tr_times = torch.tensor(self.X_train['edge_id'])
        val_times = torch.tensor(self.X_val['edge_id'])
        te_times = torch.tensor(self.X_test['edge_id'])

        # Overwrites the values we got from the original split
        self.train_indices = torch.tensor(self.train_indices)
        self.val_indices = torch.tensor(self.val_indices)
        self.test_indices = torch.tensor(self.test_indices)

        cat_tr_val_inds = torch.cat((self.train_indices, self.val_indices))
        self.train_data = Data(
            x=tr_x,
            edge_index=self.edge_index[:,self.train_indices],
            edge_attr=edge_attr[self.train_indices],
            y=self.y[self.train_indices],
            timestamps=tr_times
        )
        self.val_data = Data(
            x=val_x,
            edge_index=self.edge_index[:,cat_tr_val_inds],
            edge_attr=edge_attr[cat_tr_val_inds],
            y=self.y[cat_tr_val_inds],
            timestamps=val_times
        )
        self.test_data = Data(
            x=te_x,
            edge_index=self.edge_index,
            edge_attr=edge_attr,
            y=self.y,
            timestamps=te_times
        )
        
        self.scale_edge_features(edge_features_to_scale)

        if ports:
            self.train_data = add_ports(self.train_data)
            self.val_data = add_ports(self.val_data)
            self.test_data = add_ports(self.test_data)
            self.edge_features += ['port_in', 'port_out']
        
        if reverse_mp:
            self.train_data = create_hetero_data(self.train_data.x,  self.train_data.y,  self.train_data.edge_index,  self.train_data.edge_attr, self.train_data.timestamps, ports)
            self.val_data = create_hetero_data(self.val_data.x,  self.val_data.y,  self.val_data.edge_index,  self.val_data.edge_attr, self.val_data.timestamps, ports)
            self.test_data = create_hetero_data(self.test_data.x,  self.test_data.y,  self.test_data.edge_index,  self.test_data.edge_attr, self.test_data.timestamps, ports)
        
        self.preprocessed["train_test_val_data_split_graph"] = True
     
    def scale_edge_features(self, edge_features_to_scale: list[str]):
        """
        Scale and impute edge features in train/val/test data in place.
        Uses training set to fit scalers, then applies to all sets.
        """
        logging.info("Scaling edge features: %s", edge_features_to_scale)

        if edge_features_to_scale is None:
            self.scaled_edge_features = [
                "sent_amount_usd",
                "timestamp_int",
                "time_diff_from",
                "time_diff_to",
                "turnaround_time",
            ]
        else:
            self.scaled_edge_features = edge_features_to_scale
        
        scalers = {}       
        train_edge_attr = self.train_data.edge_attr.cpu().numpy()
        val_edge_attr = self.val_data.edge_attr.cpu().numpy()
        test_edge_attr = self.test_data.edge_attr.cpu().numpy()
        
        # Map feature name to index
        feature_idx_map = {name: i for i, name in enumerate(self.edge_features)}

        for feature in self.scaled_edge_features:
            col_idx = feature_idx_map[feature] 
            train_vals = train_edge_attr[:, col_idx]
            mask = train_vals != -1
            if np.sum(mask) == 0:
                logging.warning(f"All values -1 for {feature}; skipping.")
                continue

            median_val = np.median(train_vals[mask])
            train_vals[~mask] = median_val

            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_vals.reshape(-1, 1)).flatten()

            # Apply to val/test
            for edge_attr in [val_edge_attr, test_edge_attr]:
                col = edge_attr[:, col_idx]
                col[col == -1] = median_val
                edge_attr[:, col_idx] = scaler.transform(col.reshape(-1, 1)).flatten()

            train_edge_attr[:, col_idx] = train_scaled
            scalers[feature] = scaler # TODO: do we need to return the scalars? why do we have this?

        self.train_data.edge_attr = torch.tensor(train_edge_attr, dtype=torch.float32, device=self.device)
        self.val_data.edge_attr = torch.tensor(val_edge_attr, dtype=torch.float32, device=self.device)
        self.test_data.edge_attr = torch.tensor(test_edge_attr, dtype=torch.float32, device=self.device)

        return scalers

    def get_data_loaders(self, num_neighbors=[100,100], batch_size=8192):
        logging.info("Getting data loaders")
        Checker.graph_data_split_to_train_val_test(self)
        
        if self.train_data.isinstance(HeteroData):
            tr_edge_label_index = self.train_data['node', 'to', 'node'].edge_index
            tr_edge_label = self.train_data['node', 'to', 'node'].y


            self.train_loader =  LinkNeighborLoader(self.train_data, num_neighbors=num_neighbors, 
                                        edge_label_index=(('node', 'to', 'node'), tr_edge_label_index), 
                                        edge_label=tr_edge_label, batch_size=batch_size, shuffle=True)
            
            val_edge_label_index = self.val_data['node', 'to', 'node'].edge_index[:,self.val_indices]
            val_edge_label = self.val_data['node', 'to', 'node'].y[self.val_indices]


            self.val_loader =  LinkNeighborLoader(self.val_data, num_neighbors=num_neighbors, 
                                        edge_label_index=(('node', 'to', 'node'), val_edge_label_index), 
                                        edge_label=val_edge_label, batch_size=batch_size, shuffle=False)
            
            te_edge_label_index = self.test_data['node', 'to', 'node'].edge_index[:,self.test_indices]
            te_edge_label = self.test_data['node', 'to', 'node'].y[self.test_indices]


            self.test_loader =  LinkNeighborLoader(self.test_data, num_neighbors=num_neighbors, 
                                        edge_label_index=(('node', 'to', 'node'), te_edge_label_index), 
                                        edge_label=te_edge_label, batch_size=batch_size, shuffle=False)
        else:
            self.train_loader = LinkNeighborLoader(
                data=self.train_data,
                edge_label_index=self.edge_index[:, self.train_indices],
                edge_label=self.y[self.train_indices],
                batch_size=batch_size,
                num_neighbors=num_neighbors,
                shuffle=True,
            )

            self.val_loader = LinkNeighborLoader(
                data=self.val_data,
                edge_label_index=self.edge_index[:, self.val_indices],
                edge_label=self.y[self.val_indices],
                batch_size=batch_size,
                num_neighbors=num_neighbors,
                shuffle=False,
            )

            self.test_loader = LinkNeighborLoader(
                data=self.test_data,
                edge_label_index=self.edge_index[:, self.test_indices],
                edge_label=self.y[self.test_indices],
                batch_size=batch_size,
                num_neighbors=num_neighbors,
                shuffle=False,
            )

        self.preprocessed["got_data_loaders"] = True
                
    def run_model_specific_steps(self, edge_features=list[str], edge_features_to_scale= list[str], num_neighbors=[100,100], batch_size=8192):
        self.split_train_test_val_graph(edge_features)
        self.scale_edge_features(edge_features_to_scale)
        self.get_data_loaders(num_neighbors, batch_size)
        
    def initialize_training(
        self,
        threshold: float=0.5,
        epochs: int=50,
        patience: int=10,
    ) -> None:
        """Setup the model pipeline for training: metrics, model,
        optimizer, scheduler, and criterion
        """
        self.threshold = threshold
        self.epochs = epochs
        self.patience = patience

        # Since `initialize_training` is run after preprocessing is
        # done, we can define the node and edge features here. This
        # does assume that column ordering between data frames and
        # tensors is preserved, and it removes node and edge id
        # TODO: ran into issue with this line bc node_id has already been dropped
        if 'node_id' in self.nodes.columns:
            self.node_feature_labels = self.nodes.drop(columns="node_id").columns
        else:
            self.node_feature_labels = self.nodes.columns
        self.edge_feature_labels = self.df[self.edge_features].drop(columns="edge_id").columns

        # Model setup
        num_edge_features = self.train_data.edge_attr.shape[1]-1  # num edge feats - edge_id
        num_node_features = self.train_data.x.shape[1]
        self.model = GINe(n_node_feats=num_node_features, n_edge_feats=num_edge_features).to(self.device)
        self.trainer = GNNTrainer(self.model, self)

    def initialize_explainer(self, epochs: int=200) -> None:
        """
        Run after model training is complete to create an explainer for
        GNN interpretability
        """
        self.explainer = GNNEdgeExplainer(
            model=self.model,
            n_node_feats=self.x.size(1),
            n_edge_feats=self.data.edge_attr.size(1) - 1,  # Minus `edge_id`
            epochs=epochs,
        )

        self.gnn_explainer = GNNExplainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=epochs),  # Higher == better explanation
            explanation_type="model",               # Explains the model prediction
            node_mask_type="object",                # Masks each node
            edge_mask_type="object",                # Masks each edge
            model_config=dict(
                mode="binary_classification",       # Licit or illicit
                task_level="edge",                  # Transactions are labeled, and are edges
                return_type="raw",                  # Binary classification + logits = raw
            ),
        )

    def explain(self, target_edge: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Provides an explanation for the given `target_edge` using
        the custom explainer (returns a node and edge mask over the set
        of features for the target edge). These explanations are for use
        with the `sample_and_plot_feature_importance` method
        """
        return self.explainer(
            x=self.x,
            edge_index=self.edge_index,
            edge_attr=self.data.edge_attr[:, 1:],  # skips `edge_id`
            target_edge=target_edge,
        )

    def gnn_explain(self, target_edge: int) -> "Explanation":
        """Provides an explanation for the target edge using
        `GNNExplainer` configured to mask nodes and edges. Can be used
        to get an explanation subgraph. These explanations are for use
        with the `plot_explanation_subgraph` plotting method
        """
        return self.gnn_explainer(
            x=self.x,
            edge_index=self.edge_index,
            edge_attr=self.data.edge_attr[:, 1:],  # skips edge_id
            target=self.y,
            index=target_edge,
        )
