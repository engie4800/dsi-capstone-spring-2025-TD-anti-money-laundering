from pipeline import BaseModelPipeline


class CatBoostPipeline(BaseModelPipeline):
    def __init__(self, data_file):
        super().__init__(data_file)
        
        self.keep_acct_idx = True
    
    def should_keep_acct_idx(self):
        return True
    
    def split_train_test_val(self, X_cols=None, y_col="is_laundering", test_size=0.15, val_size=0.15, split_type="temporal"):
        return super().split_train_test_val(X_cols, y_col, test_size, val_size, split_type)
       
    def add_node_graph_feats_to_df(self, node_feat_cols=None):
        """
        Used for non-GNN model (e.g., CatBoost).
        Merge node-level graph features (e.g., pagerank, degree_centrality)
        into the transaction DataFrames (X_train, X_val, X_test) for both sender and receiver.

        Args:
            node_feat_cols (list or None): List of node feature columns to merge (excluding 'node_id').
                                        If None, will use all columns in node DataFrames except
                                        a "node_id"
        """
        if 'from_account_idx' not in self.X_train.columns:
            raise RuntimeError("To add node feats to tabular df, need from_account_idx and to_account_idx")
        
        if node_feat_cols is None:
            node_feat_cols = [col for col in self.train_nodes.columns if col != 'node_id']

        def merge_feats(txns_df, nodes_df):
            if "node_id" not in nodes_df.columns:
                raise ValueError("Each nodes_df must include a 'node_id' column")

            # Sender node features
            sender_feats = nodes_df[["node_id"] + node_feat_cols].copy()
            sender_feats = sender_feats.rename(columns={col: f"from_{col}" for col in node_feat_cols})
            sender_feats = sender_feats.rename(columns={"node_id": "from_account_idx"})

            # Receiver node features
            receiver_feats = nodes_df[["node_id"] + node_feat_cols].copy()
            receiver_feats = receiver_feats.rename(columns={col: f"to_{col}" for col in node_feat_cols})
            receiver_feats = receiver_feats.rename(columns={"node_id": "to_account_idx"})

            # Merge into transaction dataframe
            txns_df = txns_df.merge(sender_feats, on="from_account_idx", how="left")
            txns_df = txns_df.merge(receiver_feats, on="to_account_idx", how="left")
            
            txns_df = txns_df.drop(columns=['from_account_idx','to_account_idx'])

            return txns_df

        self.X_train = merge_feats(self.X_train, self.train_nodes)
        self.X_val = merge_feats(self.X_val, self.val_nodes)
        self.X_test = merge_feats(self.X_test, self.test_nodes)
            
    def scale_edge_features(self, edge_features_to_scale:list[str]):
        return super().numerical_scaling(edge_features_to_scale)
       
    def run_model_specific_steps(self, edge_features_to_scale:list[str]):
        self.scale_edge_features(edge_features_to_scale)
        self.add_node_graph_features_to_df()
