import datetime
import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython.display import display
from rich import print
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from torch_geometric.data import Data

from helpers.currency import get_usd_conversion


class ModelPipeline:

    def __init__(self, dataset_path: str):
        """
        Initialize pipeline with dataset
        """
        self.dataset_path = dataset_path
        self.df = pd.read_csv(self.dataset_path)
        self.nodes = pd.DataFrame()

        # Track if preprocessing steps have been completed
        self.preprocessed = {
            "renamed": False,
            "duplicates_removed": False,
            "unique_ids_created": False,
            "currency_normalized": False,
            "currency_features_extracted": False,
            "time_features_extracted": False,
            "additional_time_features_extracted": False,
            "cyclical_encoded": False,
            "weekend_encoded": False,
            "label_encoded": False,
            "neighbor_context_computed": False,
            "normalized": False,
            "onehot_encoded": False
        }

    def df_summary(self):
        logging.info("DATA HEAD")
        display(self.df.head())
        logging.info("\nFEATURE TYPE")
        display(self.df.info())

    def y_statistics(self):
        logging.info("Normalized Value Count: ")
        logging.info(self.df["is_laundering"].value_counts(normalize=True))

    def rename_columns(self):
        """
        Renames the columns of `self.df` to follow a consistent
        semantics, in terms of to and from, sent and received, and to
        use the more pythonic snake case
        """
        column_mapping = {
            "Timestamp": "timestamp",
            "From Bank": "from_bank",
            "Account": "from_account",
            "To Bank": "to_bank",
            "Account.1": "to_account",
            "Amount Received": "received_amount",
            "Receiving Currency": "received_currency",
            "Amount Paid": "sent_amount",
            "Payment Currency": "sent_currency",
            "Payment Format": "payment_type",
            "Is Laundering": "is_laundering",
        }

        # Ensure required columns exist
        missing_columns = [col for col in column_mapping.keys() if col not in self.df.columns]
        if missing_columns:
            raise KeyError(f"Missing expected columns in dataset: {missing_columns}")

        self.df.rename(columns=column_mapping, inplace=True)
        self.preprocessed["renamed"] = True

    def drop_duplicates(self):
        self.df.drop_duplicates(inplace=True)
        self.preprocessed["duplicates_removed"] = True

    def currency_normalization(self):
        logging.info("Normalizing currency...")
        if "sent_currency" not in self.df.columns or "received_currency" not in self.df.columns:
            raise KeyError(
                "Currency columns missing. Need to run 'rename_columns' "
                "preprocessing step first."
            )

        usd_conversion = get_usd_conversion(self.dataset_path)
        self.df["sent_amount_usd"] = self.df.apply(
            lambda row: row["sent_amount"] * usd_conversion.get(row["sent_currency"], 1),
            axis=1,
        )
        self.df["received_amount_usd"] = self.df.apply(
            lambda row: row["received_amount"] * usd_conversion.get(row["received_currency"], 1),
            axis=1,
        )
        self.preprocessed["currency_normalized"] = True

    def extract_currency_features(self):
        logging.info("Extracting currency features...")
        if "sent_currency" not in self.df.columns or "received_currency" not in self.df.columns:
            raise KeyError(
                "Currency columns missing. Need to run 'rename_columns' "
                "preprocessing step first."
            )

        self.df["currency_changed"] = (
            self.df["sent_currency"] != self.df["received_currency"]
        ).astype(int)

        self.preprocessed["currency_features_extracted"] = True

    def extract_time_features(self):
        logging.info("Extracting time features...")
        if "timestamp" not in self.df.columns:
            raise KeyError(
                "Missing 'timestamp' column, were columns renamed properly?"
            )
        if not isinstance(self.df["timestamp"], datetime.datetime):
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

        # Extract items from timestamp
        self.df["hour_of_day"] = self.df["timestamp"].dt.hour
        self.df["day_of_week"] = self.df["timestamp"].dt.weekday # 0=Monday,...,6=Sunday
        self.df["seconds_since_midnight"] = (
            self.df["timestamp"].dt.hour * 3600 +  # Convert hours to seconds
            self.df["timestamp"].dt.minute * 60 +  # Convert minutes to seconds
            self.df["timestamp"].dt.second         # Keep seconds
        )

        # Transform timestamp to raw int unix
        self.df["timestamp_int"] = self.df["timestamp"].astype('int64') / 10**9

        # Just a temp assignment, will be scaled later on
        self.df["timestamp_scaled"] = self.df["timestamp"].astype('int64') / 10**9

        self.df.drop(columns=["timestamp"], inplace= True)
        
        self.preprocessed["time_features_extracted"] = True

    def create_unique_ids(self):
        """Create unique account - ID mapping."""
        logging.info("Creating unique ids...")
        if not self.preprocessed["renamed"]:
            raise RuntimeError(
                "Columns must be renamed before creating unique IDs."
            )
        if "timestamp_int" not in self.df.columns:
            raise KeyError(
                "Timestamp column missing. Need to run 'extract_time_features' "
                "preprocessing step first."
            )

        # Sort transactions by timestamp first
        self.df = self.df.sort_values(by="timestamp_int").reset_index(drop=True)
        self.df["edge_id"] = self.df.index.astype(int)

        # Get unique account-bank combos (a couple of acct numbers found at multiple banks)
        self.df['from_account_id'] = self.df['from_bank'].astype(str) + '_' + self.df['from_account'].astype(str)
        self.df['to_account_id'] = self.df['to_bank'].astype(str) + '_' + self.df['to_account'].astype(str)
        self.df.drop(columns=["from_account", "to_account"], inplace=True)

        # Combine all accounts in the order they appear, preserving timestamp order
        accounts_ordered = pd.concat([
            self.df[["edge_id", "from_account_id"]].rename(columns={"from_account_id": "account_id"}),
            self.df[["edge_id", "to_account_id"]].rename(columns={"to_account_id": "account_id"})
        ])

        # Sort by timestamp to reflect temporal ordering
        accounts_ordered = accounts_ordered.sort_values(by="edge_id")

        # Drop duplicates to get first-seen ordering of accounts
        unique_accounts = accounts_ordered.drop_duplicates(subset="account_id")["account_id"].reset_index(drop=True)

        # Create mapping: account_id → index based on first appearance
        node_mapping = {account: idx for idx, account in enumerate(unique_accounts)}

        # Map node identifiers to integer indices
        self.df["from_account_idx"] = self.df["from_account_id"].map(node_mapping)
        self.df["to_account_idx"] = self.df["to_account_id"].map(node_mapping)

        self.preprocessed["unique_ids_created"] = True

    def get_turnaround_time(self):
        """
        Keep track of last received timestamp per account
        """
        last_received_time = {}
        turnaround_times = []
        for _, row in self.df.iterrows():
            from_id = row["from_account_idx"]
            to_id = row["to_account_idx"]
            timestamp = row["timestamp_int"]

            # Compute turnaround if account had received money earlier
            turnaround_time = timestamp - last_received_time.get(from_id, np.nan)
            turnaround_times.append(turnaround_time)

            # Update last received time for the destination
            last_received_time[to_id] = timestamp

        self.df["turnaround_time"] = turnaround_times
        self.df["turnaround_time"] = self.df["turnaround_time"].fillna(-1)  # optional

    def extract_additional_time_features(self):
        """
        Additional time features can be extracted after creating unique
        IDs, which depends on the initial time feature extraction
        """
        logging.info("Extracting additional time features...")
        if not self.preprocessed["unique_ids_created"]:
            raise RuntimeError(
                "Unique IDs must be created before extracting "
                "additional time features."
            )

        # Ensures data is sorted by timestamp
        self.df = self.df.sort_values(
            by=["from_account_idx", "timestamp_int"]
        ).reset_index(drop=True)

        # Group by account and compute time difference from previous transaction
        self.df["time_diff_from"] = self.df.groupby("from_account_idx")["timestamp_int"].diff()
        self.df["time_diff_from"] = self.df["time_diff_from"].fillna(-1)

        # Sort by 'edge_id', which is how the dataframe is sorted prior
        # to calling 'extract_additional_time_features'
        self.df = self.df.sort_values(by="edge_id").reset_index(drop=True)

        # Extract turnaround time, too
        self.get_turnaround_time()

        self.preprocessed["additional_time_features_extracted"] = True
        
    def cyclical_encoding(self):
        logging.info("Adding cyclical encoding to time feats...")
        
        if not self.preprocessed["time_features_extracted"]:
            raise RuntimeError("Time features missing, run `extract_time_features` first.")
        
        self.df["day_sin"] = np.sin(2 * np.pi * self.df["day_of_week"] / 7)
        self.df["day_cos"] = np.cos(2 * np.pi * self.df["day_of_week"] / 7)
        self.df["time_of_day_sin"] = np.sin(2 * np.pi * self.df["seconds_since_midnight"] / 86400)
        self.df["time_of_day_cos"] = np.cos(2 * np.pi * self.df["seconds_since_midnight"] / 86400)
        
        self.preprocessed["cyclical_encoded"] = True
        
    def binary_weekend(self):
        if "day_of_week" not in self.df.columns:
            raise KeyError("Day-of-week feature missing. Run `extract_time_features` first.")
        self.df["is_weekend"] = self.df["day_of_week"].isin([5, 6]).astype(int)
        self.preprocessed["weekend_encoded"] = True
    
    def apply_one_hot_encoding(self, onehot_categorical_features= None):
        """One hot encode categorical columns, handling related columns"""
        logging.info("Applying one hot encoding...")
        # Default columns for encoding
        default_categorical_features = ["payment_type", "received_currency", "sent_currency"]

        # Use provided or default
        categorical_features = onehot_categorical_features or default_categorical_features

        # Find related column groups (e.g., same suffix)
        column_groups = {}
        for col in categorical_features:
            prefix, _, suffix = col.partition("_")
            if suffix and any(other.endswith(f"_{suffix}") for other in categorical_features if other != col):
                column_groups.setdefault(suffix, []).append(col)

        # Track columns to drop and DataFrames to concat
        columns_to_drop = []
        encoded_dfs = []

        # Encode grouped columns using shared encoder
        for suffix, cols in column_groups.items():
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            unique_values = pd.concat([self.df[col] for col in cols], axis=0).drop_duplicates().to_frame()
            encoder.fit(unique_values)

            for col in cols:
                transformed = encoder.transform(self.df[[col]])
                ohe_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                encoded_df = pd.DataFrame(transformed, columns=ohe_cols, index=self.df.index)
                encoded_dfs.append(encoded_df)
                columns_to_drop.append(col)

        # Encode independent columns
        independent_cols = [col for col in categorical_features if col not in sum(column_groups.values(), [])]
        for col in independent_cols:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            transformed = encoder.fit_transform(self.df[[col]])
            ohe_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
            encoded_df = pd.DataFrame(transformed, columns=ohe_cols, index=self.df.index)
            encoded_dfs.append(encoded_df)
            columns_to_drop.append(col)

        # Drop original and concat encoded
        self.df.drop(columns=columns_to_drop, inplace=True)
        self.df = pd.concat([self.df] + encoded_dfs, axis=1)

        logging.info(f"  One hot encoding applied to columns: {categorical_features}\n")
        self.preprocessed["onehot_encoded"] = True
        
    def apply_label_encoding(self, categorical_features=None):
        """Label encode categorical columns, handling related columns"""
        logging.info("Applying label encoding...")
        # Default columns for encoding
        default_categorical_features = ["day_of_week", "from_bank", "to_bank"]

        # Determine columns to encode
        if categorical_features is None:
            categorical_features = default_categorical_features

        # Find related columns (e.g., "from_bank" and "to_bank" should use the same encoder)
        column_groups = {}
        for col in categorical_features:
            prefix, _, suffix = col.partition("_")
            if suffix and any(other.endswith(f"_{suffix}") for other in categorical_features if other != col):
                column_groups.setdefault(suffix, []).append(col)

        # Apply encoding for related columns
        for suffix, cols in column_groups.items():
            encoder = LabelEncoder()
            unique_values = pd.concat([self.df[col].drop_duplicates() for col in cols]).drop_duplicates().reset_index(drop=True)
            encoder.fit(unique_values)
            for col in cols:
                self.df[col] = encoder.transform(self.df[col])

        # Apply encoding for independent categorical columns
        independent_cols = [col for col in categorical_features if col not in sum(column_groups.values(), [])]
        for col in independent_cols:
            self.df[col] = LabelEncoder().fit_transform(self.df[col])

        logging.info(f"  Label encoding applied to columns: {categorical_features}\n")
        self.preprocessed["label_encoded"] = True
    
    def numerical_scaling(self, numerical_features):
        """Standardize Numerical Features"""

        std_scaler = StandardScaler()

        self.X_train[numerical_features] = std_scaler.fit_transform(self.X_train[numerical_features])
        self.X_test[numerical_features] = std_scaler.transform(self.X_test[numerical_features])
        self.X_val[numerical_features] = std_scaler.transform(self.X_val[numerical_features])

        return self.X_train, self.X_test, self.X_val

    def add_graph_related_features(self, weight_cols):
        """Generate graph-based neighborhood context features"""
        logging.info("Adding graph related features...")

        # Aggregate multiple edges into one per (from, to) pair
        # Otherwise DiGraph overwrites the information
        aggregated_edges = (
            self.df
            .groupby(["from_account_idx", "to_account_idx"])[weight_cols]
            .sum()
            .reset_index()
        )
        
        # Bulding the graph from aggregated edges data
        G = nx.DiGraph()
        for _, row in aggregated_edges.iterrows():
            G.add_edge(row["from_account_idx"], row["to_account_idx"], 
                       **{col: row[col] for col in weight_cols})
        
        # Compute centrality 
        degree_centrality = nx.degree_centrality(G)
        in_deg_cent = {n: d / (len(G) - 1) for n, d in G.in_degree()}
        out_deg_cent = {n: d / (len(G) - 1) for n, d in G.out_degree()}
        self.nodes["degree_centrality"] = self.nodes["node_id"].map(degree_centrality)
        self.nodes["in_degree_centrality"] = self.nodes["node_id"].map(in_deg_cent)
        self.nodes["out_degree_centrality"] = self.nodes["node_id"].map(out_deg_cent)
        
        # Compute pagerank using both sent amount and received amount
        for weight_col in weight_cols:
            pagerank = nx.pagerank(G, weight="weight")
            self.nodes[f"pagerank_{weight_col}"] = self.nodes["node_id"].map(pagerank)
            
        logging.info(f"  Graph features computed using: {weight_cols}")
        logging.info("  **Note**, previously graph-based features were calculated using only `sent_amount` as edge weight (only based on outgoing transactions). Now both sent and received amounts are included by default.")
        logging.info(f"  New feature columns added: degree_centrality, in_degree_centrality, out_degree_centrality, {', '.join([f'pagerank_{col}' for col in weight_cols])}\n")

    def add_node_features(self, node_features):
        logging.info("Adding node features...")

        # Combining nodes and their respective features (source only, destination only, or both) into several dataframes
        all_nodes = []
        for features, _, _, kind in node_features:
            if kind == 'source_agg':
                temp = self.df[['from_account_idx', features[0]]].rename(columns={'from_account_idx': 'node_id'})
            
            elif kind == 'destination_agg':
                temp = self.df[['to_account_idx', features[0]]].rename(columns={'to_account_idx': 'node_id'})

            elif kind == 'merge_agg':
                if len(features) == 1:
                    temp_from = self.df[['from_account_idx', features[0]]].rename(columns={'from_account_idx': 'node_id'})
                    temp_to = self.df[['to_account_idx', features[0]]].rename(columns={'to_account_idx': 'node_id'})
                else:
                    temp_from = self.df[['from_account_idx', features[0]]].rename(columns={'from_account_idx': 'node_id', features[0]: features[2]})
                    temp_to = self.df[['to_account_idx', features[1]]].rename(columns={'to_account_idx': 'node_id', features[1]: features[2]})

                temp = pd.concat([temp_from, temp_to])

            all_nodes.append(temp)

        # Merging all dataframes on their node_id
        temp_node_df = all_nodes[0]
        for i in all_nodes[1:]:
            temp_node_df = pd.merge(temp_node_df, i, on='node_id', how='outer')

        # Aggregating and summarizing into one dataframe containing unique nodes
        agg_funcs = {}
        rename_map = {}
        for features, method, rename_col, _ in node_features:
            feature_name = features[0] if len(features) == 1 else features[2]

            if method == 'mean':
                agg_funcs[feature_name] = 'mean'
            elif method == 'first':
                agg_funcs[feature_name] = 'first'
            elif method == 'mode':
                agg_funcs[feature_name] = lambda x: x.mode().iloc[0] if not x.mode().empty else None
            
            rename_map[feature_name] = rename_col

        temp_node_df = temp_node_df.groupby('node_id').agg(agg_funcs).reset_index()
        temp_node_df.rename(columns=rename_map, inplace=True)

        # Adding the new features to the main nodes dataframe
        self.nodes = pd.merge(self.nodes, temp_node_df, on='node_id', how='outer')

    def extract_nodes(self, node_features=None, graph_related_features=None):
        """Extract nodes (x) data that is used across splits"""

        # Ensure that unique_ids have been generated
        if not self.preprocessed["unique_ids_created"]:
            raise RuntimeError(
                "Unique account IDs must be created before computing network features"
            )

        logging.info("Extracting nodes...")

        # Creating empty node dataframe
        num_nodes = self.df[['from_account_idx', 'to_account_idx']].max().max() + 1
        logging.info(f"Creating a Data Frame containing {num_nodes} nodes")
        self.nodes = pd.DataFrame({'node_id': np.arange(num_nodes)})

        # Adding node features to the dataframe
        # 1. Graph related features (e.g. pagerank, degree_centrality)
        # 2. Aggregated node features (e.g. avg_received, avg_sent, mode_bank, etc)
        if graph_related_features is not None:
            self.add_graph_related_features(graph_related_features)
        
        if node_features is not None:
            self.add_node_features(node_features)

        if self.nodes.shape[1] == 1:
            self.nodes['placeholder'] = 1

    def generate_tensors(self, edge_features, node_features=None, edges = ["from_account_idx", "to_account_idx"]):
        """Convert data to PyTorch tensor format for GNNs"""
        logging.info("Generating tensors...")

        def create_pyg_data(X, y, dataset_name):

            edge_index = torch.LongTensor(X[edges].values.T) # [2, num_edges]
            edge_attr = torch.tensor(X[edge_features].values, dtype=torch.float) # [num_edges, num_edge_features]
            edge_labels = torch.LongTensor(y.values) # [num_edges]
            node_attr = torch.tensor(self.nodes.drop(columns='node_id').values, dtype=torch.float) # [num_nodes, num_node_features]
            
            data = Data(edge_index=edge_index, edge_attr=edge_attr, x=node_attr, y=edge_labels)

            # Print tensor shapes
            logging.info(f"\nDataset: {dataset_name}")
            logging.info(f"  Edge Index Shape: {edge_index.shape} (should be [2, num_edges])")
            logging.info(f"  Edge Attribute Shape: {edge_attr.shape} (should be [num_edges, num_edge_features])")
            logging.info(f"  Node Attribute Shape: {node_attr.shape} (should be [num_nodes, num_node_features])")
            logging.info(f"  Edge Labels Shape: {edge_labels.shape} (should be [num_edges])")
            
            return data

        # Create PyTorch Geometric datasets for train, validation, and test
        self.train_data = create_pyg_data(self.X_train, self.y_train, "train")
        self.val_data = create_pyg_data(self.X_val, self.y_val, "val")
        self.test_data = create_pyg_data(self.X_test, self.y_test, "test")

        return self.train_data, self.val_data, self.test_data
    
    def run_preprocessing(self, graph_feats=True):
        """Runs all preprocessing steps in the correct order.
           Option to not include graph_feats calculation (takes long time)
        """
        logging.info("Running preprocessing pipeline...\n")

        try:
            self.rename_columns()
            self.drop_duplicates()
            self.currency_normalization()
            self.extract_currency_features()
            self.extract_time_features()
            self.cyclical_encoding()
            self.binary_weekend()
            self.create_unique_ids()
            self.extract_additional_time_features()
            self.apply_label_encoding()
            self.apply_one_hot_encoding()
            if graph_feats:
                self.extract_graph_features()
            logging.info("Preprocessing completed successfully!")
            logging.info(self.preprocessed)

        except Exception as e:
            logging.info(f"Error in preprocessing: {e}")

    def split_train_test_val(
        self,
        X_cols=None,
        y_col="is_laundering",
        test_size=0.15,
        val_size=0.15,
        split_type="random_stratified",
    ):
        """Perform Train-Test-Validation Split

        OPTIONS: ["random_stratified", "temporal", "temporal_agg"]

        "random stratified": Data is randomized and split while keeping
            `is_laundering` label proportionate bt train/val/test.
        "temporal": Data is sorted by timestamp, and split into
            df[:t1], df[t1:t2], df[t2:]
        "temporal_agg": Data is sorted by timestamp and split into
            df[:t1], df[:t2], df[:].
        
        Note that in GNN, need to mask labels s.t. val only evaluates
        df[t1:t2] labels and test only evaluates df[t2:] labels.
        """

        # Ensure a valid split is chosen
        valid_splits = ["random_stratified", "temporal", "temporal_agg"]
        if split_type is None:
            logging.info(
                "No split type entered; using default split_type: "
                "'random_stratified'"
            )
            logging.info("Valid split_type options:\n"
                "- 'random_stratified' → Stratified random split maintaining label balance.\n"
                "- 'temporal' → Sequential split based on timestamps.\n"
                "- 'temporal_agg' → Aggregated sequential split (masking required in GNN evaluation).\n"
                "See `split_train_test_val` for more details."
            )
            split_type = "random_stratified"

        elif split_type not in valid_splits:
            raise ValueError(
                f"Invalid split_type: '{split_type}'.\n"
                f"Expected one of {valid_splits}.\n"
                "Please choose a valid option:\n"
                "- 'random_stratified'\n"
                "- 'temporal'\n"
                "- 'temporal_agg'\n"
                "See `split_train_test_val` for more details."
            )
        self.split_type = split_type

        # Allow `split_train_test_val` to default to using all columns
        # for the set of `X_cols`
        if X_cols is None:
            X_cols = sorted(list(set(self.df.columns) - "is_laundering"))
        logging.info("Using the following set of 'X_cols'")
        logging.info(X_cols)

        if self.split_type == "random_stratified":
            X = self.df[X_cols]
            y = self.df[y_col]

            self.X_train, X_temp, self.y_train, y_temp = train_test_split(
                X, y, test_size=(test_size + val_size), random_state=42, stratify=y
            )
            self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
                X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=42, stratify=y_temp
            )

        elif self.split_type == "temporal":
            if "timestamp_int" not in self.df.columns:
                raise RuntimeError("Need `timestamp_int` in df for temporal split. Review preprocessing steps.")

            # Sort by time and find indices for data split
            df_sorted = self.df.sort_values(by=["timestamp_int"])
            X = df_sorted[X_cols]
            y = df_sorted[y_col]
            t1 = int((1-(test_size+val_size))*len(self.df))
            t2 = int((1-test_size)*len(self.df))

            # Split databased on timestamp
            self.X_train, self.y_train = X[:t1], y[:t1]
            self.X_val, self.y_val = X[t1:t2], y[t1:t2]
            self.X_test, self.y_test = X[t2:], y[t2:]

        elif self.split_type == "temporal_agg":
            if "timestamp_int" not in self.df.columns:
                raise RuntimeError("Must include timestamp_int in df for temporal split")

            if "edge_id" not in self.df.columns:
                raise RuntimeError("Must include edge_id in df for temporal split")

            # Sort by time and find indices for data split
            X = self.df[X_cols]
            y = self.df[y_col]
            t1 = int((1-(test_size+val_size))*len(self.df))
            t2 = int((1-test_size)*len(self.df))

            # Temporal aggregated split (keeps earlier data but masks during GNN loss computation)
            self.X_train, self.y_train = X[:t1], y[:t1]
            self.X_val, self.y_val = X[:t2], y[:t2]
            self.X_test, self.y_test = X[:], y[:]

        logging.info(f"Data split using {self.split_type} method.")
        if self.split_type == "temporal_agg":
            logging.info("Remember to mask labels in GNN evaluation.\n"
                " - Train: no mask \n"
                " - Val: mask y_lab[:t1] (only evaluate labels y_lab[t1:t2]) \n"
                " - Test: mask y_lab[:t2] (only evaluate labels y_lab[t2:])")

        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

    def result_metrics(self, slide_title, y_train, y_train_pred, y_train_proba,
                       y_val, y_val_pred, y_val_proba,
                       y_test, y_test_pred, y_test_proba,
                       class_labels=None):
        """
        Compute and display model performance metrics for train, validation, and test sets.
        """

        def compute_metrics(y_true, y_pred, y_proba):
            """ Compute key classification metrics """
            cm = confusion_matrix(y_true, y_pred)
            accuracy = balanced_accuracy_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            logloss = log_loss(y_true, y_proba) if y_proba is not None else None
            precision = precision_score(y_true, y_pred, average="binary")
            recall = recall_score(y_true, y_pred, average="binary")

            if y_proba is not None:
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1] if len(y_proba.shape) > 1 else y_proba)
                roc_auc = roc_auc_score(y_true, y_proba[:, 1] if len(y_proba.shape) > 1 else y_proba)
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba[:, 1] if len(y_proba.shape) > 1 else y_proba)
                pr_auc = auc(recall_curve, precision_curve)
            else:
                fpr, tpr, roc_auc, precision_curve, recall_curve, pr_auc = None, None, None, None, None, None

            return {
                "confusion_matrix": cm,
                "accuracy": accuracy,
                "mcc": mcc,
                "log_loss": logloss,
                "precision": precision,
                "recall": recall,
                "roc_curve": (fpr, tpr),
                "roc_auc": roc_auc,
                "precision_recall_curve": (precision_curve, recall_curve),
                "pr_auc": pr_auc
            }

        # Compute metrics for train, validation, and test sets
        train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba)
        val_metrics = compute_metrics(y_val, y_val_pred, y_val_proba)
        test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)

        dataset_names = ["Train", "Validation", "Test"]
        metrics_dicts = [train_metrics, val_metrics, test_metrics]

        # Create figure for **3 rows, 4 columns**
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(24, 12))
        fig.suptitle(slide_title, fontsize=20, fontweight="bold")

        for i, (name, metrics) in enumerate(zip(dataset_names, metrics_dicts)):
            cm, roc_curve_vals, pr_curve_vals = metrics["confusion_matrix"], metrics["roc_curve"], metrics["precision_recall_curve"]

            # Confusion Matrix (Column 1)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i, 0])
            axes[i, 0].set_title(f"{name} Set - Confusion Matrix", fontsize=14, fontweight="bold")
            axes[i, 0].set_xlabel("Predicted Label", fontsize=12)
            axes[i, 0].set_ylabel("True Label", fontsize=12)

            # ROC Curve (Column 2)
            if metrics["roc_auc"] is not None:
                fpr, tpr = roc_curve_vals
                axes[i, 1].plot(fpr, tpr, label=f"ROC AUC = {metrics['roc_auc']:.4f}")
                axes[i, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")  # Baseline
                axes[i, 1].set_title(f"{name} Set - ROC Curve", fontsize=14, fontweight="bold")
                axes[i, 1].legend(fontsize=12)

            # Precision-Recall Curve (Column 3)
            if metrics["pr_auc"] is not None:
                precision, recall = pr_curve_vals
                axes[i, 2].plot(recall, precision, label=f"PR AUC = {metrics['pr_auc']:.4f}")
                axes[i, 2].set_title(f"{name} Set - Precision-Recall Curve", fontsize=14, fontweight="bold")
                axes[i, 2].legend(fontsize=12)

            # Convert None values to "N/A" before formatting
            log_loss_value = f"{metrics['log_loss']:.4f}" if metrics["log_loss"] is not None else "N/A"
            roc_auc_value = f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "N/A"
            pr_auc_value = f"{metrics['pr_auc']:.4f}" if metrics["pr_auc"] is not None else "N/A"

            # Text-based Metrics (Column 4)
            metrics_text = (
                f"Balanced Accuracy: {metrics['accuracy']:.4f}\n"
                f"MCC: {metrics['mcc']:.4f}\n"
                f"Precision: {metrics['precision']:.4f}\n"
                f"Recall: {metrics['recall']:.4f}\n"
                f"Log Loss: {log_loss_value}\n"
                f"AUC-ROC: {roc_auc_value}\n"
                f"PR AUC: {pr_auc_value}"
            )
            axes[i, 3].text(0.1, 0.5, metrics_text, fontsize=14, ha="left", va="center", family="monospace", fontweight="bold")
            axes[i, 3].axis("off")  # Hide axis lines for text box

        plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Better spacing for presentation
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for title
        plt.show()
