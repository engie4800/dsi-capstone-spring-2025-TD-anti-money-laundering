import datetime
import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from IPython.display import display
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
from torch_geometric.loader import LinkNeighborLoader
from torch.optim import Adam
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAveragePrecision,
)
from tqdm import tqdm

from helpers.currency import get_usd_conversion
from model import GINe


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
            "onehot_encoded": False,
            "train_test_val_data_split": False,
            "post_split_node_features": False,
            "train_test_val_data_split_graph": False,
            "got_data_loaders": False,
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
        self.nodes = pd.DataFrame({"node_id": np.arange(num_nodes)})

        # Adding node features to the dataframe
        # 1. Graph related features (e.g. pagerank, degree_centrality)
        # 2. Aggregated node features (e.g. avg_received, avg_sent, mode_bank, etc)
        if graph_related_features is not None:
            self.add_graph_related_features(graph_related_features)
        
        if node_features is not None:
            self.add_node_features(node_features)

        if self.nodes.shape[1] == 1:
            self.nodes["placeholder"] = 1

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
        self.test_size = test_size
        self.val_size = val_size

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
            X_cols = sorted(
                list(
                    set(self.df.columns) - set([
                        "from_account",
                        "from_account_id",
                        "from_account_idx",
                        "from_bank",
                        "is_laundering",
                        "to_account",
                        "to_account_id",
                        "to_account_idx",
                        "to_bank",
                    ])
                )
            )
        self.X_cols = X_cols
        logging.info("Using the following set of 'X_cols'")
        logging.info(self.X_cols)

        if self.split_type == "random_stratified":
            X = self.df[self.X_cols]
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
            X = df_sorted[self.X_cols]
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
            X = self.df[self.X_cols]
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

        self.preprocessed["train_test_val_data_split"] = True

        # We should just always get (and attach to the pipeline) the
        # split indices
        self.get_split_indices()

        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

    def get_split_indices(self):
        """
        Returns numpy arrays of appropriate indices based on validation
        and test sizes
        """
        if not self.preprocessed["train_test_val_data_split"]:
            raise RuntimeError(
                "Data must have been split into train, test, validation "
                "sets before getting split indices."
            )

        num_edges = len(self.df)

        self.t1 = int(num_edges * (1 - self.val_size - self.test_size))
        self.t2 = int(num_edges * (1 - self.test_size))
        self.train_indices = np.arange(0, self.t1)
        self.val_indices = np.arange(self.t1, self.t2)
        self.test_indices = np.arange(self.t2, num_edges)
        self.train_val_indices = np.concat([self.train_indices, self.val_indices])
        self.train_val_test_indices = np.concat([self.train_val_indices, self.test_indices])

        return (
            self.t1,
            self.t2,
            self.train_indices,
            self.val_indices,
            self.test_indices,
            self.train_val_indices,
            self.train_val_test_indices,
        )

    def compute_split_specific_node_features(
        self,
        graph_features: list[str] = ["sent_amount_usd"],
    ) -> None:
        """
        Compute node features for a specific split using `from_account_idx`
        and `to_account_idx` as node identifiers.
        """
        logging.info("Getting train-test-split-specific node features")
        if not self.preprocessed["train_test_val_data_split"]:
            raise RuntimeError(
                "Data must have been split into train, test, validation "
                "sets before getting split indices."
            )

        def get_node_features(split_edges, split_name: str, graph_features):
            # TODO: this duplicates to some extent some of the other
            # graph feature functions on the pipeline, which we should
            # clean up

            logging.info(f"Computing {split_name} node features...")

            # Aggregate edges by from-to pairs
            aggregated_edges = (
                split_edges
                .groupby(["from_account_idx", "to_account_idx"])[graph_features]
                .sum()
                .reset_index()
            )

            # Build directed graph using account indices as node ids
            G = nx.DiGraph()
            for _, row in aggregated_edges.iterrows():
                G.add_edge(
                    int(row["from_account_idx"]),
                    int(row["to_account_idx"]),
                    **{col: row[col] for col in graph_features}
                )

            # Compute graph features using account index as node_id
            degree_centrality = nx.degree_centrality(G)
            in_deg = {n: d / (len(G) - 1) for n, d in G.in_degree()}
            out_deg = {n: d / (len(G) - 1) for n, d in G.out_degree()}
            pagerank = nx.pagerank(G, weight="sent_amount_usd")

            # Collect into DataFrame
            node_df = pd.DataFrame({"node_id": list(G.nodes)})
            node_df["degree_centrality"] = node_df["node_id"].map(degree_centrality)
            node_df["in_degree"] = node_df["node_id"].map(in_deg)
            node_df["out_degree"] = node_df["node_id"].map(out_deg)
            node_df["pagerank"] = node_df["node_id"].map(pagerank)

            # Ensure completeness and ordering
            node_df.fillna(0, inplace=True)
            node_df = node_df.sort_values("node_id").reset_index(drop=True)

            logging.info(f"Computed node features for {split_name} with {len(node_df)} nodes.")
            return node_df

        self.train_nodes = get_node_features(
            split_edges=self.df.loc[self.train_indices, :],
            split_name="train",
            graph_features=graph_features
        )
        self.val_nodes = get_node_features(
            split_edges=self.df.loc[self.train_val_indices, :],
            split_name="val",
            graph_features=graph_features
        )
        self.test_nodes = get_node_features(
            split_edges=self.df.loc[self.train_val_test_indices, :],
            split_name="test",
            graph_features=graph_features
        )

        self.preprocessed["post_split_node_features"] = True

    def split_train_test_val_graph(self, edge_features=None):
        logging.info("Splitting into train, test, validation graphs")
        if not self.preprocessed["post_split_node_features"]:
            raise RuntimeError(
                "Train, test, validation graph split assumes that post-"
                "split node features have been added."
            )

        # A default set of edge features that excludes some obvious
        # features we don't want
        if edge_features is None:
            edge_features = self.X_cols  # TODO: any reason not to do this?

        # Nodes
        tr_x = torch.tensor(self.train_nodes.drop(columns="node_id").values, dtype=torch.float)
        val_x = torch.tensor(self.val_nodes.drop(columns="node_id").values, dtype=torch.float)
        te_x = torch.tensor(self.test_nodes.drop(columns="node_id").values, dtype=torch.float)

        # Labels
        self.y = torch.LongTensor(self.df["is_laundering"].to_numpy())

        # Edge index
        self.edge_index = torch.LongTensor(self.df[["from_account_idx", "to_account_idx"]].to_numpy().T)

        # Edge attr
        edge_attr = torch.tensor(self.df[edge_features].to_numpy(), dtype=torch.float)


        # Overwrites the values we got from the original split
        # TODO: Do we need to keep both?
        self.t1 = torch.tensor(self.t1)
        self.t2 = torch.tensor(self.t2)
        self.train_indices = torch.tensor(self.train_indices)
        self.val_indices = torch.tensor(self.val_indices)
        self.test_indices = torch.tensor(self.test_indices)

        cat_tr_val_inds = torch.cat((self.train_indices, self.val_indices))
        self.train_data = Data(
            x=tr_x,
            edge_index=self.edge_index[:,self.train_indices],
            edge_attr=edge_attr[self.train_indices],
            y=self.y[self.train_indices],
        )
        self.val_data = Data(
            x=val_x,
            edge_index=self.edge_index[:,cat_tr_val_inds],
            edge_attr=edge_attr[cat_tr_val_inds],
            y=self.y[cat_tr_val_inds],
        )
        self.test_data = Data(
            x=te_x,
            edge_index=self.edge_index,
            edge_attr=edge_attr,
            y=self.y,
        )

        self.preprocessed["train_test_val_data_split_graph"] = True

        return (
            self.train_indices,
            self.val_indices,
            self.test_indices,
            self.train_data,
            self.val_data,
            self.test_data,
            self.edge_index,
            self.y,
        )

    def get_data_loaders(self, num_neighbors=[100,100], batch_size=8192):
        logging.info("Getting data loaders")
        if not self.preprocessed["train_test_val_data_split_graph"]:
            raise RuntimeError(
                "Cannot get data loaders for GNN until the graph train-"
                "test split is complete."
            )

        # TODO: might be able to move this to somewhere more meaningful,
        # but it is needed here at the latest
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        def scale_data(data, cols_to_scale, device):

            edge_attr_cpu = data.edge_attr.cpu().numpy()

            for col in cols_to_scale:
                col_data = edge_attr_cpu[:, col]
                col_data = col_data.copy()

                # Mask to identify valid (non -1) values
                mask = col_data != -1
                if np.sum(mask) == 0:
                    continue  # skip column if all are -1

                # Median impute and scale
                median_val = np.median(col_data[mask])
                col_data[~mask] = median_val
                edge_attr_cpu[:, col] = StandardScaler().fit_transform(col_data.reshape(-1, 1)).flatten()

            data.edge_attr = torch.from_numpy(edge_attr_cpu).float().to(device)
            data.x = data.x.to(device)
            data.y = data.y.to(device)
            return data

        # Standard scale on CPU before sending to device
        # TODO: does it make sense to overwrite data here?
        self.train_data = scale_data(self.train_data, [1, 2, 3, 4, 5], self.device)
        self.val_data = scale_data(self.val_data, [1, 2, 3, 4, 5], self.device)
        self.test_data = scale_data(self.test_data, [1, 2, 3, 4, 5], self.device)

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

        return (
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.train_data,
            self.val_data,
            self.test_data,
        )

    def initialize_training(self):
        # TODO: does it make sense to attach this, as well as well as
        # the evaluate and train functions, onto the model pipeline?

        num_edge_features = self.train_data.edge_attr.shape[1]-1  # num edge feats - edge_id
        num_node_features = self.train_data.x.shape[1]
        self.model = GINe(n_node_feats=num_node_features, n_edge_feats=num_edge_features).to(self.device)
        self.optimizer = Adam(model.parameters(), lr=0.005)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",            # maximize the metric (e.g., F1, PR AUC)
            factor=0.5,            # reduce LR by half when triggered
            patience=3,            # wait 3 epochs without improvement
            verbose=True
        )

        pos = (self.df["is_laundering"] == 1).sum()
        neg = (self.df["is_laundering"] == 0).sum()
        pos_weight_val = 6  # neg / pos
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=self.device))

    @torch.no_grad()
    def evaluate(self, loader, inds, threshold):
        self.model.eval()
        acc_fn = BinaryAccuracy(threshold=threshold).to(self.device)
        prec_fn = BinaryPrecision(threshold=threshold).to(self.device)
        rec_fn = BinaryRecall(threshold=threshold).to(self.device)
        f1_fn = BinaryF1Score(threshold=threshold).to(self.device)
        pr_auc_fn = BinaryAveragePrecision().to(self.device)

        loss_fn = nn.BCEWithLogitsLoss()
        preds, targets, probs = [], [], []
        total_loss = 0

        for batch in loader:
            batch_input_ids = batch.input_id.detach().cpu()
            global_seed_inds = inds[batch_input_ids]
            seed_edge_ids = self.df.loc[global_seed_inds.cpu().numpy(), "edge_id"].values
            edge_ids_in_batch = batch.edge_attr[:, 0].detach().cpu().numpy()
            mask = torch.isin(torch.tensor(edge_ids_in_batch), torch.tensor(seed_edge_ids)).to(self.device)

            batch_edge_attr = batch.edge_attr[:, 1:].clone()
            batch = batch.to(self.device)

            logits = self.model(batch.x, batch.edge_index, batch_edge_attr).view(-1)[mask]
            target = batch.y[mask]
            prob = torch.sigmoid(logits)
            pred = (prob > threshold).long()

            total_loss += loss_fn(logits, target.float()).item() * logits.size(0)
            preds.append(pred); targets.append(target); probs.append(prob)

        preds = torch.cat(preds)
        targets = torch.cat(targets)
        probs = torch.cat(probs)
        total_loss /= len(targets)

        return (
            total_loss,
            acc_fn(preds, targets),
            prec_fn(preds, targets),
            rec_fn(preds, targets),
            f1_fn(preds, targets),
            pr_auc_fn(probs, targets)
        )

    def train(self, threshold=0.5, epochs=20):

        acc_fn = BinaryAccuracy(threshold=threshold).to(self.device)
        prec_fn = BinaryPrecision(threshold=threshold).to(self.device)
        rec_fn = BinaryRecall(threshold=threshold).to(self.device)
        f1_fn = BinaryF1Score(threshold=threshold).to(self.device)
        pr_auc_fn = BinaryAveragePrecision().to(self.device)

        best_val_f1 = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            train_preds, train_targets, train_probs = [], [], []

            acc_fn.reset(); prec_fn.reset(); rec_fn.reset(); f1_fn.reset(); pr_auc_fn.reset()

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training"):
                self.optimizer.zero_grad()
                batch_input_ids = batch.input_id.detach().cpu()
                global_seed_inds = self.train_indices[batch_input_ids]
                seed_edge_ids = self.df.loc[global_seed_inds.cpu().numpy(), "edge_id"].values
                edge_ids_in_batch = batch.edge_attr[:, 0].detach().cpu().numpy()
                mask = torch.isin(torch.tensor(edge_ids_in_batch), torch.tensor(seed_edge_ids)).to(self.device)

                batch_edge_attr = batch.edge_attr[:, 1:].clone()
                batch = batch.to(self.device)
                logits = self.model(batch.x, batch.edge_index, batch_edge_attr).view(-1)[mask]
                target = batch.y[mask]
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).long()

                loss = self.criterion(logits, target.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                train_loss += loss.item() * logits.size(0)
                train_preds.append(preds)
                train_targets.append(target)
                train_probs.append(probs)

            train_preds = torch.cat(train_preds)
            train_targets = torch.cat(train_targets)
            train_probs = torch.cat(train_probs)
            train_loss /= len(train_targets)

            train_acc = acc_fn(train_preds, train_targets)
            train_prec = prec_fn(train_preds, train_targets)
            train_rec = rec_fn(train_preds, train_targets)
            train_f1 = f1_fn(train_preds, train_targets)
            train_pr_auc = pr_auc_fn(train_probs, train_targets)

            # Validation
            val_loss, val_acc, val_prec, val_rec, val_f1, val_pr_auc = self.evaluate(
                self.val_loader,
                self.val_indices,
                threshold,
            )

            # Test
            test_loss, test_acc, test_prec, test_rec, test_f1, test_pr_auc = self.evaluate(
                self.test_loader,
                self.test_indices,
                threshold,
            )

            logging.info(f"Epoch {epoch+1}/{epochs}")
            logging.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Test Loss: {test_loss:.4f}")
            logging.info(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
            logging.info(f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")
            logging.info(f"Train PR-AUC: {train_pr_auc:.4f} | Val PR-AUC: {val_pr_auc:.4f} | Test PR-AUC: {test_pr_auc:.4f}")
            logging.info(f"Train Prec: {train_prec:.4f} | Val Prec: {val_prec:.4f} | Test Prec: {test_prec:.4f}")
            logging.info(f"Train Rec: {train_rec:.4f} | Val Rec: {val_rec:.4f} | Test Rec: {test_rec:.4f}")
            logging.info("-" * 80)

            self.scheduler.step(val_f1)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), "best_model.pt")

            torch.cuda.empty_cache()

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
