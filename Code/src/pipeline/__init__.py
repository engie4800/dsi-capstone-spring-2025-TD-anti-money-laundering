import datetime
import logging
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
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
from helpers.graph import scale_graph_edge_attributes
from model import GINe
from model.features import (
    add_currency_changed,
    add_day_of_week,
    add_hour_of_day,
    add_is_weekend,
    add_received_amount_usd,
    add_seconds_since_midnight,
    add_sent_amount_usd,
    add_time_diff_from,
    add_timestamp_int,
    add_timestamp_scaled,
    add_turnaround_time,
    add_unique_identifiers,
    cyclically_encode_feature
)
from pipeline.checks import Checker

if TYPE_CHECKING:
    from torch_geometric.explain import Explanation


class ModelPipeline:

    def __init__(self, dataset_path: str):
        """
        Initialize pipeline with dataset
        """
        self.dataset_path = dataset_path
        self.df = pd.read_csv(self.dataset_path)
        self.nodes = pd.DataFrame()

        # Which device will model computation occur on?
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Track if preprocessing steps have been completed
        self.preprocessed = {
            "renamed": False,
            "duplicates_removed": False,
            "checked_for_null_values": False,
            "currency_features_extracted": False,
            "time_features_extracted": False,
            "unique_ids_created": False,
            "additional_time_features_extracted": False,
            "cyclical_encoded": False,
            "weekend_encoded": False,
            "label_encoded": False,
            "neighbor_context_computed": False,
            "normalized": False,
            "onehot_encoded": False,
            "train_test_val_data_split": False,
            "post_split_node_features": False,
            "node_datasets_scaled": False,
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

    def rename_columns(self) -> None:
        """
        Renames the columns of `self.df` to follow a consistent
        semantics, in terms of using to and from, or sent and received,
        and to use the Pythonic snake case
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

        # Ensures required columns exist
        missing_columns = [
            col
            for col in column_mapping.keys()
            if col not in self.df.columns
        ]
        if missing_columns:
            raise KeyError(
                f"Missing expected columns in dataset: {missing_columns}"
            )

        self.df.rename(columns=column_mapping, inplace=True)
        self.preprocessed["renamed"] = True

    def drop_duplicates(self) -> None:
        """
        Removes any duplicate rows in the data frame
        """
        self.df.drop_duplicates(inplace=True)
        self.preprocessed["duplicates_removed"] = True

    def check_for_null(self) -> None:
        """
        Confirm that the given dataset does not contain any null values
        """
        if self.df.isnull().values.any():
            # Need to determine how to handle null values on a dataset
            # that has them
            raise ValueError(
                "Pipeline was developed on data that does not contain "
                "null values. Null values detected, remove them!"
            )
        self.preprocessed["checked_for_null_values"] = True

    def extract_currency_features(self) -> None:
        """
        Extract all currency-related features

            currency_changed: Whether the money in the transaction
                changes currency from sender to receiver
            add_sent_amount_usd: Sent amount in USD
            add_received_amount_usd: Received amount in USD

        """
        logging.info("Extracting currency features...")
        Checker.currency_columns_required(self)

        self.df = add_currency_changed(self.df)
        usd_conversion = get_usd_conversion(self.dataset_path)
        self.df = add_sent_amount_usd(self.df, usd_conversion)
        self.df = add_received_amount_usd(self.df, usd_conversion)

        self.preprocessed["currency_features_extracted"] = True

    def extract_time_features(self) -> None:
        """
        Extract initial time-related features

            hour_of_day: The hour of day of the transaction
            day_of_week: The day of week of the transaction
            seconds_since_midnight: The number of seconds that have
                passed since midnight on the day the transaction
                occurred
            timestamp_int: Integer representation of the timestamp

        """
        logging.info("Extracting time features...")
        Checker.timestamp_required(self)

        # Ensures `timestamp` is a `datetime` object
        if not isinstance(self.df["timestamp"], datetime.datetime):
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

        # Add each time feature
        self.df = add_hour_of_day(self.df)
        self.df = add_day_of_week(self.df)
        self.df = add_seconds_since_midnight(self.df)
        self.df = add_is_weekend(self.df)
        self.df = add_timestamp_int(self.df)
        self.df = add_timestamp_scaled(self.df)

        # Dropping timestamp ensures that the complex timestamp string
        # itself isn't used as a feature, as it is poorly suited to be
        # one due to its monotonicity and high cardinality
        self.df.drop(columns=["timestamp"], inplace= True)
        
        self.preprocessed["time_features_extracted"] = True

    def create_unique_ids(self) -> None:
        """Create a mapping from bank, account pairs, each of which
        should be unique, to a unique identifier. This adds the
        following data to each transaction:

            edge_id: Integer transaction (edge) identifier that
                represents the temporal ordering of the transaction
                dataset, e.g. `1` is the first transaction to appear,
                `N` is the last
            from_account_idx: Integer entity (node) identifier that
                identifies each unique bank, account pair, from
                representing the sender in each transaction
            to_account_idx: Integer entity (node) identifier that
                identifies each unique bank, account pair, to
                representing the receiver in each transaction

        """
        logging.info("Creating unique ids...")
        Checker.columns_were_renamed(self)
        Checker.time_features_were_extracted(self)
        self.df = add_unique_identifiers(self.df)
        self.preprocessed["unique_ids_created"] = True

    def extract_additional_time_features(self) -> None:
        """Additional time features can be extracted after creating
        unique identifiers, which depends on the initial time feature
        extraction. This method adds:

            time_diff_from: The time since the sender in a given
                transaction previously sent money
            turnaround_time: The time elapsed since the sender in a
                given transaction previously received money

        """
        logging.info("Extracting additional time features...")
        Checker.time_features_were_extracted(self)
        Checker.unique_ids_were_created(self)

        # TODO: does it make sense to add an analogous `time_diff_to`
        # representing the time since the receiver in a transaction
        # previously received money?
        self.df = add_time_diff_from(self.df)
        self.df = add_turnaround_time(self.df)

        self.preprocessed["additional_time_features_extracted"] = True

    def cyclical_encoding(self):
        """Adds cyclically-encoded time features. Some time features,
        like the day of week or time of day, contain an inherent
        discontinuity near zero. That is, values close to zero and
        values close to the max value appear as far apart as possible,
        but are actually very close (consider seconds before and
        seconds after midnight). Encoding these features removes this
        discontinuity.
        """
        logging.info("Adding cyclical encoding to time features...")
        Checker.time_features_were_extracted(self)
        self.df = cyclically_encode_feature(self.df, "day", "day_of_week")
        self.df = cyclically_encode_feature(self.df, "time_of_day", "seconds_since_midnight")

        # TODO: the `day_of_week` and `seconds_since_midnight` features
        # (in this case) are now represented cyclically. Should we
        # consider:
        #
        #   Removing the `day_of_week` and `seconds_since_midnight`
        #   features, as they are now represented in the cyclical features?
        #
        
        self.preprocessed["cyclical_encoded"] = True

    def apply_one_hot_encoding(self, onehot_categorical_features=None):
        """One hot encode categorical columns, handling related columns"""
        logging.info("Applying one hot encoding...")
        Checker.columns_were_renamed(self)

        # Default columns for encoding
        default_categorical_features = ["payment_type", "received_currency", "sent_currency"]

        # Use provided or default
        categorical_features = onehot_categorical_features or default_categorical_features

        # Find related column groups (e.g., same suffix)
        column_groups = {}
        for col in categorical_features:
            _, _, suffix = col.partition("_")
            if suffix and any(other.endswith(f"_{suffix}") for other in categorical_features if other != col):
                column_groups.setdefault(suffix, []).append(col)

        # Track columns to drop and DataFrames to concat
        columns_to_drop = []
        encoded_dfs = []

        # Encode grouped columns using shared encoder
        for suffix, cols in column_groups.items():
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
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
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            transformed = encoder.fit_transform(self.df[[col]])
            ohe_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
            encoded_df = pd.DataFrame(transformed, columns=ohe_cols, index=self.df.index)
            encoded_dfs.append(encoded_df)
            columns_to_drop.append(col)

        # Drop original and concat encoded
        self.df.drop(columns=columns_to_drop, inplace=True)
        self.df = pd.concat([self.df] + encoded_dfs, axis=1)

        logging.info(f"One hot encoding applied to columns: {categorical_features}\n")
        self.preprocessed["onehot_encoded"] = True
        
    def apply_label_encoding(self, categorical_features=None):
        """Label encode categorical columns, handling related columns"""
        logging.info("Applying label encoding...")
        Checker.columns_were_renamed(self)

        # Default columns for encoding
        default_categorical_features = ["day_of_week", "from_bank", "to_bank"]

        # Determine columns to encode
        if categorical_features is None:
            categorical_features = default_categorical_features

        # Find related columns (e.g., "from_bank" and "to_bank" should use the same encoder)
        column_groups = {}
        for col in categorical_features:
            _, _, suffix = col.partition("_")
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

        logging.info(f"Label encoding applied to columns: {categorical_features}\n")
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
        
        # Building the graph from aggregated edges data
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
            
        logging.info(f" Graph features computed using: {weight_cols}")
        logging.info(
            "**Note**, previously graph-based features were calculated "
            "using only `sent_amount` as edge weight (only based on "
            "outgoing transactions). Now both sent and received amounts "
            "are included by default."
        )
        logging.info(
            "New feature columns added: degree_centrality, "
            "in_degree_centrality, out_degree_centrality, "
            f"{', '.join([f'pagerank_{col}' for col in weight_cols])}\n"
        )

    def add_node_features(self, node_features):
        logging.info("Adding node features...")

        # Combining nodes and their respective features (source only, destination only, or both) into several dataframes
        all_nodes = []
        for features, _, _, kind in node_features:
            if kind == "source_agg":
                temp = self.df[["from_account_idx", features[0]]].rename(columns={"from_account_idx": "node_id"})
            
            elif kind == "destination_agg":
                temp = self.df[["to_account_idx", features[0]]].rename(columns={"to_account_idx": "node_id"})

            elif kind == "merge_agg":
                if len(features) == 1:
                    temp_from = self.df[["from_account_idx", features[0]]].rename(columns={"from_account_idx": "node_id"})
                    temp_to = self.df[["to_account_idx", features[0]]].rename(columns={"to_account_idx": "node_id"})
                else:
                    temp_from = self.df[["from_account_idx", features[0]]].rename(columns={"from_account_idx": "node_id", features[0]: features[2]})
                    temp_to = self.df[["to_account_idx", features[1]]].rename(columns={"to_account_idx": "node_id", features[1]: features[2]})

                temp = pd.concat([temp_from, temp_to])

            all_nodes.append(temp)

        # Merging all dataframes on their node_id
        temp_node_df = all_nodes[0]
        for i in all_nodes[1:]:
            temp_node_df = pd.merge(temp_node_df, i, on="node_id", how="outer")

        # Aggregating and summarizing into one dataframe containing unique nodes
        agg_funcs = {}
        rename_map = {}
        for features, method, rename_col, _ in node_features:
            feature_name = features[0] if len(features) == 1 else features[2]

            if method == "mean":
                agg_funcs[feature_name] = "mean"
            elif method == "first":
                agg_funcs[feature_name] = "first"
            elif method == "mode":
                agg_funcs[feature_name] = lambda x: x.mode().iloc[0] if not x.mode().empty else None
            
            rename_map[feature_name] = rename_col

        temp_node_df = temp_node_df.groupby("node_id").agg(agg_funcs).reset_index()
        temp_node_df.rename(columns=rename_map, inplace=True)

        # Adding the new features to the main nodes dataframe
        self.nodes = pd.merge(self.nodes, temp_node_df, on="node_id", how="outer")

    def extract_nodes(self, node_features=None, graph_related_features=None):
        """Extract nodes (x) data that is used across splits"""
        Checker.unique_ids_were_created(self)

        logging.info("Extracting nodes...")

        # Creating empty node dataframe
        num_nodes = self.df[["from_account_idx", "to_account_idx"]].max().max() + 1
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
            node_attr = torch.tensor(self.nodes.drop(columns="node_id").values, dtype=torch.float) # [num_nodes, num_node_features]
            
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
            self.check_for_null()
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
                        # Remove identifying fields, as well as the
                        # output `is_laundering`
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
            Checker.time_features_were_extracted(self)  # timestamp_int required

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
            Checker.time_features_were_extracted(self)  # timestamp_int required
            Checker.unique_ids_were_created(self)  # edge_id required

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

        # This assumes we should just always get (and attach to the
        # pipeline) the split indices
        self.get_split_indices()

        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

    def get_split_indices(self):
        """
        Returns numpy arrays of appropriate indices based on validation
        and test sizes
        """
        Checker.data_split_to_train_val_test(self)

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
        Checker.data_split_to_train_val_test(self)

        def get_node_features(split_df, split_name: str, graph_features):
            # TODO: this duplicates to some extent some of the other
            # graph feature functions on the pipeline, which we can
            # clean up?
            logging.info(f"Computing {split_name} node features...")

            # --- TRANSACTIONAL NODE FEATURES ---

            # Outgoing stats (from_account_idx)
            out_stats = (
                split_df.groupby("from_account_idx")["sent_amount_usd"]
                    .agg(["count", "sum", "mean", "std", "min", "max"])
                    .add_prefix("out_")
                    .reset_index()
                    .rename(columns={"from_account_idx": "node_id"})
            )

            # Incoming stats (to_account_idx)
            in_stats = (
                split_df.groupby("to_account_idx")["sent_amount_usd"]
                    .agg(["count", "sum", "mean", "std", "min", "max"])
                    .add_prefix("in_")
                    .reset_index()
                    .rename(columns={"to_account_idx": "node_id"})
            )

            # Number of unique partners
            unique_out = (
                split_df.groupby("from_account_idx")["to_account_idx"]
                    .nunique()
                    .reset_index()
                    .rename(columns={"from_account_idx": "node_id", "to_account_idx": "num_unique_out_partners"})
            )

            unique_in = (
                split_df.groupby("to_account_idx")["from_account_idx"]
                    .nunique()
                    .reset_index()
                    .rename(columns={"to_account_idx": "node_id", "from_account_idx": "num_unique_in_partners"})
            )

            # Merge transactional stats
            node_stat_features = out_stats.merge(in_stats, on="node_id", how="outer")
            node_stat_features = node_stat_features.merge(unique_out, on="node_id", how="outer")
            node_stat_features = node_stat_features.merge(unique_in, on="node_id", how="outer")

            # Derived features
            node_stat_features["net_flow"] = node_stat_features["out_sum"] - node_stat_features["in_sum"]
            node_stat_features["avg_txn_in"] = node_stat_features["in_mean"]
            node_stat_features["avg_txn_out"] = node_stat_features["out_mean"]

            # --- GRAPH-BASED NODE FEATURES ---

            aggregated_edges = (
                split_df
                .groupby(["from_account_idx", "to_account_idx"])[graph_features]
                .sum()
                .reset_index()
            )

            G = nx.DiGraph()
            for _, row in aggregated_edges.iterrows():
                G.add_edge(
                    int(row["from_account_idx"]),
                    int(row["to_account_idx"]),
                    **{col: row[col] for col in graph_features}
                )

            degree_centrality = nx.degree_centrality(G)
            in_deg = {n: d / (len(G) - 1) for n, d in G.in_degree()}
            out_deg = {n: d / (len(G) - 1) for n, d in G.out_degree()}
            pagerank = nx.pagerank(G, weight="sent_amount_usd")

            node_graph_df = pd.DataFrame({"node_id": list(G.nodes)})
            node_graph_df["degree_centrality"] = node_graph_df["node_id"].map(degree_centrality)
            node_graph_df["in_degree"] = node_graph_df["node_id"].map(in_deg)
            node_graph_df["out_degree"] = node_graph_df["node_id"].map(out_deg)
            node_graph_df["pagerank"] = node_graph_df["node_id"].map(pagerank)

            # --- COMBINE ALL FEATURES ---

            node_df = pd.merge(node_graph_df, node_stat_features, on="node_id", how="outer")
            node_df.fillna(0, inplace=True)
            node_df = node_df.sort_values("node_id").reset_index(drop=True)

            print(f"✅ Computed node features for {split_name} with {len(node_df)} nodes.")
            return node_df

        self.nodes = get_node_features(
            split_df=self.df,
            split_name="all",
            graph_features=graph_features,
        )
        self.train_nodes = get_node_features(
            split_df=self.df.loc[self.train_indices, :],
            split_name="train",
            graph_features=graph_features
        )
        self.val_nodes = get_node_features(
            split_df=self.df.loc[self.train_val_indices, :],
            split_name="val",
            graph_features=graph_features
        )
        self.test_nodes = get_node_features(
            split_df=self.df.loc[self.train_val_test_indices, :],
            split_name="test",
            graph_features=graph_features
        )

        self.preprocessed["post_split_node_features"] = True

    def scale_node_data_frames(self, cols_to_scale=None):
        logging.info("Scaling node data frames...")
        Checker.train_val_test_node_features_added(self)

        def preprocess_column(col_series):
            """Impute -1 values with median of the valid values."""
            mask = col_series != -1
            if mask.sum() == 0:
                return col_series, None  # No valid values to impute/scale
            median_val = np.median(col_series[mask])
            col_series = col_series.copy()
            col_series[~mask] = median_val
            return col_series, median_val

        if cols_to_scale is None:
            # Default to scaling all columns, except for the index and
            # graph features with specific definitions
            cols_to_scale = list(set(self.train_nodes.columns) - set([
                "node_id",
                "degree_centrality",
                "in_degree",
                "out_degree",
                "pagerank",
            ]))

        nodes = self.nodes.copy()
        train_nodes = self.train_nodes.copy()
        val_nodes = self.val_nodes.copy()
        test_nodes = self.test_nodes.copy()

        scalers = {}

        for col in cols_to_scale:

            # Impute -1 in train
            train_col, train_median = preprocess_column(train_nodes[col])
            train_nodes[col] = train_col

            # Fit scaler on train
            scaler = StandardScaler()
            train_nodes[col] = scaler.fit_transform(train_nodes[col].values.reshape(-1, 1)).flatten()
            scalers[col] = (scaler, train_median)

            # Impute -1 in val/test with train median.
            # TODO: we also impute all nodes here, does this make sense
            # and is it necessary? We are processing all nodes because
            # interpreting the GNN requires an input `x` that doesn't
            # seem to be defined yet. I'm interpreting the `x` we need
            # to be the one defined in `split_train_test_val_graph`,
            # but that assumption should be confirmed
            for df in [nodes, val_nodes, test_nodes]:
                if col in df.columns:
                    col_vals = df[col].copy()
                    col_vals[col_vals == -1] = train_median
                    df[col] = scalers[col][0].transform(col_vals.values.reshape(-1, 1)).flatten()

        # TODO: do we need to have ever copied these, or could we just
        # use / update them in place?
        self.nodes = nodes
        self.train_nodes = train_nodes
        self.val_nodes = val_nodes
        self.test_nodes = test_nodes

        self.preprocessed["node_datasets_scaled"] = True

        return self.train_nodes, self.val_nodes, self.test_nodes

    def split_train_test_val_graph(self, edge_features=None):
        logging.info("Splitting into train, test, validation graphs")
        Checker.train_val_test_node_features_added(self)

        # A default set of edge features that excludes some obvious
        # features we don't want
        if edge_features is None:
            # TODO: any reason not to do this?
            # edge_features = self.X_cols
            # For now, the features defined in the latest baseline:
            edge_features = [
                "edge_id",
                "sent_amount_usd",
                "received_amount_usd",
                "timestamp_scaled",
                "time_diff_from",
                "turnaround_time",
                "day_sin",
                "day_cos",
                "time_of_day_sin",
                "time_of_day_cos",
                "payment_type_ACH",
                "currency_changed",
                "received_currency_Australian Dollar",
                "received_currency_Brazil Real",
                "received_currency_Canadian Dollar",
                "received_currency_Euro",
                "received_currency_Mexican Peso",
                "received_currency_Ruble",
                "received_currency_Saudi Riyal",
                "received_currency_Shekel",
                "received_currency_Swiss Franc",
                "received_currency_UK Pound",
                "received_currency_US Dollar",
                "received_currency_Yuan",
                "sent_currency_Canadian Dollar",
                "sent_currency_Euro",
                "sent_currency_Mexican Peso",
                "sent_currency_Rupee",
                "sent_currency_Saudi Riyal",
                "sent_currency_Shekel",
                "sent_currency_Swiss Franc",
                "sent_currency_UK Pound",
                "sent_currency_US Dollar",
                "sent_currency_Yuan",
            ]

        # Nodes
        self.x = torch.tensor(self.nodes.drop(columns="node_id").values, dtype=torch.float).to(self.device)
        tr_x = torch.tensor(self.train_nodes.drop(columns="node_id").values, dtype=torch.float).to(self.device)
        val_x = torch.tensor(self.val_nodes.drop(columns="node_id").values, dtype=torch.float).to(self.device)
        te_x = torch.tensor(self.test_nodes.drop(columns="node_id").values, dtype=torch.float).to(self.device)

        # Labels
        self.y = torch.LongTensor(self.df["is_laundering"].to_numpy()).to(self.device)

        # Edge index
        self.edge_index = torch.LongTensor(self.df[["from_account_idx", "to_account_idx"]].to_numpy().T).to(self.device)

        # Edge attr
        edge_attr = torch.tensor(self.df[edge_features].to_numpy(), dtype=torch.float).to(self.device)

        # Overwrites the values we got from the original split
        self.train_indices = torch.tensor(self.train_indices).to(self.device)
        self.val_indices = torch.tensor(self.val_indices).to(self.device)
        self.test_indices = torch.tensor(self.test_indices).to(self.device)

        cat_tr_val_inds = torch.cat((self.train_indices, self.val_indices)).to(self.device)
        self.data = Data(
            x=self.x,
            edge_index=self.edge_index,
            edge_attr=edge_attr,
            y=self.y,
        ).to(self.device)
        self.train_data = Data(
            x=tr_x,
            edge_index=self.edge_index[:,self.train_indices],
            edge_attr=edge_attr[self.train_indices],
            y=self.y[self.train_indices],
        ).to(self.device)
        self.val_data = Data(
            x=val_x,
            edge_index=self.edge_index[:,cat_tr_val_inds],
            edge_attr=edge_attr[cat_tr_val_inds],
            y=self.y[cat_tr_val_inds],
        ).to(self.device)
        self.test_data = Data(
            x=te_x,
            # TODO: should this not be the full edge index, edge
            # attributes, and y?
            edge_index=self.edge_index,
            edge_attr=edge_attr,
            y=self.y,
        ).to(self.device)

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
        Checker.graph_data_split_to_train_val_test(self)

        # Standard scale on CPU before sending to device
        columns_to_scale = [1, 2, 3, 4, 5]  # TODO: what are these? Why scale these specifically?
        self.train_data = scale_graph_edge_attributes(self.train_data, columns_to_scale)
        self.val_data = scale_graph_edge_attributes(self.val_data, columns_to_scale)
        self.test_data = scale_graph_edge_attributes(self.test_data, columns_to_scale)

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

    def initialize_training(self) -> None:
        # TODO: does it make sense to attach this, as well as well as
        # the evaluate and train functions, onto the model pipeline?

        num_edge_features = self.train_data.edge_attr.shape[1]-1  # num edge feats - edge_id
        num_node_features = self.train_data.x.shape[1]
        self.model = GINe(n_node_feats=num_node_features, n_edge_feats=num_edge_features).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=0.005)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",            # maximize the metric (e.g., F1, PR AUC)
            factor=0.5,            # reduce LR by half when triggered
            patience=3,            # wait 3 epochs without improvement
            verbose=True
        )

        # pos = (self.df["is_laundering"] == 1).sum()
        # neg = (self.df["is_laundering"] == 0).sum()
        # pos_weight_val = neg / pos
        pos_weight_val = 6
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

            preds.append(pred)
            targets.append(target)
            probs.append(prob)

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

    def train(self, threshold=0.5, epochs=20, patience=10):

        acc_fn = BinaryAccuracy(threshold=threshold).to(self.device)
        prec_fn = BinaryPrecision(threshold=threshold).to(self.device)
        rec_fn = BinaryRecall(threshold=threshold).to(self.device)
        f1_fn = BinaryF1Score(threshold=threshold).to(self.device)
        pr_auc_fn = BinaryAveragePrecision().to(self.device)

        best_val_f1 = 0
        best_pr_auc = 0
        patience_counter = 0  # for early stopping
        min_epochs = 10       # don't allow early model saving

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            train_preds, train_targets, train_probs = [], [], []

            acc_fn.reset()
            prec_fn.reset()
            rec_fn.reset()
            f1_fn.reset()
            pr_auc_fn.reset()

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

            if epoch >= min_epochs and ((val_f1 > best_val_f1) or (val_pr_auc > best_pr_auc)):
                best_val_f1 = max(val_f1, best_val_f1)
                best_pr_auc = max(val_pr_auc, best_pr_auc)
                patience_counter = 0
                torch.save(self.model.state_dict(), f"best_model_epoch{epoch+1}.pt")
                print("✅ New best model saved.")
            elif epoch >= min_epochs:
                patience_counter += 1
                print(f"⚠️ No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("🛑 Early stopping triggered.")
                    break

            torch.cuda.empty_cache()

    def initialize_explainer(self, epochs: int=200) -> None:
        """
        Run after model training is complete to create an explainer for
        GNN interpretability
        """
        self.explainer = Explainer(
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

    def explain(self, edge_index: int) -> "Explanation":
        """
        Provide an `Explanation` for the given `edge_index`
        """
        return self.explainer(
            x=self.x,
            edge_index=self.edge_index,
            edge_attr=self.data.edge_attr[:, 1:],  # skips edge_id
            target=self.y,
            index=edge_index,
        )
