import datetime
import logging

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import f1_score, average_precision_score
import xgboost as xgb
import shap

from helpers.currency import get_usd_conversion
from model import GINe
from model.features import (
    add_currency_exchange,
    add_day_of_week,
    add_hour_of_day,
    add_is_weekend,
    add_received_amount_usd,
    add_seconds_since_midnight,
    add_sent_amount_usd,
    add_time_diff_from,
    add_time_diff_to,
    add_timestamp_int,
    add_timestamp_scaled,
    add_turnaround_time,
    add_unique_identifiers,
    cyclically_encode_feature
)
from pipeline.checks import Checker

# =============================================================================
# Overview of ModelPipeline Functions
# -----------------------------------------------------------------------------
# This class orchestrates data preprocessing, feature extraction, GNN tensor
# generation, and preparation of data loaders for training.
# =============================================================================

# INITIAL PREPROCESSING -------------------------------------------------------
# rename_columns:         Rename original columns to Pythonic naming.
# drop_duplicates:        Remove exact duplicate transactions.
# check_for_null:         Validate that no nulls exist in the dataset.

# FEATURE ENGINEERING: CURRENCY + TIME ---------------------------------------
# extract_currency_features:      Compute exchange rates and USD amounts.
# extract_time_features:          Add time-related features (hour, weekday, etc).
# cyclical_encoding:              Apply sine/cosine transforms to time features.
# apply_one_hot_encoding:         One-hot encode categorical variables.
# apply_label_encoding:           Label encode bank-related fields.
# extract_additional_time_features:
#                                Add time_diff_from, time_diff_to, turnaround_time.

# UNIQUE IDENTIFIERS ---------------------------------------------------------
# create_unique_ids:              Generate unique node and edge IDs.

# NODE FEATURE EXTRACTION ----------------------------------------------------
# extract_nodes:                  Initializes node feature matrix.
# add_graph_related_features:     Add graph centrality and PageRank metrics.
# add_node_features:              Aggregate transaction-level stats to nodes.
# compute_split_specific_node_features:
#                                Computes node features separately for train/val/test.

# SPLITTING DATA -------------------------------------------------------------
# split_train_test_val:           Splits edge data into train/val/test sets.
# get_split_indices:              Gets edge indices per split.

# GRAPH CONSTRUCTION (PYG) ---------------------------------------------------
# split_train_test_val_graph:     Structural setup of node & edge data as PyG Data.
# scale_node_data_frames:         Scales node-level features using StandardScaler.
# scale_edge_features:            Imputes/scales edge features (e.g. sent_amount_usd).
# generate_tensors:               (Alternative) Create PyG datasets manually.
# get_data_loaders:               Wraps final data splits into LinkNeighborLoader.

# RUNNING THE PIPELINE -------------------------------------------------------
# run_preprocessing:              Orchestrates entire preprocessing pipeline.


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
        Renames the columns of self.df to follow a consistent
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

        # Ensure required columns exist
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

            currency_exchange: exchange rate from sent to received
            add_sent_amount_usd: Sent amount in USD
            add_received_amount_usd: Received amount in USD

        """
        logging.info("Extracting currency features...")
        Checker.currency_columns_required(self)

        self.df = add_currency_exchange(self.df)
        usd_conversion = get_usd_conversion(self.dataset_path)
        self.df = add_sent_amount_usd(self.df, usd_conversion)
        # TODO: do we need received amount USD? Should be same as sent_amount_USD...
        # self.df = add_received_amount_usd(self.df, usd_conversion)

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

        # Ensures timestamp is a datetime object
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
        self.df.drop(columns=["timestamp"], inplace=True)

        self.preprocessed["time_features_extracted"] = True

    def create_unique_ids(self) -> None:
        """Create a mapping from bank, account pairs, each of which
        should be unique, to a unique identifier. This adds the
        following data to each transaction:

            edge_id: Integer transaction (edge) identifier that
                represents the temporal ordering of the transaction
                dataset, e.g. 1 is the first transaction to appear,
                N is the last
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

        # TODO: does it make sense to add an analogous time_diff_to
        # representing the time since the receiver in a transaction
        # previously received money? Sure
        self.df = add_time_diff_from(self.df)
        self.df = add_time_diff_to(self.df)
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

        # Drop after encoding
        self.df.drop(columns=["day_of_week", "seconds_since_midnight"], inplace=True)

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
            unique_values = pd.concat([self.df[col].drop_duplicates() for col in cols]).drop_duplicates().reset_index(
                drop=True)
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
            "using only sent_amount as edge weight (only based on "
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
                    temp_from = self.df[["from_account_idx", features[0]]].rename(
                        columns={"from_account_idx": "node_id"})
                    temp_to = self.df[["to_account_idx", features[0]]].rename(columns={"to_account_idx": "node_id"})
                else:
                    temp_from = self.df[["from_account_idx", features[0]]].rename(
                        columns={"from_account_idx": "node_id", features[0]: features[2]})
                    temp_to = self.df[["to_account_idx", features[1]]].rename(
                        columns={"to_account_idx": "node_id", features[1]: features[2]})

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

    # TODO: do we need this anymore? Further, how can we ensure other method also allows for no node feats?
    def extract_nodes(self, node_features=None, graph_related_features=None):
        """Extract nodes (x) data that is used across splits"""

        # Ensure that unique_ids have been generated
        if not self.preprocessed["unique_ids_created"]:
            raise RuntimeError(
                "Unique account IDs must be created before computing network features"
            )

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

    # TODO: Do we need this anymore?
    def generate_tensors(self, edge_features, node_features=None, edges=["from_account_idx", "to_account_idx"]):
        """Convert data to PyTorch tensor format for GNNs"""
        logging.info("Generating tensors...")

        def create_pyg_data(X, y, dataset_name):
            edge_index = torch.LongTensor(X[edges].values.T)  # [2, num_edges]
            edge_attr = torch.tensor(X[edge_features].values, dtype=torch.float)  # [num_edges, num_edge_features]
            edge_labels = torch.LongTensor(y.values)  # [num_edges]
            node_attr = torch.tensor(self.nodes.drop(columns="node_id").values,
                                     dtype=torch.float)  # [num_nodes, num_node_features]

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
            # TODO: do we need label encoding any more? I thought we decided
            # not to include to/from bank as feature bc of high dimensionality
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
            keep_account_idx=False
    ):
        """Perform Train-Test-Validation Split

        OPTIONS: ["random_stratified", "temporal", "temporal_agg"]

        "random stratified": Data is randomized and split while keeping
            is_laundering label proportionate bt train/val/test.
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
                         "See split_train_test_val for more details."
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
                "See split_train_test_val for more details."
            )
        self.split_type = split_type

        # Allow split_train_test_val to default to using all columns
        # for the set of X_cols
        if X_cols is None:
            X_cols = sorted(
                list(
                    set(self.df.columns) - set([
                        # Remove identifying fields, as well as the
                        # output is_laundering
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
        if not keep_account_idx:
            self.X_cols = self.X_cols + ['from_account_idx', 'to_account_idx']
            print("Keeping from_account_idx and to_account_idx (for merging node feats onto tabular data for Catboost)")

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
                raise RuntimeError("Need timestamp_int in df for temporal split. Review preprocessing steps.")

            # Sort by time and find indices for data split
            df_sorted = self.df.sort_values(by=["timestamp_int"])
            X = df_sorted[self.X_cols]
            y = df_sorted[y_col]
            t1 = int((1 - (test_size + val_size)) * len(self.df))
            t2 = int((1 - test_size) * len(self.df))

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
            t1 = int((1 - (test_size + val_size)) * len(self.df))
            t2 = int((1 - test_size) * len(self.df))

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
        self.train_val_indices = np.concatenate([self.train_indices, self.val_indices])
        self.train_val_test_indices = np.concatenate([self.train_val_indices, self.test_indices])

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
        Compute node features for a specific split using from_account_idx
        and to_account_idx as node identifiers.
        """
        logging.info("Getting train-test-split-specific node features")
        Checker.data_split_to_train_val_test(self)

        def get_node_features(split_df, split_name: str, graph_features):
            logging.info(f"Computing {split_name} node features...")

            # --- TRANSACTIONAL NODE FEATURES ---

            # Outgoing stats (from_account_idx)
            out_stats = (
                split_df.groupby("from_account_idx")["sent_amount_usd"]
                .agg(["sum", "mean", "std"])
                .add_prefix("out_")
                .reset_index()
                .rename(columns={"from_account_idx": "node_id"})
            )

            # Incoming stats (to_account_idx)
            in_stats = (
                split_df.groupby("to_account_idx")["sent_amount_usd"]
                .agg(["sum", "mean", "std"])
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
            node_stat_features["std_txn_in"] = node_stat_features["in_std"]
            node_stat_features["std_txn_out"] = node_stat_features["out_std"]

            node_stat_features = node_stat_features[[
                "node_id",
                "net_flow",
                "avg_txn_out",
                "avg_txn_in",
                "std_txn_out",
                "std_txn_in",
                "num_unique_out_partners",
                "num_unique_in_partners",
            ]]

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
            pagerank = nx.pagerank(G, weight="sent_amount_usd")

            node_graph_df = pd.DataFrame({"node_id": list(G.nodes)})
            node_graph_df["degree_centrality"] = node_graph_df["node_id"].map(degree_centrality)
            node_graph_df["pagerank"] = node_graph_df["node_id"].map(pagerank)

            # --- COMBINE ALL FEATURES ---

            node_df = pd.merge(node_graph_df, node_stat_features, on="node_id", how="outer")
            node_df.fillna(0, inplace=True)
            node_df = node_df.sort_values("node_id").reset_index(drop=True)

            print(f"✅ Computed node features for {split_name} with {len(node_df)} nodes.")
            return node_df

        # Compute for train, val, test separately
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

        scalers = {}

        for col in cols_to_scale:
            # Impute -1 in train
            train_col, train_median = preprocess_column(self.train_nodes[col])
            self.train_nodes[col] = train_col

            # Fit scaler on train
            scaler = StandardScaler()
            self.train_nodes[col] = scaler.fit_transform(self.train_nodes[col].values.reshape(-1, 1)).flatten()
            scalers[col] = (scaler, train_median)

            # Impute -1 in val/test with train median
            for df in [self.val_nodes, self.test_nodes]:
                if col in df.columns:
                    col_vals = df[col].copy()
                    col_vals[col_vals == -1] = train_median
                    df[col] = scalers[col][0].transform(col_vals.values.reshape(-1, 1)).flatten()

        self.preprocessed["node_datasets_scaled"] = True

        return self.train_nodes, self.val_nodes, self.test_nodes

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

            txns_df.drop(columns=['from_account_idx', 'to_account_idx'])

            return txns_df

        self.X_train = merge_feats(self.X_train, self.train_nodes)
        self.X_val = merge_feats(self.X_val, self.val_nodes)
        self.X_test = merge_feats(self.X_test, self.test_nodes)

        return self.X_train, self.X_val, self.X_test

    def split_train_test_val_graph(self, edge_features=None):
        """Creates graph objects for use in GNN.
        """
        logging.info("Splitting into train, test, validation graphs")
        Checker.train_val_test_node_features_added(self)

        # A default set of edge features that excludes some obvious
        # features we don't want
        if edge_features is None:
            # TODO: any reason not to do this?
            self.edge_features = ['edge_id'] + [col for col in self.X_cols if col != 'edge_id']
        else:
            self.edge_features = edge_features
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

        # Overwrites the values we got from the original split
        self.train_indices = torch.tensor(self.train_indices)
        self.val_indices = torch.tensor(self.val_indices)
        self.test_indices = torch.tensor(self.test_indices)

        cat_tr_val_inds = torch.cat((self.train_indices, self.val_indices))
        self.train_data = Data(
            x=tr_x,
            edge_index=self.edge_index[:, self.train_indices],
            edge_attr=edge_attr[self.train_indices],
            y=self.y[self.train_indices],
        )
        self.val_data = Data(
            x=val_x,
            edge_index=self.edge_index[:, cat_tr_val_inds],
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

        # TODO: again, do we need to return this stuff or nah
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

    def scale_edge_features(self, edge_features_to_scale: list[str]):
        """
        Scale and impute edge features in train/val/test data in place.
        Uses training set to fit scalers, then applies to all sets.
        """
        logging.info("Scaling edge features: %s", edge_features_to_scale)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if edge_features_to_scale is None:
            self.scaled_edge_features = [
                "sent_amount_usd",
                "timestamp_scaled",
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
            scalers[feature] = scaler  # TODO: do we need to return the scalars? why do we have this?

        self.train_data.edge_attr = torch.tensor(train_edge_attr, dtype=torch.float32, device=self.device)
        self.val_data.edge_attr = torch.tensor(val_edge_attr, dtype=torch.float32, device=self.device)
        self.test_data.edge_attr = torch.tensor(test_edge_attr, dtype=torch.float32, device=self.device)

        return scalers

    def get_data_loaders(self, num_neighbors=[100, 100], batch_size=8192):
        logging.info("Getting data loaders")
        Checker.graph_data_split_to_train_val_test(self)

        # TODO: might be able to move this to somewhere more meaningful,
        # but it is needed here at the latest
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # TODO: I add this new one - need verify
    def shap_feature_selection(self, xgb_model, top_n: int = 3):
        """
        Run SHAP on the current train/val/test split, print the top-3 in each group,
        add the “other” indicators to X_train/val/test, and return:
          - shap_importance DataFrame
          - three Python lists: top3_sent, top3_received, top3_payment
        """
        # 1) Get SHAP value df
        booster = xgb_model.get_booster()
        dtest = xgb.DMatrix(self.X_test)
        shap_vals = booster.predict(dtest, pred_contribs=True)
        feature_shap = pd.DataFrame(shap_vals[:, :-1],
                                    columns= self.X_test.columns,
                                    index= self.X_test.index)

        # 2) Mean absolute SHAP per feature
        mean_abs = feature_shap.abs().mean().sort_values(ascending=False)
        shap_importance = (
            pd.DataFrame({
                'feature': self.X_test.columns,
                'mean_abs_shap': mean_abs
            })
            .sort_values('mean_abs_shap', ascending=False)
            .reset_index(drop=True)
        )

        # 3) Build full matrix X and define your groups
        X = pd.concat([self.X_train, self.X_val, self.X_test],
                      axis=0, ignore_index=True)
        currency_features_sent = [c for c in X.columns if c.startswith("sent_currency_")]
        currency_features_received = [c for c in X.columns if c.startswith("received_currency_")]
        payment_type_features = [c for c in X.columns if c.startswith("payment_type_")]

        # 4) Pull out top-n in each
        top3_sent = shap_importance[shap_importance['feature'].isin(currency_features_sent)].nlargest(top_n,
                                                                                                      'mean_abs_shap')
        top3_received = shap_importance[shap_importance['feature'].isin(currency_features_received)].nlargest(top_n,
                                                                                                              'mean_abs_shap')
        top3_payment = shap_importance[shap_importance['feature'].isin(payment_type_features)].nlargest(top_n,
                                                                                                        'mean_abs_shap')
        # 5) Print them
        print("▶ Top 3 ‘currencies sent’ by |SHAP|:")
        print(top3_sent.to_string(index=False))
        print("\n▶ Top 3 ‘currencies received’ by |SHAP|:")
        print(top3_received.to_string(index=False))
        print("\n▶ Top 3 ‘payment types’ by |SHAP|:")
        print(top3_payment.to_string(index=False))

        # 6) Add “other” indicators to each split
        #    (sum of all in group minus sum of top3 > 0)
        sent_cols, recv_cols, pay_cols = currency_features_sent, currency_features_received, payment_type_features
        t_sent, t_recv, t_pay = top3_sent['feature'], top3_received['feature'], top3_payment['feature']

        for df in (self.X_train, self.X_val, self.X_test, X):
            df['sent_currency_other'] = ((df[sent_cols].sum(axis=1) - df[t_sent].sum(axis=1)) > 0).astype(int)
            df['received_currency_other'] = ((df[recv_cols].sum(axis=1) - df[t_recv].sum(axis=1)) > 0).astype(int)
            df['payment_type_other'] = ((df[pay_cols].sum(axis=1) - df[t_pay].sum(axis=1)) > 0).astype(int)

        # 7) Convert the top3 DataFrames into plain Python lists
        top3_sent_list = top3_sent['feature'].tolist()
        top3_received_list = top3_received['feature'].tolist()
        top3_payment_list = top3_payment['feature'].tolist()

        return shap_importance, top3_sent_list, top3_received_list, top3_payment_list

    def evaluate_feature_sets_xgb(self, feature_sets: dict):
        """
        Given a dict name→feature_list, trains an XGBClassifier
        on each and prints F1 & PR-AUC. Returns a results dict.
        """
        results = {}
        for name, feats in feature_sets.items():
            model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=10,
                learning_rate=0.1,
                objective='binary:logistic',
                scale_pos_weight=12,
                eval_metric='aucpr',
                random_state=42,
            )
            # fit on train
            model.fit(self.X_train[feats], self.y_train)

            # predict on val
            Xv = self.X_val[feats].astype(np.float32).values
            y_pred = model.predict(Xv)
            y_prob = model.predict_proba(Xv)[:, 1]

            # metrics
            f1 = f1_score(self.y_val, y_pred)
            pr_auc = average_precision_score(self.y_val, y_prob)

            results[name] = {"F1": f1, "PR AUC": pr_auc}
            print(f"{name:12s} →  F1 = {f1:.4f},  PR-AUC = {pr_auc:.4f}")

        return results

    def build_grouped_feature_sets(self,
                                   top3_sent: list[str],
                                   top3_received: list[str],
                                   top3_payment: list[str]):
        """
        1) Identifies your one-hot groups from the full feature set
        2) Adds the “other” indicator columns to self.X_train/val/test
        3) Returns the feature_sets dict exactly as in your snippet
        """
        # -- rebuild full X to scan for all one-hot cols --
        X = pd.concat([self.X_train, self.X_val, self.X_test],
                      axis=0, ignore_index=True)

        # 2) Define your groups of interest
        currency_features_sent     = [c for c in X.columns if c.startswith("sent_currency_")]
        currency_features_received = [c for c in X.columns if c.startswith("received_currency_")]
        payment_type_features      = [c for c in X.columns if c.startswith("payment_type_")]

        # 3) Add “other” flags to each split
        for df in (self.X_train, self.X_val, self.X_test):
            df['sent_currency_other']     = ((df[currency_features_sent].sum(axis=1)
                                              - df[top3_sent].sum(axis=1)) > 0).astype(int)
            df['received_currency_other'] = ((df[currency_features_received].sum(axis=1)
                                              - df[top3_received].sum(axis=1)) > 0).astype(int)
            df['payment_type_other']      = ((df[payment_type_features].sum(axis=1)
                                              - df[top3_payment].sum(axis=1)) > 0).astype(int)

        # 4) Build base_features by removing all one-hot cols
        all_cols = self.X_train.columns.tolist()
        grouped_onehot = set(currency_features_sent
                             + currency_features_received
                             + payment_type_features)
        base_features = [c for c in all_cols if c not in grouped_onehot]

        # 5) Assemble the dictionary
        feature_sets = {
            "all_onehots":        base_features
                                  + currency_features_sent
                                  + currency_features_received
                                  + payment_type_features,
            "no_sent":            base_features
                                  + currency_features_received
                                  + payment_type_features,
            "no_received":        base_features
                                  + currency_features_sent
                                  + payment_type_features,
            "no_currency":        base_features
                                  + payment_type_features,
            "top3_all":           base_features
                                  + top3_sent
                                  + top3_received
                                  + top3_payment,
            "top3_payment_only":  base_features
                                  + currency_features_sent
                                  + currency_features_received
                                  + top3_payment,
        }

        return feature_sets

    def backward_selection_xgb(self,
                               features: list[str],
                               min_features: int = 10) -> list[str]:
        """
        Iteratively drop one feature at a time from features, retrain with early stopping,
        and keep the drop if PR‐AUC does not decrease, until len(features)==min_features.
        Returns the pruned feature list.
        """
        current = features.copy()
        best_score = 0.0
        best_feats = current.copy()

        while len(current) > min_features:
            scores = {}
            for f in current:
                cand = [c for c in current if c != f]
                model = xgb.XGBClassifier(
                    n_estimators=500,
                    max_depth=4,
                    learning_rate=0.03,
                    objective='binary:logistic',
                    scale_pos_weight=12,
                    subsample=0.6,
                    colsample_bytree=0.6,
                    reg_alpha=5.0,
                    reg_lambda=5.0,
                    random_state=42,
                    eval_metric='aucpr',
                    early_stopping_rounds=50
                )
                model.fit(
                    self.X_train[cand], self.y_train,
                    eval_set=[(self.X_val[cand], self.y_val)],
                    verbose=False
                )
                y_prob = model.predict_proba(self.X_val[cand])[:, 1]
                scores[f] = average_precision_score(self.y_val, y_prob)

            drop, score = max(scores.items(), key=lambda kv: kv[1])
            print(f"Trying drop {drop}: PR‐AUC = {score:.4f}")
            if score >= best_score:
                best_score = score
                current.remove(drop)
                best_feats = current.copy()
                print(f"✅ Removed {drop}; new best PR‐AUC = {best_score:.4f}")
            else:
                print(f"🛑 Removal of {drop} hurt performance. Stopping.")
                break

        return best_feats

    def feature_selection_pipeline(self,
                                   xgb_model,
                                   top_n: int = 3,
                                   min_features: int = 10):
        """
        1) SHAP → top_n per group → add “other” flags
        2) build grouped feature‐sets & evaluate F1/PR‐AUC
        3) pick best set → backward‐select down to min_features
        4) train final XGB on that reduced set → return everything
        """


        # Stage 1: SHAP selection
        shap_imp, top3_sent, top3_recv, top3_pay = self.shap_feature_selection(xgb_model, top_n)

        # Stage 1b: build your grouped one‐hot sets
        fs = self.build_grouped_feature_sets(top3_sent, top3_recv, top3_pay)

        # Stage 1c: evaluate all of them
        evals = self.evaluate_feature_sets_xgb(fs)

        # Stage 2: pick the best initial set by PR‐AUC
        best_name = max(evals, key=lambda k: evals[k]['PR AUC'])
        initial_feats = fs[best_name]

        # Stage 2b: backward‐selection
        selected = self.backward_selection_xgb(
            initial_feats,
            min_features=min_features
        )

        # Stage 3: fit the final model
        final_model = xgb.XGBClassifier(
            n_estimators=500,  # allow more trees, we'll stop early
            max_depth=4,  # shallower trees → less variance
            learning_rate=0.03,  # smaller step size
            objective='binary:logistic',
            scale_pos_weight=12,  # keep your class‐imbalance weight
            subsample=0.6,  # each tree sees 80% of the rows
            colsample_bytree=0.6,  # each tree sees 80% of the columns
            reg_alpha=5.0,  # L1 regularization
            reg_lambda=5.0,  # L2 regularization
            random_state=42,
            eval_metric='aucpr',
            early_stopping_rounds=50  # monitor PR-AUC
        )
        final_model.fit(self.X_train[selected],self.y_train,
                        eval_set=[
                            (self.X_train[selected], self.y_train),  # monitor train + val
                            (self.X_val[selected], self.y_val)
                        ])

        # Stage 4: test‐set PR‐AUC
        test_probs = final_model.predict_proba(self.X_test[selected])[:, 1]
        test_pr = average_precision_score(self.y_test, test_probs)

        return {
            'shap_importance': shap_imp,
            'feature_sets': fs,
            'evaluation': evals,
            'best_initial_set': best_name,
            'initial_features': initial_feats,
            'selected_features': selected,
            'final_model': final_model,
            'test_pr_auc': test_pr
        }