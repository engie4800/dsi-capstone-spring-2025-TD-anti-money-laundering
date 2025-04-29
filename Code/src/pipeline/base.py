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
import logging

from helpers.currency import get_usd_conversion
from model.features import (
    add_currency_exchange,
    add_day_of_week,
    add_hour_of_day,
    add_is_weekend,
    add_seconds_since_midnight,
    add_sent_amount_usd,
    add_time_diff_from,
    add_time_diff_to,
    add_timestamp_int,
    add_turnaround_time,
    add_unique_identifiers,
    cyclically_encode_feature
)
from pipeline.checks import Checker


class BaseModelPipeline:

    def __init__(self, dataset_path: str):
        """
        Initialize pipeline with dataset
        """
        self.dataset_path = dataset_path
        self.df = pd.read_csv(self.dataset_path)
        self.nodes = pd.DataFrame()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Reconstruct dataset directory.
        # Assumes unix-like file paths.
        self.dataset_dir = "/".join(self.dataset_path.split("/")[0:-1])

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
        
        # New tracking dictionaries and lists
        self.edge_features = []
        self.node_features = []
        self.scaled_edge_features = []
        self.scaled_node_features = []
        self.onehot_features = []
        self.label_encoded_features = []
        self.engineered_features = []
        self.split_type = None
        self.train_size = None
        self.val_size = None
        self.test_size = None
        self.class_balance_stats = {}
        self.num_nodes = {}

    #----------------------
    # Data summary
    #----------------------
    
    def df_summary(self):
        logging.info("DATA HEAD")
        display(self.df.head())
        logging.info("\nFEATURE TYPE")
        display(self.df.info())

    def y_statistics(self):
        logging.info("Normalized Value Count: ")
        logging.info(self.df["is_laundering"].value_counts(normalize=True))
        
    #---------------------
    # Data preprocessing
    #---------------------

    def rename_columns(self) -> None:
        """
        Renames the columns of `self.df` to follow a consistent
        semantics, in terms of using to and from, or sent and received,
        and to use the Pythonic snake case
        """
        self.column_mapping = {
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

        self.ibm_column_header = list(self.column_mapping.keys())

        # Ensure required columns exist
        missing_columns = [
            col
            for col in self.column_mapping.keys()
            if col not in self.df.columns
        ]
        if missing_columns:
            raise KeyError(
                f"Missing expected columns in dataset: {missing_columns}"
            )

        self.df.rename(columns=self.column_mapping, inplace=True)
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

        """
        logging.info("Extracting currency features...")
        Checker.currency_columns_required(self)

        self.df = add_currency_exchange(self.df)
        usd_conversion = get_usd_conversion(self.dataset_path)
        self.df = add_sent_amount_usd(self.df, usd_conversion)

        self.engineered_features += ['sent_amount_usd','log_currency_exchange']

        self.preprocessed["currency_features_extracted"] = True

    def extract_time_features(self, keep_timestamp: bool=False) -> None:
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

        # Dropping timestamp ensures that the complex timestamp string
        # itself isn't used as a feature, as it is poorly suited to be
        # one due to its monotonicity and high cardinality
        if not keep_timestamp:
            self.df.drop(columns=["timestamp"], inplace= True)
        
        self.preprocessed["time_features_extracted"] = True

    def create_unique_ids(self, keep_intermediate_fields: bool=False) -> None:
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
        self.df = add_unique_identifiers(self.df, keep_intermediate_fields)
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

        self.df = add_time_diff_from(self.df)
        self.df = add_time_diff_to(self.df)
        self.df = add_turnaround_time(self.df)

        self.engineered_features += ['time_diff_from','time_diff_to','turnaround_time']
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
        self.df.drop(columns=["day_of_week", "hour_of_day", "seconds_since_midnight"], inplace=True)

        self.engineered_features += ['time_of_day_sin','time_of_day_cos','day_sin','day_cos']
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
        self.onehot_features = categorical_features
        self.preprocessed["onehot_encoded"] = True
        
    # TODO: need? 
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
    
    def numerical_scaling(self, numerical_features:list[str]):
        """Standardize Numerical Features"""

        self.scaled_edge_features = numerical_features
        
        std_scaler = StandardScaler()

        self.X_train[numerical_features] = std_scaler.fit_transform(self.X_train[numerical_features])
        self.X_test[numerical_features] = std_scaler.transform(self.X_test[numerical_features])
        self.X_val[numerical_features] = std_scaler.transform(self.X_val[numerical_features])

    # TODO: need? 
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

    # TODO: need? 
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
    
    def run_preprocessing(self, graph_feats: bool=False) -> None:
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
            self.create_unique_ids()
            self.extract_additional_time_features()
            self.cyclical_encoding()
            # TODO: do we need label encoding any more? I thought we decided 
            # not to include to/from bank as feature bc of high dimensionality
            # self.apply_label_encoding()
            self.apply_one_hot_encoding()
            # TODO: Do we need this? or no because done in node split process?
            # if graph_feats:
            #     self.extract_graph_features()
            logging.info("Preprocessing completed successfully!")
            logging.info(self.preprocessed)

        except Exception as e:
            logging.info(f"Error in preprocessing: {e}")

    #-----------------
    # Hooks
    #-----------------
    
    def should_keep_acct_idx(self) -> bool:
        return False

    #-----------------
    # Splitting
    #-----------------

    def get_split_indices(self):
        """
        Stores numpy arrays of appropriate indices based on validation
        and test sizes
        """

        num_edges = len(self.df)

        self.t1 = int(num_edges * (1 - self.val_size - self.test_size))
        self.t2 = int(num_edges * (1 - self.test_size))
        self.train_indices = np.arange(0, self.t1)
        self.val_indices = np.arange(self.t1, self.t2)
        self.test_indices = np.arange(self.t2, num_edges)
        self.train_val_indices = np.concatenate([self.train_indices, self.val_indices])
        self.train_val_test_indices = np.concatenate([self.train_val_indices, self.test_indices])
    
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
        split_type = split_type or "random_stratified"

        if split_type not in valid_splits:
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

        # Allow `split_train_test_val` to default to using all columns for the set of `X_cols`
        cols = X_cols or self.df.columns
         
        # Remove identifying fields, as well as the output `is_laundering`
        exclude_cols = {
            "from_bank",
            "to_bank",
            "from_account",
            "to_account",
            "from_account_idx",
            "to_account_idx",
            "from_account_id",
            "to_account_id",
            "is_laundering",
        }
        self.X_cols = list(set(cols) - exclude_cols)

        if self.should_keep_acct_idx():
            self.X_cols = list(set(self.X_cols) | {"from_account_idx", "to_account_idx"})
            print("Keeping from_account_idx and to_account_idx (for merging node feats onto tabular data for Catboost)")
        else: 
            self.X_cols = list(set(self.X_cols) - {"from_account_idx", "to_account_idx"})
        
        logging.info(f"Using the following set of 'X_cols'\n{self.X_cols}")

        if self.split_type == "random_stratified":
            X = self.df[self.X_cols]
            y = self.df[y_col]

            self.X_train, X_temp, self.y_train, y_temp = train_test_split(
                X, y, test_size=(test_size + val_size), random_state=42, stratify=y
            )
            self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
                X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=42, stratify=y_temp
            )

        else: 
            # Temporal splits ("temporal" and "temporal_agg")
            if "edge_id" not in self.df.columns:
                raise RuntimeError("Must include edge_id for temporal splits")
            
            # Sort by time (edge_id reflects first stored time sort)
            self.df = self.df.sort_values(by='edge_id').reset_index(drop=True) # precautionary sort
            X = self.df[self.X_cols]
            y = self.df[y_col]
            
            self.get_split_indices() 

            # Split databased on timestamp
            if self.split_type == "temporal":
                self.X_train, self.y_train = X.iloc[self.train_indices], y.iloc[self.train_indices]
                self.X_val, self.y_val = X.iloc[self.val_indices], y.iloc[self.val_indices]
                self.X_test, self.y_test = X.iloc[self.test_indices], y.iloc[self.test_indices]
            
            elif self.split_type == "temporal_agg":
                # Temporal aggregated split (keeps earlier data but masks during GNN loss computation)
                self.X_train, self.y_train = X.iloc[self.train_indices], y.iloc[self.train_indices]
                self.X_val, self.y_val = X.iloc[self.train_val_indices], y.iloc[self.train_val_indices]
                self.X_test, self.y_test = X, y # whole df

        logging.info(f"Data split using {self.split_type} method.")
        
        if self.split_type == "temporal_agg":
            logging.info("Remember to mask labels in GNN evaluation.\n"
                " - Train: no mask \n"
                " - Val: mask y_lab[:t1] (only evaluate labels y_lab[t1:t2]) \n"
                " - Test: mask y_lab[:t2] (only evaluate labels y_lab[t2:])")

        self.class_balance_stats = {
            "train": self.y_train.value_counts(normalize=True).to_dict(),
            "val": self.y_val.value_counts(normalize=True).to_dict(),
            "test": self.y_test.value_counts(normalize=True).to_dict()
        }
        self.train_size = len(self.X_train)
        self.val_size = len(self.X_val)
        self.test_size = len(self.X_test)
        self.preprocessed["train_test_val_data_split"] = True
    
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
        self.num_nodes = {
            "train": len(self.train_nodes),
            "val": len(self.val_nodes),
            "test": len(self.test_nodes)
        }

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
        
        self.scaled_node_features += cols_to_scale
        self.preprocessed["node_datasets_scaled"] = True
        
    def compute_placeholder_node_features(self) -> None:
        """
        Create placeholder node features when no graph-based node features desired.
        """
        logging.info("Creating placeholder node features...")
        max_node = self.df[["from_account_idx", "to_account_idx"]].max().max()
        placeholder_df = pd.DataFrame({
            "node_id": np.arange(max_node + 1),
            "placeholder": 1
        })
        self.train_nodes = placeholder_df.copy()
        self.val_nodes = placeholder_df.copy()
        self.test_nodes = placeholder_df.copy()
        self.num_nodes = {
            "train": len(self.train_nodes),
            "val": len(self.val_nodes),
            "test": len(self.test_nodes)
        }
        self.preprocessed["post_split_node_features"] = True
  
    def pipeline_summary(self):
        print("📋 Pipeline Summary:")
        print(f"- Dataset Path: {self.dataset_path}")
        print(f"- Split Type: {self.split_type}")
        print(f"- Sizes: Train={self.train_size}, Val={self.val_size}, Test={self.test_size}")
        print(f"- Laundering Proportion:")
        for split, stats in self.class_balance_stats.items():
            print(f"  {split}: {stats}")
        print(f"- Edge Features: {self.edge_features}")
        print(f"- Node Features: {self.node_features}")
        print(f"- Scaled Edge Features: {self.scaled_edge_features}")
        print(f"- Scaled Node Features: {self.scaled_node_features}")
        print(f"- One-hot Encoded Features: {self.onehot_features}")
        print(f"- Label Encoded Features: {self.label_encoded_features}")
        print(f"- Engineered Features: {self.engineered_features}")
        print()
