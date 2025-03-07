import datetime
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.data import Data

from helpers.currency import get_usd_conversion


class ModelPipeline:

    def __init__(self, dataset_path: str):
        """
        Initialize pipeline with dataset
        """
        self.dataset_path = dataset_path
        self.df = pd.read_csv(self.dataset_path)

        # Track if preprocessing steps have been completed
        self.preprocessed = {
            "renamed": False,
            "duplicates_removed": False,
            "unique_ids_created": False,
            "currency_normalized": False,
            "time_features_extracted": False,
            "cyclical_encoded": False,
            "weekend_encoded": False,
            "features_encoded": False,
            "neighbor_context_computed": False,
            "normalized": False
        }

    def df_summary(self):
        print("DATA HEAD")
        display(self.df.head())
        print("\nFEATURE TYPE")
        display(self.df.info())

    def y_statistics(self):
        print("Normalized Value Count: ")
        print(self.df["is_laundering"].value_counts(normalize=True))

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

    def create_unique_ids(self):
        """Create unique account - ID mapping."""
        if not self.preprocessed["renamed"]:
            raise RuntimeError("Columns must be renamed (run rename()) before creating unique IDs.")

        # Get unique account-bank combos (a couple of acct numbers found at multiple banks)
        self.df['from_account_id'] = self.df['from_bank'].astype(str) + '_' + self.df['from_account'].astype(str)
        self.df['to_account_id'] = self.df['to_bank'].astype(str) + '_' + self.df['to_account'].astype(str)
        self.df.drop(columns=["from_account", "to_account"], inplace=True)

        # Get list of unique account ids
        self.df = self.df.reset_index(drop=True)
        from_nodes = self.df["from_account_id"].drop_duplicates().reset_index(drop=True)
        to_nodes = self.df["to_account_id"].drop_duplicates().reset_index(drop=True)
        all_nodes = pd.concat([from_nodes, to_nodes]).drop_duplicates().reset_index(drop=True)

        # Map node identifiers to integer indices
        node_mapping = {node: idx for idx, node in enumerate(all_nodes)}
        self.df["from_account_idx"] = self.df["from_account_id"].map(node_mapping)
        self.df["to_account_idx"] = self.df["to_account_id"].map(node_mapping)

        self.preprocessed["unique_ids_created"] = True

    def currency_normalization(self):
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

    def extract_time_features(self):
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
        self.df["timestamp_int"] = self.df["timestamp"].astype(int) / 10**9

        # Just a temp assignment, will be scaled later on
        self.df["timestamp_scaled"] = self.df["timestamp"].astype(int) / 10**9

        self.df.drop(columns=["timestamp"], inplace= True)
        
        self.preprocessed["time_features_extracted"] = True
        
    def cyclical_encoding(self):
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
        
    def apply_label_encoding(self, categorical_features):
        """Label encode categorical columns"""
        for col in categorical_features:
            self.df[col] = LabelEncoder().fit_transform(self.df[col])
    
    def numerical_scaling(self, numerical_features):
        """Standardize Numerical Features"""

        std_scaler = StandardScaler()

        self.X_train[numerical_features] = std_scaler.fit_transform(self.X_train[numerical_features])
        self.X_test[numerical_features] = std_scaler.transform(self.X_test[numerical_features])
        self.X_val[numerical_features] = std_scaler.transform(self.X_val[numerical_features])

        return self.X_train, self.X_test, self.X_val

    def extract_graph_features(self, weight_col):
        """Generate graph-based neighborhood context features"""
        if not self.preprocessed["unique_ids_created"]:
            raise RuntimeError(
                "Unique account IDs must be created before computing network features"
            )

        G = nx.DiGraph()
        for _, row in self.df.iterrows():
            G.add_edge(row["from_account_idx"], row["to_account_idx"], weight=row[weight_col])

        # Add centrality and pagerank as features
        self.df["degree_centrality"] = self.df["from_account_idx"].map(nx.degree_centrality(G))
        self.df["pagerank"] = self.df["from_account_idx"].map(nx.pagerank(G))

        self.preprocessed["neighbor_context_computed"] = True

    def generate_tensors(self, edge_features, edges = ["from_account_idx", "to_account_idx"]):
        """Convert data to PyTorch tensor format for GNNs"""

        def create_pyg_data(X, y):
            node_features = torch.tensor(X[edge_features].values, dtype=torch.float)

            labels = torch.tensor(y.values, dtype=torch.long)

            edge_index = torch.tensor(X[edges].values.T, dtype=torch.long)
            
            return Data(x=node_features, edge_index=edge_index, y=labels)

        # Create PyTorch Geometric datasets for train, validation, and test
        self.train_data = create_pyg_data(self.X_train, self.y_train)
        self.val_data = create_pyg_data(self.X_val, self.y_val)
        self.test_data = create_pyg_data(self.X_test, self.y_test)

        return self.train_data, self.val_data, self.test_data
    
    def run_preprocessing(self):
        """Runs all preprocessing steps in the correct order."""
        print("Running preprocessing pipeline...")

        try:
            self.rename_columns()
            self.drop_duplicates()
            self.create_unique_ids()
            self.currency_normalization()
            self.extract_time_features()
            self.cyclical_encoding()
            self.binary_weekend()
            # self.apply_label_encoding() need args
            # self.extract_graph_features() need args
            print("Preprocessing completed successfully!")
            print(self.preprocessed)
        except Exception as e:
            print(f"Error in preprocessing: {e}")

    def split_train_test_val(self, X_cols, y_col, test_size=0.15, val_size=0.15):
        """Perform Train-Test-Validation Split"""

        X = self.df[X_cols]
        y = self.df[y_col]
        
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), random_state=42, stratify=y
        )
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=42, stratify=y_temp
        )
        
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
