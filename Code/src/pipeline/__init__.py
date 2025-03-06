import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import xgboost as xgb
from IPython.display import display
from rich import print
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from helpers.currency import get_usd_conversion


class ModelPipeline:

    def __init__(self, dataset_path: str):
        """
        Initialize pipeline with dataset
        """
        self.dataset_path = dataset_path
        self.df = pd.read_csv(self.dataset_path)

    def df_summary(self):
        print("DATA HEAD")
        display(self.df.head())
        print("\nFEATURE TYPE")
        display(self.df.info())

    def y_statistics(self):
        print("Normalized Value Count: ")
        print(self.df["Is Laundering"].value_counts(normalize=True))

    def drop_duplicates(self):
        self.df["from_account_id"] = self.df["From Bank"].astype(str) + "_" + self.df["Account"].astype(str)
        self.df["to_account_id"] = self.df["To Bank"].astype(str) + "_" + self.df["Account.1"].astype(str)

        df = df.reset_index(drop=True)
        from_nodes = df["from_account_id"].drop_duplicates().reset_index(drop=True)
        to_nodes = df["to_account_id"].drop_duplicates().reset_index(drop=True)
        all_nodes = pd.concat([from_nodes, to_nodes]).drop_duplicates().reset_index(drop=True)

    def currency_normalization(self):
        usd_conversion = get_usd_conversion(self.dataset_path)
        self.df["Amount Paid (USD)"] = self.df.apply(
            lambda row: row["Amount Paid"] * usd_conversion.get(row["Payment Currency"], 1),
            axis=1,
        )
        self.df["Amount Received (USD)"] = self.df.apply(
            lambda row: row["Amount Received"] * usd_conversion.get(row["Receiving Currency"], 1),
            axis=1,
        )

    def date_to_unix(self):
        self.df["Timestamp"] = pd.to_datetime(self.df["Timestamp"]).astype(int) / 10**9 

    def add_date_features(self, date_column="Timestamp"):
        """Extract date features"""
        
        self.df["Timestamp"] = pd.to_datetime(self.df["Timestamp"])

        self.df["time_of_day"] = self.df["Timestamp"].dt.time
        self.df["hour_of_day"] = self.df["Timestamp"].dt.hour
        self.df["day_of_week"] = self.df["Timestamp"].dt.weekday # 0=Monday,...,6=Sunday
        self.df["seconds_since_midnight"] = (
            self.df["Timestamp"].dt.hour * 3600 +  # Convert hours to seconds
            self.df["Timestamp"].dt.minute * 60 +  # Convert minutes to seconds
            self.df["Timestamp"].dt.second         # Keep seconds
        )

        # Transform timestamp to raw int unix
        self.df["timestamp_int"] = self.df["Timestamp"].astype(int) / 10**9

        # Just a temp assignment, will be scaled later on
        self.df["timestamp_scaled"] = self.df["Timestamp"].astype(int) / 10**9
        
        # Apply cyclical encoding
        self.df["day_sin"] = np.sin(2 * np.pi * self.df["day_of_week"] / 7)
        self.df["day_cos"] = np.cos(2 * np.pi * self.df["day_of_week"] / 7)
        self.df["time_of_day_sin"] = np.sin(2 * np.pi * self.df["seconds_since_midnight"] / 86400)
        self.df["time_of_day_cos"] = np.cos(2 * np.pi * self.df["seconds_since_midnight"] / 86400)

        # Create binary weekend indicator
        self.df["is_weekend"] = self.df["day_of_week"].isin([5, 6]).astype(int)
        
        self.df.drop(columns=["Timestamp"], inplace= True)

    def label_encoding(self, features_to_encode):
        return self.apply_label_encoding(features_to_encode)
        
    def apply_label_encoding(self, categorical_features):
        """Label encode categorical columns"""

        for col in categorical_features:
            self.df[col] = LabelEncoder().fit_transform(self.df[col])

    def neighbor_context(self):
        G = nx.DiGraph()

        for _, row in self.df.iterrows():
            G.add_edge(row["Account"], row["Account.1"], weight=row["Amount Paid (USD)"])

        self.df["degree_centrality"] = self.df["Account"].map(nx.degree_centrality(G))
        self.df["pagerank"] = self.df["Account"].map(nx.pagerank(G))
    
    def split_x_y(self, X_cols, y_col):
        self.X = self.df[X_cols]
        self.y = self.df[y_col]
    
    def numerical_scaling(self, numerical_features):
        """Standardize Numerical Features"""

        std_scaler = StandardScaler()

        self.X_train[numerical_features] = std_scaler.fit_transform(self.X_train[numerical_features])
        self.X_test[numerical_features] = std_scaler.transform(self.X_test[numerical_features])
        self.X_val[numerical_features] = std_scaler.transform(self.X_val[numerical_features])

        return self.X_train, self.X_test, self.X_val

    def extract_graph_features(self, weight_col):
        """Generate graph-based neighborhood context features"""

        G = nx.DiGraph()
        for _, row in self.df.iterrows():
            G.add_edge(row["Account"], row["Account.1"], weight=row[weight_col])

        # Add centrality and pagerank as features
        self.df["degree_centrality"] = self.df["Account"].map(nx.degree_centrality(G))
        self.df["pagerank"] = self.df["Account"].map(nx.pagerank(G))

    def generate_tensor(self,edge_features):
        self.train_node_features = torch.tensor(self.X_train[edge_features].values, dtype=torch.float)
        labels = torch.tensor(self.y_train.values, dtype=torch.long)
        edge_index = torch.tensor(self.X_train[["Account", "Account.1"]].values.T, dtype=torch.long)
        self.train_data = Data(x=self.train_node_features, edge_index=edge_index, y=labels)

        self.test_node_features = torch.tensor(self.X_test[edge_features].values, dtype=torch.float)
        labels = torch.tensor(self.y_test.values, dtype=torch.long)
        edge_index = torch.tensor(self.X_test[["Account", "Account.1"]].values.T, dtype=torch.long)
        self.test_data = Data(x=self.test_node_features, edge_index=edge_index, y=labels)

    def generate_tensors(self, edge_features, edges = ["Account", "Account.1"]):
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

    def split_train_test(self, test_size: float) -> None:
        """
        Use the model pipeline `random_state` and the given `test_size`
        to create a train, test split of the data and attach it to the
        model pipeline
        """
        test_size = float(test_size)
        if test_size < 0 or test_size > 1:
            raise ValueError(
                "Input 'test_size' represents a percentage as a 'float' "
                f"between 0 and 1, inclusive. Invalid input: {test_size}"
            )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=self.y,
        )

    def split_train_test_v2(self, X_cols, y_col, test_size=0.2):
        """Perform Train-Test Split"""

        X = self.df[X_cols]
        y = self.df[y_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def random_forest_classifier(self, param):      
        self.model = RandomForestClassifier(**param)
        self.model.fit(self.X_train, self.y_train)

    def xgboost_classifier(self,param):
        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self.dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        self.model = xgb.train(param, self.dtrain)

    def training_gnn_model(self, learning_rate, epoch_, gnn_model):
            
        self.model = globals()[gnn_model](input_dim=self.train_node_features.shape[1], hidden_dim=16, output_dim=2)

        # Define optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.train_data = self.train_data.to(device)

        # Training loop
        epochs = epoch_
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(self.train_data.x, self.train_data.edge_index)
            loss = criterion(out, self.train_data.y)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

    def predict_model_gnn(self):
        self.model.eval()
        with torch.no_grad():
            out_probs = self.model(self.test_data.x, self.test_data.edge_index)
            self.y_proba = out_probs.cpu().numpy()
            self.y_pred = out_probs.argmax(dim=1).cpu().numpy()

    def predict_model(self, xgboost_flag = "null"):
        if xgboost_flag == "null":
            self.y_pred = self.model.predict(self.X_test)
            self.y_proba = self.model.predict_proba(self.X_test)
        else:
            self.y_proba = self.model.predict(self.dtest)
            self.y_pred = (self.y_proba > 0.5).astype(int)

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

    def result_metrics_train_test_split(self, y_test, y_pred, y_proba=None, class_labels=None):
        """
        Compute and display model performance metrics.

        Parameters:
        - y_test: True labels
        - y_pred: Predicted labels
        - y_proba: Predicted probabilities (for AUC-ROC & PR curve)
        - class_labels: List of class names (default: inferred from y_test)

        Returns:
        - metrics_dict: Dictionary containing key evaluation metrics
        """

        print("Classification Report:")
        print(classification_report(y_test, y_pred, digits=4))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        accuracy = balanced_accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        logloss = log_loss(y_test, y_proba) if y_proba is not None else None

        print(f"Balanced Accuracy: {accuracy:.4f}")
        print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
        if logloss:
            print(f"Log Loss: {logloss:.4f}")

        # If class labels are not provided, generate default ones
        if class_labels is None:
            class_labels = [f"Class {i}" for i in range(len(set(y_test)))]

        # If probability predictions exist, compute ROC & PR AUC
        if y_proba is not None:
            # Binary classification
            if len(y_proba.shape) == 1:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = roc_auc_score(y_test, y_proba)
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                pr_auc = auc(recall, precision)
            else:  # Multi-class case
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
                pr_auc = auc(recall, precision)


            print(f"AUC-ROC Score: {roc_auc:.4f}")
            print(f"Precision-Recall AUC: {pr_auc:.4f}")

        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Confusion Matrix Plot
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
        axes[0].set_title("Confusion Matrix")
        axes[0].set_xlabel("Predicted Label")
        axes[0].set_ylabel("True Label")
        axes[0].set_xticklabels(class_labels)
        axes[0].set_yticklabels(class_labels)

        # ROC Curve Plot
        if y_proba is not None:
            axes[1].plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
            axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")  # Baseline
            axes[1].set_title("ROC Curve")
            axes[1].set_xlabel("False Positive Rate")
            axes[1].set_ylabel("True Positive Rate")
            axes[1].legend()

            # Precision-Recall Curve
            axes[2].plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
            axes[2].set_title("Precision-Recall Curve")
            axes[2].set_xlabel("Recall")
            axes[2].set_ylabel("Precision")
            axes[2].legend()

        plt.tight_layout()
        plt.show()

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
