import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import xgboost as xgb
from IPython.display import display
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

    def __init__(self, dataset_path: str, random_state: int):

        self.dataset_path = dataset_path
        self.df = pd.read_csv(self.dataset_path)
        self.random_state = random_state

    def df_summary(self):
        print("DATA HEAD")
        display(self.df.head())
        print("FEATURE TYPE")
        display(self.df.info())

    def y_statistics(self):
        print("Normalized Value Count: ")
        print(self.df["Is Laundering"].value_counts(normalize=True))

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

    def label_encoding(self, features_to_encode):
        for col in features_to_encode:
            self.df[col] = LabelEncoder().fit_transform(self.df[col])

    def neighbor_context(self):
        G = nx.DiGraph()

        for _, row in self.df.iterrows():
            G.add_edge(row["Account"], row["Account.1"], weight=row["Amount Paid (USD)"])

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
    
    def split_x_y(self, X_cols, y_col):
        self.X = self.df[X_cols]
        self.y = self.df[y_col]

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

    def result_metrics(self):

        print(classification_report(self.y_test, self.y_pred, digits=4))

        cm = confusion_matrix(self.y_test, self.y_pred)
        accuracy = balanced_accuracy_score(self.y_test, self.y_pred) 
        mcc = matthews_corrcoef(self.y_test, self.y_pred)
        logloss = log_loss(self.y_test, self.y_proba) if self.y_proba is not None else None

        print(f"Balanced Accuracy: {accuracy:.4f}")
        print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
        if logloss:
            print(f"Log Loss: {logloss:.4f}")


        if self.y_proba is not None:

            if len(self.y_proba.shape) == 1:
                fpr, tpr, _ = roc_curve(self.y_test, self.y_proba)
                roc_auc = roc_auc_score(self.y_test, self.y_proba)
                precision, recall, _ = precision_recall_curve(self.y_test, self.y_proba)
                pr_auc = auc(recall, precision)
            else:
                fpr, tpr, _ = roc_curve(self.y_test, self.y_proba[:, 1])
                roc_auc = roc_auc_score(self.y_test, self.y_proba[:, 1])
                precision, recall, _ = precision_recall_curve(self.y_test, self.y_proba[:, 1])
                pr_auc = auc(recall, precision)

            print(f"AUC-ROC Score: {roc_auc:.4f}")
            print(f"Precision-Recall AUC: {pr_auc:.4f}")

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            class_labels = ["Licit", "Illicit"] 
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
            axes[0].set_title("Confusion Matrix")
            axes[0].set_xlabel("Predicted Label")
            axes[0].set_ylabel("True Label")
            axes[0].set_xticklabels(class_labels)
            axes[0].set_yticklabels(class_labels)

            axes[1].plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
            axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")  # Baseline
            axes[1].set_title("ROC Curve")
            axes[1].set_xlabel("False Positive Rate")
            axes[1].set_ylabel("True Positive Rate")
            axes[1].legend()

            axes[2].plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
            axes[2].set_title("Precision-Recall Curve")
            axes[2].set_xlabel("Recall")
            axes[2].set_ylabel("Precision")
            axes[2].legend()

            plt.tight_layout()
            plt.show()
