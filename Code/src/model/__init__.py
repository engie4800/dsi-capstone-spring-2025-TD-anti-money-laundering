from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, PNAConv
from torch.optim import Adam
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAveragePrecision,
)
from tqdm import tqdm
import logging


class GNN(nn.Module):

    def __init__(
        self,
        n_node_feats,
        n_edge_feats,
        num_gnn_layers=2,
        n_classes=1,
        n_hidden=100,
        edge_updates=True,
        residual=True,
        dropout=0.0,
        final_dropout=0.10527690625126304,
        deg=None,
        gnn_flavor="GINe",
    ):
        if gnn_flavor not in ["GINe", "PNA"]:
            raise ValueError(
                "Unsupported GNN flavor"
            )

        super().__init__()

        # Flavor-specific setup
        if gnn_flavor == "GINe":
            self.n_hidden = n_hidden
        elif gnn_flavor == "PNA":
            n_hidden = int((n_hidden // 5) * 5)
            self.n_hidden = n_hidden
            aggregators = ["mean", "min", "max", "std"]
            scalers = ["identity", "amplification", "attenuation"]

        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout

        self.node_emb = nn.Linear(n_node_feats, n_hidden)
        self.edge_emb = nn.Linear(n_edge_feats, n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(self.num_gnn_layers):
            # Flavor-specific convolution
            if gnn_flavor == "GINe":
                conv = GINEConv(nn.Sequential(
                    nn.Linear(self.n_hidden, self.n_hidden),
                    nn.ReLU(),
                    nn.Linear(self.n_hidden, self.n_hidden)
                ), edge_dim=self.n_hidden)
            elif gnn_flavor == "PNA":
                conv = PNAConv(
                    in_channels=n_hidden,
                    out_channels=n_hidden,
                    aggregators=aggregators,
                    scalers=scalers,
                    deg=deg,
                    edge_dim=n_hidden,
                    towers=5,
                    pre_layers=1,
                    post_layers=1,
                    divide_input=False,
                )

            if self.edge_updates: self.emlps.append(nn.Sequential(
                nn.Linear(3 * self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden),
            ))
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(n_hidden))

        self.mlp = nn.Sequential(
            nn.Linear(n_hidden*3, 50),
            nn.ReLU(),
            nn.Dropout(self.final_dropout),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Dropout(self.final_dropout),
            nn.Linear(25, n_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):

            # Node update
            x_res = self.convs[i](x, edge_index, edge_attr)
            x = (x + F.relu(self.batch_norms[i](x_res))) / 2

            # Edge update
            if self.edge_updates:
                edge_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
                edge_attr = edge_attr + self.emlps[i](edge_input) / 2

        # Final prediction
        x_edge = torch.cat([x[src], x[dst], edge_attr], dim=-1)

        return self.mlp(x_edge).squeeze(-1)

class GNNTrainer:
    """Trainer class for GINe-based Graph Neural Network using PyTorch Geometric and torchmetrics.
    Handles training, evaluation, early stopping, and metric logging.
    """
    def __init__(self, model, pl, threshold: float=0.5,
                 device ="cuda" if torch.cuda.is_available() else "cpu", 
                 pos_weight_val:float=6.0, lr:float=0.005):
        """Initializes the trainer with model, data loaders, and training parameters.

        Args:
            model (nn.Module): GNN model (e.g., GINe).
            train_loader, val_loader, test_loader: PyG LinkNeighborLoaders.
            train_indices, val_indices, test_indices (torch.Tensor): Global indices used for label masking.
            df (pd.DataFrame): Original dataframe containing edge-level metadata (e.g., edge_id).
            device (str, optional): Torch device to use. Defaults to "cuda" if available.
            pos_weight_val (float, optional): Positive class weight for BCEWithLogitsLoss. Defaults to 6.0 (what multi-gnn repo used).
            lr (float, optional): Learning rate for Adam optimizer. Defaults to 0.005.
        """
        self.device = pl.device
        self.model = model.to(self.device)
        self.threshold = pl.threshold
        self.epochs = pl.epochs
        self.patience = pl.patience
        
        # Data
        self.df = pl.df
        self.train_loader = pl.train_loader
        self.val_loader = pl.val_loader
        self.test_loader = pl.test_loader
        self.train_indices = pl.train_indices
        self.val_indices = pl.val_indices
        self.test_indices = pl.test_indices

        # Loss and optimizer
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=self.device))
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",            # maximize the metric (e.g., F1, PR AUC)
            factor=0.5,            # reduce LR by half when triggered
            patience=3,            # wait 3 epochs without improvement
            verbose=True
        )
        
        # Metrics used during training and evaluation
        self.metrics = SimpleNamespace(
            accuracy=BinaryAccuracy(threshold=threshold).to(self.device),
            precision=BinaryPrecision(threshold=threshold).to(self.device),
            recall=BinaryRecall(threshold=threshold).to(self.device),
            f1_score=BinaryF1Score(threshold=threshold).to(self.device),
            pr_auc=BinaryAveragePrecision().to(self.device),
        )
        
    @torch.no_grad()
    def evaluate(self, loader, inds):
        """Evaluate model performance on a given data loader and indices.

        Args:
            loader: PyG LinkNeighborLoader.
            inds (torch.Tensor): Global indices to retrieve edge_ids for masking.
            threshold (float): Threshold to binarize predicted probabilities.

        Returns:
            Tuple of (loss, accuracy, precision, recall, F1, PR-AUC)
        """
        self.model.eval() # evaluation mode
        
        # Initialize metrics
        self.metrics.accuracy.reset()
        self.metrics.precision.reset()
        self.metrics.recall.reset()
        self.metrics.f1_score.reset()
        self.metrics.pr_auc.reset()

        loss_fn = nn.BCEWithLogitsLoss()
        preds, targets, probs = [], [], []
        total_loss = 0

        # Run evaluation
        for batch in loader:
            # Get seed transaction ids that we are evaluating this batch
            batch_input_ids = batch.input_id.detach().cpu()
            global_seed_inds = inds[batch_input_ids]
            seed_edge_ids = self.df.loc[global_seed_inds.cpu().numpy(), "edge_id"].values
            edge_ids_in_batch = batch.edge_attr[:, 0].detach().cpu().numpy()
            mask = torch.isin(torch.tensor(edge_ids_in_batch), torch.tensor(seed_edge_ids)).to(self.device)

            # Remove edge_id from attributes (now that we have identified seed ids)
            batch = batch.to(self.device)
            batch_edge_attr = batch.edge_attr[:, 1:].clone()
            
            # Forward pass
            logits = self.model(batch.x, batch.edge_index, batch_edge_attr).view(-1)[mask]
            target = batch.y[mask]
            prob = torch.sigmoid(logits)
            pred = (prob > self.threshold).long()

            total_loss += loss_fn(logits, target.float()).item() * logits.size(0)
            
            # Batch stats
            preds.append(pred)
            targets.append(target)
            probs.append(prob)

        # Epoch stats
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        probs = torch.cat(probs)
        total_loss /= len(targets)

        return (
            total_loss,
            self.metrics.accuracy(preds, targets),
            self.metrics.precision(preds, targets),
            self.metrics.recall(preds, targets),
            self.metrics.f1_score(preds, targets),
            self.metrics.pr_auc(probs, targets)
        )

    def train(self):
        """Main training loop for the model.

        Args:
            threshold (float): Classification threshold for binary decision.
            epochs (int): Number of max training epochs.
            patience (int): Early stopping patience after min_epochs is reached.
        """

        best_val_f1 = 0
        best_pr_auc = 0
        patience_counter = 0  # for early stopping
        min_epochs = 10       # don't allow early model saving

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            train_preds, train_targets, train_probs = [], [], []
            
            # Reset metrics
            self.metrics.accuracy.reset()
            self.metrics.precision.reset()
            self.metrics.recall.reset()
            self.metrics.f1_score.reset()
            self.metrics.pr_auc.reset()

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training"):
                self.optimizer.zero_grad()
                
                # Identify  batch seed transaction ids for loss calculation
                batch_input_ids = batch.input_id.detach().cpu()
                global_seed_inds = self.train_indices[batch_input_ids]
                seed_edge_ids = self.df.loc[global_seed_inds.cpu().numpy(), "edge_id"].values
                edge_ids_in_batch = batch.edge_attr[:, 0].detach().cpu().numpy()
                mask = torch.isin(torch.tensor(edge_ids_in_batch), torch.tensor(seed_edge_ids)).to(self.device)

                # Remove edge_id as attribute before running model
                batch = batch.to(self.device)
                batch_edge_attr = batch.edge_attr[:, 1:].clone()
                
                # Forward pass
                logits = self.model(batch.x, batch.edge_index, batch_edge_attr).view(-1)[mask]
                target = batch.y[mask]
                probs = torch.sigmoid(logits)
                preds = (probs > self.threshold).long()
                
                # Loss and backpropagation
                loss = self.criterion(logits, target.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                # Batch stats
                train_loss += loss.item() * logits.size(0)
                train_preds.append(preds)
                train_targets.append(target)
                train_probs.append(probs)

            # Epoch stats
            train_preds = torch.cat(train_preds)
            train_targets = torch.cat(train_targets)
            train_probs = torch.cat(train_probs)
            train_loss /= len(train_targets)

            train_acc = self.metrics.accuracy(train_preds, train_targets)
            train_prec = self.metrics.precision(train_preds, train_targets)
            train_rec = self.metrics.recall(train_preds, train_targets)
            train_f1 = self.metrics.f1_score(train_preds, train_targets)
            train_pr_auc = self.metrics.pr_auc(train_probs, train_targets)

            # Validation
            val_loss, val_acc, val_prec, val_rec, val_f1, val_pr_auc = self.evaluate(
                self.val_loader,
                self.val_indices
            )

            # Test
            test_loss, test_acc, test_prec, test_rec, test_f1, test_pr_auc = self.evaluate(
                self.test_loader,
                self.test_indices,
            )

            # Logging
            logging.info(f"Epoch {epoch+1}/{self.epochs}")
            logging.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Test Loss: {test_loss:.4f}")
            logging.info(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
            logging.info(f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")
            logging.info(f"Train PR-AUC: {train_pr_auc:.4f} | Val PR-AUC: {val_pr_auc:.4f} | Test PR-AUC: {test_pr_auc:.4f}")
            logging.info(f"Train Prec: {train_prec:.4f} | Val Prec: {val_prec:.4f} | Test Prec: {test_prec:.4f}")
            logging.info(f"Train Rec: {train_rec:.4f} | Val Rec: {val_rec:.4f} | Test Rec: {test_rec:.4f}")
            logging.info("-" * 80)

            # Modify learning rate based on chosen metric (val_f1)
            self.scheduler.step(val_f1)

            # Save best model
            if epoch >= min_epochs and ((val_f1 > best_val_f1) or (val_pr_auc > best_pr_auc)):
                best_val_f1 = max(val_f1, best_val_f1)
                best_pr_auc = max(val_pr_auc, best_pr_auc)
                patience_counter = 0
                torch.save(self.model.state_dict(), f"best_model_epoch{epoch+1}.pt")
                print("✅ New best model saved.")
            elif epoch >= min_epochs:
                patience_counter += 1
                print(f"⚠️ No improvement. Patience: {patience_counter}/{self.patience}")
                if patience_counter >= self.patience:
                    print("🛑 Early stopping triggered.")
                    break

            torch.cuda.empty_cache()
