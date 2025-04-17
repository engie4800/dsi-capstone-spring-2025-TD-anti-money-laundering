"""Note that this is a 'legacy' plotting method (added before the 
plotting module) and therefore does not (yet) operate on the pipeline
itself. This is a nice-to-have refactor
"""
import matplotlib.pyplot as plt
import seaborn as sns
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


def result_metrics(slide_title, y_train, y_train_pred, y_train_proba,
                   y_val, y_val_pred, y_val_proba,
                   y_test, y_test_pred, y_test_proba,
                   class_labels=None) -> None:
    """
    Compute and display model performance metrics for train, validation, and test sets.
    """

    def compute_metrics(y_true, y_pred, y_proba) -> dict:
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
