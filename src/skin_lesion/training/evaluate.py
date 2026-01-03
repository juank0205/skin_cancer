import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def get_predictions(model, dataloader, device):
    y_true = []
    y_probs = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            probs = torch.sigmoid(logits)

            y_true.append(y.cpu().numpy())
            y_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_probs = np.concatenate(y_probs)

    return y_true, y_probs

def compute_metrics(y_true, y_probs, threshold=0.5):
    # y_true: (N,)
    # y_probs: (N, 2)

    positive_probs = y_probs[:, 1]
    y_pred = (positive_probs > threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_true, positive_probs)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "roc_curve": (fpr, tpr),
        "roc_auc": roc_auc_score(y_true, positive_probs),
    }

def plot_confusion_matrix(cm):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def plot_roc_curve(fpr, tpr, auc):
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def plot_all(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion
    sns.heatmap(
        metrics["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0]
    )
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # ROC
    fpr, tpr = metrics["roc_curve"]
    axes[1].plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], "--")
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
