import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from skin_lesion.config.config import FEATURES_DIR, LABELS_CSV_DIR, BATCH_SIZE, LEARNING_RATE, MODEL_DIR
from skin_lesion.training import evaluate
from skin_lesion.training.build import build_feature_dataset
from skin_lesion.training.io import save_model
from skin_lesion.training.split import split_dataset
from skin_lesion.training.model import MLPClassifier
from skin_lesion.training.loop import train
from torch.utils.data import DataLoader
from skin_lesion.training.evaluate import get_predictions, compute_metrics, plot_confusion_matrix, plot_roc_curve, plot_all

def train_model():
    print("Loading features from disk...")
    dataset = build_feature_dataset(
       FEATURES_DIR,
       LABELS_CSV_DIR
    )

    train_ds, val_ds = split_dataset(dataset)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = MLPClassifier(input_dim=dataset.X.shape[1])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Training model...")
    train(model, train_loader, val_loader, criterion, optimizer)

    save_model(model, MODEL_DIR)
    print("Written file to: ", MODEL_DIR)

    evaluate_model(plot_all_plots=True)

def _print_metrics(metrics: dict) -> None:
    print("\nEvaluation metrics")
    print("-" * 30)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print(f"ROC AUC  : {metrics['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

def evaluate_model(
    plot_confusion: bool = False,
    plot_roc: bool = False,
    plot_all_plots: bool = False,
    print_metrics: bool = False,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading feature dataset...")
    dataset = build_feature_dataset(
       FEATURES_DIR,
       LABELS_CSV_DIR
    )

    _, val_dataset = split_dataset(dataset)

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    input_dim = dataset.X.shape[1]

    print("Loading trained model...")
    model = MLPClassifier(input_dim)
    model.load_state_dict(
        torch.load(MODEL_DIR, map_location=device)
    )
    model.to(device)
    model.eval()

    print("Running evaluation...")
    y_true, y_probs = get_predictions(
        model,
        val_loader,
        device
    )

    metrics = compute_metrics(y_true, y_probs)

    # ---------- SWITCH DE COMPORTAMIENTO ----------
    if plot_all_plots:
        _print_metrics(metrics)
        plot_all(metrics)
        return

    if plot_confusion:
        plot_confusion_matrix(metrics["confusion_matrix"])

    if plot_roc:
        plot_roc_curve(
            *metrics["roc_curve"],
            metrics["roc_auc"]
        )

    if print_metrics or not any(
        [plot_confusion, plot_roc, plot_all_plots]
    ):
        _print_metrics(metrics)
