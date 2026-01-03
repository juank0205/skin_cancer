from pathlib import Path
import pandas as pd
import numpy as np
import torch
from skin_lesion.training.model import MLPClassifier

def load_labels(csv_path: Path) -> dict[str, int]:
    df = pd.read_csv(csv_path)

    labels: dict[str, int] = {}

    for _, row in df.iterrows():
        image_id = row["image_id"]

        melanoma = int(row["melanoma"])
        sk = int(row["seborrheic_keratosis"])

        if melanoma == 1:
            label = 1
        elif sk == 1:
            label = 2
        else:
            label = 0  # benign

        labels[image_id] = label

    return labels

def load_features(features_path: Path):
    data = np.load(features_path, allow_pickle=True)

    X = torch.from_numpy(data["X"]).float()
    ids = list(data["ids"])

    return X, ids

def save_model(model: torch.nn.Module, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model_path: Path, input_dim: int, device: str):
    model = MLPClassifier(input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
