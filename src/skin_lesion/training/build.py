from pathlib import Path
import torch
from .io import load_labels, load_features
from skin_lesion.dataset.features import FeatureDataset


def build_feature_dataset(
    features_path: Path,
    labels_csv: Path,
) -> FeatureDataset:
    labels = load_labels(labels_csv)
    X, ids = load_features(features_path)

    y = torch.tensor(
        [labels[i] for i in ids],
        dtype=torch.long,
    )

    return FeatureDataset(X=X, y=y, ids=ids)
