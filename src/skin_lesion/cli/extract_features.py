import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import numpy as np
from skin_lesion.features.batch import extract_feature_matrix
from skin_lesion.dataset.sample import Sample
from skin_lesion.config.config import FEATURES_DIR

def extract_features(dataset) -> tuple[np.ndarray, list[str]]:
    X, ids = extract_feature_matrix(dataset)
    return X, ids

def display_superpixels(dataset: list[Sample]):
    original = dataset[0].raw / 255.0
    overlay = mark_boundaries(original, dataset[0].superpixels, color=(1, 0, 0))
    plt.imshow(overlay)
    plt.axis("off")
    plt.show()


def print_features(n: int = 5):
    data = np.load(FEATURES_DIR, allow_pickle=True)
    print(data)

    X = data["X"]
    ids = data["ids"]

    print(f"Loaded features from: {FEATURES_DIR}")
    print(f"Feature matrix shape: {X.shape}")
    print("-" * 60)

    for i in range(min(n, len(ids))):
        print(f"{ids[i]} -> {X[i]}")
