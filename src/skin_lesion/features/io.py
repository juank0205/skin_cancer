from pathlib import Path
import numpy as np


def save_features(path: Path, X: np.ndarray, ids: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        X=X,
        ids=np.asarray(ids),
    )


def load_features(path: Path) -> tuple[np.ndarray, list[str]]:
    """
    Load features from disk.
    """
    data = np.load(path, allow_pickle=True)
    return data["X"], data["ids"].tolist()
