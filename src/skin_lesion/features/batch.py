from tqdm import tqdm
from skin_lesion.dataset.sample import Sample
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable

from skin_lesion.features.extract import extract_features

def _extract_one(sample: Sample) -> tuple[str, np.ndarray]:
    return sample.id, extract_features(sample)

def extract_feature_matrix(samples: Iterable[Sample]) -> tuple[np.ndarray, list[str]]:
    features: list[np.ndarray] = []
    ids: list[str] = []

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(_extract_one, sample)
            for sample in samples
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Extracting features",
        ):
            sample_id, feat = future.result()
            ids.append(sample_id)
            features.append(feat)

    X = np.vstack(features)
    return X, ids
