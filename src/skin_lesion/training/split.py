from torch.utils.data import random_split
import torch
from skin_lesion.config.config import TRAIN_SEED, TRAIN_RATIO

def split_dataset(
    dataset,
    train_ratio: float = TRAIN_RATIO,
    seed: int = TRAIN_SEED,
):
    n_total = len(dataset)
    n_train = int(train_ratio * n_total)
    n_val = n_total - n_train

    generator = torch.Generator().manual_seed(seed)

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=generator,
    )

    return train_ds, val_ds
