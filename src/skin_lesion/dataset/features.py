import torch
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    def __init__(
        self,
        X: torch.Tensor,      # shape: (N, D)
        y: torch.Tensor,      # shape: (N,)
        ids: list[str],       
    ):
        self.X = X
        self.y = y
        self.ids = ids

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
