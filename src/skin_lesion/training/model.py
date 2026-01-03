import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()

        self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3),

                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(64, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
