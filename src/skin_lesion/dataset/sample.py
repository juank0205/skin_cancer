from dataclasses import dataclass
import numpy as np

@dataclass
class Sample: 
    id: str
    raw: np.ndarray
    segmentation: np.ndarray
    superpixels: np.ndarray
