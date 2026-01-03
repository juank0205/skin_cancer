from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class SamplePaths:
    id: str
    raw: Path
    segmentation: Path
    superpixels: Path
