from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
from skin_lesion.dataset.sample import Sample
from skin_lesion.dataset.paths import SamplePaths 
import re
from collections import defaultdict
from skimage.io import imread
from skin_lesion.preprocessing.resize import resize_sample_triplet
from skin_lesion.preprocessing.verify import verify_sample_triplet


def extract_id(filename: str) -> str:
    return filename.split('_')[0] + '_' + filename.split('_')[1]

def group_files_by_id(root: Path) -> list[SamplePaths]:
    groups: dict[str, dict[str, Path]] = defaultdict(dict)

    pattern = re.compile(
        r"(ISIC_\d+)(?:_(segmentation|superpixels))?\.(jpg|png)$"
    )

    for path in root.iterdir():
        if not path.is_file():
            continue

        match = pattern.match(path.name)
        if not match:
            continue

        lesion_id, suffix, _ = match.groups()

        if suffix is None:
            groups[lesion_id]["original"] = path
        elif suffix == "segmentation":
            groups[lesion_id]["segmentation"] = path
        elif suffix == "superpixels":
            groups[lesion_id]["superpixels"] = path

    samples: list[SamplePaths] = []

    for lesion_id, files in sorted(groups.items()):
        required = {"original", "segmentation", "superpixels"}
        missing = required - files.keys()

        if missing:
            msg = f"{lesion_id} is missing files: {missing}"
            raise ValueError(msg)

        samples.append(
            SamplePaths(
                id=lesion_id,
                raw=files["original"],
                segmentation=files["segmentation"],
                superpixels=files["superpixels"],
            )
        )

    return samples

def load_dataset(root: Path, image_size: tuple[int, int], max_workers = 8) -> list[Sample]:
    path_samples: list[SamplePaths] = group_files_by_id(root)

    tasks = [(p, image_size) for p in path_samples]
    samples: list[Sample] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for sample in tqdm(
            executor.map(_load_one_sample, tasks),
            total=len(tasks),
            desc="Loading dataset",
        ):
            samples.append(sample)

    return samples

def _load_one_sample(args: tuple[SamplePaths, tuple[int, int]]) -> Sample:
    paths, image_size = args

    raw = imread(paths.raw)
    segmentation = imread(paths.segmentation)
    superpixels = imread(paths.superpixels)

    if segmentation.ndim == 3:
        segmentation = segmentation[..., 0]
    if superpixels.ndim == 3:
        superpixels = superpixels[..., 0]

    raw, segmentation, superpixels = resize_sample_triplet(
        raw, segmentation, superpixels, image_size
    )

    verify_sample_triplet(raw, segmentation, superpixels)

    return Sample(
        id=paths.id,
        raw=raw,
        segmentation=segmentation,
        superpixels=superpixels,
    )
