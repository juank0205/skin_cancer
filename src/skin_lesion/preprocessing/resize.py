import numpy as np
from skimage.transform import resize

def resize_sample_triplet(
    raw: np.ndarray,
    segmentation: np.ndarray,
    superpixels: np.ndarray,
    size: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_h, target_w = size

    # raw
    resized_raw = np.asarray(resize(
        raw,
        (target_h, target_w, raw.shape[2]),
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True
    ))

    # segmentation
    resized_segmentation = np.asarray(resize(
        segmentation,
        (target_h, target_w),
        order=0,
        mode="reflect",
        anti_aliasing=False,
        preserve_range=True
    ))

    # superpixels
    resized_superpixels = np.asarray(resize(
        superpixels,
        (target_h, target_w),
        order=0,
        mode="reflect",
        anti_aliasing=False,
        preserve_range=True
    ))
    
    return resized_raw, resized_segmentation, resized_superpixels
