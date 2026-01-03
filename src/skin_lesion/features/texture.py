import numpy as np
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern

def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image

def _ensure_gray(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale if needed."""
    if image.ndim == 3:
        return rgb2gray(image)
    return image

def extract_lbp_superpixel_features(gray_image: np.ndarray, superpixels: np.ndarray, radius: int = 3)-> np.ndarray:
    gray = _to_uint8(_ensure_gray(gray_image))
    superpixel_gray = _to_uint8(_ensure_gray(superpixels))

    n_points = 8 * radius
    n_bins = n_points + 2

    lbp_gray = local_binary_pattern(
        gray,
        P=n_points,
        R=radius,
        method="uniform"
    )

    lbp_super = local_binary_pattern(
        superpixel_gray,
        P=n_points,
        R=radius,
        method="uniform"
    )

    hist_gray, _ = np.histogram(
        lbp_gray.ravel(),
        bins=np.arange(0, n_bins + 1),
        density=True,
    )

    hist_super, _ = np.histogram(
        lbp_super.ravel(),
        bins=np.arange(0, n_bins + 1),
        density=True,
    )

    return np.concatenate([hist_gray, hist_super]).astype(np.float32)

