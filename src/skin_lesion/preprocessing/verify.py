import numpy as np

def verify_sample_triplet(
    raw: np.ndarray,
    segmentation: np.ndarray,
    superpixels: np.ndarray,
    ) -> None:
    if raw.ndim != 3:
        raise ValueError(
            f"raw image must be 3D (H, W, C), got shape {raw.shape}"
        )

    if raw.shape[2] != 3:
        raise ValueError(
            f"raw image must have 3 channels, got {raw.shape[2]}"
        )

    if segmentation.ndim != 2:
        raise ValueError(
            f"Segmentation must be 2D (H, W), got shape {segmentation.shape}"
        )

    if superpixels.ndim != 2:
        raise ValueError(
            f"Superpixels must be 2D (H, W), got shape {superpixels.shape}"
        )

    h, w = segmentation.shape

    if raw.shape[:2] != (h, w):
        raise ValueError(
            "raw image and segmentation size mismatch: "
            f"{raw.shape[:2]} vs {(h, w)}"
        )

    if superpixels.shape != (h, w):
        raise ValueError(
            "Superpixels and segmentation size mismatch: "
            f"{superpixels.shape} vs {(h, w)}"
        )

    # ---------- Segmentation semantics ----------
    unique_seg = np.unique(segmentation)
    allowed = {0, 255}
    if not set(unique_seg).issubset(allowed):
        raise ValueError(
            f"Segmentation mask is not binary, values: {unique_seg}"
        )

    # ---------- Lesion existence ----------
    if segmentation.sum() == 0:
        raise ValueError("Segmentation mask contains no lesion pixels")

    # ---------- Superpixel semantics ----------
    if not np.issubdtype(superpixels.dtype, np.integer):
        raise TypeError("Superpixels must contain integer labels")

    if superpixels.min() < 0:
        raise ValueError("Superpixels contain negative labels")
