import numpy as np
from skimage.measure import label, regionprops

def extract_geometrical_features(segmentation: np.ndarray) -> np.ndarray:
    if segmentation.max() > 1:
        mask = segmentation > 0
    else:
        mask = segmentation.astype(bool)

    labeled = label(mask)
    regions = regionprops(labeled)

    features = np.zeros(10, dtype=np.float32)

    if not regions:
        return features

    # Assume largest connected component = lesion
    region = max(regions, key=lambda r: r.area)

    features[0] = region.area_convex
    features[1] = region.eccentricity
    features[2] = region.perimeter
    features[3] = region.equivalent_diameter_area
    features[4] = region.extent
    features[5] = region.area_filled
    features[6] = region.axis_minor_length
    features[7] = region.axis_major_length
    features[8] = (
        region.axis_major_length / region.axis_minor_length
        if region.axis_minor_length > 0
        else 0.0
    )
    features[9] = region.solidity

    return features
