from skimage.color import rgb2gray
from skin_lesion.dataset.sample import Sample
import numpy as np
from skin_lesion.features.geometry import extract_geometrical_features
from skin_lesion.features.texture import extract_lbp_superpixel_features

def extract_features(sample: Sample) -> np.ndarray:
    geom = extract_geometrical_features(sample.segmentation)
    texture = extract_lbp_superpixel_features(rgb2gray(sample.raw), sample.superpixels)

    return np.concatenate([geom, texture])
