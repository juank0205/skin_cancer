import numpy as np
import pytest
from skimage.io import imsave

@pytest.fixture
def fake_triplet(tmp_path):
    """
    Creates a minimal valid ISIC triplet on disk, without warnings.
    """
    h, w = 50, 80

    raw = np.random.randint(0, 5, size=(h, w, 3), dtype=np.uint8)       # tiny variation
    segmentation = np.zeros((h, w), dtype=np.uint8)
    segmentation[10:30, 20:40] = 1
    superpixels = np.arange(h * w, dtype=np.uint16).reshape(h, w)

    # save images
    imsave(tmp_path / "ISIC_0000000.jpg", raw)
    imsave(tmp_path / "ISIC_0000000_segmentation.png", segmentation)
    imsave(tmp_path / "ISIC_0000000_superpixels.png", superpixels)

    return tmp_path
