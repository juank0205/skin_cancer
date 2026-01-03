import numpy as np
import pytest
from src.skin_lesion.preprocessing.verify import verify_sample_triplet

def test_verify_valid_triplet():
    orig = np.zeros((64, 64, 3), dtype=np.uint8)
    seg = np.zeros((64, 64), dtype=np.uint8)
    seg[10:20, 10:20] = 1
    sp = np.zeros((64, 64), dtype=np.int32)

    verify_sample_triplet(orig, seg, sp)


def test_verify_empty_segmentation_fails():
    orig = np.zeros((64, 64, 3), dtype=np.uint8)
    seg = np.zeros((64, 64), dtype=np.uint8)
    sp = np.zeros((64, 64), dtype=np.int32)

    with pytest.raises(ValueError):
        verify_sample_triplet(orig, seg, sp)
