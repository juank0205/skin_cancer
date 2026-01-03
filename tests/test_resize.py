import numpy as np
from src.skin_lesion.preprocessing.resize import resize_sample_triplet

def test_resize_triplet_shapes():
    orig = np.zeros((100, 200, 3), dtype=np.uint8)
    seg = np.zeros((100, 200), dtype=np.uint8)
    seg[20:40, 50:70] = 1
    sp = np.zeros((100, 200), dtype=np.int32)

    ro, rs, rsp = resize_sample_triplet(orig, seg, sp, (64, 64))

    assert ro.shape == (64, 64, 3)
    assert rs.shape == (64, 64)
    assert rsp.shape == (64, 64)


def test_resize_preserves_binary_mask():
    orig = np.zeros((50, 50, 3), dtype=np.uint8)
    seg = np.zeros((50, 50), dtype=np.uint8)
    seg[10:20, 10:20] = 1
    sp = np.zeros((50, 50), dtype=np.int32)

    _, rs, _ = resize_sample_triplet(orig, seg, sp, (32, 32))

    assert set(rs.flatten()).issubset({0, 1})
