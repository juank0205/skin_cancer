from src.skin_lesion.preprocessing.io import group_files_by_id, load_dataset
import warnings

warnings.filterwarnings("ignore", message=".*low contrast image.*")

def test_group_files_by_id(fake_triplet):
    samples = group_files_by_id(fake_triplet)

    assert len(samples) == 1
    sp = samples[0]

    assert sp.__class__.__name__ == "SamplePaths"
    assert sp.id == "ISIC_0000000"
    assert sp.raw.name.endswith(".jpg")
    assert sp.segmentation.name.endswith("_segmentation.png")
    assert sp.superpixels.name.endswith("_superpixels.png")

def test_load_dataset(fake_triplet):
    dataset = load_dataset(fake_triplet, image_size=(64, 64))

    assert len(dataset) == 1

    sample = dataset[0]
    assert sample.__class__.__name__ == "Sample"

    assert sample.raw.shape == (64, 64, 3)
    assert sample.segmentation.shape == (64, 64)
    assert sample.superpixels.shape == (64, 64)

    assert sample.features is None
    assert sample.label is None
