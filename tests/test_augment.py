import pytest
import json
from PIL import Image
import numpy as np
import torch
from BlyncsySFT.augment import createAugmentedData, augmentation_types
import warnings


@pytest.fixture
def mock_coco(tmp_path):
    img_dir = tmp_path / "images"
    ann_path = tmp_path / "annotations.json"
    img_dir.mkdir()

    # Create dummy image
    img = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 255)
    img_name = "test_image.jpg"
    img_path = img_dir / img_name
    img.save(img_path)

    # Create dummy annotation
    annotations = {
        "images": [{"id": 1, "file_name": img_name}],
        "annotations": [{"id": 1, "image_id": 1, "bbox": [10, 10, 30, 30], "category_id": 1, "area": 900}],
        "categories": [{"id": 1, "name": "object"}]
    }
    with open(ann_path, "w") as f:
        json.dump(annotations, f)

    return str(img_dir), str(ann_path)


def test_dataset_length_and_loading(mock_coco):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        root, annotation = mock_coco
        dataset = createAugmentedData(
            root=root,
            annotation=annotation,
            augmentation_types=augmentation_types,
            output_dir=None,
            subset_size=1
        )
        assert len(dataset) == 1
        img, target = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert 'boxes' in target and 'labels' in target and 'image_id' in target
