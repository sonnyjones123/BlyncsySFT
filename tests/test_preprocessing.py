import pytest
import numpy as np
import json
import cv2
from unittest.mock import patch
from BlyncsySFT.preprocessing import custom_bounding_boxes, custom_normalization, calc_geometric_mean, calc_aspect_ratio


@pytest.fixture
def sample_coco_json(tmp_path):
    """Creates a temporary COCO-style annotation JSON file."""
    data = {
        "annotations": [
            {"bbox": [0, 0, 100, 50]},
            {"bbox": [0, 0, 200, 100]},
            {"bbox": [0, 0, 50, 25]},
        ]
    }
    file_path = tmp_path / "annotations.json"
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return file_path


def test_custom_bounding_boxes(sample_coco_json):
    with patch("matplotlib.pyplot.show"):  # Suppress plotting
        output = custom_bounding_boxes(str(sample_coco_json))

    assert "anchor_sizes" in output
    assert "aspect_ratios" in output
    assert isinstance(output["anchor_sizes"], tuple)
    assert isinstance(output["aspect_ratios"], tuple)
    assert all(isinstance(s, tuple) for s in output["anchor_sizes"])


def test_calc_geometric_mean():
    assert np.isclose(calc_geometric_mean([4, 9]), 6.0)


def test_calc_aspect_ratio():
    assert calc_aspect_ratio([10, 5]) == 2.0


def test_custom_normalization(tmp_path):
    # Create fake RGB images
    for i in range(10):
        img = np.ones((10, 10, 3), dtype=np.uint8) * (i * 10)
        cv2.imwrite(str(tmp_path / f"{i}.jpg"), img)

    with patch("numpy.random.choice", return_value=list(range(5))):
        mean, std = custom_normalization(str(tmp_path))

    assert isinstance(mean, list) and len(mean) == 3
    assert isinstance(std, list) and len(std) == 3
    assert all(0 <= m <= 1 for m in mean)
    assert all(s >= 0 for s in std)
