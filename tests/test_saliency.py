import torch
import numpy as np
from BlyncsySFT.saliency import (
    expected_bbox,
    normalized_center_distance_torch,
    reformat_target,
    score_images
)
from torch.utils.data import DataLoader, Dataset


# === Unit Tests for Utility Functions ===

def test_expected_bbox_center():
    saliency_map = np.zeros((100, 100))
    saliency_map[45:55, 45:55] = 1
    bbox = expected_bbox(saliency_map, k=1)
    x_min, y_min, x_max, y_max = bbox
    assert 40 <= x_min <= 50
    assert 40 <= y_min <= 50
    assert 50 <= x_max <= 60
    assert 50 <= y_max <= 60


def test_normalized_center_distance_same_boxes():
    pred = torch.tensor([[0, 0, 100, 100]])
    gt = torch.tensor([[0, 0, 100, 100]])
    score = normalized_center_distance_torch(pred, gt)
    assert torch.allclose(score, torch.tensor([1.0]), atol=1e-5)


def test_reformat_target_output():
    bbox = [10, 20, 100, 50]
    assert reformat_target(bbox) == [10, 20, 110, 70]


# === Dummy Classes for Testing score_images ===

class DummyDataset(Dataset):
    def __getitem__(self, idx):
        image = torch.rand(3, 64, 64)
        target = [{'bbox': [10, 20, 30, 40], 'image_id': 1}]
        return image, target

    def __len__(self):
        return 1


class DummyModel(torch.nn.Module):
    def forward(self, x):
        # Generate score based on mean of input (ensures grad_fn)
        score = x.mean(dim=(1, 2, 3), keepdim=False)  # Tensor of shape [1]
        return [{
            'boxes': torch.tensor([[10., 10., 40., 40.]], device=x.device),
            'scores': score,  # This has a grad_fn now
            'labels': torch.tensor([1], device=x.device)
        }]


def test_score_images_with_dummy_model():
    model = DummyModel()
    dataloader = DataLoader(DummyDataset(), batch_size=1, collate_fn=lambda x: x[0])
    results = score_images(model, dataloader, alpha=0.7)
    assert isinstance(results, list)
    assert len(results) == 1
    assert 'image_id' in results[0]
    assert 'score' in results[0]
