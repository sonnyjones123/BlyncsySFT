import pytest
import torch
from BlyncsySFT.postprocess import nms, soft_nms, dbsscan_clustering, fuse_clusters


@pytest.fixture
def boxes_scores():
    boxes = torch.tensor([
        [10, 10, 20, 20],
        [12, 12, 22, 22],
        [50, 50, 60, 60],
    ], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8, 0.75], dtype=torch.float32)
    return boxes, scores


def test_nms_output_shape(boxes_scores):
    boxes, scores = boxes_scores
    kept_boxes, kept_scores = nms(boxes, scores, iou_threshold=0.5)
    assert kept_boxes.shape[1] == 4
    assert kept_boxes.shape[0] == kept_scores.shape[0]
    assert kept_scores.ndim == 1


def test_soft_nms_linear(boxes_scores):
    boxes, scores = boxes_scores
    kept_boxes, kept_scores = soft_nms(boxes, scores.clone(), method="linear")
    assert kept_boxes.shape[1] == 4
    assert kept_boxes.shape[0] == kept_scores.shape[0]


def test_soft_nms_gaussian(boxes_scores):
    boxes, scores = boxes_scores
    kept_boxes, kept_scores = soft_nms(boxes, scores.clone(), method="gaussian")
    assert kept_boxes.shape[1] == 4
    assert kept_boxes.shape[0] == kept_scores.shape[0]


def test_soft_nms_invalid_method(boxes_scores):
    boxes, scores = boxes_scores
    with pytest.raises(ValueError):
        soft_nms(boxes, scores.clone(), method="unknown")


def test_dbscan_clusters(boxes_scores):
    boxes, scores = boxes_scores
    clusters, noise = dbsscan_clustering(boxes, scores, eps=30)
    total = sum(len(c) for c in clusters) + len(noise)
    assert total == len(boxes)
    assert isinstance(clusters, list)
    assert isinstance(noise, list)


def test_fuse_clusters(boxes_scores):
    boxes, scores = boxes_scores
    clusters, noise = dbsscan_clustering(boxes, scores, eps=30)
    fused_boxes, fused_scores = fuse_clusters(boxes, scores, clusters, noise)
    assert fused_boxes.shape[0] == fused_scores.shape[0]
    assert fused_boxes.shape[1] == 4
