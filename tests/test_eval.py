import pytest
from BlyncsySFT.eval import (
    compute_ap,
    calculate_iou_reg,
    calculate_giou,
    calculate_diou,
    calculate_ciou,
    calculate_soft_iou,
)

# Sample boxes in [xmin, ymin, width, height]
box1 = [50, 50, 100, 100]
box2 = [75, 75, 100, 100]  # overlaps with box1
box3 = [200, 200, 50, 50]   # no overlap with box1


@pytest.mark.parametrize("iou_func", [
    calculate_iou_reg,
    calculate_giou,
    calculate_diou,
    calculate_ciou,
    calculate_soft_iou
])
def test_iou_variants_overlap(iou_func):
    score = iou_func(box1, box2)
    assert 0 <= score <= 1
    assert score > 0, f"{iou_func.__name__} should give non-zero for overlapping boxes"


def test_iou_variants_non_overlap():
    assert calculate_iou_reg(box1, box3) == 0
    assert calculate_soft_iou(box1, box3) > 0  # Soft IoU returns small non-zero values
    assert calculate_giou(box1, box3) < 0      # GIoU is negative when no overlap


def test_compute_ap_perfect_match():
    preds = [{"bbox": box1, "category_id": 1, "score": 0.9}]
    gts = [{"bbox": box1, "category_id": 1}]
    ap = compute_ap(preds, gts, iou_threshold=0.5, iou_function="IoU")
    assert ap == pytest.approx(1.0), "Expected perfect AP score"


def test_compute_ap_mismatch_category():
    preds = [{"bbox": box1, "category_id": 2, "score": 0.9}]
    gts = [{"bbox": box1, "category_id": 1}]
    ap = compute_ap(preds, gts, iou_threshold=0.5, iou_function="IoU")
    assert ap == 0.0


def test_compute_ap_partial_match():
    preds = [
        {"bbox": box1, "category_id": 1, "score": 0.9},
        {"bbox": box3, "category_id": 1, "score": 0.6},
    ]
    gts = [{"bbox": box1, "category_id": 1}]
    ap = compute_ap(preds, gts, iou_threshold=0.5, iou_function="IoU")
    assert 0.5 <= round(ap, 6) <= 1.0
