import torch
from BlyncsySFT.format import (
    convert_to_coco,
    convert_to_xyxy,
    reformat_predictions,
    format_target,
)


def test_convert_to_coco():
    assert convert_to_coco([10, 20, 30, 40]) == [10, 20, 20, 20]


def test_convert_to_xyxy():
    assert convert_to_xyxy([10, 20, 20, 20]) == [10, 20, 30, 40]


def test_reformat_predictions_coco():
    preds = [{
        "image_id": 1,
        "pred_boxes": [[10, 20, 30, 40]],
        "pred_scores": [0.9],
        "pred_labels": [2]
    }]
    formatted = reformat_predictions(preds, format="coco")
    assert formatted[0]["image_id"] == 1
    assert formatted[0]["category_id"] == 2
    assert formatted[0]["bbox"] == [10, 20, 20, 20]
    assert formatted[0]["score"] == 0.9


def test_reformat_predictions_xyxy():
    preds = [{
        "image_id": 99,
        "pred_boxes": [[5, 5, 15, 25]],
        "pred_scores": [0.88],
        "pred_labels": [1]
    }]
    formatted = reformat_predictions(preds, format="xyxy")
    assert formatted[0]["bbox"] == [5, 5, 20, 30]


def test_format_target():
    target = [{
        "bbox": [10, 20, 30, 40],
        "category_id": 1,
        "area": 1200,
        "image_id": 7
    }]
    formatted = format_target(target)
    assert torch.equal(formatted["boxes"], torch.tensor([[10, 20, 40, 60]]))
    assert torch.equal(formatted["labels"], torch.tensor([1]))
    assert torch.equal(formatted["area"], torch.tensor([1200.0]))
    assert torch.equal(formatted["image_id"], torch.tensor([7]))
