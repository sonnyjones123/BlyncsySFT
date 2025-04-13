import torch
from BlyncsySFT.models import (
    custom_faster_rcnn,
    FocalLoss,
    FasterRCNNWithFocalLoss,
)


def test_model_initialization():
    model = custom_faster_rcnn(backbone_name='resnet18', num_classes=2)
    assert isinstance(model, torch.nn.Module)


def test_model_with_focal_loss():
    model = custom_faster_rcnn(
        backbone_name='resnet18',
        num_classes=2,
        focal_loss_args={'alpha': 0.25, 'gamma': 2.0}
    )
    assert isinstance(model, FasterRCNNWithFocalLoss)
    assert hasattr(model, 'focal_loss_fn')


def test_model_with_custom_anchors():
    anchors = {
        'anchor_sizes': ((32,), (64,), (128,), (256,), (512,)),
        'aspect_ratios': (0.5, 1.0, 2.0)
    }
    model = custom_faster_rcnn(
        backbone_name='resnet18',
        num_classes=2,
        custom_anchor_args=anchors
    )
    assert hasattr(model.rpn, 'anchor_generator')
    assert model.rpn.anchor_generator is not None


def test_focal_loss_output():
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    logits = torch.tensor([[2.0, 0.5], [0.5, 1.5]], requires_grad=True)
    targets = torch.tensor([0, 1])
    loss = focal_loss(logits, targets)
    assert loss.item() > 0


def test_focal_loss_no_reduction():
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')
    logits = torch.tensor([[2.0, 0.5], [0.5, 1.5]], requires_grad=True)
    targets = torch.tensor([0, 1])
    loss = focal_loss(logits, targets)
    assert loss.shape == torch.Size([2])
