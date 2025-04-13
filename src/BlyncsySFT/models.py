import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models import resnet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import box_iou


def custom_faster_rcnn(backbone_name: str = 'resnet50',
                       num_classes: int = 2,
                       focal_loss_args: dict = {},
                       custom_anchor_args: dict = {},
                       state_dict_path=None):
    """
    Getting a Custom Faster R-CNN with the specified backbone and focal loss for class imbalance.

    Args:
    - backbone_name: Backbone to use for the model
    - num_classes: Number of classes to predict
    - focal_loss_args: Creating focal loss function. For custom values, they should be provided in key : value format.
      Example:
        'alpha' : 0.025,
        'gamma' : 2.0
    - custom_anchor_args: Creating a custom anchor generator. There needs to be two arguments provided in key : value
      format.
      Example:
        'anchor_sizes' : ((50,), (77,), (108,), (134,), (177,), (204,), (272,), (308,), (470,)),
        'aspect_ratios' : (1.0, 1.5, 2.0, 3.0)
    - state_dict_path: Path to previously trained model

    Returns:
    - Custom Faster R-CNN model with the specified backbone and focal loss
    """
    # Create a mapping from backbone name to its weights enum class
    backbone_weights_map = {
        'resnet18': resnet.ResNet18_Weights,
        'resnet34': resnet.ResNet34_Weights,
        'resnet50': resnet.ResNet50_Weights,
        'resnet101': resnet.ResNet101_Weights,
        'resnet152': resnet.ResNet152_Weights,
    }

    # Get the weights enum class for the specified backbone
    weights_enum = backbone_weights_map.get(backbone_name.lower())

    # Checking if weights_enum exists
    if weights_enum is None:
        raise ValueError(f"Unsupported backbone: {backbone_name}. Supported backbones are: {list(backbone_weights_map.keys())}")

    # Initiating model_kwargs dictionary
    model_kwargs = {
        'backbone': resnet_fpn_backbone(backbone_name=backbone_name, weights=weights_enum.DEFAULT),
        'num_classes': num_classes
    }

    # Checking focal loss params
    if focal_loss_args.get('alpha') is not None and focal_loss_args.get('gamma') is not None:
        # Adding to model_kwargs
        model_kwargs['focal_loss_fn'] = FocalLoss(alpha=focal_loss_args.get('alpha'),
                                                  gamma=focal_loss_args.get('gamma'))

        # Setting create model function for Faster RCNN with Focal Loss
        create_model = FasterRCNNWithFocalLoss
    else:
        # Setting create model function for regular Faster RCNN
        create_model = FasterRCNN

    # Check for custom anchor arguments and create an AnchorGenerator if available
    if 'anchor_sizes' in custom_anchor_args and 'aspect_ratios' in custom_anchor_args:
        anchor_generator = AnchorGenerator(
            sizes=custom_anchor_args['anchor_sizes'],
            aspect_ratios=[custom_anchor_args['aspect_ratios']] * 5
        )

        # Create the RPN head
        rpn_head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

        # Adding to model_kwargs
        model_kwargs['rpn_anchor_generator'] = anchor_generator
        model_kwargs['rpn_head'] = rpn_head

    # Creating model
    model = create_model(**model_kwargs)

    # If a state dictionary path was provided
    if state_dict_path is not None:
        # Load the old weights
        old_weights = torch.load(state_dict_path)

        # Get the state dict of the new model
        new_state_dict = model.state_dict()

        # Update the new state dict with matching weights from the old state dict
        for key in new_state_dict:
            if key in old_weights and old_weights[key].shape == new_state_dict[key].shape:
                new_state_dict[key] = old_weights[key]

        # Load the updated state dict into the new model
        model.load_state_dict(new_state_dict, strict=False)  # `strict=False` allows partial loading

    # Reinitializing RPN if using custom_anchor_generator
    if 'rpn_anchor_generator' in model_kwargs.keys():
        reinitialize_rpn_head(model.rpn.head)

    return model


def reinitialize_rpn_head(base_model):
    """
    Reinitializes the final layers of the RPN head (cls_logits and bbox_pred).
    """
    # Getting RPN head from base model
    rpn_head = base_model

    # Reinitialize cls_logits layer
    if hasattr(rpn_head, 'cls_logits'):
        nn.init.normal_(rpn_head.cls_logits.weight, mean=0.0, std=0.01)
        nn.init.constant_(rpn_head.cls_logits.bias, 0)

    # Reinitialize bbox_pred layer
    if hasattr(rpn_head, 'bbox_pred'):
        nn.init.normal_(rpn_head.bbox_pred.weight, mean=0.0, std=0.01)
        nn.init.constant_(rpn_head.bbox_pred.bias, 0)


class FasterRCNNWithFocalLoss(FasterRCNN):
    def __init__(self, backbone, num_classes, focal_loss_fn, **kwargs):
        """
        Args:
            backbone: Pretrained backbone (e.g., ResNet-50-FPN)
            num_classes: Number of output classes (including background)
            focal_loss_fn: Custom Focal Loss instance
            kwargs: Additional args for FasterRCNN (e.g., min_size, max_size)
        """
        # Initialize standard FasterRCNN with the provided backbone
        super().__init__(backbone, num_classes, **kwargs)
        self.focal_loss_fn = focal_loss_fn

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            # Transform images and targets
            # Model's transform module handles preprocessing (resizing, normalization, etc.)
            transformed_images, transformed_targets = self.transform(images, targets)
            # images_tensor = transformed_images.tensors

            # Extract features
            # Pass the transformed image tensor through the backbone to obtain features
            features = self.backbone(transformed_images.tensors)
            if isinstance(features, torch.Tensor):  # Ensure OrderedDict format
                from collections import OrderedDict
                features = OrderedDict([('0', features)])

            # Generate proposals using the RPN
            # RPN uses the feature maps to generate object proposals
            # It also returns losses (rpn_losses) if targets are provided
            proposals, rpn_losses = self.rpn(transformed_images, features, transformed_targets)

            image_shapes = transformed_images.image_sizes

            # ROI Pooling: pool features for each proposal
            # Pool the features corresponding to each proposal into a fixed-size feature map
            roi_pooled_features = self.roi_heads.box_roi_pool(features, proposals, image_shapes)

            # Box Head: extract features for each ROI
            # Process the pooled features through the box head to obtain a refined feature representation
            box_features = self.roi_heads.box_head(roi_pooled_features)

            # Box Predictor: get raw classification logits
            # Pass the box features through the box predictor to obtain raw classification logits
            # The box predictor returns a tuple: (classification_logits, bbox_regression outputs)
            classification_logits, _ = self.roi_heads.box_predictor(box_features)

            # Match proposals to ground truth to generate labels for each proposal
            # For each image, match its proposals to ground truth boxes using an IoU threshold.
            # Matching function assigns a label to each proposal:
            # - a positive class label if the proposal overlaps sufficiently with a ground truth box,
            # - or 0 (background) if it doesn't
            all_matched_labels = []
            for i in range(len(transformed_targets)):
                gt_boxes = transformed_targets[i]['boxes']  # Ground truth boxes for image i
                gt_labels = transformed_targets[i]['labels']  # Corresponding class labels
                proposals_i = proposals[i]  # Proposals generated for image i

                if proposals_i.shape[0] > 0:  # Ensure there are proposals
                    matched_labels = self.match_proposals_for_image(proposals_i, gt_boxes, gt_labels)
                    all_matched_labels.append(matched_labels)

                # matched_labels = match_proposals_for_image(proposals_i, gt_boxes, gt_labels, iou_threshold=0.5)
                # all_matched_labels.append(matched_labels)

            # Concatenate matched labels from all images into a single tensor
            # all_matched_labels = torch.cat(all_matched_labels, dim=0)  # [total_proposals]
            # Ensure matched labels exist before concatenation
            if all_matched_labels:
                all_matched_labels = torch.cat(all_matched_labels, dim=0)
            else:
                all_matched_labels = torch.tensor([], dtype=torch.int64, device=classification_logits.device)

            # Compute Focal Loss using raw logits and matched labels
            # Focal loss is computed on the raw classification logits from the box predictor and the
            # aggregated ground truth labels (matched to proposals)
            focal_classification_loss = self.focal_loss_fn(classification_logits, all_matched_labels)

            # Combine losses (RPN losses and focal classification loss)
            # Combine the RPN losses (from proposal generation) and our new focal classification loss
            losses = {}
            losses.update(rpn_losses)
            losses['loss_classifier'] = focal_classification_loss

            return losses
        else:
            # When not training, simply run the standard forward pass of the pre-trained model
            return super().forward(images, targets)

    def match_proposals_for_image(self, proposals, gt_boxes, gt_labels, iou_threshold=0.5):
        """
        Match RPN proposals to ground truth using IoU-based matching.

        Args:
            proposals (Tensor): [num_proposals, 4] Proposed boxes.
            gt_boxes (Tensor): [num_gt, 4] Ground truth boxes.
            gt_labels (Tensor): [num_gt] Ground truth labels.
            iou_threshold (float): IoU threshold for matching.

        Returns:
            matched_labels (Tensor): Labels for each proposal (0 = background, 1+ = object class).
        """
        if gt_boxes.numel() == 0:  # No GT boxes in the image
            return torch.zeros(proposals.shape[0], dtype=torch.int64, device=proposals.device)

        iou_matrix = box_iou(proposals, gt_boxes)  # Compute IoU between proposals and GT boxes
        max_iou, matched_idx = iou_matrix.max(dim=1)  # Get the best match for each proposal

        # Assign background (0) to proposals with low IoU
        matched_labels = torch.full((proposals.shape[0],), 0, dtype=torch.int64, device=proposals.device)
        fg_indices = max_iou >= iou_threshold  # Proposals with IoU >= threshold are foreground
        matched_labels[fg_indices] = gt_labels[matched_idx[fg_indices]]

        return matched_labels


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Initialize the focal loss module

        Args:
        - alpha: Weighting factor for balancing class imbalance, typical values are around 0.25
        - gamma: Focusing parameter that reduces the loss for well-classified examples,
                 higher values (e.g., 2.0) increase the effect
        """
        super(FocalLoss, self).__init__()

        # Store the hyperparameters for later use in the forward pass
        self.alpha = alpha  # Balances the importance between classes
        self.gamma = gamma  # Adjusts the rate at which easy examples are down-weighted
        self.reduction = reduction  # Options: 'mean', 'sum', 'none'

    def forward(self, logits, targets):
        """
        Compute the focal loss.

        Args:
        - logits: Raw, unnormalized model outputs (before softmax), with shape [N, num_classes],
                  where N is the number of samples (or proposals) and num_classes is the number of classes
        - targets: Ground truth class labels as integers, with shape [N]

        Returns:
        - loss: Scalar loss value (the mean focal loss over the batch)
        """
        # Compute softmax probabilities
        # Converts raw scores into probabilities that sum to 1 for each sample
        log_probs = F.log_softmax(logits, dim=1)  # shape: [N, num_classes]

        # Gather the probabilities for the true classes
        # For each sample, gather the probability corresponding to the true class.
        # targets.unsqueeze(1) converts the shape from [N] to [N, 1] so that .gather() can select
        # the probability for the correct class for each sample.
        targets = targets.long()  # Ensure targets are long for indexing
        true_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # shape: [N, 1]

        probs = true_log_probs.exp()

        # Compute the focal loss scaling factor
        # (1 - true_class_probs) is high for misclassified (hard) examples and low for well-classified ones
        # Raising this term to the power gamma further emphasizes the effect on hard examples
        # Multiplying by alpha scales the loss for each class
        # focal_weight = self.alpha * (1 - true_class_probs) ** self.gamma
        focal_weight = (1 - probs) ** self.gamma

        # Compute standard cross entropy loss (without reduction)
        # (i.e., compute loss per sample)
        # reduction='none' returns the loss for each sample
        # ce_loss = F.cross_entropy(logits, targets, reduction='none').unsqueeze(1)  # shape: [N, 1]

        # Compute focal loss
        # Multiply the cross-entropy loss by the focal weighting factor
        # This means that the loss for easy examples (where true_class_probs is high) is down-weighted
        # loss = focal_weight * ce_loss
        loss = -self.alpha * focal_weight * true_log_probs

        # return loss.mean()
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # No reduction
