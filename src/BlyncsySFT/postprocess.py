import numpy as np
import torch
from torchvision.ops import box_iou
from sklearn.cluster import DBSCAN


def nms(boxes=[], scores=[], iou_threshold=0.5):
    """
    Perform Non-Maximum Suppression (NMS) on the list of bounding boxes and scores.
    This is performed under the assumption that all boxes and scores are from the same class.

    Args:
    - boxes: list of lists, numpy array, or tensor with the bounding box coordinates
    - scores: list of lists, numpy array, of tensor of scores associated with the bounding boxes
    - iou_threshold: IoU threshold for suppression

    Returns:
    - indicies: list of lists with the bounding box coordinates after NMS
    """
    # Running check on bounding box coordinates
    for box in boxes:
        if len(box) != 4:
            raise TypeError("Missing bounding box coordinates. One or more of your bounding box coordinates have less than 4 points.")

    # Running check that there are the same amount of bounding boxes and scores
    if len(boxes) != len(scores):
        raise ValueError("Len of box coordinates and scores do not match.")

    # If there are on bounding boxes
    if len(boxes) == 0:
        return []

    # Convert lists to tensors
    boxes = boxes.clone().detach() if isinstance(boxes, torch.Tensor) else torch.tensor(boxes, dtype=torch.float32)
    scores = scores.clone().detach() if isinstance(scores, torch.Tensor) else torch.tensor(scores, dtype=torch.float32)

    # Sort the boxes by confidence scores in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    keep = []

    while sorted_indices.numel() > 0:
        i = sorted_indices[0].item()
        keep.append(i)

        if sorted_indices.numel() == 1:
            break

        current_box = boxes[i].unsqueeze(0)
        remaining_indices = sorted_indices[1:]
        remaining_boxes = boxes[remaining_indices]

        ious = box_iou(current_box, remaining_boxes).squeeze(0)
        mask = ious < iou_threshold
        sorted_indices = remaining_indices[mask]

    return boxes[keep], scores[keep]


def soft_nms(boxes=[], scores=[], iou_threshold=0.5, sigma=0.5, score_threshold=0.001, method='linear'):
    """
    Perform Non-Maximum Suppression (NMS) on the list of bounding boxes and scores.
    This is performed under the assumption that all boxes and scores are from the same class.

    Args:
    - boxes: list of lists, numpy array, or tensor with the bounding box coordinates
    - scores: list of lists, numpy array, of tensor of scores associated with the bounding boxes
    - iou_threshold: IoU threshold for suppression
    - sigma: parameter for gaussian weighting
    - score_threshold: confidence score threshold to discard boxes
    - method: decaying score method ('linear' or 'gaussian')

    Returns:
    - indicies: list of lists with the bounding box coordinates after NMS
    """
    # If there are on bounding boxes
    if len(boxes) == 0:
        return torch.tensor([]), torch.tensor([])

    # Converting input to torch tensor
    boxes = boxes.clone().detach() if isinstance(boxes, torch.Tensor) else torch.tensor(boxes, dtype=torch.float32)
    scores = scores.clone().detach() if isinstance(scores, torch.Tensor) else torch.tensor(scores, dtype=torch.float32)

    # Sort scores in descending order and get the corresponding indices
    indices = torch.argsort(scores, descending=True)

    # Initialize the list of indices to keep
    keep = []

    while len(indices) > 0:
        # Select the box with the highest score
        i = indices[0]
        keep.append(i.item())

        # Compute IoU of this box with all remaining boxes
        ious = box_iou(boxes[i].unsqueeze(0), boxes[indices[1:]]).squeeze(0)

        # Decay the scores of overlapping boxes
        if method == 'linear':
            # Linear decay: score = score * (1 - iou) if iou > iou_threshold
            decay = torch.where(ious > iou_threshold, 1 - ious, torch.tensor(1.0, device=boxes.device))
        elif method == 'gaussian':
            # Gaussian decay: score = score * exp(-(iou^2) / sigma)
            decay = torch.exp(-(ious ** 2) / sigma)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Update the scores
        scores[indices[1:]] *= decay

        # Remove boxes with scores below the threshold
        remaining_indices = torch.where(scores[indices[1:]] >= score_threshold)[0]
        indices = indices[1:][remaining_indices]

    # Filter boxes and scores using the kept indices
    filtered_boxes = boxes[keep]
    filtered_scores = scores[keep]

    return filtered_boxes, filtered_scores


def dbsscan_clustering(boxes=[], scores=[], eps=10, min_samples=1):
    """
    Clustering bounding box coordinates using DBSCAN.

    Args:
    - boxes: list of lists, numpy array, or tensor with the bounding box coordinates
    - scores: list of lists, numpy array, of tensor of scores associated with the bounding boxes
    - eps: maximum distance between two samples for them to be considered in the same neighborhood
    - min_samples: minimum number of samples in a neighborhood for a point to be considered a core point
    - return_noise: return the unclustered indices. Default: false

    Returns:
    - clusters: list of clusters, where clusters contain indices of boxes
    - noise: list of noisy indices. Default: []
    """
    # If there are on bounding boxes
    if len(boxes) == 0:
        return torch.tensor([]), torch.tensor([])

    # Running check on bounding box coordinates
    for box in boxes:
        if len(box) != 4:
            raise TypeError("Missing bounding box coordinates. One or more of your bounding box coordinates have less than 4 points.")

    # Running check that there are the same amount of bounding boxes and scores
    if len(boxes) != len(scores):
        raise ValueError("Len of box coordinates and scores do not match.")

    # Convert lists to tensors
    boxes = boxes.clone().detach() if isinstance(boxes, torch.Tensor) else torch.tensor(boxes, dtype=torch.float32)
    scores = scores.clone().detach() if isinstance(scores, torch.Tensor) else torch.tensor(scores, dtype=torch.float32)

    # Convert boxes to centroids for clustering
    centroids = torch.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], dim=1).numpy()

    # Perform DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(centroids)
    labels = db.labels_

    # Group boxes into clusters
    clusters = []
    noise = []
    for label in set(labels):
        if label == -1:
            # Noise points
            noise.extend(np.where(labels == label)[0].tolist())
        else:
            # Valid clusters
            cluster_indices = np.where(labels == label)[0]
            clusters.append(cluster_indices.tolist())

    return clusters, noise


def fuse_clusters(boxes=[], scores=[], clusters=[], noise_indices=[]):
    """
    Fuse boxes within each cluster by averaging their coordinates and scores.

    Args:
    - boxes: list of lists with the bounding box coordinates
    - scores: list of scores associated with the bounding boxes
    - clusters: list of clusters, where each cluster contains indices of boxes
    - noise_indices: list of unclustered bounding boxes. Default: []

    Returns:
        fused_boxes: Fused bounding boxes.
        fused_scores: Fused confidence scores.
    """
    # If there are on bounding boxes
    if len(boxes) == 0:
        return torch.tensor([]), torch.tensor([])

    # Convert lists to tensors
    boxes = boxes.clone().detach() if isinstance(boxes, torch.Tensor) else torch.tensor(boxes, dtype=torch.float32)
    scores = scores.clone().detach() if isinstance(scores, torch.Tensor) else torch.tensor(scores, dtype=torch.float32)

    # Lists to hold boxes and scores
    fused_boxes = []
    fused_scores = []

    # Iterating through clusters
    for cluster in clusters:
        cluster_boxes = boxes[cluster]
        cluster_scores = scores[cluster]

        # Compute weighted average of boxes and scores
        weights = cluster_scores / (cluster_scores.sum() + 1e-8)
        fused_box = (cluster_boxes * weights.unsqueeze(1)).sum(dim=0)
        fused_score = cluster_scores.mean()

        # Appending to lists
        fused_boxes.append(fused_box)
        fused_scores.append(fused_score)

    # Add unclustered boxes (noise points) to the result, if provided
    if len(noise_indices) != 0:
        for idx in noise_indices:
            fused_boxes.append(boxes[idx])
            fused_scores.append(scores[idx])

    # Convert results to tensors
    fused_boxes = torch.stack(fused_boxes) if fused_boxes else torch.tensor([])
    fused_scores = torch.tensor(fused_scores) if fused_scores else torch.tensor([])

    return fused_boxes, fused_scores
