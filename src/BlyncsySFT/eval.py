import numpy as np
from collections import defaultdict
from pycocotools.cocoeval import COCOeval


def compute_mAP(ground_truths, predictions, iou_thresholds: list = []):
    """
    Calculating mAP for predicted boxes and ground truth labels.

    Args:
    - ground_truths: COCO ground truths file with bounding boxes in [xmin, ymin, width, height] format.
    - predictions: predictions in a list of dictionaries with bounding boxes in [xmin, ymin, width, height] format.
    - iou_threshoods: custom iou thresholds in list format.

    Returns:
    - mAP score for predicted boxes and ground truths
    """
    # Initializing COCOeval
    coco_dt = ground_truths.loadRes(predictions)
    coco_eval = COCOeval(ground_truths, coco_dt, 'bbox')

    # Running evaluation
    if len(iou_thresholds) > 0:
        coco_eval.params.iouThrs = iou_thresholds
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Returning All Stats
    return coco_eval.stats


def evaluate_custom_mAP(predictions, ground_truths, iou_threshold=0.5, iou_function: str = 'IoU'):
    """
    Calculating mAP for predicted boxes and ground truth labels using a custom iou_function.
    Implemented IoU Functions:
    1. IoU
    2. General Intersection Over Union (GIoU)
    3. Distance-IoU (DIoU)
    4. Complete-IoU (CIoU)
    5. Soft-IoU (SIoU)

    Args:
    - predictions: predictions in a list of dictionaries with bounding boxes in [xmin, ymin, width, height] format.
    - ground_truths: list of dictionaries with bounding boxes in [xmin, ymin, width, height] format
    - iou_threshold: the IoU threshold for making decisions for mAP (Default: 0.5)
    - iou_function: the custom IoU function for making decision for mAP (Default: IoU)

    Returns:
    - mAP score for predicted boxes and ground truths based on custom IoU Function
    """
    # Group predictions and ground truths by image_id
    predictions_grouped = group_by_image_id(predictions)
    ground_truths_grouped = group_by_image_id(ground_truths)

    # Creating list to hold average percisions
    aps = []

    # Iterating through ground truths
    for img_id in ground_truths_grouped.keys():
        # Getting image preds and ground truths
        img_predictions = predictions_grouped.get(img_id, [])
        img_ground_truths = ground_truths_grouped.get(img_id, [])

        # Computing average precision
        ap = compute_ap(img_predictions, img_ground_truths, iou_threshold, iou_function)
        aps.append(ap)

    # Returning average percision
    return np.mean(aps)


def evaluate_image_mAP(predictions, ground_truths, iou_threshold=0.5, iou_function: str = 'IoU'):
    """
    Calculating mAP for predicted boxes and ground truth labels using a custom iou_function.
    Implemented IoU Functions:
    1. IoU
    2. General Intersection Over Union (GIoU)
    3. Distance-IoU (DIoU)
    4. Complete-IoU (CIoU)
    5. Soft-IoU (SIoU)

    Args:
    - predictions: predictions in a list of dictionaries with bounding boxes in [xmin, ymin, width, height] format.
    - ground_truths: list of dictionaries with bounding boxes in [xmin, ymin, width, height] format
    - iou_threshold: the IoU threshold for making decisions for mAP (Default: 0.5)
    - iou_function: the custom IoU function for making decision for mAP (Default: IoU)

    Returns:
    - mAP score for each image
    """
    # Group predictions and ground truths by image_id
    predictions_grouped = group_by_image_id(predictions)
    ground_truths_grouped = group_by_image_id(ground_truths)

    # Creating list to hold average percisions
    aps = []

    # Iterating through ground truths
    for img_id in ground_truths_grouped.keys():
        # Getting image preds and ground truths
        img_predictions = predictions_grouped.get(img_id, [])
        img_ground_truths = ground_truths_grouped.get(img_id, [])

        # Computing average precision
        ap = compute_ap(img_predictions, img_ground_truths, iou_threshold, iou_function)
        aps.append({'image_id': img_id,
                    'ap': ap})

    # Returning image IoU scores
    return aps


def compute_ap(predictions, ground_truths, iou_threshold=0.5, iou_function: str = 'IoU'):
    """
    Calculating mAP for predicted boxes and ground truth labels using a custom iou_function.
    Implemented IoU Functions:
    1. IoU
    2. General Intersection Over Union (GIoU)
    3. Distance-IoU (DIoU)
    4. Complete-IoU (CIoU)
    5. Soft-IoU (SIoU)

    Args:
    - predictions: predictions in a list of dictionaries with bounding boxes in [xmin, ymin, width, height] format.
    - ground_truths: list of dictionaries with bounding boxes in [xmin, ymin, width, height] format
    - iou_threshold: the IoU threshold for making decisions for mAP (Default: 0.5)
    - iou_function: the custom IoU function for making decision for mAP (Default: IoU)

    Returns:
    - mAP score for predicted boxes and ground truths based on custom IoU Function
    """
    # Choosing custom function
    if iou_function.lower() == 'iou':
        calculate_iou = calculate_iou_reg
    elif iou_function.lower() == 'giou':
        calculate_iou = calculate_giou
    elif iou_function.lower() == 'diou':
        calculate_iou = calculate_diou
    elif iou_function.lower() == 'ciou':
        calculate_iou = calculate_ciou
    elif iou_function.lower() == 'siou':
        calculate_iou = calculate_soft_iou
    else:
        print("Specified IoU function now implemented. Completed implementations are:")
        print("1. General Intersection Over Union (GIoU) \n 2. Distance-IoU (DIoU) \n 3. Complete-IoU (CIoU) \n 4. Soft-IoU (SIoU)")
        return

    # Sort predictions by confidence score (descending)
    preds_sorted = sorted(predictions, key=lambda x: x["score"], reverse=True)

    # Creating list to hold true postives and false positives
    tp = []
    fp = []
    num_gts = len(ground_truths)

    # Iterating through sorted predictions
    for pred in preds_sorted:
        # Setting matched to false
        matched = False

        # Iterating through ground truths
        for gt in ground_truths:
            # Using custom iou function
            iou = calculate_iou(pred["bbox"], gt["bbox"])

            # Checking threshold and prediction category
            if iou >= iou_threshold and pred["category_id"] == gt["category_id"]:
                matched = True
                break

        # If matched
        if matched:
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    # Compute cumulative sums
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    # Compute precision and recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / num_gts

    # Compute AP using the precision-recall curve
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11

    return ap


def group_by_image_id(annotations):
    """
    Grouping bounding boxes by image
    """
    # Creating group
    grouped = defaultdict(list)

    # Iterating through annotations
    for ann in annotations:
        # Adding to group
        grouped[ann["image_id"]].append(ann)

    return grouped


def compute_precision_recall(matched_pairs, predictions, ground_truths):
    """
    Computes percision and recall given the matching pairs, the predictions, and ground_truths

    Args:
    - matched_pairs: matched ground truth and prediction boxes
    - predictions: list of predicted bounding boxes
    - ground_truths: list of ground truth boxes

    Returns:
    - precision: precision score
    - recall: recall score

    """
    # Calculating true positives, false positives, and false negatives
    tp = len(matched_pairs)
    fp = len(predictions) - tp
    fn = len(ground_truths) - tp

    # Calculating precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall


def calculate_iou_reg(box1, box2):
    """
    Calculates IoU with the provided bounding boxes.

    Args:
    - box1: the bounding box coordinates in a list: [xmin, ymin, width, height]
    - box2: the bounding box coordinates in a list: [xmin, ymin, width, height]

    Returns:
    - iou score for box1 and box2
    """
    # Reformatting into box coordinates
    box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

    # Calculating intersection coordinaties
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the area of intersection
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area != 0 else 0

    return iou


def calculate_giou(box1, box2):
    """
    Calculates General IoU with the provided bounding boxes.

    Args:
    - box1: the bounding box coordinates in a list: [xmin, ymin, width, height]
    - box2: the bounding box coordinates in a list: [xmin, ymin, width, height]

    Returns:
    - giou score for box1 and box2
    """
    # Reformatting into box coordinates
    box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

    # Calculating intersection coordinaties
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the area of intersection
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    # Calculate the area of the smallest enclosing box
    enclose_x1 = min(box1[0], box2[0])
    enclose_y1 = min(box1[1], box2[1])
    enclose_x2 = max(box1[2], box2[2])
    enclose_y2 = max(box1[3], box2[3])
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)

    # Calculate GIoU
    giou = iou - (enclose_area - union_area) / enclose_area

    return giou


def calculate_soft_iou(box1, box2, sigma=1.0):
    """
    Calculates Soft IoU with the provided bounding boxes.

    Args:
    - box1: the bounding box coordinates in a list: [xmin, ymin, width, height]
    - box2: the bounding box coordinates in a list: [xmin, ymin, width, height]

    Returns:
    - siou score for box1 and box2
    """
    # Reformatting into box coordinates
    box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

    # Calculating intersection coordinaties
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the area of intersection
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    # Calculate Soft-IoU
    soft_iou = np.exp(-(1 - iou)**2 / sigma**2)

    return soft_iou


def calculate_diou(box1, box2):
    """
    Calculates Distance IoU with the provided bounding boxes.

    Args:
    - box1: the bounding box coordinates in a list: [xmin, ymin, width, height]
    - box2: the bounding box coordinates in a list: [xmin, ymin, width, height]

    Returns:
    - diou score for box1 and box2
    """
    # Reformatting into box coordinates
    box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

    # Calculating intersection coordinaties
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the area of intersection
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    # Calculate the center distance
    center_distance = ((box1[0] + box1[2]) / 2 - (box2[0] + box2[2]) / 2)**2 + \
        ((box1[1] + box1[3]) / 2 - (box2[1] + box2[3]) / 2)**2

    # Calculate the diagonal length of the smallest enclosing box
    enclose_x1 = min(box1[0], box2[0])
    enclose_y1 = min(box1[1], box2[1])
    enclose_x2 = max(box1[2], box2[2])
    enclose_y2 = max(box1[3], box2[3])
    enclose_diagonal = (enclose_x2 - enclose_x1)**2 + (enclose_y2 - enclose_y1)**2

    # Calculate DIoU
    diou = iou - center_distance / enclose_diagonal

    return diou


def calculate_ciou(box1, box2):
    """
    Calculates Complete IoU with the provided bounding boxes.

    Args:
    - box1: the bounding box coordinates in a list: [xmin, ymin, width, height]
    - box2: the bounding box coordinates in a list: [xmin, ymin, width, height]

    Returns:
    - ciou score for box1 and box2
    """
    # Reformatting into box coordinates
    box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

    # Calculating intersection coordinaties
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the area of intersection
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    # Calculate the center distance
    center_distance = ((box1[0] + box1[2]) / 2 - (box2[0] + box2[2]) / 2)**2 + \
        ((box1[1] + box1[3]) / 2 - (box2[1] + box2[3]) / 2)**2

    # Calculate the diagonal length of the smallest enclosing box
    enclose_x1 = min(box1[0], box2[0])
    enclose_y1 = min(box1[1], box2[1])
    enclose_x2 = max(box1[2], box2[2])
    enclose_y2 = max(box1[3], box2[3])
    enclose_diagonal = (enclose_x2 - enclose_x1)**2 + (enclose_y2 - enclose_y1)**2

    # Calculate the aspect ratio penalty
    v = (4 / np.pi**2) * (np.arctan(box1[2] / box1[3]) - np.arctan(box2[2] / box2[3]))**2

    # Calculate CIoU
    alpha = v / (1 - iou + v)
    ciou = iou - (center_distance / enclose_diagonal + alpha * v)

    return ciou
