import torch


def reformat_predictions(predictions, format: str = 'coco'):
    """
    Reformats predictions from model. Current formatting functions:
    - coco
    - xyxy

    Args:
    - predictions: list of dictionaries that contain the predictions

    Returns:
    - formatted predictions for COCO eval
    """
    # Getting formatting function
    if format.lower() == 'coco':
        formatting_func = convert_to_coco
    elif format.lower() == 'xyxy':
        formatting_func = convert_to_xyxy
    else:
        print(f"Formatting function {format} not implemented. ")

    # Creating list to hold formatted predictions
    formatted_preds = []

    # Iterating through predictions
    for prediction in predictions:
        # Grabbing image_id, predicted bboxes, predicted scores, and labels
        image_id = prediction.get('image_id')
        bboxes = prediction.get('pred_boxes', prediction.get('bbox'))
        scores = prediction.get('pred_scores', prediction.get('score'))
        labels = prediction.get('pred_labels', prediction.get('category_id'))

        # Iterating through individual bboxes, scores, and labels
        for boxes, score, label in zip(bboxes, scores, labels):
            # Appending formatted dictionary to list
            formatted_preds.append({'image_id': image_id,
                                    'category_id': label,
                                    'bbox': formatting_func(boxes),
                                    'score': score})

    return formatted_preds


def convert_to_coco(bboxes):
    """
    Converts bounding boxes from the format [x_min, y_min, x_max, y_max]
    to the format [x_min, y_min, width, height].

    Args:
    - bboxes: A bounding box in the format [x_min, y_min, x_max, y_max].

    Returns:
    - A bounding box in the format [x_min, y_min, width, height].
    """
    return [bboxes[0], bboxes[1], bboxes[2] - bboxes[0], bboxes[3] - bboxes[1]]


def convert_to_xyxy(bboxes):
    """
    Converts ground truth bounding boxes from the format [x_min, y_min, width, height]
    to the format [x_min, y_min, x_max, y_max].

    Args:
    - bboxes: A bounding box in the format [x_min, y_min, width, height].

    Returns:
    - A bounding box in the format [x_min, y_min, x_max, y_max].
    """
    return [bboxes[0], bboxes[1], bboxes[2] + bboxes[0], bboxes[3] + bboxes[1]]


def format_target(target):
    """
    Format a list of object annotations for a single image into the target dictionary
    required by object detection models such as Faster R-CNN

    The input is a list where each element is a dictionary representing an object
    detected in the image. Each dictionary is expected to have at least the following keys:
      - 'bbox': List or tensor in [x, y, w, h] format, where (x, y) is the top-left
                corner and (w, h) are the width and height
      - 'category_id': Integer representing the object's class label
      - 'area': Numeric value representing the area of the object
      - 'image_id': Identifier for the image

    The function converts the bounding boxes from [x, y, w, h] format to
    [x1, y1, x2, y2] format (where x2 = x + w and y2 = y + h) and aggregates the
    labels and areas into tensors

    Args:
      target (list of dict): A list of object annotations for one image

    Returns:
      dict: A dictionary containing the following keys:
            - 'boxes': Tensor of shape (N, 4) containing bounding boxes
                       in [x1, y1, x2, y2] format, where N is the number of objects
            - 'labels': Tensor of shape (N,) containing the class labels
            - 'image_id': Tensor with a single element representing the image ID
            - 'area': Tensor of shape (N,) containing the areas of the objects
    """
    boxes = []
    labels = []
    areas = []

    for obj in target:
        # Extract the top-left coordinates (x1, y1)
        x1 = obj['bbox'][0]
        y1 = obj['bbox'][1]

        # Calculate bottom-right coordinates: x2 = x1 + width, y2 = y1 + height
        x2 = x1 + obj['bbox'][2]
        y2 = y1 + obj['bbox'][3]

        boxes.append([x1, y1, x2, y2])
        labels.append(obj['category_id'])
        areas.append(obj['area'])

    return {
        'boxes': torch.tensor(boxes, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.int64),
        'image_id': torch.tensor([target[0]['image_id']], dtype=torch.int64),
        'area': torch.tensor(areas, dtype=torch.float32)
    }
