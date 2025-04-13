import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchvision.ops import box_iou
import torchvision.transforms as T
from torchvision.transforms import ToPILImage
import time
from torch.utils.data import DataLoader
from torchvision import datasets

# Check if a CUDA-enabled GPU is available; otherwise, default to using the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score_images(model, data_loader, alpha=0.7):
    """
    Computes the scores for the provided dataset using a datalodaer. This functions uses two metrics for
    evaluating the overlap between the generated bbox from saliency and the ground truth:
        1. Box IoU overlap
        2. Normalized distance between the boxes
    The alpha parameter balances the combined scores of the two, with higher levels of alpha prioritizing
    Box IoU.

    Args:
        model: The current model to evaluate with
        data_loader: The evaluation dataset, in a dataloder with batchsize 1. Can be returned with the function
            saliency_dataloader().
        alpha (0.7): The weighting parameter between scoring metrics

    Returns:
        list of dictionaries of scored images
    """
    # Set the model to evaluation mode for inference
    model.to(device)
    model.eval()

    total_batches = len(data_loader)  # Total number of batches in validation set
    score_list = []

    start_time = time.time()  # Start tracking validation time

    for batch_idx, batch in enumerate(data_loader):
        # batch is a single item due to batch_size=1 and collate_fn
        image, target = batch

        # Apply transformation and move to device
        input_tensor = image.unsqueeze(0).to(device)  # Add batch dimension
        input_tensor.requires_grad_(True)  # Enable gradient computation for the input

        # Perform inference
        outputs = model(input_tensor)

        # Get the score for the "damage" class (class index 1)
        # For Faster R-CNN, the outputs are dictionaries containing 'boxes', 'labels', and 'scores'
        # We use the highest scoring box's class score for saliency
        scores = outputs[0]['scores']

        # Checking scores
        if scores.numel() == 0:
            print(f"Image {target[0]['image_id']} has no predicted boxes")
            continue

        max_score_index = torch.argmax(scores).item()
        max_score = scores[max_score_index]

        # Compute gradients of the max score with respect to the input image
        model.zero_grad()
        max_score.backward()

        # Get the gradients of the input tensor
        gradients = input_tensor.grad[0].cpu().numpy()

        # Compute the saliency map by taking the maximum absolute value of the gradients across channels
        saliency_map = np.max(np.abs(gradients), axis=0)

        # Normalize the saliency map for visualization
        saliency_min = saliency_map.min()
        saliency_max = saliency_map.max()
        if saliency_max != saliency_min:
            saliency_map = (saliency_map - saliency_min) / (saliency_max - saliency_min)
        else:
            saliency_map = np.zeros_like(saliency_map)

        # Calculating bbox
        pred_bbox = torch.tensor([expected_bbox(saliency_map)], dtype=torch.float).to(device)
        gt_bbox = torch.tensor([reformat_target(target[0]['bbox'])], dtype=torch.float).to(device)

        # Calculating overlap metrics
        iou_scores = box_iou(pred_bbox, gt_bbox)  # Shape: [1, N]
        dist_scores = normalized_center_distance_torch(pred_bbox, gt_bbox)

        # Combining scores
        combined_scores = alpha * iou_scores + (1 - alpha) * dist_scores

        # Appending to score list
        score_list.append({
            'image_id': target[0]['image_id'],
            'index': batch_idx,
            'score': combined_scores.cpu().item()})

        # Print progress every 50 batches
        if batch_idx % 500 == 0:
            elapsed_time = time.time() - start_time  # Compute elapsed time
            print(f"Processed {batch_idx + 1}/{total_batches} batches - Elapsed time: {elapsed_time:.2f}s")

    total_elapsed_time = time.time() - start_time  # Compute total validation time
    print(f"Validation completed in {total_elapsed_time:.2f} seconds")

    return score_list


def plot_saliency_bbox(model, image, target, save_path=None):
    """
    Plots the current image, the generated bounding box from the saliency_map, and the target that
    contains the original bounding box.

    Args:
        model: The current model to plot
        image: The current image from the dataset to analyze
        target: Targets from dataset
        save_path: Which file path to save the image to
    """
    # Load the image for visualization
    to_pil = ToPILImage()
    image_rgb = to_pil(image)

    # Apply transformation and move to device
    input_tensor = image.unsqueeze(0).to(device)  # Add batch dimension
    input_tensor.requires_grad_(True)  # Enable gradient computation for the input

    # Perform inference
    outputs = model(input_tensor)

    # Get the score for the "damage" class (class index 1)
    # For Faster R-CNN, the outputs are dictionaries containing 'boxes', 'labels', and 'scores'
    # We use the highest scoring box's class score for saliency
    scores = outputs[0]['scores']
    max_score_index = torch.argmax(scores).item()
    max_score = scores[max_score_index]

    # Compute gradients of the max score with respect to the input image
    model.zero_grad()
    max_score.backward()

    # Get the gradients of the input tensor
    gradients = input_tensor.grad[0].cpu().numpy()

    # Compute the saliency map by taking the maximum absolute value of the gradients across channels
    saliency_map = np.max(np.abs(gradients), axis=0)

    # Normalize the saliency map for visualization
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

    # Visualize the saliency map
    visualize_saliency_and_bbox(image_rgb, saliency_map, target, save_path=None)


def expected_bbox(saliency_map, k=1):
    """
    Compute expected bounding box from a saliency map.

    Args:
        saliency_map: 2D array, non-negative saliency values.
        k: Multiplier for standard deviation (default=1).

    Returns:
        bbox: (x_min, y_min, x_max, y_max)
    """
    total_saliency = np.sum(saliency_map)
    if total_saliency == 0:
        # Return default center box or full image box if no saliency is detected
        h, w = saliency_map.shape
        return (w//4, h//4, 3*w//4, 3*h//4)

    # Normalize saliency to a probability distribution
    p = saliency_map / np.sum(saliency_map)

    # Compute weighted mean (center of mass)
    h, w = saliency_map.shape
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    x_mean = np.sum(x_grid * p)
    y_mean = np.sum(y_grid * p)

    # Compute weighted standard deviation (spread)
    x_std = np.sqrt(np.sum((x_grid - x_mean)**2 * p))
    y_std = np.sqrt(np.sum((y_grid - y_mean)**2 * p))

    # Define bounding box
    x_min = max(0, int(x_mean - k * x_std))
    x_max = min(w, int(x_mean + k * x_std))
    y_min = max(0, int(y_mean - k * y_std))
    y_max = min(h, int(y_mean + k * y_std))

    return (x_min, y_min, x_max, y_max)


def visualize_saliency_and_bbox(image, saliency_map, target, save_path=None):
    """
    Plots the current image, the generated bounding box from the saliency_map, and the target that
    contains the original bounding box.

    Args:
        image: Input image from dataset
        saliency_map: The normalized saliency map
        target: Targets from dataset
        save_path: Which file path to save the image to
    """
    # Creating figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 12))

    # Getting ground truth bounding box
    gt_bbox = target[0]['bbox']
    image_id = target[0]['image_id']

    # Plot the original image
    axes[0].imshow(image)
    axes[0].set_title(f'Original Image : ID{image_id}')
    axes[0].axis('off')

    # Drawing ground truth bounding box
    x_min, y_min, width, height = gt_bbox
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2,
                             edgecolor='r', facecolor='none', label='Ground Truth')
    axes[0].add_patch(rect)

    # Calculating bbox
    bbox = expected_bbox(saliency_map)

    # Draw bounding box
    x_min, y_min, x_max, y_max = bbox
    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2,
                             edgecolor='g', facecolor='none', label='Generate')
    axes[0].add_patch(rect)

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys())

    # Plot the saliency map
    axes[1].imshow(saliency_map, cmap='hot')
    axes[1].set_title('Saliency Map')
    axes[1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def normalized_center_distance_torch(pred_box, gt_boxes, image_width=1920, image_height=1080):
    """
    Calculates the normalized center distances between the predicted bounding boxes and the
    ground truth bounding boxes.

    Args:
        pred_box(tensor): predicted bounding boxes
        gt_boxes(tesnor): ground truth bounding boxes
        image_width: width of image in pixels
        image_height: height of image in pixels

    Returns:
        normalized distance metric (0 - 1)
    """
    # Compute centers
    pred_center = (pred_box[:, :2] + pred_box[:, 2:]) / 2  # Shape: [1, 2]
    gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2   # Shape: [N, 2]

    # Euclidean distance
    distances = torch.sqrt(((pred_center - gt_centers) ** 2).sum(dim=1))  # Shape: [N]

    # Normalize by image diagonal
    image_diagonal = torch.sqrt(torch.tensor(image_width ** 2 + image_height ** 2))
    return 1 - torch.clamp(distances / image_diagonal, max=1.0)  # Shape: [N]

# -----------------------------------------------------------------------------------
# --- Helper Functions
# -----------------------------------------------------------------------------------


def reformat_target(gt_bbox):
    return [gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]]


def saliency_dataloader(image_folder, annotation_file, num_workers=4):
    """
    Returns a correctly formatted PyTorch DataLoader for the saliency scoring.

    Args:
        image_folder: Path to image folder
        annotation_file: Path to annotation file
        num_workers: Number of workers for data loading

    Returns:
        DataLoader
    """
    # Create a CocoDetection dataset
    dataset = datasets.CocoDetection(root=image_folder, annFile=annotation_file, transform=T.ToTensor())

    # Configure DataLoader with multiple workers
    data_loader = DataLoader(
        dataset,
        batch_size=1,                           # Process one image at a time
        shuffle=False,                          # No need to shuffle for validation
        num_workers=num_workers,                # Number of parallel workers (adjust based on CPU cores)
        pin_memory=torch.cuda.is_available(),   # Faster data transfer to GPU (if available)
        collate_fn=lambda x: x[0]               # Handle batch_size=1 explicitly
    )

    return data_loader
