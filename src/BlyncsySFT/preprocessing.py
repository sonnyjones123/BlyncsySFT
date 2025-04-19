import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
from pathlib import Path
import json


def custom_bounding_boxes(image_paths,
                          image_size: tuple = (1920, 1080),
                          percentiles_scales=[10, 25, 50, 75, 90],
                          percentiles_ratios=[10, 50, 90],
                          plot = False):
    """
    Grabbing bounding boxes using percentiles.

    Args:
    - image_paths: path to image directory
    - image_size: tuple of image size
    - percentiles_scales: percentiles of scales to extract. Default: [10, 25, 50, 75, 90]
    - percentiles_ratios: percentiles of ratios to extract. Default: [10, 50, 90]

    Returns:
    - anchor sizes
    - aspect ratios
    """
    # Opening and loading image paths
    with open(image_paths, 'r') as f:
        train_dataset = json.load(f)

    # Converting ground truths to width and height scale
    gt_bbox_list = [[item['bbox'][2], item['bbox'][3]] for item in train_dataset['annotations']]

    # Lists to hold stats
    widths = []
    heights = []
    areas = []
    aspect_ratios = []

    # Iterating through gt_bbox_list
    for gt in gt_bbox_list:
        # Calculating width and height
        width = gt[0]
        height = gt[1]

        # Appending to lists
        widths.append(width)
        heights.append(height)
        areas.append(width * height)
        aspect_ratios.append(width / height)

    # Compute statistics
    print("Average width:", np.mean(widths))
    print("Average height:", np.mean(heights))
    print("Average area:", np.mean(areas))
    print("Median aspect ratio:", np.median(aspect_ratios))

    # Plotting
    if plot:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.hist(areas, bins=50)
        plt.title("Object Areas")

        plt.subplot(1, 2, 2)
        plt.hist(aspect_ratios, bins=50)
        plt.title("Aspect Ratios (width/height)")
        plt.show()

    # Recommending anchors based on percentiles
    area_percentiles = np.round(np.percentile(areas, percentiles_scales))
    scales = np.sqrt(area_percentiles)
    print(f"Anchor Sizes: {scales}")

    # Recommending aspect ratios based on percentiles
    ratios = np.round(np.percentile(aspect_ratios, percentiles_ratios), 1)

    # Round down the minimum and round up the maximum to nearest 0.5
    min_val = np.floor(min(ratios) / 0.5) * 0.5
    max_val = np.ceil(max(ratios) / 0.5) * 0.5

    # Generate the list of values
    degraded_ratios = np.arange(min_val, max_val + 0.5, 0.5)
    print(f"Ratios: {degraded_ratios}")

    return {'anchor_sizes': tuple([(item,) for item in scales]),
            'aspect_ratios': tuple(degraded_ratios)}


def custom_normalization(image_paths):
    """
    Calculating the custom normalization values for image normalization.

    Args:
    - image_paths: path to image directory

    Returns:
    - mean of 3 channels
    - std of 3 channels
    """
    # Grabbing files using glob
    paths = glob(str(Path(image_paths) / "*.jpg"))

    if len(paths) == 0:
        raise ValueError(f"No images found in path: {image_paths}")

    # Calculate 5% of the total number of images
    sample_size = int(0.05 * len(paths))

    # Randomly sample without replacement
    sampled_paths = np.random.choice(len(paths), sample_size, replace=False)

    # Temp variable to holding pixel values
    # pixel_values = []

    # Initialize variables for running stats
    mean = np.zeros(3)
    std = np.zeros(3)
    total_pixels = 0

    # Iterating through dataset
    for index in sampled_paths:
        # Read image (returns None if corrupted)
        path = paths[index]
        img = cv2.imread(paths[index])

        if img is None:
            print(f"Skipping corrupted image: {path}")
            continue

        # Convert BGR to RGB and normalize to [0, 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        # Update running mean and std
        batch_pixels = img.shape[0] * img.shape[1]  # H * W
        batch_mean = np.mean(img, axis=(0, 1))      # Mean per channel (R, G, B)
        batch_var = np.var(img, axis=(0, 1))        # Variance per channel

        # Update global mean and variance incrementally
        new_total = total_pixels + batch_pixels
        mean = (mean * total_pixels + batch_mean * batch_pixels) / new_total
        std = (std * total_pixels + batch_var * batch_pixels) / new_total  # Running variance
        total_pixels = new_total

    # Final std = sqrt(running variance)
    std = np.sqrt(std)

    return mean.tolist(), std.tolist()

# -----------------------------------------------------------------------------------
# --- Helper Functions
# -----------------------------------------------------------------------------------


def calc_geometric_mean(row):
    return np.sqrt(row[0] * row[1])


def calc_aspect_ratio(row):
    return row[0]/row[1]
