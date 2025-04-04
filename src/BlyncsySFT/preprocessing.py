import numpy as np
import cv2 
from glob import glob
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
import math

def custom_bounding_boxes(gt_bbox_list, 
                          image_size: tuple = (1920, 1080), 
                          percentiles_scales = [10, 25, 50, 75, 90],
                          percentiles_ratios = [10, 50, 90])
    """
    Grabbing bounding boxes using percentiles.

    Args:
    - gt_bbox_list: list of dictionaries containing the image metadata and bounding box information in the 
                    [xmin, ymin, xmax, ymax] format.
    - image_size: tuple of image size
    - percentiles_scales: percentiles of scales to extract. Default: [10, 25, 50, 75, 90]
    - percentiles_ratios: percentiles of ratios to extract. Default: [10, 50, 90]

    Returns:
    - optimal anchor sizes
    - optmial aspect ratios
    """
    # Lists to hold stats
    widths = []
    heights = []
    areas = []
    aspect_ratios = []

    # Iterating through gt_bbox_list
    for gt in gt_bbox_list:
        # Calculating width and height
        width = gt[2] - gt[0]
        height = gt[3] - gt[1]

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

    ratios = np.round(np.percentile(aspect_ratios, percentiles_ratios), 1)
    print(f"Ratios: {ratios}")

    return {'anchor_sizes' : tuple([(item,) for item in scales]),
            'aspect_ratios' : tuple(ratios)}


def optimal_bounding_boxes(gt_bbox_list, image_size: tuple = (1920, 1080)):
    """
    Uses BayesianGaussianMixture and KMeans clustering to determine the optimal bounding boxes.

    Args:
    - gt_bbox_list: list of dictionaries containing the image metadata and bounding box information in the 
                    [xmin, ymin, xmax, ymax] format.
    - image_size: tuple of image size

    Returns:
    - optimal anchor sizes
    - optmial aspect ratios
    """
    # Reformatting bounding boxes
    reformatted = []

    # Iterating through bounding boxes
    for gt in gt_bbox_list:
        reformatted.append([gt[2] - gt[0], gt[3] - gt[1]])

    # Normalize box dimensions
    boxes = np.array(gt_bbox_list) / image_size

    # Fit BayesianGaussianMixture (set a high max k)
    bgm = BayesianGaussianMixture(n_components=15, random_state=42)  
    bgm.fit(boxes)

    # Optimal k = number of significant clusters (weights > threshold)
    optimal_k = np.sum(bgm.weights_ > 0.01)  # Adjust threshold as needed
    print("Optimal k (BayesianGaussianMixture):", optimal_k)

    # Now run K-Means with this k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(boxes)

    # Get cluster centroids (optimal anchor box sizes)
    anchor_boxes = kmeans.cluster_centers_ * image_size
    print("Optimal anchor boxes:", anchor_boxes)

    # Apply the function to each row
    bbox_anchor_sizes = np.apply_along_axis(calc_geometric_mean, 1, anchor_boxes).sort()
    bbox_aspect_ratios = np.apply_along_axis(calc_aspect_ratio, 1, anchor_boxes).sort()

    # Creating optimal aspect_ratios
    min_val = min(bbox_aspect_sizes)
    max_val = max(bbox_aspect_ratios)

    # Calculate first bin start
    first_bin = math.floor(min_val / bin_width) * bin_width

    # Generate all bin starts
    bin_starts = []
    current = first_bin
    while current <= max_val:
        bin_starts.append(current)
        current += bin_width

    print(f"Anchor Sizes: {bbox_anchor_sizes}")
    print(f"Ratios: {bin_starts}")

    return {'anchor_sizes' : tuple([(item,) for item in bbox_anchor_sizes]),
            'aspect_ratios' : tuple(bin_starts)}


def custom_normalization(image_paths)
    """
    Calculating the custom normalization values for image normalization.

    Args:
    - image_paths: path to image directory

    Returns:
    - mean of 3 channels
    - std of 3 channels
    """
    # Grabbing files using glob
    paths = glob((Path(image_paths) / "*.jpg"))

    # Calculate 5% of the total number of images
    sample_size = int(0.05 * len(paths))

    # Randomly sample without replacement
    sampled_paths = np.random.choice(len(paths), sample_size, replace = False)

    # Temp variable to holding pixel values
    pixel_values = []

    # Initialize variables for running stats
    mean = np.zeros(3)
    std = np.zeros(3)
    total_pixels = 0

    # Iterating through dataset
    for index in tqdm(sampled_paths):
        # Read image (returns None if corrupted)
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

#-----------------------------------------------------------------------------------
#--- Helper Functions
#-----------------------------------------------------------------------------------

def calc_geometric_mean(row):
    return np.sqrt(row[0] * row[1])

def calc_aspect_ratio(row):
    return row[0]/row[1]