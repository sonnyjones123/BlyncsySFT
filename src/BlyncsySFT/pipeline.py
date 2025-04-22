import time
import json
import logging
from tqdm import tqdm
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from torch.optim.lr_scheduler import StepLR
from BlyncsySFT.models import custom_faster_rcnn
from BlyncsySFT.format import format_target, reformat_predictions
from BlyncsySFT.eval import compute_mAP
from pathlib import Path
from BlyncsySFT.augment import createAugmentedData, augmentation_types


def run_auto_training_pipeline(project_dir, cfg, verbose):
    """
    Runs fine-tuning pipeline for CLI interface for the Faster R-CNN with FPN using focal loss and automatic
    dataset configuration. Returns checkpointed model variables and a final model variable. Additionally runs
    metrics on a validation set.

    Args:
        project_dir: The project directory where the training and validation folders are
        cfg: The env variables loaded with dotenv.load_dotenv
        verbose: Printing variable
    """
    # -----------------------------------------------------------------------------------
    # --- Configuring training pipeline
    # -----------------------------------------------------------------------------------

    # Grabbing training parameters from cfg
    training_run = cfg['TRAINING_RUN']
    num_epochs = int(cfg['EPOCHS'])
    batch_size = int(cfg['BATCH_SIZE'])
    workers = int(cfg['WORKERS'])
    num_classes = int(cfg['NUM_CLASSES'])
    backbone = cfg['BACKBONE']
    save_every = int(cfg['SAVE_EVERY'])

    # Default training parameters
    learning_rate = cfg.get('LEARNING_RATE', 1e-3)
    momentum = cfg.get('MOMENTUM', 0.95)
    weight_decay = cfg.get('WEIGHT_DECAY', 1e-4)
    alpha = cfg.get('ALPHA', 0.25)
    gamma = cfg.get('GAMMA', 2.0)

    # Augmentation Parameters
    use_augmentations = cfg.get("USE_AUGMENTATIONS", False)
    shuffle_dataset = cfg.get('SHUFFLE_DATASET', True)
    augmentation_prob = cfg.get('AUGMENTATION_PROB', 0.5)
    subset_size = cfg.get('SUBSET_SIZE', None)
    subset_size = int(subset_size) if subset_size is not None else None
    mixup_lambda = cfg.get('MIXUP_LAMBDA', 0.5)
    random_seed = cfg.get('RANDOM_SEED', None)
    set_random_seed = cfg.get('SET_RANDOM_SEED', False)

    # Custom Transforms
    norm_mean = cfg.get('IMAGE_MEAN', [0.485, 0.456, 0.406])
    norm_std = cfg.get('IMAGE_STD', [0.229, 0.224, 0.225])

    # Paths
    train_image_path = (Path(project_dir) / cfg.get('TRAIN_IMAGE_PATH', "images/train"))
    train_annot_path = (Path(project_dir) / cfg.get('TRAIN_ANNOT_PATH', "annotations/train.json"))

    # Optional
    # state_dict_path = cfg.get('STATE_DICT_PATH', None)

    # Grabbing model with params
    try:
        model = custom_faster_rcnn(backbone_name=backbone,
                                   num_classes=num_classes,
                                   focal_loss_args={'alpha': alpha, 'gamma': gamma},
                                   state_dict_path=None)

    except Exception as e:
        print('Unable to load model:')
        print(e)
        return

    # Check if a CUDA-enabled GPU is available; otherwise, default to using the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define transformations (apply them to the image)
    transform = transforms.Compose([
                transforms.ToTensor(),  # Convert images to tensor
                transforms.Normalize(mean=norm_mean, std=norm_std),
                # Add more transformations like resizing, normalization, etc.
    ])

    # Create the custom augmentation dataset
    if use_augmentations:
        train_dataset = createAugmentedData(
            root=train_image_path,
            annotation=train_annot_path,
            augmentation_types=augmentation_types,  # the list you defined
            seed=random_seed,
            set_random_seed=set_random_seed,
            transforms=None,  # uses default: ToTensorV2
            shuffle_dataset=shuffle_dataset,
            aug_probability=augmentation_prob,
            subset_size=subset_size,  # or None for full dataset
            mixup_lambda=mixup_lambda
        )
    # Creating regular dataset
    else:
        train_dataset = datasets.CocoDetection(
            root=train_image_path,
            annFile=train_annot_path,
            transform=transform
        )

    # Create a DataLoader for batching the dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=workers,
        pin_memory=torch.cuda.is_available()
    )

    # -----------------------------------------------------------------------------------
    # --- Training pipeine
    # -----------------------------------------------------------------------------------

    # Set up logging to log both to a file and the console
    log_path = (Path(project_dir) / "logging" / f"model{training_run}_log.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_path,  # Log file name
        level=logging.INFO,  # Log level
        format='%(asctime)s - %(message)s',  # Log format
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Also log to console
    if verbose:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger().addHandler(console)

    # Logging information to training log
    logging.info("===== Training Configuration =====")
    logging.info(f"Training run: model{training_run}")
    logging.info("Loss function: focal loss")
    logging.info(f"Alpha: {alpha}")
    logging.info(f"Gamma: {gamma}")
    logging.info(f"Learning rate: {learning_rate}")
    logging.info(f"Momentum: {momentum}")
    logging.info(f"Weight decay: {weight_decay}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Epochs: {num_epochs}")
    logging.info(f"Device: {device}")
    logging.info("===================================")
    logging.info("")
    logging.info("======== Transform Params =========")
    logging.info(f"Mean: {norm_mean}, std: {norm_std}")
    logging.info("===================================")

    # Set up optimizer using SGD and specified learning rate, momentum, and weight decay
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    # Record the start time to track overall training duration
    start_time = time.time()

    # Main training loop
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()

        # Initialize accumlative loss
        total_loss = 0

        num_batches = len(train_loader)  # Total number of batches

        # Wrap training data loader with tqdm to display a progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        # Iterate over each batch provided by the train_loader
        for batch_idx, (images, targets) in enumerate(progress_bar):
            # Move each image tensor in the batch to the device (GPU/CPU)
            images = [image.to(device) for image in images]

            # If Using Augmentations
            if use_augmentations:
                formatted_targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            else:
                # Formatting targets for each image
                formatted_targets = [format_target(t) for t in targets]

                # Move each element in the formatted targets to the correct device
                formatted_targets = [{k: v.to(device) for k, v in t.items()} for t in formatted_targets]

            # Forward pass: compute the loss dictionary by passing images and formatted targets to the model
            loss_dict = model(images, formatted_targets)

            # Sum all the losses in the dictionary, ignoring any None values
            loss = sum(loss for loss in loss_dict.values() if loss is not None)

            # Zero out the gradients before backpropagation
            optimizer.zero_grad()

            # Backward pass: compute gradients with respect to the loss
            loss.backward()

            # Update model parameters using the optimizer
            optimizer.step()

            # Accumulate the loss for this batch into the total loss for the epoch
            total_loss += loss.item()

            # Update the progress bar to display the current loss value
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Compute average loss for the epoch
        avg_loss = total_loss / num_batches

        # Compute the elapsed time since training started
        elapsed = time.time() - start_time

        # Stepping scheduler
        scheduler.step()

        # Print the epoch number, total loss for the epoch, and elapsed time
        # Log and print the epoch summary
        epoch_log_msg = f"Epoch [{epoch+1}/{num_epochs}] Total Loss: {total_loss:.4f}, Avg Loss: {avg_loss:.4f} - Time: {elapsed:.2f} seconds"
        logging.info(epoch_log_msg)

        # **Save model checkpoint**
        if (epoch + 1) % save_every == 0:
            (Path(project_dir) / f"model{training_run}_log.txt")
            checkpoint_path = (Path(project_dir) / "models" / f"model{training_run}_epoch{epoch+1}.pth")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss
            }, checkpoint_path)

    # Save only the state dictionary
    torch.save(model.state_dict(), (Path(project_dir) / "models" / f"model{training_run}.pth"))

    # -----------------------------------------------------------------------------------
    # --- Validation scoring
    # -----------------------------------------------------------------------------------
    # Running validation pipeline
    run_validation_pipeline(model, project_dir, cfg, verbose)


def run_training_pipeline(model, project_dir, cfg, verbose=True):
    """
    Runs fine-tuning pipeline for the inputted model and automatic
    dataset configuration. Returns checkpointed model variables and a final model variable.

    Args:
        model: Provided pytorch model
        project_dir: The project directory where the training and validation folders are
        cfg: The env variables loaded with dotenv.load_dotenv
        verbose: Printing variable

    Returns:
        trained model
    """

    # -----------------------------------------------------------------------------------
    # --- Configuring training pipeline
    # -----------------------------------------------------------------------------------

    # Grabbing training parameters from cfg
    training_run = cfg['TRAINING_RUN']
    num_epochs = int(cfg['EPOCHS'])
    batch_size = int(cfg['BATCH_SIZE'])
    workers = int(cfg['WORKERS'])
    save_every = int(cfg['SAVE_EVERY'])

    # Default training parameters
    learning_rate = cfg.get('LEARNING_RATE', 1e-3)
    momentum = cfg.get('MOMENTUM', 0.95)
    weight_decay = cfg.get('WEIGHT_DECAY', 1e-4)
    alpha = cfg.get('ALPHA', 0.25)
    gamma = cfg.get('GAMMA', 2.0)

    # Augmentation Parameters
    use_augmentations = cfg.get("USE_AUGMENTATIONS", False)
    shuffle_dataset = cfg.get('SHUFFLE_DATASET', True)
    augmentation_prob = cfg.get('AUGMENTATION_PROB', 0.5)
    subset_size = cfg.get('SUBSET_SIZE', None)
    subset_size = int(subset_size) if subset_size is not None else None
    mixup_lambda = cfg.get('MIXUP_LAMBDA', 0.5)
    random_seed = cfg.get('RANDOM_SEED', None)
    set_random_seed = cfg.get('SET_RANDOM_SEED', False)

    # Custom Transforms
    norm_mean = cfg.get('IMAGE_MEAN', [0.485, 0.456, 0.406])
    norm_std = cfg.get('IMAGE_STD', [0.229, 0.224, 0.225])

    # Paths
    train_image_path = (Path(project_dir) / cfg.get('TRAIN_IMAGE_PATH', "images/train"))
    train_annot_path = (Path(project_dir) / cfg.get('TRAIN_ANNOT_PATH', "annotations/train.json"))

    # Check if a CUDA-enabled GPU is available; otherwise, default to using the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define transformations (apply them to the image)
    transform = transforms.Compose([
                transforms.ToTensor(),  # Convert images to tensor
                transforms.Normalize(mean=norm_mean, std=norm_std),
                # Add more transformations like resizing, normalization, etc.
    ])

    # Create the custom augmentation dataset
    if use_augmentations:
        train_dataset = createAugmentedData(
            root=train_image_path,
            annotation=train_annot_path,
            augmentation_types=augmentation_types,  # the list you defined
            seed=random_seed,
            set_random_seed=set_random_seed,
            transforms=None,  # uses default: ToTensorV2
            shuffle_dataset=shuffle_dataset,
            aug_probability=augmentation_prob,
            subset_size=subset_size,  # or None for full dataset
            mixup_lambda=mixup_lambda
        )
    # Creating regular dataset
    else:
        train_dataset = datasets.CocoDetection(
            root=train_image_path,
            annFile=train_annot_path,
            transform=transform
        )

    # Create a DataLoader for batching the dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=workers,
        pin_memory=torch.cuda.is_available()
    )

    # -----------------------------------------------------------------------------------
    # --- Training pipeine
    # -----------------------------------------------------------------------------------

    # Set up logging to log both to a file and the console
    log_path = (Path(project_dir) / "logging" / f"model{training_run}_log.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_path,  # Log file name
        level=logging.INFO,  # Log level
        format='%(asctime)s - %(message)s',  # Log format
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Also log to console
    if verbose:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger().addHandler(console)

    # Logging information to training log
    logging.info("===== Training Configuration =====")
    logging.info(f"Training run: model{training_run}")
    logging.info("Loss function: focal loss")
    logging.info(f"Alpha: {alpha}")
    logging.info(f"Gamma: {gamma}")
    logging.info(f"Learning rate: {learning_rate}")
    logging.info(f"Momentum: {momentum}")
    logging.info(f"Weight decay: {weight_decay}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Epochs: {num_epochs}")
    logging.info(f"Device: {device}")
    logging.info("===================================")
    logging.info("")
    logging.info("======== Transform Params =========")
    logging.info(f"Mean: {norm_mean}, std: {norm_std}")
    logging.info("===================================")

    # Set up optimizer using SGD and specified learning rate, momentum, and weight decay
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    # Record the start time to track overall training duration
    start_time = time.time()

    # Main training loop
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()

        # Initialize accumlative loss
        total_loss = 0

        num_batches = len(train_loader)  # Total number of batches

        # Wrap training data loader with tqdm to display a progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        # Iterate over each batch provided by the train_loader
        for batch_idx, (images, targets) in enumerate(progress_bar):
            # Move each image tensor in the batch to the device (GPU/CPU)
            images = [image.to(device) for image in images]

            # If Using Augmentations
            if use_augmentations:
                formatted_targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            else:
                # Formatting targets for each image
                formatted_targets = [format_target(t) for t in targets]

                # Move each element in the formatted targets to the correct device
                formatted_targets = [{k: v.to(device) for k, v in t.items()} for t in formatted_targets]

            # Forward pass: compute the loss dictionary by passing images and formatted targets to the model
            loss_dict = model(images, formatted_targets)

            # Sum all the losses in the dictionary, ignoring any None values
            loss = sum(loss for loss in loss_dict.values() if loss is not None)

            # Zero out the gradients before backpropagation
            optimizer.zero_grad()

            # Backward pass: compute gradients with respect to the loss
            loss.backward()

            # Update model parameters using the optimizer
            optimizer.step()

            # Accumulate the loss for this batch into the total loss for the epoch
            total_loss += loss.item()

            # Update the progress bar to display the current loss value
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Compute average loss for the epoch
        avg_loss = total_loss / num_batches

        # Compute the elapsed time since training started
        elapsed = time.time() - start_time

        # Stepping scheduler
        scheduler.step()

        # Print the epoch number, total loss for the epoch, and elapsed time
        # Log and print the epoch summary
        epoch_log_msg = f"Epoch [{epoch+1}/{num_epochs}] Total Loss: {total_loss:.4f}, Avg Loss: {avg_loss:.4f} - Time: {elapsed:.2f} seconds"
        logging.info(epoch_log_msg)

        # **Save model checkpoint**
        if (epoch + 1) % save_every == 0:
            (Path(project_dir) / f"model{training_run}_log.txt")
            checkpoint_path = (Path(project_dir) / "models" / f"model{training_run}_epoch{epoch+1}.pth")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss
            }, checkpoint_path)

    # Save only the state dictionary
    torch.save(model.state_dict(), (Path(project_dir) / "models" / f"model{training_run}.pth"))

    return model


def run_validation_pipeline(model, project_dir, cfg, verbose=True):
    """
    Runs the validation loop for an object detection model and collects predictions.

    Args:
        model: The object detection model to evaluate.
        project_dir: The project directory where the training and validation folders are
        cfg: The env variables loaded with dotenv.load_dotenv
        verbose: If outputs are wanted.

    Returns:
        predictions in xyxy format
    """
    # Grabbing training parameters from cfg
    training_run = cfg['TRAINING_RUN']
    batch_size = int(cfg['BATCH_SIZE'])
    workers = int(cfg['WORKERS'])

    # Custom Transforms
    norm_mean = cfg.get('IMAGE_MEAN', [0.485, 0.456, 0.406])
    norm_std = cfg.get('IMAGE_STD', [0.229, 0.224, 0.225])

    # Optional
    # state_dict_path = cfg.get('STATE_DICT_PATH', None)

    # Check if a CUDA-enabled GPU is available; otherwise, default to using the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Paths
    val_image_path = (Path(project_dir) / cfg.get('VAL_IMAGE_PATH', "images/validation"))
    val_annot_path = (Path(project_dir) / cfg.get('VAL_ANNOT_PATH', "annotations/validation.json"))

    # Define transformations (apply them to the image)
    transform = transforms.Compose([
                transforms.ToTensor(),  # Convert images to tensor
                transforms.Normalize(mean=norm_mean, std=norm_std),
                # Add more transformations like resizing, normalization, etc.
    ])

    # Create a CocoDetection dataset
    val_dataset = datasets.CocoDetection(root=val_image_path, annFile=val_annot_path, transform=transform)

    # Create a DataLoader for batching the dataset
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=workers,
        pin_memory=torch.cuda.is_available()
    )

    model.eval()  # Set model to evaluation mode (disables dropout, batch norm updates)
    all_boxes = []  # Stores predicted bounding boxes
    all_labels = []  # Stores predicted class labels
    all_scores = []  # Stores confidence scores for each prediction
    all_image_ids = []  # Stores image ids

    start_time = time.time()  # Start tracking validation time

    with torch.no_grad():  # Disable gradient computation for efficiency
        total_batches = len(val_loader)  # Total number of batches in validation set

        for batch_idx, (images, targets) in enumerate(val_loader):
            images = [image.to(device) for image in images]  # Move images to the specified device (CPU/GPU)
            batch_ids = [t[0]['image_id'] for t in targets]  # Grabbing image ids from targets
            all_image_ids.extend(batch_ids)

            # Make predictions
            predictions = model(images)

            # Store results for each prediction in the batch
            for prediction in predictions:
                all_boxes.append(prediction['boxes'].cpu().numpy().tolist())  # Store bounding boxes
                all_labels.append(prediction['labels'].cpu().numpy().tolist())  # Store class labels
                all_scores.append(prediction['scores'].cpu().numpy().tolist())  # Store confidence scores

            # Print progress every 10 batches
            if batch_idx % 10 == 0 and verbose:
                elapsed_time = time.time() - start_time  # Compute elapsed time
                print(f"Processed {batch_idx + 1}/{total_batches} batches - Elapsed time: {elapsed_time:.2f}s")

    total_elapsed_time = time.time() - start_time  # Compute total validation time
    print(f"Validation completed in {total_elapsed_time:.2f} seconds") if verbose else None

    # Loading predictions into coco format
    predictions = []  # List to store formatted prediction data

    # Iterate through all predictions and match them with dataset samples
    for image_id, boxes, labels, scores in zip(all_image_ids, all_boxes, all_labels, all_scores):
        # Store predictions in a structured dictionary
        predictions.append({
            "image_id": image_id,
            "bbox": boxes,
            "category_id": labels,
            "score": scores
        })

    # Saving Predictions
    pred_save_location = (Path(project_dir) / "predictions" / f"run{training_run}_preds.json")
    pred_save_location.parent.mkdir(parents=True, exist_ok=True)

    with open(pred_save_location, 'w') as f:
        json.dump(predictions, f)

    # Evaluating predictions
    reformatted_preds = reformat_predictions(predictions, format='coco')
    gt_bbox = COCO(val_annot_path)
    compute_mAP(gt_bbox, reformatted_preds)

    return predictions
