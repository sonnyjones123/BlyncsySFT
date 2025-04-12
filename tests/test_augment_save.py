from BlyncsySFT.augment import createAugmentedData, augmentation_types

# Paths
root = "../BlyncsyRL/guardrail-damage/images/train"  # Folder with your original images
annotation = "../BlyncsyRL/guardrail-damage/annotation_files/train_filtered.json"  # COCO-format JSON file
output_dir = "../BlyncsyRL/guardrail-damage/images/train_augmented"  # This will store your saved augmentations

# Instantiate dataset
dataset = createAugmentedData(
    root=root,
    annotation=annotation,
    output_dir=output_dir,
    augmentation_types=augmentation_types,
    shuffle_dataset=False,
    augmentation_split=0.5,
    subset_size=10,  # keep this small to test quickly
    mixup_lambda=0.3
)

# Trigger a few samples to force augmentation and saving
for i in range(len(dataset)):
    sample = dataset[i]
    if sample is None:
        continue
    print(f"Processed sample {i+1}/{len(dataset)}")
