from pathlib import Path
import torch
from torchvision import datasets
from torch.utils.data import DataLoader


def get_dataloader(cfg, project_dir, transform, use_augmentations):
    train_image_path = Path(project_dir) / cfg.get('TRAIN_IMAGE_PATH', "images/train")
    train_annot_path = Path(project_dir) / cfg.get('TRAIN_ANNOT_PATH', "annotations/train.json")
    if use_augmentations:
        dataset = createAugmentedData(
            root=train_image_path,
            annotation=train_annot_path,
            augmentation_types=augmentation_types,
            seed=cfg.get('RANDOM_SEED', None),
            set_random_seed=cfg.get('SET_RANDOM_SEED', False),
            transforms=None,
            shuffle_dataset=cfg.get('SHUFFLE_DATASET', True),
            augmentation_split=cfg.get('AUGMENTATION_SPLIT', 0.5),
            subset_size=int(cfg.get('SUBSET_SIZE')) if cfg.get('SUBSET_SIZE') else None,
            mixup_lambda=cfg.get('MIXUP_LAMBDA', 0.2)
        )
    else:
        dataset = datasets.CocoDetection(
            root=train_image_path,
            annFile=train_annot_path,
            transform=transform
        )
    return DataLoader(
        dataset,
        batch_size=int(cfg['BATCH_SIZE']),
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=int(cfg['WORKERS']),
        pin_memory=torch.cuda.is_available()
    )
