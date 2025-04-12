import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from typing import Optional, Dict, List
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Got processor for bboxes, but no transform to process it.")

augmentation_types = [
    {
        'name': 'mixup',
        'ratio': 0.2,  # Allocate 20% of augmentations to mixup
        'transform': None  # Mixup is handled specially, so no Albumentations transform needed
    },
    {
        'name': 'flip',
        'ratio': 0.15,  # Adjusted ratios to accommodate mixup
        'transform': A.Compose([
            A.HorizontalFlip(p=1)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    },
    {
        'name': 'blur',
        'ratio': 0.15,
        'transform': A.Compose([
            A.GaussianBlur(blur_limit=(3, 7), p=1)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    },
    {
        'name': 'color',
        'ratio': 0.1,
        'transform': A.Compose([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    },
    {
        'name': 'brightness_contrast',
        'ratio': 0.1,
        'transform': A.Compose([
            A.RandomBrightnessContrast(p=1)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    },
    {
        'name': 'grid_shuffle',
        'ratio': 0.1,
        'transform': A.Compose([
            A.RandomGridShuffle(grid=(2, 2), p=1)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels'], min_area=32, min_visibility=0.9))
    },
    {
        'name': 'grayscale',
        'ratio': 0.2,
        'transform': A.Compose([
            A.ToGray(p=1)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    }
]


class createAugmentedData(Dataset):
    def __init__(self,
                 root: str,
                 annotation: str,
                 output_dir: Optional[str] = None,
                 augmentation_types: Optional[List[Dict]] = None,
                 transforms: Optional[A.Compose] = None,
                 seed: int = 42,
                 set_random_seed: Optional[bool] = True,
                 shuffle_dataset: Optional[bool] = True,
                 augmentation_split: float = 0.5,
                 subset_size: Optional[int] = None,
                 mixup_lambda: float = 0.2
                 ):

        """
        Class for generating augmented images
        """
        self.root = root
        self.annotation = annotation
        self.transforms = transforms
        self.seed = seed
        # self.output_dir = output_dir
        self.output_dir = Path(output_dir) if output_dir else None
        self.augmentation_types = augmentation_types
        self.augmentation_split = augmentation_split
        self.mixup_lambda = mixup_lambda
        self.set_random_seed = set_random_seed
        self.shuffle_dataset = shuffle_dataset
        self.coco = COCO(annotation)

        # Set random seeds
        if self.set_random_seed:
            self._set_seeds()

        if self.output_dir:
            # Create output directory if it doesn't exist
            self._prepare_output_dirs()

        # Get all image ids and shuffle
        self.ids = list(sorted(self.coco.imgs.keys()))

        if self.shuffle_dataset:
            random.shuffle(self.ids)
        # random.shuffle(all_ids)

        # Apply subset if specified
        if subset_size is not None:
            self.ids = self.ids[:subset_size]

        # Set transforms (must change to ToTensorV2)
        self.transforms = transforms if transforms else A.Compose(
            [ToTensorV2()],
            bbox_params=A.BboxParams(format='coco', label_fields=['labels'])
            )

        # Clean up and recreate output directories
        # self._prepare_output_dirs()

        # split the dataset
        self._split_dataset()

    def _prepare_output_dirs(self):
        for aug in self.augmentation_types or []:
            (self.output_dir / aug['name']).mkdir(parents=True, exist_ok=True)

        (self.output_dir / 'original').mkdir(parents=True, exist_ok=True)

    def _set_seeds(self):
        """ Setting all random seeds for reproducibility """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _split_dataset(self):
        """
        splitting data into augmented and non_augmented
        """
        # only split if the data is present

        if self.augmentation_types and self.augmentation_split > 0:
            # for storing augemtation details
            self.aug_type_details = {}

            num_total_imgs = len(self.ids)
            # print(f"total no of images  -*10: {num_total_imgs}")
            num_augmentation_imgs = int(num_total_imgs * self.augmentation_split)
            # print(f"total no of aug images  -*10: {num_augmentation_imgs}")

            num_original_imgs = num_total_imgs - num_augmentation_imgs
            # print(f"total no of original images  -*10: {num_original_imgs}")

            # original image subset
            self.original_ids = self.ids[:num_original_imgs]

            # augmented subset
            self.augmented_ids = self.ids[num_original_imgs:]

            # Normalize augmentation ratios amongst all the augmentations
            total_ratio = sum(aug['ratio'] for aug in self.augmentation_types)
            if total_ratio > 0:
                for aug in self.augmentation_types:
                    aug['ratio'] = aug['ratio'] / total_ratio

            # Split augmented part according to augmentation ratios
            current_idx = 0
            num_augmented = len(self.augmented_ids)

            # Normalize augmentation ratios amongst all the augmentations
            # to divide the no of augmentations for each transformation
            total_ratio = sum(aug['ratio'] for aug in self.augmentation_types)

            if total_ratio > 0:
                for aug in self.augmentation_types:
                    aug['ratio'] = aug['ratio'] / total_ratio

            # Split augmented part according to augmentation ratios
            current_idx = 0
            num_augmented = len(self.augmented_ids)

            # now get numbers for each augemtation and also put everything left in last block
            for i, aug_type in enumerate(self.augmentation_types):

                # if on last category
                if i == len(self.augmentation_types) - 1:
                    split_size = num_augmented - current_idx
                else:
                    split_size = int(num_augmented * aug_type['ratio'])

                if split_size > 0:
                    split_ids = self.augmented_ids[current_idx:current_idx + split_size]
                    self.aug_type_details[aug_type['name']] = split_ids
                    current_idx += split_size

        else:
            self.original_ids = self.ids
            self.aug_type_details = {}

        # print("Split dataset done")

    def _load_image_and_annotations(self, img_id):
        """
        Method to load image and its annotations using image id"
        """
        coco = self.coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]["file_name"]
        try:
            img = Image.open(os.path.join(self.root, path)).convert('RGB')
            img = np.array(img)

        except Exception as e:
            print(f"Warning: Image file '{path}' not found: {e}")
            return None, None, None, None

        # Now get bboxes and labels
        boxes = []
        labels = []

        for annotation in coco_annotation:
            x, y, width, height = annotation['bbox']
            boxes.append([x, y, width, height])
            labels.append(annotation['category_id'])

        # print("LOAD IMAGE DONE")

        return img, boxes, labels, path

    def _get_augmentation_folder(self, img_id: int):
        for aug_type in self.augmentation_types:
            if img_id in self.aug_type_details.get(aug_type['name'], []):
                return aug_type['name']
        return 'original'

    def _apply_mixup(self, img1, boxes1, labels1, img_id):
        """
        to apply mixup
        """
        # mixup_img_ids = self.aug_type_details.get('mixup',[])
        # print(mixup_img_ids)

        # Only select from images that are in the mixup category
        mixup_ids = self.aug_type_details.get('mixup', [])

        # If the current img_id is the only one in mixup_ids, or there are no other mixup images,
        # fallback to random selection from the entire dataset
        valid_mixup_ids = [id for id in mixup_ids if id != img_id]

        if valid_mixup_ids:
            # Select from other images designated for mixup
            random_img_id = random.choice(valid_mixup_ids)
        else:
            # Fallback to the original random selection from all images
            random_idx = random.randint(0, len(self.ids) - 1)
            random_img_id = self.ids[random_idx]

            random_idx = random.randint(0, len(self.ids) - 1)
            random_img_id = self.ids[random_idx]

        # load second image
        img2, boxes2, labels2, _ = self._load_image_and_annotations(random_img_id)

        if img2 is None:
            return img1, boxes1, labels1

        # making image shapes consistent
        if img1.shape != img2.shape:
            img2 = np.array(Image.fromarray(img2).resize((img1.shape[1], img1.shape[0])))

        lam = self.mixup_lambda

        mixed_img = lam * img1 + (1-lam) * img2
        mixed_img = mixed_img.astype(np.uint8)

        mixed_boxes = boxes1 + boxes2
        mixed_labels = labels1 + labels2

        return mixed_img, mixed_boxes, mixed_labels

    def _augment_image(self, img, boxes, labels, img_id: int):

        """
        Apply augmentations to an image and its bounding boxes
        """
        # c onverting to numpy array as albumentations work on that
        img_np = np.array(img)

        # default transform
        transform = self.transforms
        aug_type = "original"

        # First check if this is a mixup image
        if self.augmentation_types:
            for aug_type in self.augmentation_types:
                if aug_type['name'] == 'mixup' and img_id in self.aug_type_details.get('mixup', []):
                    # Apply mixup transformation
                    img_np, boxes, labels = self._apply_mixup(img_np, boxes, labels, img_id)
                    break

        if self.augmentation_types:
            for aug_type in self.augmentation_types:
                if aug_type['name'] == 'mixup':
                    continue
                if img_id in self.aug_type_details.get(aug_type['name'], []):
                    try:
                        # Add clipBboxes=True to keep bounding boxes within [0, 1] range
                        transform = A.Compose([
                            aug_type['transform'],
                            *self.transforms.transforms
                        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
                        break
                    except Exception as e:
                        print(f"Error creating transform for {aug_type['name']}: {str(e)}")
                        continue

        try:
            # Check and filter out invalid boxes before transformation
            valid_boxes = []
            valid_labels = []
            img_height, img_width = img_np.shape[:2]

            for box, label in zip(boxes, labels):
                x, y, w, h = box
                if (x >= 0 and y >= 0 and x + w <= img_width and y + h <= img_height and w > 0 and h > 0):
                    valid_boxes.append(box)
                    valid_labels.append(label)
                else:
                    print(f"Skipping invalid box: {box} for image {img_id}")

            transformed = transform(image=img_np, bboxes=valid_boxes, labels=valid_labels)
            img_aug = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']

            # Convert back to PIL Image
            if isinstance(img_aug, torch.Tensor):
                img_aug = img_aug.cpu().numpy()
                if img_aug.shape[0] == 3:  # C,H,W -> H,W,C
                    img_aug = img_aug.transpose(1, 2, 0)

            # Ensure img_aug is converted to uint8 before creating PIL Image
            if img_aug.dtype != np.uint8:
                img_aug = img_aug.astype(np.uint8)

            # Save the augmented image if output_dir is specified
            if self.output_dir:
                aug_folder = self._get_augmentation_folder(img_id)
                file_name = self.coco.loadImgs(img_id)[0]['file_name']
                save_path = self.output_dir / aug_folder / file_name
                Image.fromarray(img_aug).save(save_path)

            return Image.fromarray(img_aug), boxes, labels
        except Exception as e:
            print(f"Error augmenting image {img_id}: {str(e)}")
            return img, boxes, labels

    def __getitem__(self, index: int):
        """
        For getting single item for dataset
        """

        img_id = self.ids[index]

        img, box, labels, img_filename = self._load_image_and_annotations(img_id)

        if img is None:
            return None

        # Perform the augmentations
        img, boxes, labels = self._augment_image(img, box, labels, img_id)

        if isinstance(img, Image.Image):
            img = np.array(img)

        # Ensure image is in uint8 format and HWC layout
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        # Convert to tensor and normalize to [0,1]
        img = torchvision.transforms.functional.to_tensor(img)

        return img, {
            'boxes': torch.as_tensor([
                [x, y, x + w, y + h] for (x, y, w, h) in boxes
            ], dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_id])
        }

    def __len__(self):
        return len(self.ids)
