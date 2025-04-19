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
        'name': 'horizontal_flip',
        'transform': A.HorizontalFlip(p=1.0),
        'probability': 0.5
    },
    {
        'name': 'random_brightness',
        'transform': A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        'probability': 0.5  
    },
    # {
    #     'name': 'mixup',
    #     'transform': None,  # Mixup is handled separately
    #     'probability': 0.5  
    # },
    {
        'name': 'blur',
        'probability': 0.5,
        'transform': A.Compose([
            A.GaussianBlur(blur_limit=(3, 7), p=1)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    },
    {
        'name': 'color',
        'probability': 0.5,
        'transform': A.Compose([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    },
    {
        'name': 'brightness_contrast',
        'probability': 0.5,
        'transform': A.Compose([
            A.RandomBrightnessContrast(p=1)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    },
    {
        'name': 'grid_shuffle',
        'probability': 0.5,
        'transform': A.Compose([
            A.RandomGridShuffle(grid=(2, 2), p=1)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    },
    {
        'name': 'grayscale',
        'probability': 0.5,
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
                 seed: int =42,
                 set_random_seed: Optional[bool] = True,
                 shuffle_dataset: Optional[bool] = True,
                 subset_size: Optional[int]= None,
                 aug_probability: float = 0.5,
                 mixup_lambda: float = 0.5
                ):

        """
        Class for generating augmented images
        """

        self.root = root
        self.annotation = annotation
        self.transforms = transforms
        self.seed = seed
        # self.output_dir = output_dir
        self.augmentation_types = augmentation_types
        self.aug_probability = aug_probability
        self.mixup_lambda = mixup_lambda
        self.set_random_seed = set_random_seed
        self.shuffle_dataset = shuffle_dataset
        self.coco = COCO(annotation)

        # Set random seeds
        if self.set_random_seed:
            self._set_seeds()
        

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


    def _set_seeds(self):
        """ Setting all random seeds for reproducibility """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)


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
            x,y,width, height = annotation['bbox']
            boxes.append([x,y,width, height])
            labels.append(annotation['category_id'])

        # print("LOAD IMAGE DONE")

        return img, boxes, labels, path

    
    def _get_augmentation_folder(self, img_id: int):
    #         """
    #         Determine which augmentation folder an image belongs to
    #         """
            for aug_type in self.augmentation_types:
                if img_id in self.aug_type_details.get(aug_type['name'], []):
                    return aug_type['name']
            return 'original'

    def _apply_mixup(self, img1, boxes1, labels1, img_id):
        """
        to apply mixup
        """
        random_idx = random.randint(0, len(self.ids) -1)
        random_img_id = self.ids[random_idx]

        #load second image
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
        
        
    def _augment_image(self, 
                   img, 
                   boxes,
                   labels, 
                   img_id: int):
    
        """
        Apply augmentations to an image and its bounding boxes
        """
        #converting to numpy array as albumentations work on that 
        img_np = np.array(img)
    
        #default transform
        transform = self.transforms

        # Decide whether to apply any augmentation based on overall probability
        if random.random() < self.aug_probability and self.augmentation_types:
            # First check for mixup specifically as it's a special case
            mixup_type = next((aug for aug in self.augmentation_types if aug['name'] == 'mixup'), None)
            
            if mixup_type and random.random() < mixup_type.get('probability', 0):
                # Apply mixup transformation
                img_np, boxes, labels = self._apply_mixup(img_np, boxes, labels, img_id)
            
            # Now select a non-mixup augmentation based on individual probabilities
            non_mixup_augs = [aug for aug in self.augmentation_types if aug['name'] != 'mixup']

            for aug_type in non_mixup_augs:
                    # Apply this augmentation based on its individual probability
                    if random.random() < aug_type.get('probability', 0):
                        try:
                            # Create a composite transform with the selected augmentation
                            transform = A.Compose([
                                aug_type['transform'],
                                *self.transforms.transforms
                            ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
                            break  # Only apply one non-mixup augmentation
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
                if (x >= 0 and y >= 0 and 
                    x + w <= img_width and y + h <= img_height and 
                    w > 0 and h > 0):
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


        #Perform the augmentations
        img, boxes, labels = self._augment_image(img, box,labels, img_id)

        if isinstance(img, Image.Image):
            img = np.array(img)
    
        # Ensure image is in uint8 format and HWC layout
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        # Convert to tensor and normalize to [0,1]
        img = torchvision.transforms.functional.to_tensor(img)

        # Convert boxes to tensor format [x, y, x+w, y+h]
        boxes_tensor = torch.tensor([
            [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            for box in boxes
        ], dtype=torch.float32)
        
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        
        return( img, {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([img_id])
        }) 


    def __len__(self):
        return len(self.ids)
