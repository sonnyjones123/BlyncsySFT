import time
import json
import torch
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pycocotools.coco import COCO

from BlyncsySFT.format import reformat_predictions
from BlyncsySFT.eval import compute_mAP


def run_validation_pipeline(model, project_dir, cfg, verbose=True):
    val_image_path = Path(project_dir) / cfg.get('VAL_IMAGE_PATH', "images/validation")
    val_annot_path = Path(project_dir) / cfg.get('VAL_ANNOT_PATH', "annotations/validation.json")
    batch_size = int(cfg['BATCH_SIZE'])
    workers = int(cfg['WORKERS'])
    norm_mean = cfg.get('IMAGE_MEAN', [0.485, 0.456, 0.406])
    norm_std = cfg.get('IMAGE_STD', [0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    val_dataset = datasets.CocoDetection(root=val_image_path, annFile=val_annot_path, transform=transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=workers,
        pin_memory=torch.cuda.is_available()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_boxes, all_labels, all_scores, all_image_ids = [], [], [], []
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = [img.to(device) for img in images]
            batch_ids = [t[0]['image_id'] for t in targets]
            all_image_ids.extend(batch_ids)

            predictions = model(images)
            for pred in predictions:
                all_boxes.append(pred['boxes'].cpu().numpy().tolist())
                all_labels.append(pred['labels'].cpu().numpy().tolist())
                all_scores.append(pred['scores'].cpu().numpy().tolist())

            if batch_idx % 10 == 0 and verbose:
                print(f"Processed {batch_idx + 1}/{len(val_loader)} batches")

    total_elapsed_time = time.time() - start_time
    if verbose:
        print(f"Validation completed in {total_elapsed_time:.2f} seconds")

    predictions_out = []
    for image_id, boxes, labels, scores in zip(all_image_ids, all_boxes, all_labels, all_scores):
        predictions_out.append({
            "image_id": image_id,
            "bbox": boxes,
            "category_id": labels,
            "score": scores
        })

    pred_path = Path(project_dir) / "predictions" / f"run{cfg['TRAINING_RUN']}_preds.json"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_path, 'w') as f:
        json.dump(predictions_out, f)

    gt_coco = COCO(val_annot_path)
    coco_formatted_preds = reformat_predictions(predictions_out, format='coco')
    compute_mAP(gt_coco, coco_formatted_preds)

    return predictions_out
