import torch
from torch.optim.lr_scheduler import StepLR
import time
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from .dataloader import get_dataloader
from .logger import setup_logger, log_training_metadata


def run_training_loop(model, dataloader, cfg, project_dir, device, logger, log_metadata=True):
    if log_metadata:
        log_training_metadata(logger, cfg, device)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.get('LEARNING_RATE', 1e-3),
                                momentum=cfg.get('MOMENTUM', 0.95),
                                weight_decay=cfg.get('WEIGHT_DECAY', 1e-4))
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    start_time = time.time()

    for epoch in range(int(cfg['EPOCHS'])):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg['EPOCHS']}")

        for images, targets in progress_bar:
            images = [img.to(device) for img in images]
            formatted_targets = [
                {k: v.to(device) for k, v in t.items()} for t in (
                    targets if cfg.get("USE_AUGMENTATIONS") else [format_target(t) for t in targets]
                )
            ]
            loss_dict = model(images, formatted_targets)
            loss = sum(v for v in loss_dict.values() if v is not None)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        elapsed = time.time() - start_time
        logger.info(f"Epoch [{epoch+1}/{cfg['EPOCHS']}] Total Loss: {total_loss:.4f}, Avg Loss: {avg_loss:.4f} - Time: {elapsed:.2f}s")

        if (epoch + 1) % int(cfg['SAVE_EVERY']) == 0:
            model_dir = Path(project_dir) / "models"
            model_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss
            }, model_dir / f"model{cfg['TRAINING_RUN']}_epoch{epoch+1}.pth")

    torch.save(model.state_dict(), Path(project_dir) / "models" / f"model{cfg['TRAINING_RUN']}.pth")

    return model


def run_auto_training_pipeline(project_dir, cfg, verbose):
    logger = setup_logger(project_dir, cfg['TRAINING_RUN'], verbose)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_training_metadata(logger, cfg, device)

    model = build_model(cfg).to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.get('IMAGE_MEAN', [0.485, 0.456, 0.406]),
                             std=cfg.get('IMAGE_STD', [0.229, 0.224, 0.225]))
    ])

    dataloader = get_dataloader(cfg, project_dir, transform, cfg.get("USE_AUGMENTATIONS", False))
    model = run_training_loop(model, dataloader, cfg, project_dir, device, logger, log_metadata=False)

    from .validate import run_validation_pipeline
    run_validation_pipeline(model, project_dir, cfg, verbose)

    return model
