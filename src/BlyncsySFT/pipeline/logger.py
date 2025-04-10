import logging
from pathlib import Path


def setup_logger(project_dir, training_run, verbose):
    log_path = Path(project_dir) / "logging" / f"model{training_run}_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("BlyncsyLogger")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers to prevent duplicate logs

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    return logger


def log_training_metadata(logger, cfg, device):
    logger.info("===== Training Configuration =====")
    logger.info(f"Training run: model{cfg['TRAINING_RUN']}")
    logger.info("Loss function: focal loss")
    logger.info(f"Alpha: {cfg.get('ALPHA', 0.25)}")
    logger.info(f"Gamma: {cfg.get('GAMMA', 2.0)}")
    logger.info(f"Learning rate: {cfg.get('LEARNING_RATE', 1e-3)}")
    logger.info(f"Momentum: {cfg.get('MOMENTUM', 0.95)}")
    logger.info(f"Weight decay: {cfg.get('WEIGHT_DECAY', 1e-4)}")
    logger.info(f"Batch size: {cfg['BATCH_SIZE']}")
    logger.info(f"Epochs: {cfg['EPOCHS']}")
    logger.info(f"Device: {device}")
    logger.info("===================================")
    logger.info("")
    logger.info("======== Transform Params =========")
    logger.info(f"Mean: {cfg.get('IMAGE_MEAN', [0.485, 0.456, 0.406])}")
    logger.info(f"Std: {cfg.get('IMAGE_STD', [0.229, 0.224, 0.225])}")
    logger.info("===================================")
