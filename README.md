# 📦 BlyncsySFT

**Supervised Fine-Tuning for Faster R-CNN with Focal Loss and Custom Augmentations**  
An extensible training and evaluation framework for object detection on COCO-format datasets.

---

## 🚀 Features

- 🧠 Fine-tuning with Focal Loss for class imbalance
- 🎯 Customizable anchor boxes and backbone networks
- 🧪 Augmentation pipeline (MixUp, transforms, etc.)
- 📊 Validation pipeline with mAP evaluation (COCO)
- 🛠 CLI interface for training automation
- 🗂 Compatible with COCO-style datasets

---

## 📁 Installation

```bash
pip install blyncsysft
```

or

```bash
git clone https://github.com/your-username/BlyncsySFT.git
cd BlyncsySFT
pip install .
```

## 🧩 Quick Start

1. **Prepare your dataset and project structure**: Ensure your dataset is in COCO format. The directory structure should look like this:

    ```pgsql
    your_project/
    ├── images/
    │   ├── train/
    │   └── validation/
    ├── annotations/
    │   ├── train.json
    │   └── validation.json
    └── .env
    ```

2. **Create a `.env` file**: This file should contain the following variables:

    ```bash
    TRAINING_RUN=test01
    EPOCHS=20
    BATCH_SIZE=4
    WORKERS=2
    NUM_CLASSES=2
    BACKBONE=resnet50
    SAVE_EVERY=5
    TRAIN_IMAGE_PATH=images/train
    TRAIN_ANNOT_PATH=annotations/train.json
    VAL_IMAGE_PATH=images/validation
    VAL_ANNOT_PATH=annotations/validation.json
    ```

## 🧪 Usage

Run training directly from the command line:

```bash
python -m BlyncsySFT.cli train /path/to/your_project/ --verbose
```

This command:

- Valdiates your .env file
- Loads the dataset
- Builds and trains the model
- Logs the training process
- Saves the model checkpoints

Or use the Python API:

```python
from BlyncsySFT.pipeline import run_auto_training_pipeline
from BlyncsySFT.config import load_and_validate_env

# Step 1: Load config
cfg = load_and_validate_env(env_file="/path/to/your_project/.env")

# Step 2: Define project directory
project_dir = "/path/to/your_project"

# Step 3: Run the training pipeline
run_auto_training_pipeline(project_dir, cfg, verbose=True)
```

## 📄 License

MIT License. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repo and create your branch (`git checkout -b feature/YourFeature`).
2. Make your changes, add tests, and commit them (`git commit -m 'Add some feature'`).
3. Submit a pull request and describe your changes.

## 👥 Contributors

- **Sonny Jones** – [sonny.jones@utah.edu](mailto:sonny.jones@utah.edu)  
- **Tony Le** – [anthony.le@utah.edu](mailto:anthony.le@utah.edu)  
- **Rohit Raj** – [rohitrraj284@gmail.com](mailto:rohitrraj284@gmail.com)
