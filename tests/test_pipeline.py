import pytest
import torch
from unittest.mock import patch, MagicMock
from BlyncsySFT.pipeline import run_auto_training_pipeline


@pytest.fixture
def dummy_config(tmp_path):
    return {
        "TRAINING_RUN": "test01",
        "EPOCHS": "1",
        "BATCH_SIZE": "2",
        "WORKERS": "0",
        "NUM_CLASSES": "2",
        "BACKBONE": "resnet18",
        "SAVE_EVERY": "1",
        "TRAIN_IMAGE_PATH": "images/train",
        "TRAIN_ANNOT_PATH": "annotations/train.json",
        "VAL_IMAGE_PATH": "images/validation",
        "VAL_ANNOT_PATH": "annotations/validation.json"
    }


@pytest.fixture
def dummy_project_dir(tmp_path):
    (tmp_path / "images/train").mkdir(parents=True)
    (tmp_path / "annotations").mkdir(parents=True)
    (tmp_path / "images/validation").mkdir(parents=True)
    (tmp_path / "annotations/train.json").write_text('{"images": [], "annotations": [], "categories": []}')
    (tmp_path / "annotations/validation.json").write_text('{"images": [], "annotations": [], "categories": []}')
    return tmp_path


@patch("BlyncsySFT.pipeline.torch.save")  # Add this decorator
@patch("BlyncsySFT.pipeline.run_validation_pipeline")
@patch("BlyncsySFT.pipeline.datasets.CocoDetection")
@patch("BlyncsySFT.pipeline.DataLoader")
@patch("BlyncsySFT.pipeline.custom_faster_rcnn")
def test_run_auto_training_pipeline_runs(
    mock_model_fn,
    mock_loader,
    mock_coco,
    mock_validation,
    mock_save,  # <-- New argument
    dummy_config,
    dummy_project_dir
):
    mock_model = MagicMock()
    mock_model.return_value = {"loss_classifier": torch.tensor(1.0, requires_grad=True)}
    mock_model.train.return_value = None
    mock_model.eval.return_value = None
    mock_model.to.return_value = mock_model
    mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
    mock_model_fn.return_value = mock_model

    # Return valid batch for DataLoader
    mock_loader.return_value = [(
        [torch.rand(3, 224, 224)],
        [[  # <-- Note: extra list here!
            {
                "bbox": [10, 10, 30, 30],
                "category_id": 1,
                "image_id": 1,
                "area": 400
            }
        ]]
    )]

    run_auto_training_pipeline(dummy_project_dir, dummy_config, verbose=False)

    assert mock_model_fn.called
    assert mock_validation.called


@patch("BlyncsySFT.pipeline.torch.save")  # 👈 Patch torch.save here
@patch("BlyncsySFT.pipeline.run_validation_pipeline")
@patch("BlyncsySFT.pipeline.datasets.CocoDetection")
@patch("BlyncsySFT.pipeline.DataLoader")
@patch("BlyncsySFT.pipeline.custom_faster_rcnn")
def test_logging_output(
    mock_model_fn,
    mock_loader,
    mock_coco,
    mock_validation,
    mock_save,  # 👈 Include this argument to receive the mock
    dummy_config,
    dummy_project_dir,
    caplog
):
    mock_model = MagicMock()
    mock_model.return_value = {"loss_classifier": torch.tensor(1.0, requires_grad=True)}
    mock_model.train.return_value = None
    mock_model.eval.return_value = None
    mock_model.to.return_value = mock_model
    mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
    mock_model.state_dict.return_value = {"layer": torch.tensor([1.0])}  # Optional
    mock_model_fn.return_value = mock_model

    mock_loader.return_value = [(
        [torch.rand(3, 224, 224)],
        [[  # <-- Note: extra list here!
            {
                "bbox": [10, 10, 30, 30],
                "category_id": 1,
                "image_id": 1,
                "area": 400
            }
        ]]
    )]

    with caplog.at_level("INFO"):
        run_auto_training_pipeline(dummy_project_dir, dummy_config, verbose=True)

    assert "Training run: modeltest01" in caplog.text
