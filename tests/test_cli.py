import pytest
from click.testing import CliRunner
from BlyncsySFT.cli import cli
from BlyncsySFT.config import ConfigValidationError


@pytest.fixture
def temp_env_file(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join([
            "TRAINING_RUN=test",
            "EPOCHS=10",
            "BATCH_SIZE=4",
            "WORKERS=2",
            "NUM_CLASSES=2",
            "BACKBONE=resnet18",
            "SAVE_EVERY=1"
        ])
    )
    return tmp_path


def test_train_command_success(monkeypatch, temp_env_file):
    # Mock the pipeline so we don't actually train
    monkeypatch.setattr("BlyncsySFT.cli.run_auto_training_pipeline", lambda *args, **kwargs: None)

    runner = CliRunner()
    result = runner.invoke(cli, ["train", str(temp_env_file)])

    assert result.exit_code == 0
    assert "Training completed successfully" in result.output


def test_train_command_verbose(monkeypatch, temp_env_file):
    monkeypatch.setattr("BlyncsySFT.cli.run_auto_training_pipeline", lambda *args, **kwargs: None)

    runner = CliRunner()
    result = runner.invoke(cli, ["train", str(temp_env_file), "-v"])

    assert result.exit_code == 0
    assert "✅ Loaded valid configuration from .env" in result.output


def test_train_command_missing_env():
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "nonexistent_dir"])
    assert result.exit_code != 0
    assert "Error" in result.output


def test_train_command_missing_keys(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("EPOCHS=10")

    monkeypatch.setattr(
        "BlyncsySFT.cli.load_and_validate_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(ConfigValidationError("Missing required configuration keys: TRAINING_RUN"))
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["train", str(tmp_path)])

    assert result.exit_code != 0
    assert "Missing required configuration keys" in result.output
