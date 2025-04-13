import pytest
from BlyncsySFT.config import load_and_validate_env, ConfigValidationError


def test_load_env_success(tmp_path, monkeypatch):
    # Create temporary .env file
    env_path = tmp_path / ".env"
    env_path.write_text("API_KEY=testkey\nMODEL_PATH=/models/final.pt")

    # Clear relevant environment before loading
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("MODEL_PATH", raising=False)

    # Load and validate
    env_vars = load_and_validate_env(env_file=str(env_path), required_keys=["API_KEY", "MODEL_PATH"])
    assert env_vars["API_KEY"] == "testkey"
    assert env_vars["MODEL_PATH"] == "/models/final.pt"


def test_load_env_missing_file():
    with pytest.raises(FileNotFoundError):
        load_and_validate_env(env_file="nonexistent.env", required_keys=["SOME_KEY"])


def test_load_env_missing_required_keys(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("API_KEY=testkey")  # SECRET_KEY is not provided

    with pytest.raises(ConfigValidationError) as exc_info:
        load_and_validate_env(env_file=str(env_path), required_keys=["API_KEY", "SECRET_KEY"])

    assert "Missing required configuration keys" in str(exc_info.value)


def test_load_env_without_required_keys(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("OPTIONAL_VAR=value")

    monkeypatch.delenv("OPTIONAL_VAR", raising=False)

    env_vars = load_and_validate_env(env_file=str(env_path))
    assert "OPTIONAL_VAR" in env_vars
