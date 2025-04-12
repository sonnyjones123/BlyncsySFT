from dotenv import load_dotenv
import os
from typing import Dict


class ConfigValidationError(Exception):
    """Raised when required configuration is missing or invalid"""
    pass


def load_and_validate_env(env_file: str = ".env", required_keys: list[str] = None) -> Dict[str, str]:
    """
    Load environment variables from file and validate required keys are present.

    Args:
        env_file: Path to the .env file
        required_keys: List of required environment variable keys

    Returns:
        Dictionary of loaded environment variables

    Raises:
        ConfigValidationError: If required keys are missing
        FileNotFoundError: If env_file doesn't exist
    """
    if required_keys is None:
        required_keys = []

    # Load environment variables from file
    if not os.path.exists(env_file):
        raise FileNotFoundError(f"Environment file not found: {env_file}")

    load_dotenv(env_file, override=True)

    # Get all loaded environment variables
    env_vars = {key: value for key, value in os.environ.items()
                if not key.startswith(("_", "."))}  # Filter out some system vars

    # Validate required keys
    missing_keys = [key for key in required_keys if key not in env_vars]
    if missing_keys:
        raise ConfigValidationError(
            f"Missing required configuration keys: {', '.join(missing_keys)}"
        )

    return env_vars
