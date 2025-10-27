"""
Common utility functions and decorators for the InferSwitch API.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from functools import wraps

from fastapi import HTTPException


# Logger setup utility
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with consistent setup."""
    return logging.getLogger(name)


# Common header validation
def validate_required_headers(
    x_api_key: Optional[str] = None,
    anthropic_version: Optional[str] = None,
) -> None:
    """
    Validate required headers for API requests.

    Args:
        x_api_key: API key header (optional when OAuth is configured)
        anthropic_version: Anthropic version header

    Raises:
        HTTPException: If required headers are missing
    """
    # Note: x-api-key is optional when OAuth is configured
    # The actual authentication will be handled by the backend
    # which will use OAuth if no API key is provided

    if not anthropic_version:
        raise HTTPException(status_code=400, detail="Missing anthropic-version header")


# Header validation decorator
def require_headers(func: Callable) -> Callable:
    """
    Decorator to validate required headers for API endpoints.

    The decorated function should have x_api_key and anthropic_version parameters.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract header values from kwargs
        x_api_key = kwargs.get("x_api_key")
        anthropic_version = kwargs.get("anthropic_version")

        # Validate headers
        validate_required_headers(x_api_key, anthropic_version)

        return await func(*args, **kwargs)

    return wrapper


# Common configuration loading
def load_config_file(config_path: str = "inferswitch.config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file with error handling.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration data, empty dict if file doesn't exist
    """
    logger = get_logger(__name__)
    config_file = Path(config_path)

    if not config_file.exists():
        return {}

    try:
        with open(config_file) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        return {}


# Common error handling for backend operations
def handle_backend_error(operation: str, backend_name: str) -> Callable:
    """
    Decorator for common backend error handling.

    Args:
        operation: Description of the operation being performed
        backend_name: Name of the backend
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            logger = get_logger(__name__)
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {operation} for {backend_name}: {str(e)}")
                raise

        return wrapper

    return decorator


# Common token estimation fallback
def estimate_tokens_fallback(
    messages: list, system: Optional[str] = None
) -> Dict[str, int]:
    """
    Fallback token estimation when backend token counting fails.

    Args:
        messages: List of messages
        system: Optional system message

    Returns:
        Dictionary with estimated token counts
    """
    char_count = sum(len(str(msg)) for msg in messages)
    if system:
        char_count += len(system)

    # Rough estimation: ~4 characters per token
    estimated_tokens = char_count // 4

    return {"input_tokens": estimated_tokens, "output_tokens": 0}


# Common request data validation
def validate_request_data(request_dict: Dict[str, Any]) -> None:
    """
    Validate common request data fields.

    Args:
        request_dict: Request data dictionary

    Raises:
        HTTPException: If validation fails
    """
    if not request_dict.get("messages"):
        raise HTTPException(status_code=400, detail="Missing messages in request")

    if not request_dict.get("model"):
        raise HTTPException(status_code=400, detail="Missing model in request")


# Common model validation patterns
def is_model_supported(model: str, supported_models: Optional[list] = None) -> bool:
    """
    Check if a model is supported by a backend.

    Args:
        model: Model name to check
        supported_models: List of supported models, None means all models supported

    Returns:
        True if model is supported, False otherwise
    """
    if supported_models is None:
        return True

    return model in supported_models


# Common patterns for checking model failure indicators
def is_model_not_supported_error(error_message: str) -> bool:
    """
    Check if an error message indicates model is not supported.

    Args:
        error_message: Error message to check

    Returns:
        True if error indicates unsupported model
    """
    error_msg = error_message.lower()
    unsupported_patterns = ["model", "not supported", "not found", "invalid model"]
    return any(pattern in error_msg for pattern in unsupported_patterns)
