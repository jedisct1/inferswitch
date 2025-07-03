"""
Backend implementations for InferSwitch.
"""

from .base import BaseBackend, BackendConfig, BackendResponse
from .errors import (
    BackendError,
    ModelNotFoundError,
    RateLimitError,
    AuthenticationError,
)
from .registry import BackendRegistry, backend_registry
from .router import BackendRouter
from .normalizer import ResponseNormalizer
from .anthropic import AnthropicBackend
from .openai import OpenAIBackend

__all__ = [
    "BaseBackend",
    "BackendConfig",
    "BackendResponse",
    "BackendError",
    "ModelNotFoundError",
    "RateLimitError",
    "AuthenticationError",
    "BackendRegistry",
    "backend_registry",
    "BackendRouter",
    "ResponseNormalizer",
    "AnthropicBackend",
    "OpenAIBackend",
]
