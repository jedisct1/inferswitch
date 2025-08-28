"""
Unified error handling for backends.
"""

from typing import Optional, Dict, Any


class BackendError(Exception):
    """Base exception for backend errors."""

    def __init__(
        self,
        message: str,
        backend: Optional[str] = None,
        status_code: Optional[int] = None,
        error_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.backend = backend
        self.status_code = status_code
        self.error_type = error_type
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses."""
        return {
            "error": {
                "type": self.error_type or "backend_error",
                "message": self.message,
                "backend": self.backend,
                "details": self.details,
            }
        }


class AuthenticationError(BackendError):
    """Authentication/API key error."""

    def __init__(self, message: str, backend: Optional[str] = None):
        super().__init__(
            message=message,
            backend=backend,
            status_code=401,
            error_type="authentication_error",
        )


class RateLimitError(BackendError):
    """Rate limit exceeded error."""

    def __init__(
        self,
        message: str,
        backend: Optional[str] = None,
        retry_after: Optional[int] = None,
    ):
        details = {}
        if retry_after:
            details["retry_after"] = retry_after

        super().__init__(
            message=message,
            backend=backend,
            status_code=429,
            error_type="rate_limit_error",
            details=details,
        )


class ModelNotFoundError(BackendError):
    """Model not found or not supported error."""

    def __init__(
        self,
        message: str,
        model: str,
        backend: Optional[str] = None,
        available_models: Optional[list] = None,
    ):
        details = {"requested_model": model}
        if available_models:
            details["available_models"] = available_models

        super().__init__(
            message=message,
            backend=backend,
            status_code=404,
            error_type="model_not_found",
            details=details,
        )


class BackendUnavailableError(BackendError):
    """Backend service unavailable error."""

    def __init__(self, message: str, backend: Optional[str] = None):
        super().__init__(
            message=message,
            backend=backend,
            status_code=503,
            error_type="backend_unavailable",
        )


class InvalidRequestError(BackendError):
    """Invalid request format or parameters."""

    def __init__(
        self, message: str, backend: Optional[str] = None, field: Optional[str] = None
    ):
        details = {}
        if field:
            details["field"] = field

        super().__init__(
            message=message,
            backend=backend,
            status_code=400,
            error_type="invalid_request",
            details=details,
        )


class ContextWindowExceededError(BackendError):
    """Request exceeds model context window."""

    def __init__(
        self,
        message: str,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        current_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        messages: Optional[list] = None,
    ):
        details = {}
        if model:
            details["model"] = model
        if current_tokens:
            details["current_tokens"] = current_tokens
        if max_tokens:
            details["max_tokens"] = max_tokens
        if messages:
            details["message_count"] = len(messages)

        super().__init__(
            message=message,
            backend=backend,
            status_code=400,
            error_type="context_window_exceeded",
            details=details,
        )
        self.messages = messages  # Store original messages for compression


def convert_backend_error(error: Exception, backend: str) -> BackendError:
    """
    Convert backend-specific errors to unified BackendError.

    Args:
        error: Original exception from backend
        backend: Backend name

    Returns:
        Unified BackendError instance
    """
    error_str = str(error).lower()

    # Context window errors - check first as they're often reported as "invalid request"
    if any(
        phrase in error_str
        for phrase in [
            "context_length_exceeded",
            "max_tokens_exceeded",
            "request_too_large",
            "context window",
            "maximum context",
            "token limit exceeded",
            "context length",
            "exceeds maximum",
            "too many tokens",
            "message too long",
        ]
    ):
        return ContextWindowExceededError(str(error), backend)

    # Authentication errors
    if any(
        phrase in error_str
        for phrase in ["api key", "authentication", "unauthorized", "invalid key"]
    ):
        return AuthenticationError(str(error), backend)

    # Rate limit errors
    if any(
        phrase in error_str
        for phrase in ["rate limit", "too many requests", "quota exceeded"]
    ):
        return RateLimitError(str(error), backend)

    # Model not found
    if any(
        phrase in error_str
        for phrase in ["model not found", "unknown model", "invalid model"]
    ):
        return ModelNotFoundError(str(error), model="", backend=backend)

    # Service unavailable
    if any(
        phrase in error_str
        for phrase in ["service unavailable", "connection error", "timeout"]
    ):
        return BackendUnavailableError(str(error), backend)

    # Invalid request
    if any(
        phrase in error_str
        for phrase in ["invalid request", "bad request", "validation error"]
    ):
        return InvalidRequestError(str(error), backend)

    # Default backend error
    return BackendError(str(error), backend)
