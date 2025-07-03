"""
Utility functions for InferSwitch.
"""

from .logging import log_request, log_chat_template
from .chat_template import (
    convert_to_chat_template,
    apply_chat_template,
    truncate_chat_template_to_fit,
    remove_oldest_message_pair,
)
from .helpers import estimate_tokens
from .streaming import generate_sse_events
from .cache import get_cache
from .common import (
    get_logger,
    validate_required_headers,
    require_headers,
    load_config_file,
    handle_backend_error,
    estimate_tokens_fallback,
    validate_request_data,
    is_model_supported,
    is_model_not_supported_error,
)
from .auth import (
    get_auth_credentials,
    should_use_oauth,
    get_anthropic_auth_headers,
    get_openai_auth_headers,
    AuthenticationError,
    validate_authentication,
)
from .proxy import (
    start_proxy_server,
    stop_proxy_server,
    get_proxy_server,
)

__all__ = [
    "log_request",
    "log_chat_template",
    "convert_to_chat_template",
    "apply_chat_template",
    "truncate_chat_template_to_fit",
    "remove_oldest_message_pair",
    "estimate_tokens",
    "generate_sse_events",
    "get_cache",
    "get_logger",
    "validate_required_headers",
    "require_headers",
    "load_config_file",
    "handle_backend_error",
    "estimate_tokens_fallback",
    "validate_request_data",
    "is_model_supported",
    "is_model_not_supported_error",
    "get_auth_credentials",
    "should_use_oauth",
    "get_anthropic_auth_headers",
    "get_openai_auth_headers",
    "AuthenticationError",
    "validate_authentication",
    "start_proxy_server",
    "stop_proxy_server",
    "get_proxy_server",
]
