"""
InferSwitch - An Anthropic API proxy server with logging and chat template support.
"""

__version__ = "0.1.0"

# Export main components for easier imports
from .client import AnthropicClient
from .models import (
    ContentBlock,
    Message,
    MessagesRequest,
    MessagesResponse,
    CountTokensRequest,
    CountTokensResponse,
    Usage
)
from .utils import (
    log_request,
    log_chat_template,
    convert_to_chat_template,
    apply_chat_template,
    truncate_chat_template_to_fit,
    remove_oldest_message_pair,
    estimate_tokens
)

__all__ = [
    "AnthropicClient",
    "ContentBlock",
    "Message", 
    "MessagesRequest",
    "MessagesResponse",
    "CountTokensRequest",
    "CountTokensResponse",
    "Usage",
    "log_request",
    "log_chat_template",
    "convert_to_chat_template",
    "apply_chat_template",
    "truncate_chat_template_to_fit",
    "remove_oldest_message_pair",
    "estimate_tokens"
]