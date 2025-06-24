"""
Utility functions for InferSwitch.
"""

from .logging import log_request, log_chat_template
from .chat_template import (
    convert_to_chat_template,
    apply_chat_template,
    truncate_chat_template_to_fit,
    remove_oldest_message_pair
)
from .helpers import estimate_tokens
from .streaming import generate_sse_events, generate_sse_from_response
from .cache import get_cache, RequestCache

__all__ = [
    "log_request",
    "log_chat_template",
    "convert_to_chat_template",
    "apply_chat_template",
    "truncate_chat_template_to_fit",
    "remove_oldest_message_pair",
    "estimate_tokens",
    "generate_sse_events",
    "generate_sse_from_response",
    "get_cache",
    "RequestCache"
]