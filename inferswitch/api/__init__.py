"""
API endpoint handlers for InferSwitch.
"""

from .messages_v2 import create_message_v2
from .tokens import count_tokens
from .chat_template import get_chat_template

__all__ = [
    "create_message_v2",
    "count_tokens", 
    "get_chat_template"
]