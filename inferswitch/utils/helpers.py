"""
Helper utility functions.
"""

from typing import Union, List, Optional
from ..models import ContentBlock
from ..config import MODEL_MAX_TOKENS


def get_default_max_tokens(model: str) -> int:
    """
    Get the default max_tokens value for a model.

    Uses the model's maximum supported output tokens as the default.
    Falls back to 4096 if the model is not recognized.

    Args:
        model: The model name/ID

    Returns:
        The appropriate default max_tokens value for the model
    """
    # First, try exact match
    if model in MODEL_MAX_TOKENS:
        return MODEL_MAX_TOKENS[model]

    # Try partial matches (e.g., "claude-3-5-haiku" matches "claude-3-5-haiku-20241022")
    for model_key, max_tokens in MODEL_MAX_TOKENS.items():
        if model_key in model or model in model_key:
            return max_tokens

    # Fall back to default
    return MODEL_MAX_TOKENS.get("default", 4096)


def estimate_tokens(content: Union[str, List[ContentBlock]]) -> int:
    """
    Estimate the number of tokens in content.

    Uses a simple heuristic of ~4 characters per token.

    Args:
        content: String content or list of ContentBlock objects

    Returns:
        Estimated token count
    """
    if isinstance(content, str):
        return len(content) // 4
    else:
        total = 0
        for block in content:
            if block.type == "text" and block.text:
                total += len(block.text) // 4
        return total
