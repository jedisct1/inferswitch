"""
Helper utility functions.
"""

from typing import Union, List
from ..models import ContentBlock


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