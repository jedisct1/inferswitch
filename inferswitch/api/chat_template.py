"""
Chat template endpoint handler.
"""

import logging
from typing import Optional

from fastapi import HTTPException, Header

from ..models import MessagesRequest
from ..utils import log_request, convert_to_chat_template, apply_chat_template

logger = logging.getLogger(__name__)


async def get_chat_template(
    request: MessagesRequest,
    x_api_key: Optional[str] = Header(None),
    anthropic_version: Optional[str] = Header(None),
):
    """
    Convert an Anthropic messages request to chat template format.
    This is a custom endpoint not part of the official Anthropic API.
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing x-api-key header")

    if not anthropic_version:
        raise HTTPException(status_code=400, detail="Missing anthropic-version header")

    request_dict = request.model_dump(exclude_none=True)

    try:
        # Convert to chat template format
        chat_messages = convert_to_chat_template(request_dict)

        # Generate different formats
        response = {
            "chat_messages": chat_messages,
            "formatted": {
                "chatml": apply_chat_template(
                    chat_messages, add_generation_prompt=True
                ),
                "chatml_no_prompt": apply_chat_template(
                    chat_messages, add_generation_prompt=False
                ),
            },
            "message_count": len(chat_messages),
            "roles": [msg["role"] for msg in chat_messages],
        }

        # Log the request
        log_request("/v1/messages/chat-template", request_dict)

        return response

    except Exception as e:
        logger.error(f"Error converting to chat template: {e}")
        raise HTTPException(status_code=500, detail=str(e))
