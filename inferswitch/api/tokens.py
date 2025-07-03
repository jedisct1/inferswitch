"""
Token counting endpoint handler.
"""

import json
from typing import Optional
import httpx

from fastapi import HTTPException, Header

from ..config import PROXY_MODE
from ..backends import backend_registry, BackendError
from ..models import CountTokensRequest, CountTokensResponse
from ..utils import (
    log_request,
    estimate_tokens,
    get_logger,
    validate_request_data,
)
from ..utils.oauth import oauth_manager

logger = get_logger(__name__)


async def count_tokens(
    request: CountTokensRequest,
    x_api_key: Optional[str] = Header(None),
    anthropic_version: Optional[str] = Header(None),
    anthropic_beta: Optional[str] = Header(None),
):
    """Handle POST /v1/messages/count_tokens requests."""
    # Check for OAuth token first
    oauth_token = await oauth_manager.get_valid_token()

    # Validate authentication (OAuth or API key)
    if not oauth_token and not x_api_key:
        raise HTTPException(
            status_code=401, detail="Missing x-api-key header or OAuth token"
        )

    # Validate anthropic version header
    if not anthropic_version:
        raise HTTPException(status_code=400, detail="Missing anthropic-version header")

    request_dict = request.model_dump(exclude_none=True)

    # Validate request data
    validate_request_data(request_dict)

    if PROXY_MODE:
        # Use Anthropic backend directly for token counting
        try:
            # Get the Anthropic backend
            backend = backend_registry.get_backend("anthropic")

            # Prepare headers (OAuth will be handled inside the backend)
            headers = await backend._prepare_request_headers(
                x_api_key or "",  # Pass empty string if no API key
                anthropic_version,
                anthropic_beta,
            )

            # Make direct request to count_tokens endpoint
            response = await backend.client.post(
                f"{backend.base_url}/v1/messages/count_tokens",
                json=request_dict,
                headers=headers,
            )

            if response.status_code == 200:
                return response.json()
            else:
                error_detail = response.text
                try:
                    error_json = response.json()
                    error_detail = error_json
                except (ValueError, json.JSONDecodeError):
                    pass
                raise HTTPException(
                    status_code=response.status_code, detail=error_detail
                )
        except httpx.RequestError as e:
            logger.error(f"Error forwarding request to Anthropic: {e}")
            raise HTTPException(
                status_code=500, detail="Error connecting to Anthropic API"
            )
        except BackendError as e:
            raise HTTPException(status_code=e.status_code or 500, detail=e.to_dict())
    else:
        # Return estimated token count
        log_request("/v1/messages/count_tokens", request_dict)

        total_tokens = sum(estimate_tokens(msg.content) for msg in request.messages)
        if request.system:
            if isinstance(request.system, str):
                total_tokens += estimate_tokens(request.system)
            else:
                # System is an array of objects
                for sys_obj in request.system:
                    if isinstance(sys_obj, dict) and "text" in sys_obj:
                        total_tokens += estimate_tokens(sys_obj["text"])

        return CountTokensResponse(input_tokens=total_tokens)
