"""
Anthropic API client for forwarding requests.
"""

import json
from datetime import datetime
import httpx

from ..config import ANTHROPIC_API_BASE, REQUEST_TIMEOUT, LOG_FILE
from ..utils import log_request, log_chat_template


class AnthropicClient:
    """Client for forwarding requests to the Anthropic API."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)

    async def forward_request(
        self,
        endpoint: str,
        request_data: dict,
        headers: dict,
        skip_request_logging: bool = False,
    ):
        """Forward request to Anthropic API (always non-streaming) and log the response."""
        # Log the incoming request (unless already logged with difficulty rating)
        if not skip_request_logging:
            log_request(endpoint, request_data)

        # Log chat template representation
        if endpoint == "/v1/messages":
            log_chat_template(endpoint, request_data)

        # Prepare headers for Anthropic API
        forward_headers = {
            "x-api-key": headers.get("x-api-key"),
            "anthropic-version": headers.get("anthropic-version"),
            "content-type": "application/json",
        }

        if headers.get("anthropic-beta"):
            forward_headers["anthropic-beta"] = headers["anthropic-beta"]

        # Make request to Anthropic API (always non-streaming)
        url = f"{ANTHROPIC_API_BASE}{endpoint}"

        # Force streaming to false for the Anthropic API request (only for messages endpoint)
        request_data_copy = request_data.copy()
        if endpoint == "/v1/messages" and "stream" in request_data_copy:
            request_data_copy["stream"] = False

        response = await self.client.post(
            url, json=request_data_copy, headers=forward_headers
        )

        # Log the response
        with open(LOG_FILE, "a") as f:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            f.write(f"\n[RESPONSE] {timestamp}\n")
            f.write(f"Status: {response.status_code}\n")
            if response.status_code == 200:
                f.write("Response Body:\n")
                f.write(json.dumps(response.json(), indent=2)[:5000])
                if len(json.dumps(response.json())) > 5000:
                    f.write("\n... (truncated)")
            else:
                f.write(f"Error: {response.text[:1000]}\n")
                # Also log what we sent for debugging 400 errors
                if response.status_code == 400:
                    f.write("\nSent to Anthropic:\n")
                    f.write(json.dumps(request_data_copy, indent=2)[:2000])
                    f.write("\n\nHeaders sent:\n")
                    f.write(json.dumps(dict(forward_headers), indent=2))
            f.write("\n")

        return response

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


__all__ = ["AnthropicClient"]
