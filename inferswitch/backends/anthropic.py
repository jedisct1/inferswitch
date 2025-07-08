"""
Anthropic backend implementation.
"""

import httpx
import json
import time
from typing import Dict, Any, List, Optional, AsyncIterator
from datetime import datetime
from .base import BaseBackend, BackendConfig, BackendResponse
from .errors import BackendError, convert_backend_error
from ..utils.logging import log_request, log_chat_template
from ..utils import get_logger, estimate_tokens_fallback
from ..config import LOG_FILE, MODEL_MAX_TOKENS
from ..utils.oauth import oauth_manager

logger = get_logger(__name__)


class AnthropicBackend(BaseBackend):
    """Backend implementation for Anthropic API."""

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self.client = httpx.AsyncClient(
            timeout=config.timeout, headers=self._get_headers()
        )

    def _get_headers(self) -> Dict[str, str]:
        """Get base headers for Anthropic API requests."""
        headers = {
            "content-type": "application/json",
        }

        if self.config.headers:
            headers.update(self.config.headers)

        return headers

    async def _prepare_request_headers(
        self,
        x_api_key: str,
        anthropic_version: str,
        anthropic_beta: Optional[str] = None,
    ) -> Dict[str, str]:
        """Prepare headers for a specific request."""
        headers = {
            "anthropic-version": anthropic_version,
            "content-type": "application/json",
        }

        # Check if we have an OAuth token first
        oauth_token = await oauth_manager.get_valid_token()
        if oauth_token:
            # Using OAuth - don't include any API key, only the Bearer token
            headers["authorization"] = f"Bearer {oauth_token}"
            # OAuth requires the beta header - combine with any additional beta headers
            beta_headers = ["oauth-2025-04-20"]
            if anthropic_beta:
                beta_headers.append(anthropic_beta)
            headers["anthropic-beta"] = ",".join(beta_headers)
            logger.debug("Using OAuth token for authentication")
        else:
            # No OAuth token, fall back to API key
            if x_api_key:
                headers["x-api-key"] = x_api_key
                logger.debug("Using API key for authentication")
            else:
                logger.warning(
                    "No authentication method available (no OAuth token or API key)"
                )

            # Add beta header for non-OAuth requests
            if anthropic_beta:
                headers["anthropic-beta"] = anthropic_beta

        return headers

    async def create_message(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs,
    ) -> BackendResponse:
        """Create a message using Anthropic API."""
        # Get the effective model to use
        effective_model = self.get_effective_model(model)

        # Check if this model needs thinking support
        anthropic_beta = kwargs.get("anthropic_beta")
        thinking_models = [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-4-opus-20250514",
            "claude-4-sonnet-20250514",
            # Note: claude-3-5-sonnet-20241022 and claude-3-5-haiku-20241022 do not support thinking mode
        ]

        if effective_model in thinking_models:
            # These models need the interleaved-thinking beta header
            if not anthropic_beta:
                anthropic_beta = "interleaved-thinking-2025-05-14"
            elif "interleaved-thinking-2025-05-14" not in anthropic_beta:
                anthropic_beta = f"{anthropic_beta},interleaved-thinking-2025-05-14"
            kwargs["anthropic_beta"] = anthropic_beta

        # Build request data
        request_data = {
            "model": effective_model,
            "messages": messages,
            "stream": False,  # Always non-streaming for base method
        }

        # Check if we're using OAuth - if so, we need to emulate Claude Code
        oauth_token = await oauth_manager.get_valid_token()
        if oauth_token:
            # When using OAuth, we must identify as Claude Code
            claude_code_system = {
                "type": "text",
                "text": "You are Claude Code, Anthropic's official CLI for Claude.",
            }

            if system:
                # Combine user's system prompt with Claude Code identification
                if isinstance(system, str):
                    request_data["system"] = [
                        claude_code_system,
                        {"type": "text", "text": system},
                    ]
                elif isinstance(system, list):
                    request_data["system"] = [claude_code_system] + system
                else:
                    request_data["system"] = [claude_code_system]
            else:
                request_data["system"] = [claude_code_system]
        else:
            # Regular API key authentication - use system as provided
            if system:
                request_data["system"] = system

        # Handle max_tokens with model-specific limits
        if max_tokens:
            # Get the maximum allowed tokens for this model
            model_max = MODEL_MAX_TOKENS.get(
                effective_model, MODEL_MAX_TOKENS["default"]
            )

            if max_tokens > model_max:
                logger.warning(
                    f"Requested max_tokens ({max_tokens}) exceeds limit for {effective_model} ({model_max}). "
                    f"Capping to {model_max}."
                )
                request_data["max_tokens"] = model_max
            else:
                request_data["max_tokens"] = max_tokens

        if temperature is not None:
            request_data["temperature"] = temperature

        # Add any additional parameters (excluding internal ones)
        internal_params = [
            "x_api_key",
            "anthropic_version",
            "anthropic_beta",
            "difficulty_rating",
        ]

        # Filter out thinking parameter for models that don't support it
        non_thinking_models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]

        # Parameters that should be filtered out for all Anthropic models
        filtered_params = ["container", "mcp_servers"]

        for key, value in kwargs.items():
            if key not in request_data and key not in internal_params:
                # Skip thinking parameter for models that don't support it
                if key == "thinking" and effective_model in non_thinking_models:
                    logger.debug(
                        f"Filtering out 'thinking' parameter for model {effective_model}"
                    )
                    continue
                # Skip parameters that aren't supported by Anthropic API
                if key in filtered_params:
                    logger.debug(
                        f"Filtering out '{key}' parameter (not supported by Anthropic API)"
                    )
                    continue
                request_data[key] = value

        # Extract API headers from kwargs
        x_api_key = kwargs.get("x_api_key", self.api_key)
        anthropic_version = kwargs.get("anthropic_version", "2023-06-01")
        anthropic_beta = kwargs.get("anthropic_beta")

        # Log request
        log_request("/v1/messages", request_data, kwargs.get("difficulty_rating"))
        log_chat_template("/v1/messages", request_data)

        # Try the request with automatic OAuth token refresh on 401 errors
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Prepare headers
                headers = await self._prepare_request_headers(
                    x_api_key, anthropic_version, anthropic_beta
                )

                # Log headers for debugging (excluding sensitive data)
                safe_headers = {
                    k: v if k != "authorization" else "Bearer ***"
                    for k, v in headers.items()
                }
                logger.debug(f"Request headers (attempt {attempt + 1}): {safe_headers}")

                # Make request with progress logging
                import asyncio
                from ..utils import log_streaming_progress

                # Start progress logging task
                start_time = time.time()
                progress_interval = 30.0
                stop_progress = False

                async def log_progress():
                    last_log_time = start_time
                    while not stop_progress:
                        await asyncio.sleep(1)  # Check every second
                        current_time = time.time()
                        elapsed = current_time - start_time
                        if current_time - last_log_time >= progress_interval:
                            log_streaming_progress(
                                elapsed,
                                0,  # No token count available yet
                                effective_model,
                            )
                            last_log_time = current_time

                # Start progress logging in background
                progress_task = asyncio.create_task(log_progress())

                try:
                    response = await self.client.post(
                        f"{self.base_url}/v1/messages",
                        json=request_data,
                        headers=headers,
                    )
                finally:
                    # Stop progress logging
                    stop_progress = True
                    progress_task.cancel()
                    try:
                        await progress_task
                    except asyncio.CancelledError:
                        pass

                # Log response
                self._log_response(response, request_data, headers)

                # Check for OAuth-specific errors before raising
                if response.status_code == 400:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", {}).get("message", "")
                        if (
                            "This credential is only authorized for use with Claude Code"
                            in error_msg
                        ):
                            logger.error(
                                "OAuth token is restricted to Claude Code. This token cannot be used for general API access."
                            )
                            logger.info(
                                "Please use API keys for general Anthropic API access, or obtain an OAuth token with broader permissions."
                            )
                    except (ValueError, json.JSONDecodeError):
                        pass

                # Handle 401 errors (token expired) with automatic refresh
                if response.status_code == 401 and attempt < max_retries - 1:
                    oauth_token = await oauth_manager.get_valid_token()
                    if oauth_token:
                        logger.info(
                            f"Received 401 error, attempting OAuth token refresh (attempt {attempt + 1}/{max_retries})"
                        )
                        try:
                            # Force refresh the token
                            stored_token = oauth_manager.load_token()
                            if stored_token and stored_token.refresh_token:
                                await oauth_manager.refresh_access_token(
                                    stored_token.refresh_token
                                )
                                logger.info(
                                    "OAuth token refreshed successfully, retrying request"
                                )
                                continue  # Retry the request with new token
                            else:
                                logger.error(
                                    "No refresh token available for OAuth token refresh"
                                )
                        except Exception as refresh_error:
                            logger.error(
                                f"Failed to refresh OAuth token: {refresh_error}"
                            )
                    else:
                        logger.error("No OAuth token available for refresh")

                response.raise_for_status()

                # Parse response
                response_data = response.json()

                # Clean usage data - only keep integer values
                usage_data = response_data.get("usage", {})
                clean_usage = {}
                for key, value in usage_data.items():
                    if isinstance(value, int):
                        clean_usage[key] = value

                # Return as BackendResponse
                return BackendResponse(
                    content=response_data.get("content", []),
                    model=response_data.get("model", effective_model),
                    stop_reason=response_data.get("stop_reason"),
                    usage=clean_usage if clean_usage else None,
                    raw_response=response_data,
                )

            except httpx.HTTPStatusError as e:
                # If this is the last attempt or not a 401 error, re-raise
                if attempt == max_retries - 1 or e.response.status_code != 401:
                    error = convert_backend_error(e, self.name)
                    raise error
                # Otherwise, continue to next attempt for 401 errors
                logger.warning(
                    f"HTTP {e.response.status_code} error on attempt {attempt + 1}, retrying..."
                )

            except Exception as e:
                # Non-HTTP errors should not be retried
                raise BackendError(
                    f"Anthropic backend error: {str(e)}", backend=self.name
                )

    async def create_message_stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Create a streaming message - Anthropic always returns non-streaming, so we convert."""
        # Remove 'stream' from kwargs if present to avoid duplicate
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop("stream", None)

        # Get the non-streaming response
        response = await self.create_message(
            messages=messages,
            model=model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            **kwargs_copy,
        )

        # Convert to SSE stream events
        # Message start
        yield {
            "type": "message_start",
            "message": {
                "id": response.raw_response.get("id", "msg_unknown"),
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": response.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": response.usage or {"input_tokens": 0, "output_tokens": 0},
            },
        }

        # Content blocks
        for idx, content_block in enumerate(response.content):
            # Content block start
            yield {
                "type": "content_block_start",
                "index": idx,
                "content_block": content_block,
            }

            # Content block delta (send the whole text as one delta)
            if content_block.get("type") == "text":
                yield {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {
                        "type": "text_delta",
                        "text": content_block.get("text", ""),
                    },
                }

            # Content block stop
            yield {"type": "content_block_stop", "index": idx}

        # Message delta
        yield {
            "type": "message_delta",
            "delta": {"stop_reason": response.stop_reason},
            "usage": {
                "output_tokens": response.usage.get("output_tokens", 0)
                if response.usage
                else 0
            },
        }

        # Message stop
        yield {"type": "message_stop"}

    async def count_tokens(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, int]:
        """Count tokens using Anthropic's token counting endpoint."""
        request_data = {"model": model, "messages": messages}

        if system:
            request_data["system"] = system

        # Extract API headers from kwargs
        x_api_key = kwargs.get("x_api_key", self.api_key)
        anthropic_version = kwargs.get("anthropic_version", "2023-06-01")
        anthropic_beta = kwargs.get("anthropic_beta")

        # Try the request with automatic OAuth token refresh on 401 errors
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Prepare headers
                headers = await self._prepare_request_headers(
                    x_api_key, anthropic_version, anthropic_beta
                )

                # Make request
                response = await self.client.post(
                    f"{self.base_url}/v1/messages/count_tokens",
                    json=request_data,
                    headers=headers,
                )

                # Handle 401 errors (token expired) with automatic refresh
                if response.status_code == 401 and attempt < max_retries - 1:
                    oauth_token = await oauth_manager.get_valid_token()
                    if oauth_token:
                        logger.info(
                            f"Received 401 error in count_tokens, attempting OAuth token refresh (attempt {attempt + 1}/{max_retries})"
                        )
                        try:
                            # Force refresh the token
                            stored_token = oauth_manager.load_token()
                            if stored_token and stored_token.refresh_token:
                                await oauth_manager.refresh_access_token(
                                    stored_token.refresh_token
                                )
                                logger.info(
                                    "OAuth token refreshed successfully, retrying count_tokens request"
                                )
                                continue  # Retry the request with new token
                            else:
                                logger.error(
                                    "No refresh token available for OAuth token refresh"
                                )
                        except Exception as refresh_error:
                            logger.error(
                                f"Failed to refresh OAuth token: {refresh_error}"
                            )
                    else:
                        logger.error("No OAuth token available for refresh")

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                # If this is the last attempt or not a 401 error, fall back to estimation
                if attempt == max_retries - 1 or e.response.status_code != 401:
                    logger.warning(
                        f"Token counting failed: {e}, falling back to estimation"
                    )
                    break
                # Otherwise, continue to next attempt for 401 errors
                logger.warning(
                    f"HTTP {e.response.status_code} error on count_tokens attempt {attempt + 1}, retrying..."
                )

            except Exception as e:
                logger.warning(
                    f"Token counting failed: {e}, falling back to estimation"
                )
                break

        # Fallback: estimate tokens using common utility
        return estimate_tokens_fallback(messages, system)

    def supports_model(self, model: str) -> bool:
        """Check if this backend supports a given model."""
        if self.config.models:
            return model in self.config.models

        # Default Anthropic models
        anthropic_models = [
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-4-opus-20250514",
            "claude-4-sonnet-20250514",
        ]
        return model in anthropic_models

    def _log_response(
        self, response: httpx.Response, request_data: dict, headers: dict
    ):
        """Log response details."""
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
                    f.write(json.dumps(request_data, indent=2)[:2000])
                    f.write("\n\nHeaders sent:\n")
                    f.write(json.dumps(dict(headers), indent=2))
            f.write("\n")

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
