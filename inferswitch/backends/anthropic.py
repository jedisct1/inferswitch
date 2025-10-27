"""
Anthropic backend implementation.
"""

import httpx
import json
import time
from typing import Dict, Any, List, Optional, AsyncIterator
from datetime import datetime
from .base import BaseBackend, BackendConfig, BackendResponse
from .errors import BackendError, convert_backend_error, ContextWindowExceededError
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

        # Filter out interleaved-thinking beta for models that don't support it
        if effective_model in non_thinking_models and anthropic_beta:
            # Remove interleaved-thinking from beta header
            beta_parts = [b.strip() for b in anthropic_beta.split(",")]
            beta_parts = [b for b in beta_parts if "interleaved-thinking" not in b]
            anthropic_beta = ",".join(beta_parts) if beta_parts else None
            logger.debug(
                f"Filtered out interleaved-thinking beta for model {effective_model}"
            )

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

                # We now check for context window errors before raise_for_status()

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

                # Check for context window errors before raising
                if response.status_code == 400:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", {}).get("message", "")

                        # Check for context window exceeded errors
                        context_error_indicators = [
                            "maximum context length",
                            "context_length_exceeded",
                            "max_tokens_exceeded",
                            "request_too_large",
                            "exceeds maximum context length",
                            "message length exceeds limit",
                            "input is too long",
                            "token limit",
                            "context window",
                        ]

                        if any(
                            indicator in error_msg.lower()
                            for indicator in context_error_indicators
                        ):
                            logger.warning(
                                f"Context window exceeded detected in Anthropic response: {error_msg}"
                            )
                            raise ContextWindowExceededError(
                                message=error_msg,
                                backend=self.name,
                                model=effective_model,
                                messages=messages,  # Store original messages for compression
                            )
                    except (ValueError, json.JSONDecodeError):
                        pass

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

            except ContextWindowExceededError:
                # Re-raise context window errors without wrapping
                raise
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
        """Create a streaming message using Anthropic's streaming API."""
        # Get the effective model to use
        effective_model = self.get_effective_model(model)

        # Check if this model needs thinking support
        anthropic_beta = kwargs.get("anthropic_beta")
        thinking_models = [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-4-opus-20250514",
            "claude-4-sonnet-20250514",
        ]

        if effective_model in thinking_models:
            if not anthropic_beta:
                anthropic_beta = "interleaved-thinking-2025-05-14"
            elif "interleaved-thinking-2025-05-14" not in anthropic_beta:
                anthropic_beta = f"{anthropic_beta},interleaved-thinking-2025-05-14"
            kwargs["anthropic_beta"] = anthropic_beta

        # Build request data with streaming enabled
        request_data = {
            "model": effective_model,
            "messages": messages,
            "stream": True,  # Enable streaming
        }

        # Check if we're using OAuth
        oauth_token = await oauth_manager.get_valid_token()
        if oauth_token:
            claude_code_system = {
                "type": "text",
                "text": "You are Claude Code, Anthropic's official CLI for Claude.",
            }

            if system:
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
            if system:
                request_data["system"] = system

        # Handle max_tokens with model-specific limits
        if max_tokens:
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

        # Add any additional parameters
        internal_params = [
            "x_api_key",
            "anthropic_version",
            "anthropic_beta",
            "difficulty_rating",
        ]
        filtered_params = ["container", "mcp_servers"]
        non_thinking_models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]

        for key, value in kwargs.items():
            if key not in request_data and key not in internal_params:
                if key == "thinking" and effective_model in non_thinking_models:
                    logger.debug(
                        f"Filtering out 'thinking' parameter for model {effective_model}"
                    )
                    continue
                if key in filtered_params:
                    logger.debug(
                        f"Filtering out '{key}' parameter (not supported by Anthropic API)"
                    )
                    continue
                request_data[key] = value

        # Extract API headers
        x_api_key = kwargs.get("x_api_key", self.api_key)
        anthropic_version = kwargs.get("anthropic_version", "2023-06-01")
        anthropic_beta = kwargs.get("anthropic_beta")

        # Filter out interleaved-thinking beta for models that don't support it
        if effective_model in non_thinking_models and anthropic_beta:
            # Remove interleaved-thinking from beta header
            beta_parts = [b.strip() for b in anthropic_beta.split(",")]
            beta_parts = [b for b in beta_parts if "interleaved-thinking" not in b]
            anthropic_beta = ",".join(beta_parts) if beta_parts else None
            logger.debug(
                f"Filtered out interleaved-thinking beta for streaming model {effective_model}"
            )

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

                # Log headers for debugging
                safe_headers = {
                    k: v if k != "authorization" else "Bearer ***"
                    for k, v in headers.items()
                }
                logger.debug(f"Streaming request headers (attempt {attempt + 1}): {safe_headers}")

                # Make streaming request
                async with self.client.stream(
                    "POST",
                    f"{self.base_url}/v1/messages",
                    json=request_data,
                    headers=headers,
                ) as response:
                    # Handle 401 errors with OAuth token refresh
                    if response.status_code == 401 and attempt < max_retries - 1:
                        oauth_token = await oauth_manager.get_valid_token()
                        if oauth_token:
                            logger.info(
                                f"Received 401 error in streaming, attempting OAuth token refresh (attempt {attempt + 1}/{max_retries})"
                            )
                            try:
                                stored_token = oauth_manager.load_token()
                                if stored_token and stored_token.refresh_token:
                                    await oauth_manager.refresh_access_token(
                                        stored_token.refresh_token
                                    )
                                    logger.info(
                                        "OAuth token refreshed successfully, retrying streaming request"
                                    )
                                    continue
                            except Exception as refresh_error:
                                logger.error(
                                    f"Failed to refresh OAuth token: {refresh_error}"
                                )

                    # Check for errors before streaming
                    if response.status_code != 200:
                        error_text = await response.aread()
                        try:
                            error_data = json.loads(error_text)
                            error_msg = error_data.get("error", {}).get("message", "")

                            # Check for context window errors
                            if response.status_code == 400:
                                context_error_indicators = [
                                    "maximum context length",
                                    "context_length_exceeded",
                                    "max_tokens_exceeded",
                                    "request_too_large",
                                    "exceeds maximum context length",
                                    "message length exceeds limit",
                                    "input is too long",
                                    "token limit",
                                    "context window",
                                ]
                                if any(
                                    indicator in error_msg.lower()
                                    for indicator in context_error_indicators
                                ):
                                    logger.warning(
                                        f"Context window exceeded in streaming: {error_msg}"
                                    )
                                    raise ContextWindowExceededError(
                                        message=error_msg,
                                        backend=self.name,
                                        model=effective_model,
                                        messages=messages,
                                    )
                        except (ValueError, json.JSONDecodeError):
                            pass

                        response.raise_for_status()

                    # Stream the response line by line
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue

                        # Parse SSE format: "event: <type>" and "data: <json>"
                        if line.startswith("event:"):
                            continue  # Skip event type lines, we get type from data

                        if line.startswith("data:"):
                            data_str = line[5:].strip()  # Remove "data:" prefix
                            if data_str == "[DONE]":
                                break

                            try:
                                event_data = json.loads(data_str)
                                yield event_data
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse SSE data: {e}, line: {data_str}")
                                continue

                # Success, exit retry loop
                break

            except httpx.HTTPStatusError as e:
                if attempt == max_retries - 1 or e.response.status_code != 401:
                    error = convert_backend_error(e, self.name)
                    raise error
                logger.warning(
                    f"HTTP {e.response.status_code} error in streaming (attempt {attempt + 1}), retrying..."
                )

            except ContextWindowExceededError:
                raise
            except Exception as e:
                raise BackendError(
                    f"Anthropic streaming error: {str(e)}", backend=self.name
                )

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

                # Check for context window errors before raising
                if response.status_code == 400:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", {}).get("message", "")

                        # Check for context window exceeded errors
                        context_error_indicators = [
                            "maximum context length",
                            "context_length_exceeded",
                            "max_tokens_exceeded",
                            "request_too_large",
                            "exceeds maximum context length",
                            "message length exceeds limit",
                            "input is too long",
                            "token limit",
                            "context window",
                        ]

                        if any(
                            indicator in error_msg.lower()
                            for indicator in context_error_indicators
                        ):
                            logger.warning(
                                f"Context window exceeded detected in count_tokens: {error_msg}"
                            )
                            raise ContextWindowExceededError(
                                message=error_msg,
                                backend=self.name,
                                model=model,
                                messages=messages,  # Store original messages for compression
                            )
                    except (ValueError, json.JSONDecodeError):
                        pass

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

            except ContextWindowExceededError:
                # Re-raise context window errors from count_tokens
                raise
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
            "claude-haiku-4-5-20251001",
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
