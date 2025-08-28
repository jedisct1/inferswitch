"""
OpenAI/LM-Studio backend implementation.
"""

import httpx
import json
from typing import Dict, Any, List, Optional, AsyncIterator
from .base import BaseBackend, BackendConfig, BackendResponse
from .normalizer import ResponseNormalizer
from .errors import BackendError, convert_backend_error, ContextWindowExceededError
from ..utils.logging import log_request
from ..utils import estimate_tokens_fallback, get_logger

logger = get_logger(__name__)


class OpenAIBackend(BaseBackend):
    """Backend implementation for OpenAI API and compatible servers (e.g., LM-Studio)."""

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self.client = httpx.AsyncClient(
            base_url=self.base_url, timeout=config.timeout, headers=self._get_headers()
        )
        self._available_models: Optional[List[str]] = None

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for OpenAI API requests."""
        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self.config.headers:
            headers.update(self.config.headers)

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
        """Create a chat completion using OpenAI API."""
        try:
            # Get the effective model to use
            effective_model = self.get_effective_model(model)

            # Convert from Anthropic format to OpenAI format
            openai_messages = ResponseNormalizer.anthropic_to_openai_messages(
                messages, system
            )

            # Build request
            request_data = {
                "model": effective_model,
                "messages": openai_messages,
                "stream": False,  # Always non-streaming for base method
            }

            if max_tokens:
                request_data["max_tokens"] = max_tokens
            if temperature is not None:
                request_data["temperature"] = temperature

            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in request_data:
                    request_data[key] = value

            # Log request
            log_request("/v1/chat/completions", request_data)

            # Make request
            response = await self.client.post("/v1/chat/completions", json=request_data)

            # Check for context window errors before raising
            if response.status_code == 400:
                try:
                    error_data = response.json()
                    error_msg = ""

                    # OpenAI API error format
                    if "error" in error_data:
                        error_info = error_data["error"]
                        error_msg = error_info.get("message", "")
                    else:
                        error_msg = str(error_data)

                    # Check for context window exceeded errors
                    context_error_indicators = [
                        "context_length_exceeded",
                        "maximum context length",
                        "max_tokens_exceeded",
                        "exceeds maximum context length",
                        "request too large",
                        "token limit exceeded",
                        "context window",
                        "input is too long",
                        "message length exceeds",
                        "too many tokens",
                    ]

                    if any(
                        indicator in error_msg.lower()
                        for indicator in context_error_indicators
                    ):
                        logger.warning(
                            f"Context window exceeded detected in OpenAI response: {error_msg}"
                        )
                        raise ContextWindowExceededError(
                            message=error_msg,
                            backend=self.name,
                            model=effective_model,
                            messages=messages,  # Store original messages for compression
                        )

                except (ValueError, json.JSONDecodeError) as parse_error:
                    logger.debug(f"Could not parse error response: {parse_error}")
                    pass

            response.raise_for_status()

            # Parse response
            response_data = response.json()

            # Convert to Anthropic format
            anthropic_response = ResponseNormalizer.openai_to_anthropic(response_data)

            # Return normalized response
            return BackendResponse(
                content=anthropic_response["content"],
                model=anthropic_response["model"],
                stop_reason=anthropic_response["stop_reason"],
                usage=anthropic_response["usage"],
                raw_response=response_data,
            )

        except httpx.HTTPStatusError as e:
            error = convert_backend_error(e, self.name)
            raise error
        except ContextWindowExceededError:
            # Re-raise context window errors without wrapping
            raise
        except Exception as e:
            raise BackendError(f"OpenAI backend error: {str(e)}", backend=self.name)

    async def create_message_stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Create a streaming chat completion."""
        try:
            # Get the effective model to use
            effective_model = self.get_effective_model(model)

            # Convert from Anthropic format to OpenAI format
            openai_messages = ResponseNormalizer.anthropic_to_openai_messages(
                messages, system
            )

            # Build request
            request_data = {
                "model": effective_model,
                "messages": openai_messages,
                "stream": True,
            }

            if max_tokens:
                request_data["max_tokens"] = max_tokens
            if temperature is not None:
                request_data["temperature"] = temperature

            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in request_data:
                    request_data[key] = value

            # Log request
            log_request("/v1/chat/completions", request_data)

            # Make streaming request
            async with self.client.stream(
                "POST", "/v1/chat/completions", json=request_data
            ) as response:
                # Check for context window errors before raising
                if response.status_code == 400:
                    try:
                        # For streaming, we need to read content first
                        content = await response.aread()
                        error_data = json.loads(content)
                        error_msg = ""

                        # OpenAI API error format
                        if "error" in error_data:
                            error_info = error_data["error"]
                            error_msg = error_info.get("message", "")
                        else:
                            error_msg = str(error_data)

                        # Check for context window exceeded errors
                        context_error_indicators = [
                            "context_length_exceeded",
                            "maximum context length",
                            "max_tokens_exceeded",
                            "exceeds maximum context length",
                            "request too large",
                            "token limit exceeded",
                            "context window",
                            "input is too long",
                            "message length exceeds",
                            "too many tokens",
                        ]

                        if any(
                            indicator in error_msg.lower()
                            for indicator in context_error_indicators
                        ):
                            logger.warning(
                                f"Context window exceeded detected in OpenAI streaming: {error_msg}"
                            )
                            raise ContextWindowExceededError(
                                message=error_msg,
                                backend=self.name,
                                model=effective_model,
                                messages=messages,  # Store original messages for compression
                            )

                    except (ValueError, json.JSONDecodeError) as parse_error:
                        logger.debug(
                            f"Could not parse streaming error response: {parse_error}"
                        )
                        pass

                response.raise_for_status()

                # Yield Anthropic-style SSE events
                first_chunk = True
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix

                        if data == "[DONE]":
                            # End of stream
                            yield {"type": "message_stop"}
                            break

                        try:
                            chunk = json.loads(data)

                            # Ensure chunk is a dictionary
                            if not isinstance(chunk, dict):
                                continue

                            if first_chunk:
                                # Send message_start event
                                first_chunk = False
                                yield {
                                    "type": "message_start",
                                    "message": {
                                        "id": chunk.get("id", "msg_unknown"),
                                        "type": "message",
                                        "role": "assistant",
                                        "content": [],
                                        "model": chunk.get("model", effective_model),
                                        "usage": {
                                            "input_tokens": 0,
                                            "output_tokens": 0,
                                        },
                                    },
                                }

                                # Send content_block_start
                                yield {
                                    "type": "content_block_start",
                                    "index": 0,
                                    "content_block": {"type": "text", "text": ""},
                                }

                            # Convert chunk to Anthropic format
                            normalized = ResponseNormalizer.normalize_streaming_chunk(
                                chunk, "openai"
                            )

                            # Ensure normalized is also a dictionary
                            if isinstance(normalized, dict):
                                if normalized.get("type") == "content_block_delta":
                                    yield normalized
                                elif normalized.get("type") == "message_delta":
                                    yield normalized

                        except json.JSONDecodeError:
                            continue

        except httpx.HTTPStatusError as e:
            error = convert_backend_error(e, self.name)
            raise error
        except ContextWindowExceededError:
            # Re-raise context window errors without wrapping
            raise
        except Exception as e:
            raise BackendError(f"OpenAI streaming error: {str(e)}", backend=self.name)

    async def count_tokens(
        self, messages: List[Dict[str, Any]], model: str, system: Optional[str] = None
    ) -> Dict[str, int]:
        """Count tokens in messages."""
        # OpenAI doesn't have a separate token counting endpoint
        # We'll make a completion request with max_tokens=1 to get token counts
        try:
            response = await self.create_message(
                messages=messages,
                model=model,
                system=system,
                max_tokens=1,
                temperature=0,
            )

            return response.usage or {"input_tokens": 0, "output_tokens": 0}

        except ContextWindowExceededError:
            # Re-raise context window errors from create_message
            raise
        except Exception:
            # Fallback: estimate tokens using common utility
            return estimate_tokens_fallback(messages, system)

    def supports_model(self, model: str) -> bool:
        """Check if this backend supports a given model."""
        # For OpenAI, check against known models
        if self.config.models:
            return model in self.config.models

        # For LM-Studio, accept any model (it's dynamic)
        if self.name == "lm-studio":
            return True

        # Default OpenAI models
        openai_models = [
            "gpt-4-turbo-preview",
            "gpt-4",
            "gpt-4-32k",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
        ]
        return model in openai_models

    async def health_check(self) -> Dict[str, Any]:
        """Check backend health by listing models."""
        try:
            response = await self.client.get("/v1/models")
            response.raise_for_status()

            data = response.json()
            models = [m["id"] for m in data.get("data", [])]

            # Cache available models
            self._available_models = models

            return {
                "status": "ok",
                "backend": self.name,
                "available_models": len(models),
                "models": models[:5],  # Show first 5 models
            }

        except Exception as e:
            return {"status": "error", "backend": self.name, "error": str(e)}

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
