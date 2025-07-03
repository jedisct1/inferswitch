"""
Response normalization utilities for converting between backend formats.
"""

from typing import Dict, Any, List, Optional


class ResponseNormalizer:
    """Normalizes responses from different backends to Anthropic format."""

    @staticmethod
    def openai_to_anthropic(openai_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert OpenAI response format to Anthropic format.

        Args:
            openai_response: Response from OpenAI API

        Returns:
            Response in Anthropic format
        """
        # Extract the message content
        choice = openai_response["choices"][0]
        message = choice.get("message", {})
        content = message.get("content", "")

        # Map finish reason
        finish_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "function_call": "tool_use",
            "content_filter": "stop_sequence",
        }
        finish_reason = choice.get("finish_reason", "stop")
        stop_reason = finish_reason_map.get(finish_reason, "end_turn")

        # Map usage
        usage = openai_response.get("usage", {})
        anthropic_usage = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }

        # Build Anthropic response
        anthropic_response = {
            "id": openai_response.get("id", "msg_unknown"),
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": content}],
            "model": openai_response.get("model", "unknown"),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": anthropic_usage,
        }

        return anthropic_response

    @staticmethod
    def openai_to_anthropic_messages(
        openai_messages: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Convert OpenAI messages format to Anthropic format.

        Args:
            openai_messages: List of OpenAI-style messages

        Returns:
            Tuple of (messages, system_prompt)
        """
        anthropic_messages = []
        system_prompt = None

        for msg in openai_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Anthropic uses separate system parameter
                if system_prompt:
                    system_prompt += "\n\n" + content
                else:
                    system_prompt = content
            elif role in ["user", "assistant"]:
                # Convert to Anthropic format
                anthropic_msg = {
                    "role": role,
                    "content": content
                    if isinstance(content, list)
                    else [{"type": "text", "text": content}],
                }
                anthropic_messages.append(anthropic_msg)

        return anthropic_messages, system_prompt

    @staticmethod
    def anthropic_to_openai_messages(
        anthropic_messages: List[Dict[str, Any]], system: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Convert Anthropic messages format to OpenAI format.

        Args:
            anthropic_messages: List of Anthropic-style messages
            system: System prompt (if any)

        Returns:
            List of OpenAI-style messages
        """
        openai_messages = []

        # Add system message if present
        if system:
            # Handle system as string or list of content blocks
            if isinstance(system, str):
                openai_messages.append({"role": "system", "content": system})
            elif isinstance(system, list):
                # Extract text from content blocks
                system_text = ""
                for block in system:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            system_text += block.get("text", "")
                        elif "text" in block:  # Handle {"text": "..."} format
                            system_text += block.get("text", "")
                    elif isinstance(block, str):
                        system_text += block

                if system_text:
                    openai_messages.append({"role": "system", "content": system_text})

        # Convert messages
        for msg in anthropic_messages:
            role = msg.get("role", "user")

            # Extract text content
            content = ""
            msg_content = msg.get("content")

            if isinstance(msg_content, list):
                for block in msg_content:
                    if isinstance(block, dict):
                        # Handle Anthropic format: {"type": "text", "text": "..."}
                        if block.get("type") == "text":
                            content += block.get("text", "")
                        # Handle simplified format: {"text": "..."}
                        elif "text" in block and "type" not in block:
                            content += block.get("text", "")
                    elif isinstance(block, str):
                        content += block
            elif isinstance(msg_content, str):
                content = msg_content
            else:
                content = str(msg_content) if msg_content else ""

            openai_messages.append({"role": role, "content": content})

        return openai_messages

    @staticmethod
    def normalize_streaming_chunk(
        chunk: Dict[str, Any], source_format: str
    ) -> Dict[str, Any]:
        """
        Normalize streaming chunks to Anthropic SSE format.

        Args:
            chunk: Streaming chunk from backend
            source_format: Source format ("openai" or "anthropic")

        Returns:
            Normalized chunk in Anthropic SSE format
        """
        # Ensure chunk is a dictionary
        if not isinstance(chunk, dict):
            return {}

        if source_format == "openai":
            # OpenAI stream format
            if (
                chunk.get("choices")
                and isinstance(chunk.get("choices"), list)
                and len(chunk["choices"]) > 0
                and chunk["choices"][0].get("delta")
            ):
                delta = chunk["choices"][0]["delta"]
                content = delta.get("content", "")

                if content:
                    # Content delta event
                    return {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": content},
                    }
                elif delta.get("role"):
                    # Start event
                    return {
                        "type": "message_start",
                        "message": {
                            "id": chunk.get("id", "msg_unknown"),
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": chunk.get("model", "unknown"),
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        },
                    }

            # Check for end
            if (
                chunk.get("choices")
                and isinstance(chunk.get("choices"), list)
                and len(chunk["choices"]) > 0
                and chunk["choices"][0].get("finish_reason")
            ):
                return {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 0},
                }

        # Already in Anthropic format or unknown
        return chunk
