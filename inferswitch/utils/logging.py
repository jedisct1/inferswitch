"""
Logging utilities for request/response tracking.
"""

import json
import logging
from datetime import datetime

from ..config import LOG_FILE, DEFAULT_TRUNCATION_LIMIT
from .chat_template import (
    convert_to_chat_template,
    apply_chat_template,
    truncate_chat_template_to_fit,
)

logger = logging.getLogger(__name__)


def log_request(endpoint: str, request_data: dict, difficulty_rating: float = None):
    """Log an incoming request to the log file."""
    with open(LOG_FILE, "a") as f:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        f.write(f"\n{'=' * 80}\n")
        f.write(f"[REQUEST] {timestamp}\n")
        f.write(f"Endpoint: {endpoint}\n")

        if difficulty_rating is not None:
            f.write(f"Difficulty Rating: {difficulty_rating:.1f}/5.0\n")

        # Check for cache_control presence
        has_cache_control = False
        if "system" in request_data and isinstance(request_data["system"], list):
            for item in request_data["system"]:
                if isinstance(item, dict) and "cache_control" in item:
                    has_cache_control = True
                    break

        if "messages" in request_data:
            for msg in request_data["messages"]:
                if isinstance(msg.get("content"), list):
                    for content in msg["content"]:
                        if isinstance(content, dict) and "cache_control" in content:
                            has_cache_control = True
                            break

        if has_cache_control:
            f.write("Cache Control: Present\n")

        f.write("Request Body:\n")
        f.write(json.dumps(request_data, indent=2)[:5000])  # Limit to 5000 chars
        if len(json.dumps(request_data)) > 5000:
            f.write("\n... (truncated)")
        f.write("\n")


def log_chat_template(endpoint: str, request_dict: dict):
    """Log the chat template representation of a request."""
    try:
        chat_messages = convert_to_chat_template(request_dict)

        # Check if truncation is needed (using character count as approximation)
        total_size = sum(len(msg.get("content", "")) for msg in chat_messages)

        # If too large, truncate
        if total_size > DEFAULT_TRUNCATION_LIMIT:
            original_count = len(chat_messages)
            chat_messages = truncate_chat_template_to_fit(
                chat_messages,
                max_context_size=DEFAULT_TRUNCATION_LIMIT,
                model_name=request_dict.get("model", "claude-3-opus-20240229"),
            )
            truncated_count = original_count - len(chat_messages)
        else:
            truncated_count = 0

        chat_string = apply_chat_template(chat_messages, add_generation_prompt=True)

        with open(LOG_FILE, "a") as f:
            f.write("\n[CHAT TEMPLATE]\n")
            f.write(f"Messages: {len(chat_messages)}")
            if truncated_count > 0:
                f.write(f" (truncated {truncated_count} messages)")
            f.write("\n")
            f.write(f"Formatted:\n{chat_string[:1000]}")
            if len(chat_string) > 1000:
                f.write("\n... (truncated)")
            f.write("\n")
    except Exception as e:
        logger.error(f"Error generating chat template: {e}")


def log_streaming_progress(
    elapsed_seconds: float, tokens_received: int = 0, model: str = None
):
    """Log progress for long-running streaming responses."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Build progress message
    progress_msg = f"Elapsed: {elapsed_seconds:.1f}s"
    if tokens_received > 0:
        progress_msg += f", Tokens received: ~{tokens_received}"
    if model:
        progress_msg += f", Model: {model}"
    progress_msg += " - Response still streaming..."

    # Log to file with full timestamp
    with open(LOG_FILE, "a") as f:
        f.write(f"\n[STREAMING PROGRESS] {timestamp}\n")
        f.write(f"{progress_msg}\n")

    # Log to console using logger (will appear on stderr)
    logger.info(f"[STREAMING PROGRESS] {progress_msg}")
