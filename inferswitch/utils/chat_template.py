"""
Chat template conversion and manipulation utilities.
"""

import json
from typing import List, Dict
import logging

from ..config import MODEL_CONTEXT_SIZES, TRUNCATION_BUFFER
from .tool_validation import validate_tool_pairs, remove_orphaned_tool_results

logger = logging.getLogger(__name__)


def convert_to_chat_template(request_dict: dict) -> List[Dict[str, str]]:
    """
    Convert an Anthropic API request to Hugging Face chat template format.

    The chat template format is a list of dictionaries with 'role' and 'content' keys.
    System messages are included as a message with role 'system'.

    Args:
        request_dict: The Anthropic API request as a dictionary

    Returns:
        List of message dictionaries in chat template format
    """
    chat_messages = []

    # Add system message if present
    system = request_dict.get("system")
    if system:
        if isinstance(system, str):
            # Simple string system message
            chat_messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # Array of system message objects - concatenate all text
            system_texts = []
            for sys_obj in system:
                if isinstance(sys_obj, dict) and "text" in sys_obj:
                    system_texts.append(sys_obj["text"])
            if system_texts:
                chat_messages.append(
                    {"role": "system", "content": "\n\n".join(system_texts)}
                )

    # Add user and assistant messages
    messages = request_dict.get("messages", [])
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Convert content to string format
        if isinstance(content, str):
            content_text = content
        elif isinstance(content, list):
            # Extract text from content blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text" and "text" in block:
                        text_parts.append(block["text"])
                    elif block.get("type") == "image" and "source" in block:
                        # For images, add a placeholder
                        text_parts.append("[Image]")
                    elif block.get("type") == "tool_use":
                        # For tool use, add the tool information
                        tool_name = block.get("name", "unknown_tool")
                        tool_input = block.get("input", {})
                        text_parts.append(
                            f"[Tool Use: {tool_name}]\n{json.dumps(tool_input, indent=2)}"
                        )
                    elif block.get("type") == "tool_result":
                        # For tool results
                        tool_use_id = block.get("tool_use_id", "unknown")
                        content = block.get("content", "")
                        text_parts.append(f"[Tool Result: {tool_use_id}]\n{content}")
            content_text = "\n\n".join(text_parts)
        else:
            content_text = str(content)

        chat_messages.append({"role": role, "content": content_text})

    return chat_messages


def apply_chat_template(
    chat_messages: List[Dict[str, str]],
    add_generation_prompt: bool = False,
    tokenize: bool = False,
) -> str:
    """
    Apply a simple chat template to format messages as a string.
    This mimics the Hugging Face transformers chat template functionality.

    Args:
        chat_messages: List of message dictionaries with 'role' and 'content'
        add_generation_prompt: Whether to add a prompt for the assistant to respond
        tokenize: If True, return a note that tokenization would happen (we don't actually tokenize)

    Returns:
        Formatted string representation of the chat
    """
    if tokenize:
        # In a real implementation, this would return token IDs
        return f"[Tokenization requested but not implemented. Would tokenize {len(chat_messages)} messages]"

    # Simple chat template format similar to ChatML
    formatted_parts = []

    for message in chat_messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        # Format each message
        formatted_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    # Add generation prompt if requested
    if add_generation_prompt:
        formatted_parts.append("<|im_start|>assistant\n")

    return "\n".join(formatted_parts)


def truncate_chat_template_to_fit(
    chat_messages: List[Dict[str, str]],
    max_context_size: int = 200000,
    model_name: str = "claude-3-opus-20240229",
) -> List[Dict[str, str]]:
    """
    Intelligently truncate chat messages to fit within context size.

    Args:
        chat_messages: List of message dictionaries with 'role' and 'content'
        max_context_size: Maximum context size in characters (approximation of tokens)
        model_name: Model name to estimate appropriate context size

    Returns:
        Truncated list of messages that fits within context
    """
    # Adjust max context based on model
    for model_prefix, context_tokens in MODEL_CONTEXT_SIZES.items():
        if model_prefix in model_name:
            max_context_size = context_tokens * 4  # ~4 chars per token
            break
    else:
        # Default for non-matching models
        max_context_size = max_context_size * 4

    # Calculate current size
    def calculate_size(messages):
        return sum(
            len(msg.get("content", "")) + len(msg.get("role", "")) + 50
            for msg in messages
        )

    current_size = calculate_size(chat_messages)

    # If it fits, return as-is
    if current_size <= max_context_size:
        return chat_messages

    # Separate system messages and conversation
    system_messages = []
    conversation_messages = []

    for msg in chat_messages:
        if msg.get("role") == "system":
            system_messages.append(msg)
        else:
            conversation_messages.append(msg)

    # Always keep system messages
    truncated_messages = system_messages.copy()
    system_size = calculate_size(system_messages)

    # If system messages alone exceed limit, truncate system content
    if system_size > max_context_size * 0.5:  # Don't let system take more than half
        for msg in truncated_messages:
            if msg.get("role") == "system":
                # Truncate system message content
                max_system_len = int((max_context_size * 0.5) / len(system_messages))
                msg["content"] = (
                    msg["content"][:max_system_len] + "\n... (system message truncated)"
                )
        system_size = calculate_size(truncated_messages)

    # Calculate remaining space for conversation
    remaining_space = max_context_size - system_size - TRUNCATION_BUFFER

    # Find how many recent messages we can keep
    # Start from the end and work backwards
    kept_conversation = []
    current_conv_size = 0

    # Always try to keep the last message if it's from the user
    if conversation_messages and conversation_messages[-1].get("role") == "user":
        last_msg = conversation_messages[-1]
        last_msg_size = len(last_msg.get("content", "")) + 50
        if last_msg_size < remaining_space:
            kept_conversation.append(last_msg)
            current_conv_size += last_msg_size
            conversation_messages = conversation_messages[:-1]

    # Now add message pairs (user + assistant) from most recent
    i = len(conversation_messages) - 1
    while i >= 1 and current_conv_size < remaining_space:
        # Look for assistant message
        if conversation_messages[i].get("role") == "assistant" and i > 0:
            # Check if previous is user message
            if conversation_messages[i - 1].get("role") == "user":
                pair_size = (
                    len(conversation_messages[i].get("content", ""))
                    + len(conversation_messages[i - 1].get("content", ""))
                    + 100
                )

                if current_conv_size + pair_size < remaining_space:
                    # Add the pair
                    kept_conversation.insert(0, conversation_messages[i - 1])
                    kept_conversation.insert(1, conversation_messages[i])
                    current_conv_size += pair_size
                    i -= 2
                else:
                    break
            else:
                i -= 1
        else:
            i -= 1

    # If we removed messages, add a truncation notice
    if len(kept_conversation) < len(conversation_messages):
        truncation_notice = {
            "role": "system",
            "content": f"[Previous {len(conversation_messages) - len(kept_conversation)} messages truncated to fit context]",
        }
        truncated_messages.append(truncation_notice)

    # Combine system messages with kept conversation
    truncated_messages.extend(kept_conversation)

    # Validate and fix tool_use/tool_result pairs
    if not validate_tool_pairs(truncated_messages):
        logger.warning(
            "Truncated messages have invalid tool_use/tool_result pairs. Cleaning up..."
        )
        truncated_messages = remove_orphaned_tool_results(truncated_messages)

    return truncated_messages


def remove_oldest_message_pair(
    chat_messages: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Remove the oldest user/assistant message pair from the chat messages.

    This function finds and removes the first user/assistant pair in the conversation,
    preserving system messages and any standalone messages.

    Args:
        chat_messages: List of message dictionaries with 'role' and 'content'

    Returns:
        List of messages with the oldest user/assistant pair removed
    """
    if not chat_messages:
        return chat_messages

    # Create a copy to avoid modifying the original
    messages = chat_messages.copy()

    # Find the first user message index (skipping system messages)
    first_user_idx = -1
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            first_user_idx = i
            break

    # If no user message found, return as-is
    if first_user_idx == -1:
        return messages

    # Check if there's an assistant message immediately after
    if (
        first_user_idx + 1 < len(messages)
        and messages[first_user_idx + 1].get("role") == "assistant"
    ):
        # Remove both the user and assistant messages
        del messages[first_user_idx : first_user_idx + 2]
    else:
        # Just remove the user message if no assistant follows
        del messages[first_user_idx]

    return messages
