"""
Tool use/result validation utilities.

Ensures that tool_result blocks always have corresponding tool_use blocks.
"""

from typing import List, Dict, Any, Set
import logging

logger = logging.getLogger(__name__)


def extract_tool_use_ids(message: Dict[str, Any]) -> Set[str]:
    """
    Extract all tool_use IDs from a message.

    Args:
        message: A message dictionary

    Returns:
        Set of tool_use IDs found in the message
    """
    tool_use_ids = set()
    content = message.get("content", [])

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_id = block.get("id")
                if tool_id:
                    tool_use_ids.add(tool_id)

    return tool_use_ids


def extract_tool_result_ids(message: Dict[str, Any]) -> Set[str]:
    """
    Extract all tool_use_ids referenced by tool_result blocks.

    Args:
        message: A message dictionary

    Returns:
        Set of tool_use_ids referenced by tool_result blocks
    """
    tool_result_ids = set()
    content = message.get("content", [])

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tool_use_id = block.get("tool_use_id")
                if tool_use_id:
                    tool_result_ids.add(tool_use_id)

    return tool_result_ids


def validate_tool_pairs(messages: List[Dict[str, Any]]) -> bool:
    """
    Validate that all tool_result blocks have corresponding tool_use blocks.

    Args:
        messages: List of message dictionaries

    Returns:
        True if all tool_result blocks have corresponding tool_use blocks,
        False otherwise
    """
    # Track all tool_use IDs seen so far
    seen_tool_use_ids = set()

    for i, msg in enumerate(messages):
        # Collect tool_use IDs from this message
        tool_use_ids = extract_tool_use_ids(msg)
        seen_tool_use_ids.update(tool_use_ids)

        # Check tool_result IDs in this message
        tool_result_ids = extract_tool_result_ids(msg)

        # For each tool_result, check if we've seen the corresponding tool_use
        for result_id in tool_result_ids:
            # According to Anthropic API, tool_result must reference a tool_use
            # from the PREVIOUS message (not this one, not earlier)
            if i == 0:
                # First message can't have valid tool_results
                logger.warning(
                    f"Message {i} has tool_result for {result_id} but it's the first message"
                )
                return False

            # Check if the previous message has this tool_use_id
            prev_msg = messages[i - 1]
            prev_tool_use_ids = extract_tool_use_ids(prev_msg)

            if result_id not in prev_tool_use_ids:
                logger.warning(
                    f"Message {i} has tool_result for {result_id} but previous message doesn't have it"
                )
                return False

    return True


def filter_messages_preserving_tool_pairs(
    messages: List[Dict[str, Any]], keep_indices: Set[int]
) -> List[Dict[str, Any]]:
    """
    Filter messages while preserving tool_use/tool_result pairs.

    If a message with tool_result is kept, the previous message with
    corresponding tool_use must also be kept.

    Args:
        messages: List of message dictionaries
        keep_indices: Set of indices to keep

    Returns:
        Filtered list of messages with tool pairs preserved
    """
    if not messages:
        return []

    # Ensure we keep tool_use messages for any kept tool_result messages
    adjusted_keep_indices = set(keep_indices)

    for i in keep_indices:
        if i >= len(messages):
            continue

        msg = messages[i]
        tool_result_ids = extract_tool_result_ids(msg)

        if tool_result_ids and i > 0:
            # This message has tool_results, we need to keep the previous message
            prev_msg = messages[i - 1]
            prev_tool_use_ids = extract_tool_use_ids(prev_msg)

            # Check if any of the tool_results reference the previous message
            if tool_result_ids & prev_tool_use_ids:
                # Previous message has the tool_use, we must keep it
                adjusted_keep_indices.add(i - 1)
                logger.debug(
                    f"Preserving message {i-1} (has tool_use) for message {i} (has tool_result)"
                )

    # Filter messages
    filtered = [msg for i, msg in enumerate(messages) if i in adjusted_keep_indices]

    # Validate the result
    if not validate_tool_pairs(filtered):
        logger.error("Filtered messages still have invalid tool pairs!")
        # In this case, we should probably be more aggressive and remove
        # the problematic tool_result messages
        return remove_orphaned_tool_results(filtered)

    return filtered


def remove_orphaned_tool_results(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove tool_result blocks that don't have corresponding tool_use blocks.

    This is a last resort to fix invalid message sequences.

    Args:
        messages: List of message dictionaries

    Returns:
        Messages with orphaned tool_results removed
    """
    if not messages:
        return []

    cleaned_messages = []

    for i, msg in enumerate(messages):
        # Get tool_result IDs in this message
        tool_result_ids = extract_tool_result_ids(msg)

        if not tool_result_ids:
            # No tool_results, keep as-is
            cleaned_messages.append(msg)
            continue

        # Check which tool_results are valid
        valid_tool_result_ids = set()
        if i > 0:
            prev_tool_use_ids = extract_tool_use_ids(messages[i - 1])
            valid_tool_result_ids = tool_result_ids & prev_tool_use_ids

        # If all tool_results are valid, keep message as-is
        if valid_tool_result_ids == tool_result_ids:
            cleaned_messages.append(msg)
            continue

        # Filter out invalid tool_result blocks
        content = msg.get("content", [])
        if isinstance(content, list):
            filtered_content = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_use_id = block.get("tool_use_id")
                    if tool_use_id in valid_tool_result_ids:
                        filtered_content.append(block)
                    else:
                        logger.warning(
                            f"Removing orphaned tool_result block for {tool_use_id} in message {i}"
                        )
                else:
                    # Keep non-tool_result blocks
                    filtered_content.append(block)

            # Only keep message if it has content left
            if filtered_content:
                cleaned_msg = msg.copy()
                cleaned_msg["content"] = filtered_content
                cleaned_messages.append(cleaned_msg)
            else:
                logger.warning(f"Message {i} had only orphaned tool_results, removing entirely")

        else:
            # Content is not a list, keep as-is
            cleaned_messages.append(msg)

    return cleaned_messages
