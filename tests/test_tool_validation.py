"""
Test tool use/result validation and preservation.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inferswitch.utils.tool_validation import (
    validate_tool_pairs,
    extract_tool_use_ids,
    extract_tool_result_ids,
    filter_messages_preserving_tool_pairs,
    remove_orphaned_tool_results,
)


def test_extract_tool_use_ids():
    """Test extracting tool_use IDs from messages."""
    msg = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me help you"},
            {"type": "tool_use", "id": "toolu_123", "name": "calculator", "input": {}},
            {"type": "tool_use", "id": "toolu_456", "name": "search", "input": {}},
        ],
    }

    ids = extract_tool_use_ids(msg)
    assert ids == {"toolu_123", "toolu_456"}

    # Test with no tool_use blocks
    msg_no_tools = {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
    assert extract_tool_use_ids(msg_no_tools) == set()


def test_extract_tool_result_ids():
    """Test extracting tool_result IDs from messages."""
    msg = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_123", "content": "42"},
            {"type": "tool_result", "tool_use_id": "toolu_456", "content": "result"},
        ],
    }

    ids = extract_tool_result_ids(msg)
    assert ids == {"toolu_123", "toolu_456"}


def test_validate_tool_pairs_valid():
    """Test validation with valid tool pairs."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Calculate 2+2"}],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me calculate"},
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "calculator",
                    "input": {"expression": "2+2"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_123", "content": "4"}
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "The result is 4"}],
        },
    ]

    assert validate_tool_pairs(messages) is True
    print("✓ Valid tool pairs validated correctly")


def test_validate_tool_pairs_invalid():
    """Test validation with invalid tool pairs (missing tool_use)."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Calculate 2+2"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Let me calculate"}],
        },
        {
            "role": "user",
            "content": [
                # This tool_result references a tool_use that doesn't exist
                {"type": "tool_result", "tool_use_id": "toolu_123", "content": "4"}
            ],
        },
    ]

    assert validate_tool_pairs(messages) is False
    print("✓ Invalid tool pairs detected correctly")


def test_filter_messages_preserving_tool_pairs():
    """Test filtering messages while preserving tool pairs."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Old message"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Old response"}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Calculate 2+2"}],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me calculate"},
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "calculator",
                    "input": {},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_123", "content": "4"}
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "The result is 4"}],
        },
    ]

    # Try to keep only messages 4 and 5 (tool_result and final response)
    # The function should automatically add message 3 (tool_use) to preserve the pair
    keep_indices = {4, 5}

    filtered = filter_messages_preserving_tool_pairs(messages, keep_indices)

    # Should include message 3 (tool_use), 4 (tool_result), and 5 (final response)
    assert len(filtered) == 3
    assert validate_tool_pairs(filtered) is True
    print("✓ Tool pairs preserved during filtering")


def test_remove_orphaned_tool_results():
    """Test removing orphaned tool_result blocks."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Some text"},
                # This tool_result is orphaned (no corresponding tool_use)
                {"type": "tool_result", "tool_use_id": "toolu_999", "content": "result"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Response"}],
        },
    ]

    cleaned = remove_orphaned_tool_results(messages)

    # The orphaned tool_result should be removed
    assert len(cleaned) == 3
    assert validate_tool_pairs(cleaned) is True

    # Check that the orphaned tool_result was removed from message 1
    msg_content = cleaned[1]["content"]
    assert len(msg_content) == 1
    assert msg_content[0]["type"] == "text"
    print("✓ Orphaned tool_results removed correctly")


def test_complex_scenario():
    """Test a complex scenario with multiple tool pairs and truncation."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Calculate 1+1"}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_001",
                    "name": "calculator",
                    "input": {},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_001", "content": "2"}
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "The result is 2"}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Calculate 2+2"}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_002",
                    "name": "calculator",
                    "input": {},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_002", "content": "4"}
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "The result is 4"}],
        },
    ]

    # Simulate truncation - try to keep only the last 4 messages
    keep_indices = {4, 5, 6, 7}

    filtered = filter_messages_preserving_tool_pairs(messages, keep_indices)

    # Should preserve the second tool pair (messages 4-7)
    assert validate_tool_pairs(filtered) is True
    assert len(filtered) == 4
    print("✓ Complex tool pair scenario handled correctly")


def main():
    """Run all tests."""
    print("Testing tool validation utilities...\n")

    test_extract_tool_use_ids()
    test_extract_tool_result_ids()
    test_validate_tool_pairs_valid()
    test_validate_tool_pairs_invalid()
    test_filter_messages_preserving_tool_pairs()
    test_remove_orphaned_tool_results()
    test_complex_scenario()

    print("\n✅ All tool validation tests passed!")


if __name__ == "__main__":
    main()
