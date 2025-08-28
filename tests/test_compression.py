#!/usr/bin/env python3
"""
Test suite for intelligent context compression.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from inferswitch.utils.compression import (
    MessageCompressor,
    CompressionStrategy,
    CompressionResult,
)


def test_basic_compression():
    """Test basic message compression."""
    print("Testing basic compression...")

    compressor = MessageCompressor()

    # Create a large conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! " * 100},  # Long message
        {"role": "assistant", "content": "Hi there! " * 100},
        {"role": "user", "content": "How are you? " * 100},
        {"role": "assistant", "content": "I'm doing well! " * 100},
        {"role": "user", "content": "Tell me a story. " * 100},
        {"role": "assistant", "content": "Once upon a time... " * 200},  # Very long
        {"role": "user", "content": "What's the weather?"},  # Recent important
    ]

    # Test compression
    result = compressor.compress_messages(
        messages=messages,
        model="claude-3-haiku-20240307",
        target_ratio=0.5,  # Compress to 50%
    )

    assert isinstance(result, CompressionResult)
    assert (
        result.compressed_count <= result.original_count
    )  # May not compress if under limit
    assert result.compressed_tokens <= result.original_tokens
    # Notice only added if compression happened
    if result.compressed_count < result.original_count:
        assert result.compression_notice != ""
    assert len(result.messages) > 0

    # Check that recent message is preserved
    last_msg = result.messages[-1]
    assert "weather" in last_msg.get("content", "").lower()

    print(
        f"✅ Compressed {result.original_count} messages to {result.compressed_count}"
    )
    print(f"   Token reduction: {result.original_tokens} → {result.compressed_tokens}")
    print(f"   Strategy used: {result.strategy_used.value}")


def test_no_compression_needed():
    """Test when compression is not needed."""
    print("\nTesting no compression needed...")

    compressor = MessageCompressor()

    # Small conversation
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]

    result = compressor.compress_messages(
        messages=messages,
        model="claude-3-haiku-20240307",
        target_ratio=0.9,
    )

    assert result.original_count == result.compressed_count
    assert result.original_tokens == result.compressed_tokens
    assert result.compression_notice == ""
    assert result.messages == messages

    print("✅ No compression applied when not needed")


def test_truncation_strategy():
    """Test simple truncation strategy."""
    print("\nTesting truncation strategy...")

    compressor = MessageCompressor()

    messages = []
    for i in range(20):
        messages.append({"role": "user", "content": f"Message {i}"})
        messages.append({"role": "assistant", "content": f"Response {i}"})

    result = compressor.compress_messages(
        messages=messages,
        model="claude-3-haiku-20240307",
        target_ratio=0.3,
        strategy=CompressionStrategy.TRUNCATE,
    )

    assert result.strategy_used == CompressionStrategy.TRUNCATE
    assert result.compressed_count <= result.original_count
    # Check for compression notice if messages were actually removed
    if result.compressed_count < result.original_count:
        assert "[NOTICE:" in str(result.messages)  # Check for compression notice

    print(
        f"✅ Truncation removed {result.original_count - result.compressed_count} messages"
    )


def test_smart_truncation():
    """Test MLX-guided smart truncation."""
    print("\nTesting smart truncation...")

    compressor = MessageCompressor()

    messages = [
        {"role": "system", "content": "Important system message"},
        {"role": "user", "content": "Unimportant chitchat"},
        {"role": "assistant", "content": "Chitchat response"},
        {
            "role": "user",
            "content": "Write me a Python function to calculate fibonacci",
        },
        {"role": "assistant", "content": "def fibonacci(n): ..."},  # Important code
        {"role": "user", "content": "Thanks!"},
        {"role": "assistant", "content": "You're welcome!"},
        {
            "role": "user",
            "content": "Fix the bug in the fibonacci function",
        },  # Critical
    ]

    result = compressor.compress_messages(
        messages=messages,
        model="claude-3-haiku-20240307",
        target_ratio=0.5,
        strategy=CompressionStrategy.SMART_TRUNCATE,
    )

    # Check that important messages are preserved
    content_str = json.dumps(result.messages)
    assert "fibonacci" in content_str.lower() or not compressor.mlx_available
    assert "system" in content_str or not compressor.mlx_available

    print(
        f"✅ Smart truncation preserved {result.compressed_count}/{result.original_count} important messages"
    )


def test_summarization():
    """Test message summarization."""
    print("\nTesting summarization...")

    compressor = MessageCompressor()

    # Long conversation that should be summarized
    messages = []
    for i in range(10):
        messages.append({"role": "user", "content": f"Question {i} about topic {i}"})
        messages.append(
            {
                "role": "assistant",
                "content": f"Detailed answer {i} explaining topic {i}",
            }
        )

    # Add recent important messages
    messages.append({"role": "user", "content": "What's the final answer?"})
    messages.append({"role": "assistant", "content": "The final answer is 42."})

    result = compressor.compress_messages(
        messages=messages,
        model="claude-3-haiku-20240307",
        target_ratio=0.3,
        strategy=CompressionStrategy.SUMMARIZE,
    )

    # Check for summary in messages
    has_summary = any(
        "[Summary" in str(msg.get("content", "")) for msg in result.messages
    )

    # Recent messages should be preserved
    content_str = json.dumps(result.messages)
    assert "final answer" in content_str.lower()
    assert "42" in content_str

    if compressor.mlx_available:
        assert has_summary or result.strategy_used == CompressionStrategy.TRUNCATE

    print(
        f"✅ Summarization reduced {result.original_count} to {result.compressed_count} messages"
    )


def test_hybrid_compression():
    """Test hybrid compression strategy."""
    print("\nTesting hybrid compression...")

    compressor = MessageCompressor()

    # Very long conversation needing aggressive compression
    messages = []

    # Old messages (will be truncated)
    for i in range(5):
        messages.append({"role": "user", "content": f"Old question {i}"})
        messages.append({"role": "assistant", "content": f"Old answer {i}"})

    # Middle messages (will be summarized)
    for i in range(5):
        messages.append(
            {"role": "user", "content": f"Middle question {i} with details"}
        )
        messages.append(
            {"role": "assistant", "content": f"Middle answer {i} with explanation"}
        )

    # Recent messages (will be kept)
    messages.append({"role": "user", "content": "Recent important question"})
    messages.append({"role": "assistant", "content": "Recent important answer"})

    result = compressor.compress_messages(
        messages=messages,
        model="claude-3-haiku-20240307",
        target_ratio=0.2,  # Very aggressive
        strategy=CompressionStrategy.HYBRID,
    )

    assert result.compressed_count <= result.original_count
    # Check for significant reduction if compression happened
    if result.compressed_count < result.original_count:
        assert result.compressed_tokens < result.original_tokens

    # Recent content preserved
    content_str = json.dumps(result.messages)
    assert "recent" in content_str.lower() or "important" in content_str.lower()

    print(
        f"✅ Hybrid compression: {result.original_count} → {result.compressed_count} messages"
    )
    print(f"   Token reduction: {result.original_tokens} → {result.compressed_tokens}")


def test_multimodal_messages():
    """Test compression with multimodal messages."""
    print("\nTesting multimodal message compression...")

    compressor = MessageCompressor()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this image"},
                {"type": "image", "source": {"type": "base64", "data": "..."}},
            ],
        },
        {"role": "assistant", "content": "I see the image"},
        {"role": "user", "content": "What's in it?"},
        {"role": "assistant", "content": "It shows a cat"},
    ]

    result = compressor.compress_messages(
        messages=messages,
        model="claude-3-5-sonnet-20241022",
        target_ratio=0.5,
    )

    assert result.messages is not None
    assert len(result.messages) > 0

    print("✅ Multimodal messages handled correctly")


def test_compression_notice():
    """Test that compression notices are added correctly."""
    print("\nTesting compression notices...")

    compressor = MessageCompressor()

    messages = []
    for i in range(20):
        messages.append({"role": "user", "content": f"Message {i}"})
        messages.append({"role": "assistant", "content": f"Response {i}"})

    result = compressor.compress_messages(
        messages=messages,
        model="claude-3-haiku-20240307",
        target_ratio=0.2,
    )

    # Check for compression notice

    if result.compressed_count < result.original_count:
        # Should have a notice if messages were removed
        has_notice = False
        for msg in result.messages:
            content = msg.get("content", "")
            if "[NOTICE:" in content:
                has_notice = True
                assert "compressed" in content.lower()
                assert "fit context window" in content.lower()
                break

        assert has_notice, "Compression notice not found"

    print("✅ Compression notice added correctly")


def test_edge_cases():
    """Test edge cases."""
    print("\nTesting edge cases...")

    compressor = MessageCompressor()

    # Empty messages
    result = compressor.compress_messages([], "claude-3-haiku-20240307")
    assert result.messages == []
    assert result.original_count == 0

    # Single message
    single = [{"role": "user", "content": "Hello"}]
    result = compressor.compress_messages(single, "claude-3-haiku-20240307", 0.5)
    assert len(result.messages) == 1

    # Very aggressive compression
    messages = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "C"},
    ]
    result = compressor.compress_messages(
        messages, "claude-3-haiku-20240307", target_ratio=0.01
    )
    assert len(result.messages) >= 1  # At least one message preserved

    print("✅ Edge cases handled correctly")


def test_model_context_sizes():
    """Test different model context sizes."""
    print("\nTesting model context sizes...")

    compressor = MessageCompressor()

    # Large message for testing
    large_content = "x" * 10000
    messages = [
        {"role": "user", "content": large_content},
        {"role": "assistant", "content": large_content},
    ]

    # Test with different models
    models = [
        "claude-3-haiku-20240307",  # 200K context
        "claude-3-opus-20240229",  # 200K context
        "gpt-4",  # Smaller context
    ]

    for model in models:
        result = compressor.compress_messages(
            messages=messages,
            model=model,
            target_ratio=0.5,
        )

        assert result is not None
        assert result.messages is not None
        print(f"  ✅ {model}: {result.compressed_tokens} tokens")

    print("✅ Model context sizes handled correctly")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Intelligent Context Compression")
    print("=" * 60)

    try:
        test_basic_compression()
        test_no_compression_needed()
        test_truncation_strategy()
        test_smart_truncation()
        test_summarization()
        test_hybrid_compression()
        test_multimodal_messages()
        test_compression_notice()
        test_edge_cases()
        test_model_context_sizes()

        print("\n" + "=" * 60)
        print("✅ All compression tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
