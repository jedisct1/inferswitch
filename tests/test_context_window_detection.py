#!/usr/bin/env python3
"""
Test context window error detection in backends.
"""

from unittest.mock import Mock, AsyncMock
from inferswitch.backends.anthropic import AnthropicBackend
from inferswitch.backends.openai import OpenAIBackend
from inferswitch.backends.base import BackendConfig
from inferswitch.backends.errors import ContextWindowExceededError


def test_anthropic_context_window_detection():
    """Test that Anthropic backend detects context window errors."""
    config = BackendConfig(
        name="anthropic", base_url="https://api.anthropic.com", api_key="test-key"
    )
    backend = AnthropicBackend(config)

    # Mock response for context window error
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.text = "Request exceeds maximum context length of 100000 tokens"
    mock_response.json.return_value = {
        "error": {
            "type": "invalid_request_error",
            "message": "Request exceeds maximum context length of 100000 tokens",
        }
    }

    # Mock the client to return the response directly
    backend.client = AsyncMock()
    backend.client.post = AsyncMock(return_value=mock_response)

    # Test that ContextWindowExceededError is raised
    import asyncio

    async def test_context_error():
        try:
            await backend.create_message(
                messages=[{"role": "user", "content": "Test message"}],
                model="claude-3-haiku-20240307",
            )
            assert False, "Expected ContextWindowExceededError"
        except ContextWindowExceededError as e:
            assert e.backend == "anthropic"
            assert "context length" in e.message.lower()
            assert e.messages is not None
            print("✓ Anthropic context window error detected correctly")

    asyncio.run(test_context_error())


def test_openai_context_window_detection():
    """Test that OpenAI backend detects context window errors."""
    config = BackendConfig(
        name="openai", base_url="https://api.openai.com", api_key="test-key"
    )
    backend = OpenAIBackend(config)

    # Mock response for context window error
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.text = "This model's maximum context length is 4097 tokens."
    mock_response.json.return_value = {
        "error": {
            "message": "This model's maximum context length is 4097 tokens. However, you requested 5000 tokens.",
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
        }
    }

    # Mock the client to return the response directly
    backend.client = AsyncMock()
    backend.client.post = AsyncMock(return_value=mock_response)

    # Test that ContextWindowExceededError is raised
    import asyncio

    async def test_context_error():
        try:
            await backend.create_message(
                messages=[{"role": "user", "content": "Test message"}],
                model="gpt-3.5-turbo",
            )
            assert False, "Expected ContextWindowExceededError"
        except ContextWindowExceededError as e:
            assert e.backend == "openai"
            assert "context length" in e.message.lower()
            assert e.messages is not None
            print("✓ OpenAI context window error detected correctly")

    asyncio.run(test_context_error())


def test_anthropic_count_tokens_context_window_detection():
    """Test that Anthropic count_tokens detects context window errors."""
    config = BackendConfig(
        name="anthropic", base_url="https://api.anthropic.com", api_key="test-key"
    )
    backend = AnthropicBackend(config)

    # Mock response for context window error
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.text = "Input is too long for token counting"
    mock_response.json.return_value = {
        "error": {
            "type": "invalid_request_error",
            "message": "Input is too long for token counting",
        }
    }

    # Mock the client to return the response directly
    backend.client = AsyncMock()
    backend.client.post = AsyncMock(return_value=mock_response)

    # Test that ContextWindowExceededError is raised
    import asyncio

    async def test_count_tokens_context_error():
        try:
            await backend.count_tokens(
                messages=[{"role": "user", "content": "Test message"}],
                model="claude-3-haiku-20240307",
            )
            assert False, "Expected ContextWindowExceededError"
        except ContextWindowExceededError as e:
            assert e.backend == "anthropic"
            assert "too long" in e.message.lower()
            assert e.messages is not None
            print("✓ Anthropic count_tokens context window error detected correctly")

    asyncio.run(test_count_tokens_context_error())


if __name__ == "__main__":
    print("Testing context window error detection...\n")

    try:
        test_anthropic_context_window_detection()
        test_openai_context_window_detection()
        test_anthropic_count_tokens_context_window_detection()
        print("\n✅ All context window detection tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
