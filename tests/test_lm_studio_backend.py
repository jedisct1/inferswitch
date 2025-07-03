"""
Test LM-Studio backend integration.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inferswitch.backends import OpenAIBackend, BackendConfig


async def test_lm_studio_backend():
    """Test basic LM-Studio backend functionality."""

    # Configure LM-Studio backend
    config = BackendConfig(
        name="lm-studio",
        base_url="http://127.0.0.1:1234",
        api_key="lm-studio",  # LM-Studio doesn't require real API key
        timeout=30,
    )

    backend = OpenAIBackend(config)

    try:
        # 1. Test health check
        print("1. Testing health check...")
        health = await backend.health_check()
        print(f"Health check result: {health}")

        if health["status"] != "ok":
            print(
                "LM-Studio server not available. Make sure it's running on port 1234."
            )
            return

        # 2. Test non-streaming message
        print("\n2. Testing non-streaming message...")
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Say hello in one word"}],
            }
        ]

        # Use first available model or default
        models = health.get("models", [])
        model = models[0] if models else "qwen/qwen3-1.7b"
        print(f"Using model: {model}")

        response = await backend.create_message(
            messages=messages, model=model, max_tokens=10, temperature=0.1
        )

        print(f"Response content: {response.content}")
        print(f"Response model: {response.model}")
        print(f"Stop reason: {response.stop_reason}")
        print(f"Usage: {response.usage}")

        # 3. Test streaming message
        print("\n3. Testing streaming message...")
        stream_content = ""
        event_count = 0

        async for event in backend.create_message_stream(
            messages=messages, model=model, max_tokens=10, temperature=0.1
        ):
            event_count += 1
            event_type = event.get("type", "")

            if event_type == "content_block_delta":
                delta_text = event.get("delta", {}).get("text", "")
                stream_content += delta_text
                print(f"Received delta: '{delta_text}'")
            elif event_type == "message_start":
                print("Stream started")
            elif event_type == "message_stop":
                print("Stream ended")

        print(f"Total streamed content: '{stream_content}'")
        print(f"Total events: {event_count}")

        # 4. Test model support
        print("\n4. Testing model support...")
        print(
            f"Supports 'qwen/qwen3-1.7b': {backend.supports_model('qwen/qwen3-1.7b')}"
        )
        print(f"Supports 'gpt-4': {backend.supports_model('gpt-4')}")
        print(f"Supports 'claude-3-opus': {backend.supports_model('claude-3-opus')}")

        # 5. Test capabilities
        print("\n5. Backend capabilities:")
        capabilities = backend.get_capabilities()
        for cap, supported in capabilities.items():
            print(f"  {cap}: {supported}")

        print("\nAll tests completed successfully!")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()

    finally:
        await backend.close()


async def test_with_inferswitch_api():
    """Test LM-Studio through InferSwitch API."""
    import httpx

    print("\n\nTesting LM-Studio through InferSwitch API...")

    # Set environment variable to use LM-Studio
    os.environ["INFERSWITCH_BACKEND"] = "lm-studio"

    async with httpx.AsyncClient() as client:
        # Test with explicit backend header
        response = await client.post(
            "http://localhost:1235/v1/messages",
            headers={
                "x-api-key": "test-key",
                "anthropic-version": "2023-06-01",
                "x-backend": "lm-studio",  # Explicit backend selection
            },
            json={
                "model": "qwen/qwen3-1.7b",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "max_tokens": 20,
            },
        )

        if response.status_code == 200:
            print("Success! Response:")
            print(response.json())
        else:
            print(f"Error: {response.status_code}")
            print(response.text)


if __name__ == "__main__":
    # Run the direct backend test
    asyncio.run(test_lm_studio_backend())

    # Uncomment to test through InferSwitch API (requires server to be running)
    # asyncio.run(test_with_inferswitch_api())
