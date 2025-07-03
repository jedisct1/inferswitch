"""Test LM-Studio API compatibility."""

import requests
import json


def test_lm_studio_chat_completion():
    """Test basic chat completion with LM-Studio."""
    url = "http://127.0.0.1:1234/v1/chat/completions"

    payload = {
        "model": "qwen/qwen3-1.7b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in exactly 3 words"},
        ],
        "max_tokens": 20,
        "temperature": 0.7,
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        print("Response:", json.dumps(data, indent=2))

        # Verify response structure
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "content" in data["choices"][0]["message"]
        assert "usage" in data

        print("\n✅ Basic chat completion test passed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

    return True


def test_lm_studio_streaming():
    """Test streaming chat completion with LM-Studio."""
    url = "http://127.0.0.1:1234/v1/chat/completions"

    payload = {
        "model": "qwen/qwen3-1.7b",
        "messages": [{"role": "user", "content": "Count from 1 to 5"}],
        "max_tokens": 50,
        "stream": True,
    }

    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()

        print("\nStreaming response:")
        chunks = []
        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        print("Stream completed")
                        break

                    chunk = json.loads(data_str)
                    chunks.append(chunk)

                    # Print content if available
                    if "choices" in chunk and chunk["choices"]:
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            print(delta["content"], end="", flush=True)

        print(f"\n\nReceived {len(chunks)} chunks")
        print("\n✅ Streaming test passed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

    return True


def test_model_list():
    """Test model listing endpoint."""
    url = "http://127.0.0.1:1234/v1/models"

    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        print("\nAvailable models:")
        for model in data:
            print(f"  - {model['id']} (owned by: {model['owned_by']})")

        print("\n✅ Model listing test passed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

    return True


def compare_with_anthropic_format():
    """Show the transformation needed."""
    print("\n" + "=" * 60)
    print("TRANSFORMATION EXAMPLE")
    print("=" * 60)

    openai_request = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ],
        "max_tokens": 100,
        "temperature": 0.7,
    }

    print("\nOpenAI Format:")
    print(json.dumps(openai_request, indent=2))

    # Transform to Anthropic
    anthropic_request = {
        "model": "claude-3-opus-20240229",
        "system": "You are helpful.",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ],
        "max_tokens": 100,
        "temperature": 0.7,
    }

    print("\nTransformed to Anthropic Format:")
    print(json.dumps(anthropic_request, indent=2))


if __name__ == "__main__":
    print("Testing LM-Studio API Compatibility")
    print("=" * 60)

    # Run tests
    test_model_list()
    test_lm_studio_chat_completion()
    test_lm_studio_streaming()
    compare_with_anthropic_format()
