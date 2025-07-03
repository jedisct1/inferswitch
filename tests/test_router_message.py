#!/usr/bin/env python3
"""Test script to verify router message appears in responses."""

import requests
import json

# Test configuration
BASE_URL = "http://localhost:1235"
API_KEY = "test-key"  # Replace with your actual API key


def test_non_streaming():
    """Test non-streaming request."""
    print("Testing non-streaming request...")

    response = requests.post(
        f"{BASE_URL}/v1/messages",
        headers={
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": "claude-3-haiku-20240307",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 100,
        },
    )

    if response.status_code == 200:
        data = response.json()
        print("Response content:")
        for content_block in data.get("content", []):
            if content_block.get("type") == "text":
                print(f"  {content_block.get('text')}")
        print()
    else:
        print(f"Error: {response.status_code} - {response.text}")


def test_streaming():
    """Test streaming request."""
    print("Testing streaming request...")

    response = requests.post(
        f"{BASE_URL}/v1/messages",
        headers={
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": "claude-3-haiku-20240307",
            "messages": [{"role": "user", "content": "What is 3+3?"}],
            "max_tokens": 100,
            "stream": True,
        },
        stream=True,
    )

    if response.status_code == 200:
        print("Streaming response:")
        full_text = ""
        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    try:
                        data = json.loads(line_str[6:])
                        if data.get("type") == "content_block_delta":
                            text = data.get("delta", {}).get("text", "")
                            full_text += text
                            print(f"  Block {data.get('index')}: {text[:50]}...")
                    except json.JSONDecodeError:
                        pass
        print(f"\nFull response: {full_text}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


if __name__ == "__main__":
    print("Testing InferSwitch router messages...\n")
    test_non_streaming()
    print("-" * 50)
    test_streaming()
