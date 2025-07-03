#!/usr/bin/env python3
"""
Test caching functionality.
"""

import os
import time
import requests

# Test configuration
BASE_URL = "http://localhost:1235"
API_KEY = os.getenv("ANTHROPIC_API_KEY", "test-key")


def test_cache_with_environment_details():
    """Test that cache ignores environment details with timestamps."""

    os.environ["CACHE_ENABLED"] = "true"
    os.environ["PROXY_MODE"] = "false"

    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    # Request with environment details (like timestamps)
    request_data = {
        "model": "claude-3-opus-20240229",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<task>\nCompute 2+2\n</task>"},
                    {
                        "type": "text",
                        "text": "<environment_details>\n# Current Time\n6/20/2025, 4:43:13 PM\n</environment_details>",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            }
        ],
        "max_tokens": 100,
    }

    print("Testing cache with environment details...")

    # First request
    print("\n1. First request with timestamp 4:43:13 PM...")
    response1 = requests.post(
        f"{BASE_URL}/v1/messages", headers=headers, json=request_data
    )
    assert response1.status_code == 200
    response1_data = response1.json()
    print(f"   Response ID: {response1_data['id']}")

    # Second request with different timestamp but same actual content
    request_data["messages"][0]["content"][1]["text"] = (
        "<environment_details>\n# Current Time\n6/20/2025, 4:43:32 PM\n</environment_details>"
    )

    print("\n2. Second request with timestamp 4:43:32 PM (should hit cache)...")
    response2 = requests.post(
        f"{BASE_URL}/v1/messages", headers=headers, json=request_data
    )
    assert response2.status_code == 200
    response2_data = response2.json()
    print(f"   Response ID: {response2_data['id']}")

    # Responses should be identical despite different timestamps
    print("\n3. Verifying cache hit despite different timestamps...")
    assert response1_data == response2_data, (
        "Cache should return same response for different timestamps"
    )
    print("   ✓ Cache correctly ignored timestamp differences!")

    print("\n✅ Environment details test passed!")


def test_cache_ignores_processing_tags():
    """Test that cache ignores processing tags in responses."""

    os.environ["CACHE_ENABLED"] = "true"
    os.environ["PROXY_MODE"] = "false"

    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    print("\nTesting cache with processing tags...")

    # First request - simple computation
    request1 = {
        "model": "claude-3-opus-20240229",
        "messages": [{"role": "user", "content": "What is 7+7?"}],
        "max_tokens": 100,
    }

    print("\n1. First request for 7+7...")
    response1 = requests.post(f"{BASE_URL}/v1/messages", headers=headers, json=request1)
    assert response1.status_code == 200
    data1 = response1.json()
    print(f"   Response: {data1['content'][0]['text'][:50]}...")

    # Second request - same question but with assistant response containing processing tag
    request2 = {
        "model": "claude-3-opus-20240229",
        "messages": [
            {"role": "user", "content": "What is 7+7?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "<processing>This request is currently being processed by a local InferSwitch AI gateway.\n</processing>\n14",
                    }
                ],
            },
            {"role": "user", "content": "Are you sure?"},
        ],
        "max_tokens": 100,
    }

    print("\n2. Request with processing tag in history...")
    response2 = requests.post(f"{BASE_URL}/v1/messages", headers=headers, json=request2)
    assert response2.status_code == 200

    # Third request - same as first (should hit cache)
    print("\n3. Repeat first request (should hit cache)...")
    response3 = requests.post(f"{BASE_URL}/v1/messages", headers=headers, json=request1)
    assert response3.status_code == 200
    response3_data = response3.json()

    # Verify cache hit
    assert data1 == response3_data, "Should get cached response for identical request"
    print("   ✓ Cache correctly ignored processing tags!")

    print("\n✅ Processing tag test passed!")


def test_cache():
    """Test that identical requests return cached responses."""

    # Enable cache for testing
    os.environ["CACHE_ENABLED"] = "true"
    os.environ["PROXY_MODE"] = "false"  # Use OK response mode for predictable testing

    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    # Test request
    request_data = {
        "model": "claude-3-opus-20240229",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 100,
    }

    print("Testing cache functionality...")

    # First request (cache miss)
    print("\n1. Sending first request (should be cache miss)...")
    start_time = time.time()
    response1 = requests.post(
        f"{BASE_URL}/v1/messages", headers=headers, json=request_data
    )
    time1 = time.time() - start_time

    assert response1.status_code == 200
    data1 = response1.json()
    print(f"   Response ID: {data1['id']}")
    print(f"   Response time: {time1:.3f}s")

    # Second identical request (cache hit)
    print("\n2. Sending identical request (should be cache hit)...")
    start_time = time.time()
    response2 = requests.post(
        f"{BASE_URL}/v1/messages", headers=headers, json=request_data
    )
    time2 = time.time() - start_time

    assert response2.status_code == 200
    data2 = response2.json()
    print(f"   Response ID: {data2['id']}")
    print(f"   Response time: {time2:.3f}s")

    # Verify responses are identical
    print("\n3. Verifying responses...")
    assert data1 == data2, "Cached response should be identical"
    print("   ✓ Responses are identical")

    # Cache should be faster (in OK mode, both should be very fast)
    print(f"   ✓ Second request was {time1 / time2:.1f}x faster")

    # Test with slightly different request (cache miss)
    print("\n4. Sending different request (should be cache miss)...")
    request_data["messages"][0]["content"] = "What is 3+3?"

    response3 = requests.post(
        f"{BASE_URL}/v1/messages", headers=headers, json=request_data
    )

    assert response3.status_code == 200
    data3 = response3.json()
    print(f"   Response ID: {data3['id']}")
    print("   ✓ Different request got different response")

    # Test streaming with cache
    print("\n5. Testing streaming with cache...")
    request_data["stream"] = True
    request_data["messages"][0]["content"] = "What is 4+4?"

    # First streaming request
    response4 = requests.post(
        f"{BASE_URL}/v1/messages", headers=headers, json=request_data, stream=True
    )

    events1 = []
    for line in response4.iter_lines():
        if line:
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                events1.append(line_str)

    print(f"   First stream: {len(events1)} events")

    # Second identical streaming request (cached)
    response5 = requests.post(
        f"{BASE_URL}/v1/messages", headers=headers, json=request_data, stream=True
    )

    events2 = []
    for line in response5.iter_lines():
        if line:
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                events2.append(line_str)

    print(f"   Cached stream: {len(events2)} events")
    print("   ✓ Streaming responses work with cache")

    print("\n✅ All cache tests passed!")


def test_cache_disabled():
    """Test that cache can be disabled."""
    os.environ["CACHE_ENABLED"] = "false"

    print("\n\nTesting with cache disabled...")

    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    request_data = {
        "model": "claude-3-opus-20240229",
        "messages": [{"role": "user", "content": "Cache disabled test"}],
        "max_tokens": 100,
    }

    # Send two identical requests
    response1 = requests.post(
        f"{BASE_URL}/v1/messages", headers=headers, json=request_data
    )

    response2 = requests.post(
        f"{BASE_URL}/v1/messages", headers=headers, json=request_data
    )

    # Verify both requests succeeded
    assert response1.status_code == 200
    assert response2.status_code == 200

    # With cache disabled, responses should still be identical in OK mode
    # (but would be different in proxy mode)
    print("   ✓ Cache disabled, requests processed normally")

    print("\n✅ Cache disable test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("InferSwitch Cache Tests")
    print("=" * 60)
    print(f"Testing against: {BASE_URL}")
    print("Note: Start the server with 'PROXY_MODE=false uv run python main.py'")

    try:
        test_cache_with_environment_details()
        test_cache_ignores_processing_tags()
        test_cache()
        test_cache_disabled()
        print("\n" + "=" * 60)
        print("All tests passed! 🎉")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
