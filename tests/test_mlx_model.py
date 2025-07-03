#!/usr/bin/env python3
"""Test MLX model integration with InferSwitch."""

import requests
import json
import time


def test_mlx_status():
    """Check MLX model status."""
    try:
        response = requests.get("http://localhost:1235/mlx/status")
        if response.status_code == 200:
            data = response.json()
            print("MLX Model Status:")
            print(json.dumps(data, indent=2))
            return data.get("loaded", False)
        else:
            print(f"Error: Status code {response.status_code}")
            return False
    except Exception as e:
        print(f"Connection error: {e}")
        return False


def test_difficulty_rating():
    """Test difficulty rating with sample queries."""
    test_queries = [
        {"query": "What is 2+2?", "expected": "easy"},
        {
            "query": "Write a Python function to calculate factorial",
            "expected": "medium",
        },
        {
            "query": "Implement a distributed cache with consistent hashing",
            "expected": "hard",
        },
        {
            "query": "Debug this React component that has a memory leak",
            "expected": "hard",
        },
    ]

    for test in test_queries:
        request_body = {
            "model": "claude-3-haiku-20240307",
            "messages": [{"role": "user", "content": test["query"]}],
            "max_tokens": 100,
        }

        try:
            response = requests.post(
                "http://localhost:1235/v1/messages",
                json=request_body,
                headers={"x-api-key": "test-key"},
            )

            if response.status_code == 200:
                # Check response headers or logs for difficulty rating
                print(f"\nQuery: {test['query']}")
                print(f"Expected difficulty: {test['expected']}")
                # The difficulty rating would be in the logs
            else:
                print(f"Error for query '{test['query']}': {response.status_code}")

        except Exception as e:
            print(f"Error testing query: {e}")


def main():
    """Main test function."""
    print("Waiting for server to start...")

    # Wait for server with retries
    for i in range(30):  # Try for 30 seconds
        if test_mlx_status():
            print("\nMLX model loaded successfully!")
            break
        time.sleep(1)
    else:
        print("\nMLX model failed to load within timeout")
        return

    print("\nTesting difficulty rating...")
    test_difficulty_rating()


if __name__ == "__main__":
    main()
