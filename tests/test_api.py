import requests
import json

BASE_URL = "http://localhost:1235"
HEADERS = {
    "x-api-key": "test-key",
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}


def test_messages_endpoint():
    print("Testing /v1/messages endpoint...")

    data = {
        "model": "claude-3-opus-20240229",
        "messages": [{"role": "user", "content": "Hello, Claude!"}],
        "max_tokens": 100,
    }

    response = requests.post(f"{BASE_URL}/v1/messages", headers=HEADERS, json=data)

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_count_tokens_endpoint():
    print("Testing /v1/messages/count_tokens endpoint...")

    data = {
        "model": "claude-3-opus-20240229",
        "messages": [
            {"role": "user", "content": "Hello, world"},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
        ],
    }

    response = requests.post(
        f"{BASE_URL}/v1/messages/count_tokens", headers=HEADERS, json=data
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_missing_headers():
    print("Testing with missing headers...")

    data = {
        "model": "claude-3-opus-20240229",
        "messages": [{"role": "user", "content": "Test"}],
        "max_tokens": 50,
    }

    response = requests.post(f"{BASE_URL}/v1/messages", json=data)

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    print()


if __name__ == "__main__":
    test_messages_endpoint()
    test_count_tokens_endpoint()
    test_missing_headers()
