import requests
import json

# Test count tokens endpoint with API key
api_key_headers = {
    "x-api-key": "test-key",
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}

# Test count tokens endpoint with OAuth (no API key)
oauth_headers = {"anthropic-version": "2023-06-01", "content-type": "application/json"}

# Simple test request
data = {
    "model": "claude-3-opus-20240229",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
}

print("Testing /v1/messages/count-tokens endpoint with API key...")
print(f"Request: {json.dumps(data, indent=2)}")

try:
    response = requests.post(
        "http://localhost:1235/v1/messages/count-tokens",
        headers=api_key_headers,
        json=data,
    )

    print(f"\nStatus Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")

    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")

except Exception as e:
    print(f"Error: {e}")

# Test with OAuth (no API key)
print("\n\nTesting /v1/messages/count-tokens endpoint with OAuth...")
print("(No x-api-key header, should use OAuth token if available)")

try:
    response = requests.post(
        "http://localhost:1235/v1/messages/count-tokens",
        headers=oauth_headers,
        json=data,
    )

    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print("✓ OAuth authentication worked!")
    elif response.status_code == 401:
        print(f"Error: {response.text}")
        print("ℹ️  OAuth token not available or expired")
    else:
        print(f"Error: {response.text}")

except Exception as e:
    print(f"Error: {e}")

# Test with complex content
print("\n\nTesting with complex content format...")
data2 = {
    "model": "claude-3-opus-20240229",
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Hello, world!"}]}
    ],
    "system": "You are a helpful assistant.",
}

print(f"Request: {json.dumps(data2, indent=2)}")

try:
    response = requests.post(
        "http://localhost:1235/v1/messages/count-tokens",
        headers=api_key_headers,
        json=data2,
    )

    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")

except Exception as e:
    print(f"Error: {e}")
