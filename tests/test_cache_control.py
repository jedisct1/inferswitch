import requests
import json

# Test cache_control forwarding
headers = {
    "x-api-key": "test-key",
    "anthropic-version": "2023-06-01",
    "anthropic-beta": "prompt-caching-2024-07-31",
    "content-type": "application/json",
}

# Test with cache_control in system messages
print("Testing cache_control in system messages...")
data = {
    "model": "claude-3-opus-20240229",
    "system": [
        {
            "text": "You are a helpful assistant.",
            "type": "text",
            "cache_control": {"type": "ephemeral"},
        }
    ],
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
}

print(f"Request: {json.dumps(data, indent=2)}")

response = requests.post(
    "http://localhost:1235/v1/messages", headers=headers, json=data
)

print(f"\nStatus Code: {response.status_code}")
if response.status_code != 200:
    print(f"Error: {response.text}")

# Test with cache_control in message content
print("\n\nTesting cache_control in message content...")
data2 = {
    "model": "claude-3-opus-20240229",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Tell me about caching.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ],
    "max_tokens": 100,
}

print(f"Request: {json.dumps(data2, indent=2)}")

response = requests.post(
    "http://localhost:1235/v1/messages", headers=headers, json=data2
)

print(f"\nStatus Code: {response.status_code}")
if response.status_code != 200:
    print(f"Error: {response.text}")

# Test count_tokens with cache_control
print("\n\nTesting cache_control in count_tokens...")
data3 = {
    "model": "claude-3-opus-20240229",
    "system": [
        {
            "text": "You are helpful.",
            "type": "text",
            "cache_control": {"type": "ephemeral"},
        }
    ],
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Count my tokens",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ],
}

print(f"Request: {json.dumps(data3, indent=2)}")

response = requests.post(
    "http://localhost:1235/v1/messages/count-tokens", headers=headers, json=data3
)

print(f"\nStatus Code: {response.status_code}")
if response.status_code == 200:
    print(f"Response: {json.dumps(response.json(), indent=2)}")
else:
    print(f"Error: {response.text}")

print(
    "\nCheck requests.log to verify cache_control is being forwarded to Anthropic API"
)
