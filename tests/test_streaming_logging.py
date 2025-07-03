import requests
import json

# Test streaming with logging
headers = {
    "x-api-key": "test-key",
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}

data = {
    "model": "claude-3-opus-20240229",
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "max_tokens": 100,
    "stream": True,
}

print("Testing streaming response with event logging...")
print(f"Request: {json.dumps(data, indent=2)}")

try:
    response = requests.post(
        "http://localhost:1235/v1/messages", headers=headers, json=data, stream=True
    )

    print(f"\nStatus: {response.status_code}")
    print("\nStreaming events:")

    for line in response.iter_lines():
        if line:
            decoded = line.decode()
            print(
                f"Received: {decoded[:100]}..."
                if len(decoded) > 100
                else f"Received: {decoded}"
            )

except Exception as e:
    print(f"Error: {e}")
