import requests
import json

headers = {
    "x-api-key": "test-key",
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
}

# Test non-streaming
print("Testing non-streaming response...")
data = {
    "model": "claude-3-opus-20240229",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100,
    "stream": False
}

response = requests.post("http://localhost:1235/v1/messages", headers=headers, json=data)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test streaming
print("\n\nTesting streaming response...")
data["stream"] = True

response = requests.post("http://localhost:1235/v1/messages", headers=headers, json=data, stream=True)
print(f"Status: {response.status_code}")
print("Stream events:")
for line in response.iter_lines():
    if line:
        print(line.decode())