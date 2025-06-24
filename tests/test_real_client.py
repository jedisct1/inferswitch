import requests
import json

# Test with a real Anthropic client format
headers = {
    "x-api-key": "test-key",
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
}

# This is the exact format the Anthropic client sends
data = {
    "model": "claude-3-opus-20240229",
    "messages": [
        {
            "role": "user",
            "content": "Hello, world!"
        }
    ],
    "max_tokens": 1024
}

print("Request:")
print(json.dumps(data, indent=2))
print("\nHeaders:")
print(json.dumps(headers, indent=2))

try:
    response = requests.post(
        "http://localhost:1235/v1/messages",
        headers=headers,
        json=data
    )
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 422:
        print("\nValidation Error Details:")
        print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")