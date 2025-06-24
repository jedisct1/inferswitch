import requests
import json

# Test with the exact format from the example file
headers = {
    "x-api-key": "sk-ant-api03-test",
    "anthropic-version": "2023-06-01",
    "anthropic-beta": "prompt-caching-2024-07-31",
    "content-type": "application/json"
}

data = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 32768,
    "temperature": 1,
    "thinking": {
        "type": "enabled",
        "budget_tokens": 8192
    },
    "system": [
        {
            "text": "You are a helpful assistant.",
            "type": "text",
            "cache_control": {
                "type": "ephemeral"
            }
        }
    ],
    "messages": [
        {
            "role": "user",
            "content": "Hello!"
        }
    ],
    "stream": False
}

print("Testing with example format...")
print(f"Request: {json.dumps(data, indent=2)[:200]}...")

try:
    response = requests.post(
        "http://localhost:1235/v1/messages",
        headers=headers,
        json=data
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        print(f"Success! Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error Response: {response.text}")
        if response.status_code == 422:
            print("\nValidation Error Details:")
            print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")