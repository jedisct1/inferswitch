import requests
import json

# Test with complex content format like in the original example
headers = {
    "x-api-key": "test-key",
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
}

# Complex content with objects
data = {
    "model": "claude-3-opus-20240229",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Hello, world!"
                }
            ]
        }
    ],
    "max_tokens": 100
}

print("Testing with complex content format...")
print(f"Request: {json.dumps(data, indent=2)}")

try:
    response = requests.post(
        "http://localhost:1235/v1/messages",
        headers=headers,
        json=data
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")
        
except Exception as e:
    print(f"Error: {e}")