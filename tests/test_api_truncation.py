#!/usr/bin/env python3
"""
Test the API with a large conversation that triggers automatic truncation.
"""

import requests

# Test with a conversation that will be auto-truncated
headers = {
    "x-api-key": "test-key",
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
}

# Create a very large conversation
messages = []
for i in range(100):
    messages.append({
        "role": "user",
        "content": f"Question {i+1}: Can you explain concept number {i+1} in detail? " * 5
    })
    messages.append({
        "role": "assistant",
        "content": f"Answer {i+1}: Here's a detailed explanation of concept {i+1}. " * 10
    })

data = {
    "model": "claude-3-opus-20240229",
    "system": "You are a helpful AI assistant with extensive knowledge.",
    "messages": messages,
    "max_tokens": 100
}

# Calculate approximate size
total_chars = len(data['system'])
for msg in messages:
    total_chars += len(msg['content'])

print("Testing API with large conversation")
print("=" * 60)
print(f"Total conversation size: ~{total_chars:,} characters")
print(f"Number of messages: {len(messages)}")
print()

print("Sending request to /v1/messages...")
print("Check requests.log to see the automatic truncation in action!")
print()

try:
    response = requests.post(
        "http://localhost:1235/v1/messages",
        headers=headers,
        json=data
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("Response received successfully!")
        print(f"Response content: {result['content'][0]['text']}")
        
        print("\n" + "=" * 60)
        print("Check requests.log to see:")
        print("1. The original request size")
        print("2. The [CHAT TEMPLATE] section showing truncation")
        print("3. How many messages were kept vs truncated")
    else:
        print(f"Error: {response.text}")
        
except Exception as e:
    print(f"Error: {e}")
    print("\nMake sure the server is running:")
    print("  uv run python main.py")