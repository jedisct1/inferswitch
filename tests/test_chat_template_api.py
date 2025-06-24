import requests
import json

# Test the chat template API endpoint
headers = {
    "x-api-key": "test-key",
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
}

# Test with a complex conversation
data = {
    "model": "claude-3-opus-20240229",
    "system": "You are a helpful AI assistant specialized in Python programming.",
    "messages": [
        {
            "role": "user",
            "content": "What's the difference between a list and a tuple in Python?"
        },
        {
            "role": "assistant",
            "content": "The main differences between lists and tuples in Python are:\n\n1. **Mutability**: Lists are mutable (can be changed), tuples are immutable\n2. **Syntax**: Lists use square brackets [], tuples use parentheses ()\n3. **Performance**: Tuples are slightly faster than lists\n4. **Use cases**: Lists for collections that might change, tuples for fixed collections"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Can you show me an example?"
                }
            ]
        }
    ],
    "max_tokens": 500
}

print("Testing /v1/messages/chat-template endpoint...")
print(f"Request preview: {json.dumps(data, indent=2)[:200]}...\n")

try:
    response = requests.post(
        "http://localhost:1235/v1/messages/chat-template",
        headers=headers,
        json=data
    )
    
    print(f"Status Code: {response.status_code}\n")
    
    if response.status_code == 200:
        result = response.json()
        
        print("Chat Messages:")
        print(json.dumps(result['chat_messages'], indent=2))
        
        print(f"\nMessage count: {result['message_count']}")
        print(f"Roles: {result['roles']}")
        
        print("\nFormatted ChatML (with generation prompt):")
        print(result['formatted']['chatml'])
        
        print("\n" + "="*60)
        
        # Now make the same request to the regular messages endpoint
        print("\nMaking the same request to /v1/messages...")
        print("Check requests.log to see the chat template logged there too!")
        
        messages_response = requests.post(
            "http://localhost:1235/v1/messages",
            headers=headers,
            json=data
        )
        
        print(f"Messages endpoint status: {messages_response.status_code}")
        
    else:
        print(f"Error: {response.text}")
        
except Exception as e:
    print(f"Error: {e}")