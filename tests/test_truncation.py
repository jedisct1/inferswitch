#!/usr/bin/env python3
"""
Test the intelligent chat template truncation functionality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import convert_to_chat_template, truncate_chat_template_to_fit

def create_large_conversation(num_exchanges=50):
    """Create a large conversation with many exchanges."""
    messages = []
    
    for i in range(num_exchanges):
        messages.append({
            "role": "user",
            "content": f"Question {i+1}: This is a somewhat long user message that contains various details about topic {i+1}. " * 10
        })
        messages.append({
            "role": "assistant", 
            "content": f"Answer {i+1}: This is a detailed assistant response explaining the answer to question {i+1}. " * 15
        })
    
    return messages

print("Testing Chat Template Truncation")
print("=" * 80)

# Test 1: Small conversation that doesn't need truncation
print("\nTest 1: Small conversation (no truncation needed)")
print("-" * 50)

small_request = {
    "model": "claude-3-opus-20240229",
    "system": "You are a helpful assistant.",
    "messages": [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"},
        {"role": "user", "content": "What's the weather like?"}
    ]
}

chat_messages = convert_to_chat_template(small_request)
truncated = truncate_chat_template_to_fit(chat_messages, max_context_size=10000)

print(f"Original messages: {len(chat_messages)}")
print(f"After truncation: {len(truncated)}")
print(f"Messages truncated: {len(chat_messages) - len(truncated)}")

# Test 2: Large conversation requiring truncation
print("\n\nTest 2: Large conversation (truncation required)")
print("-" * 50)

large_request = {
    "model": "claude-3-opus-20240229",
    "system": "You are an AI assistant specializing in detailed technical explanations.",
    "messages": create_large_conversation(30)
}

chat_messages = convert_to_chat_template(large_request)
# Use a smaller context size to force truncation
truncated = truncate_chat_template_to_fit(chat_messages, max_context_size=5000)

print(f"Original messages: {len(chat_messages)}")
print(f"After truncation: {len(truncated)}")
print(f"Messages truncated: {len(chat_messages) - len(truncated)}")

print("\nTruncated conversation preview:")
for i, msg in enumerate(truncated):
    role = msg['role']
    content_preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
    print(f"{i+1}. {role}: {content_preview}")

# Test 3: Very large system message
print("\n\nTest 3: Very large system message")
print("-" * 50)

large_system_request = {
    "model": "claude-3-opus-20240229",
    "system": "This is an extremely detailed system prompt that contains extensive instructions. " * 200,
    "messages": [
        {"role": "user", "content": "Can you help me?"},
        {"role": "assistant", "content": "Of course! What do you need help with?"},
        {"role": "user", "content": "I need to understand truncation."}
    ]
}

chat_messages = convert_to_chat_template(large_system_request)
truncated = truncate_chat_template_to_fit(chat_messages, max_context_size=5000)

print(f"Original messages: {len(chat_messages)}")
print(f"After truncation: {len(truncated)}")

for msg in truncated:
    if msg['role'] == 'system':
        if '(system message truncated)' in msg['content']:
            print("System message was truncated")
        if '[Previous' in msg['content'] and 'messages truncated' in msg['content']:
            print(f"Truncation notice: {msg['content']}")

# Test 4: Different model context sizes
print("\n\nTest 4: Model-specific context sizes")
print("-" * 50)

models = [
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "gpt-4"  # Non-Claude model
]

test_messages = convert_to_chat_template({
    "messages": create_large_conversation(20)
})

for model in models:
    truncated = truncate_chat_template_to_fit(
        test_messages, 
        max_context_size=50000,  # Base size before model adjustment
        model_name=model
    )
    
    total_chars = sum(len(msg['content']) for msg in truncated)
    print(f"{model}: {len(truncated)} messages, ~{total_chars:,} chars")

# Test 5: Preserving the last user message
print("\n\nTest 5: Preserving the last user message")
print("-" * 50)

conversation_ending_with_user = {
    "model": "claude-3-opus-20240229",
    "system": "You are helpful.",
    "messages": create_large_conversation(10) + [
        {"role": "user", "content": "This is the final important user question that must be preserved!"}
    ]
}

chat_messages = convert_to_chat_template(conversation_ending_with_user)
truncated = truncate_chat_template_to_fit(chat_messages, max_context_size=3000)

print(f"Original messages: {len(chat_messages)}")
print(f"After truncation: {len(truncated)}")

# Check if the last user message was preserved
last_msg = truncated[-1]
print(f"\nLast message role: {last_msg['role']}")
print(f"Last message content: {last_msg['content']}")

# Test 6: Real API usage simulation
print("\n\nTest 6: Simulating real API usage with auto-truncation")
print("-" * 50)


# Create a very large request
huge_request = {
    "model": "claude-3-opus-20240229",
    "system": [
        {
            "text": "You are Claude, an AI assistant created by Anthropic. " * 50,
            "type": "text",
            "cache_control": {"type": "ephemeral"}
        }
    ],
    "messages": create_large_conversation(100),
    "max_tokens": 100
}

# Calculate the size
chat_messages = convert_to_chat_template(huge_request)
total_size = sum(len(msg.get('content', '')) for msg in chat_messages)
print(f"Total chat template size: {total_size:,} characters")
print("This would be auto-truncated in log_chat_template() since it exceeds 100,000 chars")

# Show what the truncation would look like
if total_size > 100000:
    truncated = truncate_chat_template_to_fit(
        chat_messages,
        max_context_size=100000,
        model_name=huge_request['model']
    )
    print(f"After auto-truncation: {len(truncated)} messages")
    print(f"Truncated {len(chat_messages) - len(truncated)} messages")

print("\n" + "=" * 80)
print("Truncation testing complete!")
print("\nThe truncate_chat_template_to_fit() function intelligently:")
print("1. Preserves system messages (truncating if needed)")
print("2. Keeps the most recent conversation pairs")
print("3. Tries to preserve the last user message")
print("4. Adds a truncation notice when messages are removed")
print("5. Adjusts context size based on the model")