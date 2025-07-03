#!/usr/bin/env python3
"""
Demonstrate the intelligent chat template truncation functionality.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import convert_to_chat_template, truncate_chat_template_to_fit


def create_conversation(num_exchanges=10):
    """Create a conversation with specified number of exchanges."""
    messages = []

    for i in range(num_exchanges):
        messages.append(
            {"role": "user", "content": f"Question {i + 1}: What about topic {i + 1}?"}
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"Answer {i + 1}: Here's information about topic {i + 1}.",
            }
        )

    return messages


print("Demonstrating Intelligent Chat Template Truncation")
print("=" * 80)

# Demo 1: Truncation with small context window
print("\nDemo 1: Conversation exceeding small context window")
print("-" * 50)

request = {
    "model": "claude-3-opus-20240229",
    "system": "You are a helpful AI assistant.",
    "messages": create_conversation(20),  # 40 messages total
}

chat_messages = convert_to_chat_template(request)
print(f"Original conversation: {len(chat_messages)} messages")

# Calculate original size
original_size = sum(len(msg.get("content", "")) for msg in chat_messages)
print(f"Original size: {original_size} characters")

# Force truncation with very small context (500 chars / 4 = 125 chars effective)
truncated = truncate_chat_template_to_fit(chat_messages, max_context_size=125)

print(f"\nAfter truncation: {len(truncated)} messages")
truncated_size = sum(len(msg.get("content", "")) for msg in truncated)
print(f"Truncated size: {truncated_size} characters")

print("\nTruncated messages:")
for i, msg in enumerate(truncated):
    print(
        f"  {i + 1}. [{msg['role']}] {msg['content'][:60]}{'...' if len(msg['content']) > 60 else ''}"
    )

# Demo 2: Very large system message truncation
print("\n\nDemo 2: Large system message requiring truncation")
print("-" * 50)

large_system = "This is a very detailed system prompt. " * 50  # ~2000 chars

request2 = {
    "model": "gpt-4",  # Non-Claude model, won't get 4x multiplier
    "system": large_system,
    "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ],
}

chat_messages2 = convert_to_chat_template(request2)
print(f"System message size: {len(chat_messages2[0]['content'])} characters")

# Use small context to force system truncation
truncated2 = truncate_chat_template_to_fit(chat_messages2, max_context_size=200)

print("\nAfter truncation:")
for i, msg in enumerate(truncated2):
    if msg["role"] == "system":
        if "(system message truncated)" in msg["content"]:
            print(f"  System message was truncated to {len(msg['content'])} characters")
        elif "[Previous" in msg["content"]:
            print(f"  Truncation notice: {msg['content']}")
        else:
            print(f"  System: {msg['content'][:50]}...")

# Demo 3: Real-world example with gradual truncation
print("\n\nDemo 3: Gradual truncation as context fills up")
print("-" * 50)

# Create a long conversation
long_conv = {
    "model": "claude-3-opus-20240229",
    "system": "You are Claude, an AI assistant.",
    "messages": [],
}

# Add 50 exchanges
for i in range(50):
    long_conv["messages"].append(
        {
            "role": "user",
            "content": f"This is message {i + 1}. Tell me about artificial intelligence topic number {i + 1}.",
        }
    )
    long_conv["messages"].append(
        {
            "role": "assistant",
            "content": f"Regarding topic {i + 1}: AI is fascinating because it involves machine learning, neural networks, and complex algorithms. This is a detailed response about AI topic {i + 1}.",
        }
    )

chat_messages3 = convert_to_chat_template(long_conv)
print(f"Full conversation: {len(chat_messages3)} messages")

# Test different context sizes
context_sizes = [50, 100, 200, 500]  # Very small to force truncation

for size in context_sizes:
    truncated = truncate_chat_template_to_fit(chat_messages3, max_context_size=size)
    kept_conv = len([m for m in truncated if m["role"] in ["user", "assistant"]])
    has_notice = any(
        "[Previous" in m.get("content", "") for m in truncated if m["role"] == "system"
    )

    print(f"\nContext size {size} chars:")
    print(f"  Kept {len(truncated)} messages ({kept_conv} conversation messages)")
    print(f"  Truncation notice: {'Yes' if has_notice else 'No'}")

    # Show the last few messages
    if kept_conv > 0:
        last_user_idx = -1
        for i in range(len(truncated) - 1, -1, -1):
            if truncated[i]["role"] == "user":
                last_user_idx = i
                break
        if last_user_idx >= 0:
            print(
                f'  Last user message: "{truncated[last_user_idx]["content"][:50]}..."'
            )

# Demo 4: API logging simulation
print("\n\nDemo 4: Automatic truncation in API logging")
print("-" * 50)

# This simulates what happens in log_chat_template()
large_request = {
    "model": "claude-3-opus-20240229",
    "messages": create_conversation(200),  # 400 messages
}

chat_messages4 = convert_to_chat_template(large_request)
total_size = sum(len(msg.get("content", "")) for msg in chat_messages4)

print(f"Chat template size: {total_size:,} characters")

if total_size > 100000:
    print("This exceeds the 100,000 character limit in log_chat_template()")
    print("The function would automatically truncate to fit...")

    truncated4 = truncate_chat_template_to_fit(
        chat_messages4, max_context_size=100000, model_name=large_request["model"]
    )

    new_size = sum(len(msg.get("content", "")) for msg in truncated4)
    print("\nAfter automatic truncation:")
    print(f"  Messages: {len(chat_messages4)} -> {len(truncated4)}")
    print(f"  Size: {total_size:,} -> {new_size:,} characters")
    print(f"  Removed: {len(chat_messages4) - len(truncated4)} messages")

print("\n" + "=" * 80)
print("Truncation demonstration complete!")
print("\nKey features demonstrated:")
print("✓ Intelligent removal of old message pairs")
print("✓ Preservation of system messages")
print("✓ Truncation of oversized system messages")
print("✓ Addition of truncation notices")
print("✓ Preservation of recent conversation context")
