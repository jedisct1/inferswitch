#!/usr/bin/env python3
"""
Demonstrate the truncation functionality working correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import truncate_chat_template_to_fit

# Create a simple test case
print("Testing Chat Template Truncation - Working Example")
print("=" * 60)

# Create messages that will definitely exceed a small context
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
]

# Add many message pairs
for i in range(20):
    messages.append({
        "role": "user",
        "content": f"User message {i+1}: " + "This is a long message. " * 20
    })
    messages.append({
        "role": "assistant",
        "content": f"Assistant response {i+1}: " + "This is a detailed response. " * 30
    })

# Calculate total size
total_size = sum(len(msg['content']) for msg in messages)
print("\nOriginal conversation:")
print(f"  Total messages: {len(messages)}")
print(f"  Total size: {total_size:,} characters")
print(f"  First user message: {messages[1]['content'][:50]}...")
print(f"  Last user message: {messages[-2]['content'][:50]}...")

# Now truncate to a very small size that will force truncation
# Since we're not specifying a model, it won't be a Claude model, 
# so max_context_size will be used directly (* 4)
print("\n\nTruncating to fit in 1000 character context...")
print("(This becomes 4000 chars after conversion)")

truncated = truncate_chat_template_to_fit(
    messages.copy(), 
    max_context_size=1000,
    model_name="gpt-4"  # Non-Claude model
)

truncated_size = sum(len(msg['content']) for msg in truncated)
print("\nAfter truncation:")
print(f"  Total messages: {len(truncated)}")
print(f"  Total size: {truncated_size:,} characters")
print(f"  Messages removed: {len(messages) - len(truncated)}")

print("\nTruncated message list:")
for i, msg in enumerate(truncated):
    content_preview = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
    print(f"  {i+1}. [{msg['role']:9}] {content_preview}")

# Check for truncation notice
has_truncation_notice = any(
    '[Previous' in msg['content'] and 'truncated' in msg['content'] 
    for msg in truncated if msg['role'] == 'system'
)
print(f"\nTruncation notice added: {has_truncation_notice}")

# Now test with an even smaller context to show system message truncation
print("\n\n" + "=" * 60)
print("Testing with very large system message...")

messages2 = [
    {"role": "system", "content": "System instructions: " + "Very detailed instructions. " * 100},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},
]

system_size = len(messages2[0]['content'])
print(f"\nOriginal system message size: {system_size:,} characters")

# Truncate with small context
truncated2 = truncate_chat_template_to_fit(
    messages2.copy(),
    max_context_size=500,  # Will become 2000 after *4
    model_name="gpt-4"
)

print("\nAfter truncation:")
for msg in truncated2:
    if msg['role'] == 'system':
        if '(system message truncated)' in msg['content']:
            print(f"  System message truncated to: {len(msg['content'])} characters")
            print(f"  Ends with: ...{msg['content'][-50:]}")
        elif '[Previous' not in msg['content']:
            print(f"  System message size: {len(msg['content'])} characters")

# Finally, let's show the exact behavior in the API
print("\n\n" + "=" * 60)
print("Demonstrating API auto-truncation behavior...")

# Create messages that exceed 100k chars
huge_messages = [{"role": "system", "content": "AI Assistant"}]
for i in range(500):
    huge_messages.append({
        "role": "user", 
        "content": f"Question {i}: " + "x" * 200
    })
    huge_messages.append({
        "role": "assistant",
        "content": f"Answer {i}: " + "y" * 200  
    })

huge_size = sum(len(msg['content']) for msg in huge_messages)
print(f"\nHuge conversation: {huge_size:,} characters")

if huge_size > 100000:
    print("This would trigger auto-truncation in log_chat_template()")
    
    # Simulate what log_chat_template does
    truncated_api = truncate_chat_template_to_fit(
        huge_messages,
        max_context_size=100000,
        model_name="claude-3-opus-20240229"
    )
    
    final_size = sum(len(msg['content']) for msg in truncated_api)
    print("\nAfter API truncation (100k limit for claude-3-opus):")
    print(f"  Original: {len(huge_messages)} messages, {huge_size:,} chars")
    print(f"  Truncated: {len(truncated_api)} messages, {final_size:,} chars")
    print(f"  Fits in limit: {final_size <= 100000 * 4}")  # *4 for Claude model