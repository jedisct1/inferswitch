#!/usr/bin/env python3
"""
Demonstrate how remove_oldest_message_pair and truncate_chat_template_to_fit
can work together for different truncation strategies.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    remove_oldest_message_pair,
    truncate_chat_template_to_fit,
    convert_to_chat_template,
)


def calculate_size(messages):
    """Calculate total character size of messages."""
    return sum(len(msg.get("content", "")) for msg in messages)


print("Combined Truncation Strategies Demo")
print("=" * 70)

# Create a conversation
request = {
    "model": "claude-3-opus-20240229",
    "system": "You are an AI assistant helping with a long technical discussion.",
    "messages": [],
}

# Add 20 exchanges
for i in range(20):
    request["messages"].append(
        {
            "role": "user",
            "content": f"Question {i + 1}: Please explain technical concept number {i + 1} in detail.",
        }
    )
    request["messages"].append(
        {
            "role": "assistant",
            "content": f"Answer {i + 1}: Here's a comprehensive explanation of concept {i + 1}. "
            * 5,
        }
    )

chat_messages = convert_to_chat_template(request)
original_size = calculate_size(chat_messages)

print("Original conversation:")
print(f"  Messages: {len(chat_messages)}")
print(f"  Size: {original_size:,} characters")
print()

# Strategy 1: Gradual removal with remove_oldest_message_pair
print("Strategy 1: Gradual removal (one pair at a time)")
print("-" * 50)

strategy1_messages = chat_messages.copy()
target_size = 1500
removals = 0

print(f"Target size: {target_size} characters")
print("Removing oldest pairs one by one...")

while calculate_size(strategy1_messages) > target_size and len(strategy1_messages) > 1:
    strategy1_messages = remove_oldest_message_pair(strategy1_messages)
    removals += 1
    current_size = calculate_size(strategy1_messages)
    print(
        f"  Removal {removals}: {len(strategy1_messages)} messages, {current_size} chars"
    )

print(f"\nResult: Removed {removals} pairs")
print(
    f"Final: {len(strategy1_messages)} messages, {calculate_size(strategy1_messages)} chars"
)

# Strategy 2: Bulk truncation with truncate_chat_template_to_fit
print("\n\nStrategy 2: Intelligent bulk truncation")
print("-" * 50)

# Use a small context to force truncation
strategy2_messages = truncate_chat_template_to_fit(
    chat_messages.copy(),
    max_context_size=375,  # Will be *4 = 1500 chars
    model_name="gpt-4",  # Non-Claude model
)

print("Target size: ~1500 characters (375 * 4)")
print(
    f"Result: {len(strategy2_messages)} messages, {calculate_size(strategy2_messages)} chars"
)

# Check for truncation notice
has_notice = any(
    "[Previous" in msg["content"]
    for msg in strategy2_messages
    if msg["role"] == "system"
)
print(f"Truncation notice added: {has_notice}")

# Strategy 3: Hybrid approach
print("\n\nStrategy 3: Hybrid approach")
print("-" * 50)
print(
    "First use bulk truncation for large reduction, then fine-tune with gradual removal"
)

# Start with a large conversation
large_request = {
    "model": "claude-3-opus-20240229",
    "system": "AI Assistant for technical discussions",
    "messages": [],
}

for i in range(50):
    large_request["messages"].extend(
        [
            {"role": "user", "content": f"Q{i + 1}: " + "Technical question. " * 10},
            {"role": "assistant", "content": f"A{i + 1}: " + "Detailed answer. " * 20},
        ]
    )

hybrid_messages = convert_to_chat_template(large_request)
print(
    f"\nStarting with: {len(hybrid_messages)} messages, {calculate_size(hybrid_messages):,} chars"
)

# Step 1: Bulk truncation to get close to target
intermediate_target = 2000
hybrid_messages = truncate_chat_template_to_fit(
    hybrid_messages, max_context_size=intermediate_target // 4, model_name="gpt-4"
)
print(
    f"After bulk truncation: {len(hybrid_messages)} messages, {calculate_size(hybrid_messages)} chars"
)

# Step 2: Fine-tune with gradual removal
final_target = 1800
fine_tune_removals = 0

while calculate_size(hybrid_messages) > final_target and len(hybrid_messages) > 2:
    hybrid_messages = remove_oldest_message_pair(hybrid_messages)
    fine_tune_removals += 1

print(
    f"After fine-tuning ({fine_tune_removals} removals): {len(hybrid_messages)} messages, {calculate_size(hybrid_messages)} chars"
)

# Comparison
print("\n\nComparison of Strategies")
print("-" * 50)
print(f"Original: {len(chat_messages)} messages, {original_size:,} chars")
print(
    f"Strategy 1 (gradual): {len(strategy1_messages)} messages, {calculate_size(strategy1_messages)} chars"
)
print(
    f"Strategy 2 (bulk): {len(strategy2_messages)} messages, {calculate_size(strategy2_messages)} chars"
)
print(
    f"Strategy 3 (hybrid): {len(hybrid_messages)} messages, {calculate_size(hybrid_messages)} chars"
)

print("\n\nKey Differences:")
print("- remove_oldest_message_pair: Simple, predictable, removes one pair at a time")
print(
    "- truncate_chat_template_to_fit: Intelligent, preserves recent context, bulk operation"
)
print("- Hybrid: Best of both worlds for precise control over final size")
