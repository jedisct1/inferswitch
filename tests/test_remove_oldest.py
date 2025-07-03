#!/usr/bin/env python3
"""
Test the remove_oldest_message_pair function.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import remove_oldest_message_pair


def print_messages(messages, title="Messages"):
    """Helper to print messages nicely."""
    print(f"\n{title} ({len(messages)} total):")
    for i, msg in enumerate(messages):
        content_preview = (
            msg["content"][:60] + "..." if len(msg["content"]) > 60 else msg["content"]
        )
        print(f"  {i + 1}. [{msg['role']:9}] {content_preview}")


print("Testing remove_oldest_message_pair Function")
print("=" * 70)

# Test 1: Basic conversation
print("\nTest 1: Basic conversation with system message")
messages1 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a high-level programming language."},
    {"role": "user", "content": "What about Java?"},
    {
        "role": "assistant",
        "content": "Java is an object-oriented programming language.",
    },
]

print_messages(messages1, "Original")
reduced1 = remove_oldest_message_pair(messages1)
print_messages(reduced1, "After removing oldest pair")

# Test 2: Multiple removals
print("\n\nTest 2: Removing multiple pairs sequentially")
messages2 = messages1.copy()

for i in range(3):
    print(f"\nIteration {i + 1}:")
    messages2 = remove_oldest_message_pair(messages2)
    print_messages(messages2, f"After removal {i + 1}")
    if len(messages2) <= 1:
        print("  (No more pairs to remove)")
        break

# Test 3: Conversation with only user messages
print("\n\nTest 3: Conversation with standalone user messages")
messages3 = [
    {"role": "system", "content": "System prompt"},
    {"role": "user", "content": "First question"},
    {"role": "user", "content": "Second question"},
    {"role": "assistant", "content": "Answer to second question"},
]

print_messages(messages3, "Original")
reduced3 = remove_oldest_message_pair(messages3)
print_messages(reduced3, "After removing oldest pair")

# Test 4: No user messages
print("\n\nTest 4: Only system messages")
messages4 = [
    {"role": "system", "content": "System message 1"},
    {"role": "system", "content": "System message 2"},
]

print_messages(messages4, "Original")
reduced4 = remove_oldest_message_pair(messages4)
print_messages(reduced4, "After removal attempt")
print("  (No change - no user/assistant pairs to remove)")

# Test 5: Complex conversation
print("\n\nTest 5: Complex conversation with mixed messages")
messages5 = [
    {"role": "system", "content": "You are Claude."},
    {"role": "system", "content": "Be helpful and harmless."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "Tell me a joke."},
    {"role": "assistant", "content": "Why did the chicken cross the road?"},
    {"role": "user", "content": "I don't know, why?"},
    {"role": "assistant", "content": "To get to the other side!"},
    {"role": "user", "content": "That's funny!"},
]

print_messages(messages5, "Original")

# Remove pairs one by one
print("\nRemoving oldest pairs one by one:")
current = messages5.copy()
removal_count = 0

while True:
    prev_len = len(current)
    current = remove_oldest_message_pair(current)

    if len(current) == prev_len:
        print(f"\nNo more pairs to remove after {removal_count} removals")
        break

    removal_count += 1
    print(f"\nAfter removal {removal_count}:")
    print_messages(current)

# Test 6: Empty list
print("\n\nTest 6: Empty message list")
messages6 = []
reduced6 = remove_oldest_message_pair(messages6)
print(f"Original: {messages6}")
print(f"After removal: {reduced6}")

# Test 7: Using with truncate_chat_template_to_fit
print("\n\nTest 7: Comparison with truncate_chat_template_to_fit")
print("The remove_oldest_message_pair function provides a simpler alternative")
print("for gradually reducing conversation size by removing one pair at a time,")
print("while truncate_chat_template_to_fit does intelligent bulk truncation.")

# Practical example
print("\n\nPractical Example: Gradual size reduction")
messages7 = [
    {"role": "system", "content": "AI Assistant"},
]

# Add 10 exchanges
for i in range(10):
    messages7.append({"role": "user", "content": f"Question {i + 1}"})
    messages7.append({"role": "assistant", "content": f"Answer {i + 1}"})

total_size = sum(len(msg["content"]) for msg in messages7)
print(f"\nStarting with {len(messages7)} messages, {total_size} total characters")

# Remove pairs until we're under a certain size
target_size = 100
while total_size > target_size and len(messages7) > 1:
    messages7 = remove_oldest_message_pair(messages7)
    total_size = sum(len(msg["content"]) for msg in messages7)
    print(f"After removal: {len(messages7)} messages, {total_size} characters")

print("\nFinal messages:")
print_messages(messages7)
