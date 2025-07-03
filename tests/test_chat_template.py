#!/usr/bin/env python3
"""
Test and demonstrate the chat template conversion functionality.
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import convert_to_chat_template, apply_chat_template

# Example 1: Simple conversation
print("Example 1: Simple conversation")
print("-" * 50)

request1 = {
    "model": "claude-3-opus-20240229",
    "messages": [
        {"role": "user", "content": "Hello, how are you?"},
        {
            "role": "assistant",
            "content": "I'm doing well, thank you! How can I help you today?",
        },
        {"role": "user", "content": "Can you explain quantum computing?"},
    ],
    "max_tokens": 100,
}

chat_messages1 = convert_to_chat_template(request1)
print("Chat messages:", json.dumps(chat_messages1, indent=2))
print("\nFormatted chat template:")
print(apply_chat_template(chat_messages1, add_generation_prompt=True))

# Example 2: With system message
print("\n\nExample 2: With system message")
print("-" * 50)

request2 = {
    "model": "claude-3-opus-20240229",
    "system": "You are a helpful coding assistant who explains concepts clearly.",
    "messages": [{"role": "user", "content": "What is a Python decorator?"}],
    "max_tokens": 200,
}

chat_messages2 = convert_to_chat_template(request2)
print("Chat messages:", json.dumps(chat_messages2, indent=2))
print("\nFormatted chat template:")
print(apply_chat_template(chat_messages2, add_generation_prompt=True))

# Example 3: Complex content blocks
print("\n\nExample 3: Complex content blocks")
print("-" * 50)

request3 = {
    "model": "claude-3-opus-20240229",
    "system": [
        {
            "text": "You are an AI assistant.",
            "type": "text",
            "cache_control": {"type": "ephemeral"},
        },
        {"text": "Be concise in your responses.", "type": "text"},
    ],
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Look at this image and describe what you see.",
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "base64_encoded_image_data",
                    },
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "I can see an image has been provided. Let me analyze it.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Now use the calculator tool to add 5 + 3"}
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "calculator",
                    "input": {"operation": "add", "a": 5, "b": 3},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tool_1", "content": "8"}
            ],
        },
    ],
    "max_tokens": 100,
}

chat_messages3 = convert_to_chat_template(request3)
print("Chat messages:", json.dumps(chat_messages3, indent=2))
print("\nFormatted chat template:")
print(apply_chat_template(chat_messages3, add_generation_prompt=True))

# Example 4: Tokenization request (simulated)
print("\n\nExample 4: Tokenization request")
print("-" * 50)

print("With tokenize=True:")
print(apply_chat_template(chat_messages1, add_generation_prompt=True, tokenize=True))

# Example 5: Different chat template styles
print("\n\nExample 5: Custom chat template format")
print("-" * 50)


def apply_llama_style_template(chat_messages):
    """Apply a Llama-style chat template."""
    formatted_parts = []

    for message in chat_messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        if role == "system":
            formatted_parts.append(f"<<SYS>>\n{content}\n<</SYS>>\n")
        elif role == "user":
            formatted_parts.append(f"[INST] {content} [/INST]")
        elif role == "assistant":
            formatted_parts.append(f"{content}")

    return "\n".join(formatted_parts)


print("Llama-style template:")
print(apply_llama_style_template(chat_messages2))


def apply_alpaca_style_template(chat_messages):
    """Apply an Alpaca-style chat template."""
    formatted_parts = []

    system_content = ""
    for message in chat_messages:
        if message["role"] == "system":
            system_content = message["content"]
            break

    if system_content:
        formatted_parts.append(f"### Instruction:\n{system_content}\n")

    for message in chat_messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        if role == "user":
            formatted_parts.append(f"### Input:\n{content}\n")
        elif role == "assistant":
            formatted_parts.append(f"### Response:\n{content}\n")

    return "\n".join(formatted_parts)


print("\n\nAlpaca-style template:")
print(apply_alpaca_style_template(chat_messages2))
