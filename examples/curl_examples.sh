#!/bin/bash
# Example curl commands for using InferSwitch with different backends

echo "=== InferSwitch Backend Examples ==="
echo

# 1. Using default backend (set by INFERSWITCH_BACKEND env var)
echo "1. Using default backend:"
curl -X POST http://localhost:1235/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "qwen/qwen3-1.7b",
    "messages": [
      {"role": "user", "content": "Hello, what is 2+2?"}
    ],
    "max_tokens": 50
  }'

echo -e "\n\n"

# 2. Explicitly using LM-Studio backend
echo "2. Explicitly using LM-Studio backend:"
curl -X POST http://localhost:1235/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -H "anthropic-version": "2023-06-01" \
  -H "x-backend: lm-studio" \
  -d '{
    "model": "mistralai/mistral-small-3.2",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 30
  }'

echo -e "\n\n"

# 3. Using Anthropic backend explicitly
echo "3. Using Anthropic backend (requires valid API key):"
curl -X POST http://localhost:1235/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-ant-api03-xxxxx" \
  -H "anthropic-version": "2023-06-01" \
  -H "x-backend: anthropic" \
  -d '{
    "model": "claude-3-haiku-20240307",
    "messages": [
      {"role": "user", "content": "Say hello"}
    ],
    "max_tokens": 10
  }'

echo -e "\n\n"

# 4. Streaming request with LM-Studio
echo "4. Streaming with LM-Studio:"
curl -X POST http://localhost:1235/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -H "anthropic-version: 2023-06-01" \
  -H "x-backend: lm-studio" \
  -d '{
    "model": "qwen/qwen3-1.7b",
    "messages": [
      {"role": "user", "content": "Count from 1 to 5"}
    ],
    "max_tokens": 50,
    "stream": true
  }'

echo -e "\n\n"

# 5. Check backend status
echo "5. Check backend status:"
curl http://localhost:1235/backends/status | jq '.'

echo -e "\n\n"

# 6. Using system message
echo "6. Using system message with LM-Studio:"
curl -X POST http://localhost:1235/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -H "anthropic-version: "2023-06-01" \
  -H "x-backend: lm-studio" \
  -d '{
    "model": "qwen/qwen3-1.7b",
    "system": "You are a helpful assistant that speaks like a pirate.",
    "messages": [
      {"role": "user", "content": "Tell me about the weather"}
    ],
    "max_tokens": 100
  }'