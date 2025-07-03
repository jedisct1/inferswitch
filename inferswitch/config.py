"""
Configuration settings for InferSwitch.
"""

import os
from pathlib import Path

# Logging configuration
LOG_FILE = Path("requests.log")

# API configuration
ANTHROPIC_API_BASE = "https://api.anthropic.com"

# Server configuration
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 1235

# Mode configuration
# Set PROXY_MODE=true to forward requests to real Anthropic API
# Set PROXY_MODE=false to return "OK" responses
PROXY_MODE = os.getenv("PROXY_MODE", "true").lower() == "true"

# Request timeout
REQUEST_TIMEOUT = 600.0

# Truncation settings
DEFAULT_TRUNCATION_LIMIT = 100000  # characters (~25k tokens)
TRUNCATION_BUFFER = 1000  # buffer to leave when truncating

# Model context sizes (in tokens, will be multiplied by 4 for chars)
MODEL_CONTEXT_SIZES = {
    # Claude 3 (legacy)
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    # Claude 3.5
    "claude-3-5-sonnet": 200000,
    "claude-3-5-sonnet-20240620": 200000,
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-haiku": 200000,
    "claude-3-5-haiku-20241022": 200000,
    # Claude 4
    "claude-4": 200000,
    "claude-4-20250514": 200000,
    "claude-sonnet-4": 200000,
    "claude-sonnet-4-20250514": 200000,
    "claude-opus-4": 200000,
    "claude-4-opus": 200000,
    "default": 50000,
}

# Model max output tokens limits
MODEL_MAX_TOKENS = {
    # Claude 3 (legacy)
    "claude-3-opus-20240229": 4096,
    "claude-3-sonnet-20240229": 4096,
    "claude-3-haiku-20240307": 4096,
    # Claude 3.5
    "claude-3-5-sonnet-20241022": 8192,
    "claude-3-5-haiku-20241022": 8192,
    # Claude 4 (these models support higher limits)
    "claude-opus-4-20250514": 32768,
    "claude-sonnet-4-20250514": 32768,
    "claude-4-opus-20250514": 32768,
    "claude-4-sonnet-20250514": 32768,
    # Default for unknown models
    "default": 4096,
}

# Cache configuration
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))  # 1 hour default

# HTTP/HTTPS Proxy configuration
PROXY_HOST = os.getenv("PROXY_HOST", "127.0.0.1")
PROXY_PORT = int(os.getenv("PROXY_PORT", "1236"))
PROXY_ENABLED = os.getenv("PROXY_ENABLED", "false").lower() == "true"  # Disabled by default
PROXY_TIMEOUT = int(os.getenv("PROXY_TIMEOUT", "30"))  # seconds
