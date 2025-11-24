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
    "claude-haiku-4-5-20251001": 200000,
    "claude-4": 200000,
    "claude-4-20250514": 200000,
    "claude-sonnet-4": 1000000,
    "claude-sonnet-4-20250514": 200000,
    "claude-sonnet-4-1": 200000,
    "claude-sonnet-4-5": 200000,
    "claude-sonnet-4-5-20250929": 200000,
    "claude-opus-4": 200000,
    "claude-4-opus": 200000,
    "claude-opus-4-1": 200000,
    "claude-opus-4-1-20250805": 200000,
    "claude-opus-4-5": 200000,
    "claude-opus-4-5-20251101": 200000,
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
    "claude-haiku-4-5-20251001": 64000,
    "claude-opus-4-20250514": 32000,
    "claude-opus-4-1-20250805": 32000,
    "claude-4-opus-20250514": 32000,
    "claude-4-sonnet-20250514": 64000,
    "claude-sonnet-4-20250514": 64000,
    "claude-sonnet-4-1": 64000,
    "claude-sonnet-4-5": 64000,
    "claude-sonnet-4-5-20250929": 64000,
    "claude-opus-4-5": 32000,
    "claude-opus-4-5-20251101": 32000,
    # Default for unknown models
    "default": 4096,
}

# Cache configuration
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))  # 1 hour default
