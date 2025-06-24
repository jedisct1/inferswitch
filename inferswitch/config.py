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
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "default": 50000
}

# Cache configuration
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))  # 1 hour default