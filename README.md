# InferSwitch

<div align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-orange.svg" alt="FastAPI">
</div>

<img src="https://raw.github.com/jedisct1/inferswitch/master/.media/logo.png" alt="InferSwitch logo" width="600">

## Unified Gateway for Multiple LLM Providers

InferSwitch is an intelligent API gateway that seamlessly routes requests between multiple Large Language Model (LLM) providers. It acts as a drop-in replacement for the Anthropic API while providing automatic failover, smart caching, and the ability to use local models for appropriate queries.

### Key Features

🚀 **Multi-Provider Support** - Route between Anthropic Claude, OpenAI GPT, OpenRouter models, local LM-Studio models, and any OpenAI-compatible endpoints

🧠 **Intelligent Routing** - Automatically sends simple queries to fast local models and complex queries to powerful cloud models

💾 **Smart Caching** - Reduce costs and latency with intelligent response caching that ignores irrelevant metadata

🔄 **Model Overrides** - Transparently replace expensive models with cheaper alternatives for development and testing

🔐 **OAuth Support** - Use your Claude.ai account instead of managing API keys

📊 **Difficulty Assessment** - Built-in MLX model rates query complexity to optimize routing decisions

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
- [Advanced Usage](#advanced-usage)
- [Development](#development)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8 or higher
- (Optional) LM-Studio for local model support
- (Optional) MLX framework for difficulty assessment on Apple Silicon

### Install with uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/jedisct1/inferswitch.git
cd inferswitch

# Install dependencies with uv
uv sync

# Run the server
uv run python main.py
```

### Install with pip

```bash
# Clone the repository
git clone https://github.com/jedisct1/inferswitch.git
cd inferswitch

# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
```

### Install as a Package

```bash
# Install InferSwitch as a command-line tool
pip install -e .

# Run from anywhere
inferswitch
```

## Quick Start

### Basic Usage

1. **Start InferSwitch** (defaults to Anthropic backend):
```bash
uv run python main.py
```

2. **Make your first request**:
```bash
curl -X POST http://localhost:1235/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello, Claude!"}],
    "max_tokens": 100
  }'
```

### Using Local Models with LM-Studio

1. **Start LM-Studio** and load a model
2. **Start InferSwitch** with LM-Studio backend:
```bash
INFERSWITCH_BACKEND=lm-studio uv run python main.py
```

3. **All requests now route to your local model**:
```bash
# Same API call, but now uses your local model!
curl -X POST http://localhost:1235/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Using OpenRouter Models

1. **Get an OpenRouter API key** from [openrouter.ai](https://openrouter.ai)
2. **Start InferSwitch** with OpenRouter backend:
```bash
OPENROUTER_API_KEY=your_key INFERSWITCH_BACKEND=openrouter uv run python main.py
```

3. **Access hundreds of models** through OpenRouter:
```bash
# Use any OpenRouter model
curl -X POST http://localhost:1235/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-3.5-sonnet",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Smart Routing Example

Configure InferSwitch to automatically route based on query difficulty:

1. **Create `inferswitch.config.json`**:
```json
{
  "force_difficulty_routing": true,
  "difficulty_models": {
    "0-1": ["all-hands_openhands-lm-32b-v0.1", "gpt-3.5-turbo"],
    "2": ["claude-3-5-haiku-20241022", "gpt-4"],
    "3": ["claude-3-7-sonnet-20250219", "gpt-4-turbo"],
    "4": ["claude-sonnet-4-20250514"],
    "5": ["claude-opus-4-20250514"]
  },
  "model_providers": {
    "qwen/qwen3-1.7b": "lm-studio",
    "qwen/qwen3-30b-a3b": "lm-studio",
    "all-hands_openhands-lm-32b-v0.1": "lm-studio",
    "claude-3-5-haiku-20241022": "anthropic",
    "claude-3-5-sonnet-20241022": "anthropic",
    "claude-3-7-sonnet-20250219": "anthropic",
    "claude-sonnet-4-20250514": "anthropic",
    "claude-opus-4-20250514": "anthropic"
  },
  "fallback": {
    "provider": "lm-studio",
    "model": "all-hands_openhands-lm-32b-v0.1"
  },
  "providers_auth": {
    "anthropic": {
      "oauth": {}
    }
  }
}
```

2. **Now InferSwitch automatically routes**:
   - "What is 2+2?" → Local model (difficulty ~0)
   - "Write a Python function" → Sonnet (difficulty ~3)
   - "Design a distributed system" → Opus (difficulty ~10)

## Configuration

### Configuration Methods

InferSwitch can be configured through (in order of precedence):

1. **Request headers** - Per-request overrides
2. **Environment variables** - Runtime configuration
3. **Configuration file** - `inferswitch.config.json` in working directory
4. **Default values** - Built-in defaults

### Configuration File Structure

Create `inferswitch.config.json` in your working directory:

```json
{
  // Backend configurations
  "backends": {
    "anthropic": {
      "api_key": "sk-ant-...",  // Optional if using env var
      "timeout": 300
    },
    "lm-studio": {
      "base_url": "http://127.0.0.1:1234",
      "timeout": 600
    },
    "openai": {
      "api_key": "sk-...",
      "base_url": "https://api.openai.com/v1"
    },
    "openrouter": {
      "api_key": "sk-or-...",
      "base_url": "https://openrouter.ai/api/v1"
    }
  },

  // Model to backend mappings
  "model_providers": {
    "claude-3-haiku-20240307": "lm-studio",
    "gpt-3.5-turbo": "openai",
    "my-custom-model": "lm-studio"
  },

  // Model override mappings
  "model_overrides": {
    "claude-3-5-sonnet-20241022": "claude-3-haiku-20240307",
    "gpt-4": "gpt-3.5-turbo"
  },

  // Difficulty-based routing
  "difficulty_models": {
    "0-3": "claude-3-haiku-20240307",
    "3-7": "claude-3-5-sonnet-20241022",
    "7-10": "claude-3-opus-20240229"
  },

  // OAuth configuration (optional)
  "providers_auth": {
    "anthropic": {
      "oauth": {}
    }
  },

  // Fallback configuration
  "fallback": {
    "provider": "anthropic",
    "model": "claude-3-haiku-20240307"
  }
}
```

### Environment Variables

| Variable                     | Description                            | Default                 |
| ---------------------------- | -------------------------------------- | ----------------------- |
| `INFERSWITCH_BACKEND`        | Force all requests to specific backend | `anthropic`             |
| `ANTHROPIC_API_KEY`          | Anthropic API key                      | Required for Anthropic  |
| `OPENAI_API_KEY`             | OpenAI API key                         | Required for OpenAI     |
| `OPENROUTER_API_KEY`         | OpenRouter API key                     | Required for OpenRouter |
| `LM_STUDIO_BASE_URL`         | LM-Studio server URL                   | `http://127.0.0.1:1234` |
| `OPENROUTER_BASE_URL`        | OpenRouter server URL                  | `https://openrouter.ai/api/v1` |
| `INFERSWITCH_MODEL_OVERRIDE` | Model override mappings                | None                    |
| `INFERSWITCH_DEFAULT_MODEL`  | Override all models                    | None                    |
| `CACHE_ENABLED`              | Enable response caching                | `true`                  |
| `CACHE_MAX_SIZE`             | Maximum cache entries                  | `1000`                  |
| `CACHE_TTL_SECONDS`          | Cache time-to-live                     | `3600`                  |
| `LOG_LEVEL`                  | Logging verbosity                      | `INFO`                  |
| `PROXY_MODE`                 | Enable proxy mode                      | `true`                  |
| `INFERSWITCH_MODEL_DISABLE_DURATION` | Seconds to disable failed models | `300`                   |

## Core Concepts

### Backend Priority

InferSwitch selects backends using this priority order:

1. **Explicit Header** - `x-backend: lm-studio` in request
2. **Environment Override** - `INFERSWITCH_BACKEND=lm-studio`
3. **Difficulty Routing** - Based on query complexity (if configured)
4. **Model Mapping** - Direct model → backend mapping
5. **Pattern Matching** - e.g., "claude-*" → anthropic
6. **Fallback** - Configured fallback or default to anthropic

### Difficulty-Based Routing

InferSwitch can analyze each query and route to appropriate backends:

- **Difficulty 0-1**: Simple queries (definitions, basic math)
  - Examples: "What is 2+2?", "Define HTTP"
  - Best for: Fast local models

- **Difficulty 1-3**: Moderate queries (explanations, simple code)
  - Examples: "Explain recursion", "Write a sorting function"
  - Best for: Capable local or mid-tier cloud models

- **Difficulty 3-5**: Complex queries (architecture, debugging)
  - Examples: "Design a microservices system", "Debug this race condition"
  - Best for: State-of-the-art cloud models

### Model Overrides

Replace any model transparently:

```bash
# Replace all Claude requests with local model
INFERSWITCH_MODEL_OVERRIDE="claude-3-5-sonnet-20241022:llama-3.1-8b" uv run python main.py

# Replace ALL models with a single model
INFERSWITCH_DEFAULT_MODEL="llama-3.1-8b" uv run python main.py
```

### Automatic Model Fallback

InferSwitch provides automatic fallback when models fail due to rate limits or insufficient credits:

```json
{
  "difficulty_models": {
    "0-3": ["claude-3-haiku-20240307", "gpt-3.5-turbo", "llama-3.1-8b"],
    "3-5": ["claude-3-5-sonnet-20241022", "gpt-4-turbo", "claude-3-opus-20240229"]
  },
  "model_availability": {
    "disable_duration_seconds": 300  // 5 minutes
  }
}
```

Features:
- **Automatic Retry**: When a model fails, the next model in the list is tried
- **Temporary Disabling**: Failed models are disabled for a configurable duration
- **Self-Healing**: Models automatically re-enable after the cooldown period
- **Smart Detection**: Recognizes rate limits (429) and credit errors (402)

## API Reference

### Anthropic-Compatible Endpoints

#### POST /v1/messages
Send messages to any configured backend using Anthropic's format.

```bash
curl -X POST http://localhost:1235/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "stream": true
  }'
```

#### POST /v1/messages/count_tokens
Count tokens in messages without making a completion request.

### OpenAI-Compatible Endpoints

#### POST /v1/chat/completions
OpenAI-style chat completions (automatically converted to/from Anthropic format).

### InferSwitch Extensions

#### GET /backends/status
Check health and capabilities of all configured backends.

#### GET /cache/stats
View cache performance metrics.

#### POST /cache/clear
Clear the response cache.

#### POST /v1/messages/chat-template
Convert messages to Hugging Face chat template format.

### OAuth Endpoints (Anthropic)

#### GET /oauth/authorize
Start OAuth authentication flow.

#### GET /oauth/callback
Complete OAuth authentication.

#### GET /oauth/status
Check current OAuth status.

#### POST /oauth/refresh
Manually refresh OAuth tokens.

#### POST /oauth/logout
Clear stored OAuth tokens.

## Advanced Usage

### OAuth Authentication

Use your Claude.ai account instead of API keys:

1. **Configure OAuth** in `inferswitch.config.json`:
```json
{
  "providers_auth": {
    "anthropic": {
      "oauth": {
        "client_id": "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
      }
    }
  }
}
```

2. **Start InferSwitch** - it will prompt for authentication:
```bash
uv run python main.py
# Follow the prompts to authenticate via Claude.ai
```

3. **Use without API keys**:
```bash
# No x-api-key header needed!
curl -X POST http://localhost:1235/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model": "claude-3-5-sonnet-20241022", "messages": [...]}'
```

### Custom Backend Integration

Add any OpenAI-compatible endpoint:

```json
{
  "backends": {
    "my-backend": {
      "base_url": "https://my-llm-api.com/v1",
      "api_key": "my-api-key",
      "timeout": 300
    }
  },
  "model_providers": {
    "my-model": "my-backend"
  }
}
```

### Performance Optimization

1. **Enable caching** for repeated queries:
```bash
CACHE_ENABLED=true CACHE_TTL_SECONDS=7200 uv run python main.py
```

2. **Use difficulty routing** to minimize costs:
   - Configure local models for simple queries
   - Reserve expensive models for complex tasks

3. **Monitor performance**:
```bash
# Check cache hit rate
curl http://localhost:1235/cache/stats

# Enable debug logging
LOG_LEVEL=DEBUG uv run python main.py
```

## Development

### Running Tests

```bash
# Run all tests
for test in tests/test_*.py; do 
  echo "Running $test"
  uv run python "$test" || break
done

# Run specific test suites
uv run python tests/test_api.py              # Core API tests
uv run python tests/test_difficulty_routing.py # Routing logic
uv run python tests/test_cache.py            # Caching functionality
uv run python tests/test_streaming.py        # Streaming responses
```

### Benchmarking

```bash
# Benchmark MLX difficulty classifier
python benchmarks/benchmark_mlx_classifier.py

# Compare different MLX models
python benchmarks/benchmark_mlx_models.py
```

### Adding a New Backend

1. Create a new file in `inferswitch/backends/`
2. Inherit from `BaseBackend`
3. Implement required methods
4. Register in `main.py`

See `inferswitch/backends/openai.py` for a complete example.

## Architecture

InferSwitch uses a modular, extensible architecture:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client    │────▶│ InferSwitch  │────▶│  Anthropic  │
└─────────────┘     │              │     └─────────────┘
                    │   Router     │     ┌─────────────┐
                    │      +       │────▶│   OpenAI    │
                    │   Cache      │     └─────────────┘
                    │      +       │     ┌─────────────┐
                    │ Normalizer   │────▶│  LM-Studio  │
                    └──────────────┘     └─────────────┘
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- MLX support for Apple Silicon
- Compatible with Anthropic, OpenAI, and LM-Studio APIs
