# InferSwitch

<div align="center">
  <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-orange.svg" alt="FastAPI">
</div>

<img src="https://raw.github.com/jedisct1/inferswitch/master/.media/logo.png" alt="InferSwitch logo" width="600">

## Unified Gateway for Multiple LLM Providers

InferSwitch is an intelligent API gateway that seamlessly routes requests between multiple Large Language Model (LLM) providers. It acts as a drop-in replacement for the Anthropic API while providing expertise-based routing, automatic failover, smart caching, and the ability to use local models through LM-Studio or other backends.

### Key Features

🚀 **Multi-Provider Support** - Route between Anthropic Claude, OpenAI GPT, OpenRouter models, local LM-Studio models, and any OpenAI-compatible endpoints

🧠 **Custom Expert Routing** - Define your own AI experts with custom descriptions and let MLX intelligently route queries to the most appropriate specialist - no hardcoded patterns needed

🎯 **MLX-Powered Classification** - Local AI models (default: `jedisct1/arch-router-1.5b`, optimized for routing) analyze queries to match them with your custom expert definitions using pure AI classification

💾 **Smart Caching** - Reduce costs and latency with intelligent response caching that ignores irrelevant metadata

🔄 **Model Overrides** - Transparently replace expensive models with cheaper alternatives for development and testing

🔐 **OAuth Support** - Use your Claude.ai account instead of managing API keys

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

- Python 3.12 or higher
- (Optional) LM-Studio for local model support
- (Optional) MLX framework for intelligent routing (automatically installed on Apple Silicon)

### Install with uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/jedisct1/inferswitch.git
cd inferswitch

# Install dependencies with uv
uv sync

# Run the server
uv run python main.py
# or
uv run python -m inferswitch.main
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
# or
python -m inferswitch.main
```

### Install as a Package

```bash
# Install InferSwitch as a command-line tool
pip install -e .

# Run from anywhere
inferswitch
```

## Quick Start

### 5-Minute Setup

1. **Clone and install**:
```bash
git clone https://github.com/jedisct1/inferswitch.git
cd inferswitch
uv sync
```

2. **Set your API key**:
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

3. **Start InferSwitch**:
```bash
uv run python main.py
```

4. **Test it works**:
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

You should see Claude respond normally. **That's it!** InferSwitch is now running and ready to intelligently route your requests.

### Advanced Quick Start Examples

#### Enable Streaming
```bash
curl -X POST http://localhost:1235/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Write a short story about a robot"}],
    "max_tokens": 300,
    "stream": true
  }'
```

#### Test Expert Routing
```bash
# Create a simple expert config
cat > inferswitch.config.json << 'EOF'
{
  "force_expert_routing": true,
  "expert_definitions": {
    "coding_expert": "A programming specialist for code-related tasks",
    "general_assistant": "A general-purpose assistant for other tasks"
  },
  "expert_models": {
    "coding_expert": ["claude-3-5-sonnet-20241022"],
    "general_assistant": ["claude-3-haiku-20240307"]
  },
  "model_providers": {
    "claude-3-5-sonnet-20241022": "anthropic",
    "claude-3-haiku-20240307": "anthropic"
  }
}
EOF

# Restart InferSwitch
uv run python main.py

# Test code query (should route to Sonnet)
curl -X POST http://localhost:1235/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Debug this Python function: def add(a, b): return a + b + 1"}],
    "max_tokens": 200
  }'

# Test general query (should route to Haiku)
curl -X POST http://localhost:1235/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 50
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

## Custom Expert System

InferSwitch's most powerful feature is its custom expert routing system. Unlike traditional rule-based routing, InferSwitch uses MLX language models to intelligently classify queries and route them to your custom-defined experts.

### Why Custom Experts?

- **No Hardcoded Patterns**: MLX models understand context and meaning, not just keywords
- **Domain-Specific Routing**: Create experts for any field - medical, legal, technical, creative
- **Flexible Definitions**: Define experts with natural language descriptions
- **Intelligent Matching**: Queries are matched based on semantic similarity to expert descriptions
- **Multi-Backend Support**: Route different experts to different model providers

### Expert Setup Example

1. **Create `inferswitch.config.json`** with your expert definitions:
```json
{
  "force_expert_routing": true,
  "expert_definitions": {
    "coding_specialist": "A coding-focused AI model optimized for programming tasks including writing code, debugging, code review, refactoring, explaining algorithms, and solving complex programming problems across multiple languages and frameworks.",
    "vision_analyst": "A vision-capable multimodal AI model that can analyze images, screenshots, diagrams, charts, UI mockups, and visual content, providing detailed descriptions and insights about visual elements.",
    "documentation_writer": "A model optimized for creating clear, comprehensive documentation including API docs, README files, technical guides, code comments, user manuals, and converting complex technical concepts into readable content.",
    "reasoning_engine": "A model optimized for complex reasoning, mathematical problem-solving, logical analysis, step-by-step thinking, research tasks, and handling queries requiring deep analytical thinking.",
    "fast_responder": "A lightweight, fast model optimized for quick responses to simple questions, basic coding tasks, quick explanations, and scenarios where speed is prioritized over complexity.",
    "general_assistant": "A well-rounded generalist model capable of handling diverse tasks across multiple domains when no specific model capability is clearly required for the query."
  },
  "expert_models": {
    "coding_specialist": ["claude-3-5-sonnet-20241022", "qwen/qwen-2.5-coder-32b"],
    "vision_analyst": ["claude-3-5-sonnet-20241022", "gpt-4-vision-preview"],
    "documentation_writer": ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"],
    "reasoning_engine": ["claude-3-opus-20240229", "claude-3-5-sonnet-20241022"],
    "fast_responder": ["claude-3-haiku-20240307", "qwen/qwen-2.5-3b"],
    "general_assistant": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"]
  },
  "model_providers": {
    "claude-3-5-sonnet-20241022": "anthropic",
    "claude-3-opus-20240229": "anthropic",
    "claude-3-5-haiku-20241022": "anthropic",
    "claude-3-haiku-20240307": "anthropic",
    "gpt-4": "openai",
    "qwen/qwen-2.5-coder-32b": "lm-studio"
  },
  "fallback": {
    "provider": "anthropic",
    "model": "claude-3-haiku-20240307"
  }
}
```

2. **Start InferSwitch** - it automatically routes using MLX classification:
   - "Debug this React component" → `coding_specialist` → Claude 3.5 Sonnet
   - "Analyze this screenshot" → `vision_analyst` → Claude 3.5 Sonnet
   - "Write API documentation" → `documentation_writer` → Claude 3.5 Haiku
   - "Solve this complex math problem" → `reasoning_engine` → Claude 3 Opus
   - "What's 2+2?" → `fast_responder` → Claude 3 Haiku
   - "Help me plan my day" → `general_assistant` → Claude 3.5 Sonnet

The MLX model analyzes each query against your expert descriptions and routes to the best match - no keyword patterns or hardcoded rules needed!

See the [Custom Expert Documentation](docs/custom_experts.md) for detailed configuration examples.

## Configuration

### Configuration Methods

InferSwitch can be configured through (in order of precedence):

1. **Request headers** - Per-request overrides
2. **Environment variables** - Runtime configuration
3. **Configuration file** - `inferswitch.config.json` in working directory
4. **Default values** - Built-in defaults

### Configuration File Structure

Create `inferswitch.config.json` in your working directory for advanced configuration:

```json
{
  "backends": {
    "anthropic": { "api_key": "sk-ant-...", "timeout": 300 },
    "lm-studio": { "base_url": "http://127.0.0.1:1234", "timeout": 600 },
    "openai": { "api_key": "sk-...", "base_url": "https://api.openai.com/v1" },
    "openrouter": { "api_key": "sk-or-...", "base_url": "https://openrouter.ai/api/v1" }
  },
  "model_providers": {
    "claude-3-haiku-20240307": "lm-studio",
    "gpt-3.5-turbo": "openai"
  },
  "model_overrides": {
    "claude-3-5-sonnet-20241022": "claude-3-haiku-20240307"
  },
  "expert_definitions": {
    "coding_expert": "Programming specialist for development and debugging tasks",
    "data_analyst": "Expert in data analysis, visualization, and insights"
  },
  "expert_models": {
    "coding_expert": ["claude-3-5-sonnet-20241022"],
    "data_analyst": ["claude-3-opus-20240229"]
  },
  "fallback": {
    "provider": "anthropic",
    "model": "claude-3-haiku-20240307"
  }
}
```

### Environment Variables

| Variable                             | Description                                 | Default                        |
| ------------------------------------ | ------------------------------------------- | ------------------------------ |
| `INFERSWITCH_BACKEND`                | Force all requests to specific backend      | `anthropic`                    |
| `INFERSWITCH_FORCE_EXPERT_ROUTING`   | Force expert-based routing for all requests | `false`                        |
| `INFERSWITCH_MLX_MODEL`              | MLX model for expert classification         | `jedisct1/arch-router-1.5b`    |
| `ANTHROPIC_API_KEY`                  | Anthropic API key                           | Required for Anthropic         |
| `OPENAI_API_KEY`                     | OpenAI API key                              | Required for OpenAI            |
| `OPENROUTER_API_KEY`                 | OpenRouter API key                          | Required for OpenRouter        |
| `LM_STUDIO_BASE_URL`                 | LM-Studio server URL                        | `http://127.0.0.1:1234`        |
| `OPENROUTER_BASE_URL`                | OpenRouter server URL                       | `https://openrouter.ai/api/v1` |
| `INFERSWITCH_MODEL_OVERRIDE`         | Model override mappings                     | None                           |
| `INFERSWITCH_DEFAULT_MODEL`          | Override all models                         | None                           |
| `CACHE_ENABLED`                      | Enable response caching                     | `true`                         |
| `CACHE_MAX_SIZE`                     | Maximum cache entries                       | `1000`                         |
| `CACHE_TTL_SECONDS`                  | Cache time-to-live                          | `3600`                         |
| `LOG_LEVEL`                          | Logging verbosity                           | `INFO`                         |
| `PROXY_MODE`                         | Enable proxy mode                           | `true`                         |
| `INFERSWITCH_MODEL_DISABLE_DURATION` | Seconds to disable failed models            | `300`                          |

## Core Concepts

### Backend Priority

InferSwitch selects backends using this priority order:

1. **Explicit Header** - `x-backend: lm-studio` in request
2. **Environment Override** - `INFERSWITCH_BACKEND=lm-studio`
3. **Expert Routing** - Based on custom expert classification (if configured)
4. **Expertise Routing** - Based on predefined expertise areas (legacy)
5. **Difficulty Routing** - Based on query complexity (legacy)
6. **Model Mapping** - Direct model → backend mapping
7. **Fallback** - Configured fallback or default to anthropic

### Custom Expert-Based Routing

InferSwitch uses MLX models to intelligently classify queries and route them to your custom-defined experts:

- **Capability-Based Experts**: Define experts based on actual AI model capabilities
  - Examples: "coding_specialist", "vision_analyst", "reasoning_engine", "fast_responder"
  - MLX analyzes queries against model capabilities to find the best match

- **No Hardcoded Patterns**: Pure AI-based classification without keyword matching
  - Query: "Debug this memory leak in Python" → Expert: "coding_specialist"
  - Query: "Analyze this screenshot" → Expert: "vision_analyst"
  - Query: "Solve this complex equation" → Expert: "reasoning_engine"

- **Optimized Model Assignment**: Route each expert to models with matching capabilities
  - Coding tasks → Code-optimized models (Sonnet, Qwen Coder, OpenHands)
  - Vision tasks → Multimodal models (Claude Sonnet, GPT-4 Vision)
  - Simple tasks → Fast, lightweight models (Haiku, small local models)

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
  "expert_models": {
    "ai_researcher": ["claude-opus-4-20250514", "claude-3-opus-20240229", "claude-3-5-sonnet-20241022"],
    "code_developer": ["claude-3-5-sonnet-20241022", "qwen/qwen-2.5-coder-32b", "claude-3-haiku-20240307"]
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

2. **Use expert routing** to minimize costs:
   - Configure local models for simple queries via general assistant experts
   - Reserve expensive models for specialized expert tasks

3. **Monitor performance**:
```bash
# Check cache hit rate
curl http://localhost:1235/cache/stats

# Enable debug logging
LOG_LEVEL=DEBUG uv run python main.py
```

### Performance Monitoring

InferSwitch provides built-in monitoring and performance metrics. Example cache stats output:
```json
{
  "enabled": true,
  "size": 150,
  "max_size": 1000,
  "hit_rate": 0.67,
  "total_requests": 450,
  "cache_hits": 302,
  "cache_misses": 148,
  "ttl_seconds": 3600
}
```

### Cost Optimization

1. **Use expert routing** to route simple queries to cheaper models
2. **Enable caching** to avoid repeated API calls
3. **Configure model fallbacks** to use cheaper models when expensive ones fail
4. **Monitor cache hit rates** to ensure effective caching

```bash
# Example cost-optimized configuration
cat > inferswitch.config.json << 'EOF'
{
  "force_expert_routing": true,
  "expert_definitions": {
    "simple_qa": "Quick questions and simple explanations",
    "complex_tasks": "Complex reasoning, detailed analysis, and difficult problems"
  },
  "expert_models": {
    "simple_qa": ["claude-3-haiku-20240307"],
    "complex_tasks": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"]
  },
  "model_providers": {
    "claude-3-haiku-20240307": "anthropic",
    "claude-3-5-sonnet-20241022": "anthropic"
  }
}
EOF

# Enable caching for cost savings
CACHE_ENABLED=true CACHE_TTL_SECONDS=7200 uv run python main.py
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Server Won't Start

**Problem**: InferSwitch fails to start with port binding errors.

**Solution**:
```bash
# Check if port 1235 is already in use
lsof -i :1235

# Kill existing process or use a different port
INFERSWITCH_PORT=1236 uv run python main.py
```

#### 2. Authentication Errors

**Problem**: `401 Unauthorized` or `403 Forbidden` errors.

**Solutions**:
```bash
# Check if API key is set correctly
echo $ANTHROPIC_API_KEY

# Verify API key format (should start with 'sk-ant-')
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# For OAuth users, check token status
curl http://localhost:1235/oauth/status
```

#### 3. Model Not Found Errors

**Problem**: Model routing fails with "model not found" errors.

**Solution**:
```bash
# Check backend status
curl http://localhost:1235/backends/status

# Verify model mappings in config
cat inferswitch.config.json | jq '.model_providers'

# Use explicit backend header
curl -H "x-backend: anthropic" http://localhost:1235/v1/messages
```

#### 4. MLX Installation Issues (Apple Silicon)

**Problem**: MLX models fail to load on Apple Silicon.

**Solution**:
```bash
# Install MLX explicitly
pip install mlx mlx-lm

# Check MLX model status
python -c "from inferswitch.mlx_model import mlx_model_manager; print(mlx_model_manager.is_available())"

# Use fallback routing if MLX fails
INFERSWITCH_FORCE_DIFFICULTY_ROUTING=true uv run python main.py
```

#### 5. LM-Studio Connection Issues

**Problem**: Cannot connect to LM-Studio backend.

**Solution**:
```bash
# Check LM-Studio is running and accessible
curl http://127.0.0.1:1234/v1/models

# Verify LM-Studio base URL
LM_STUDIO_BASE_URL=http://127.0.0.1:1234 uv run python main.py

# Check if model is loaded in LM-Studio
curl http://127.0.0.1:1234/v1/chat/completions \
  -d '{"model": "llama", "messages": [{"role": "user", "content": "test"}]}'
```

#### 6. High Memory Usage

**Problem**: InferSwitch consumes excessive memory.

**Solution**:
```bash
# Reduce cache size
CACHE_MAX_SIZE=100 uv run python main.py

# Disable caching entirely
CACHE_ENABLED=false uv run python main.py

# Use lightweight MLX model
INFERSWITCH_MLX_MODEL="mlx-community/Qwen2.5-3B-8bit" uv run python main.py
```

#### 7. Slow Response Times

**Problem**: API responses are slower than expected.

**Solution**:
```bash
# Enable cache for repeated queries
CACHE_ENABLED=true uv run python main.py

# Use faster models for simple queries
# Configure expert routing with fast models for basic tasks

# Check backend response times
curl -w "@curl-format.txt" http://localhost:1235/backends/status
```

#### 8. Configuration Not Loading

**Problem**: Configuration file changes aren't applied.

**Solution**:
```bash
# Verify config file location
ls -la inferswitch.config.json

# Check config file syntax
python -m json.tool inferswitch.config.json

# Restart server after config changes
# Config is loaded at startup, not runtime
```

### Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Enable debug logging with `LOG_LEVEL=DEBUG`
2. **Verify your setup**: Use the example configurations in `examples/`
3. **Test individual components**: Run specific tests from the `tests/` directory
4. **Report bugs**: Create an issue on the GitHub repository

### Debug Commands

```bash
# Check server health
curl http://localhost:1235/backends/status

# View cache statistics
curl http://localhost:1235/cache/stats

# Test specific backend
curl -H "x-backend: anthropic" -H "Content-Type: application/json" \
  http://localhost:1235/v1/messages -d '{"model": "claude-3-haiku", "messages": [{"role": "user", "content": "test"}], "max_tokens": 10}'

# Check MLX model loading
python -c "from inferswitch.mlx_model import mlx_model_manager; mlx_model_manager.load_model()"

# Validate configuration
python -c "from inferswitch.config import load_config; print(load_config())"
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
uv run python tests/test_custom_experts.py   # Custom expert routing
uv run python tests/test_difficulty_routing.py # Legacy routing logic
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

InferSwitch uses a modular, extensible architecture built on FastAPI:

```
┌─────────────┐     ┌──────────────────────────────────────┐     ┌─────────────┐
│   Client    │────▶│            InferSwitch               │────▶│  Anthropic  │
└─────────────┘     │                                      │     └─────────────┘
                    │  ┌─────────────┐  ┌─────────────┐    │     ┌─────────────┐
                    │  │ FastAPI     │  │ MLX Expert  │    │────▶│   OpenAI    │
                    │  │ Endpoints   │  │ Classifier  │    │     └─────────────┘
                    │  └─────────────┘  └─────────────┘    │     ┌─────────────┐
                    │  ┌─────────────┐  ┌─────────────┐    │────▶│  LM-Studio  │
                    │  │ Smart Cache │  │ Backend     │    │     └─────────────┘
                    │  │ & Logging   │  │ Registry    │    │     ┌─────────────┐
                    │  └─────────────┘  └─────────────┘    │────▶│ OpenRouter  │
                    │                                      │     └─────────────┘
                    └──────────────────────────────────────┘     ┌─────────────┐
                                                                 │   Custom    │
                                                                 │  Backends   │
                                                                 └─────────────┘
```

### Key Components

1. **Request Processing**:
   - FastAPI handles HTTP requests and validation
   - Request logging and response caching
   - Streaming and non-streaming response support

2. **Intelligent Routing**:
   - MLX-powered expert classification
   - Custom expert definitions with natural language
   - Fallback chains for high availability

3. **Backend Abstraction**:
   - Unified interface for all LLM providers
   - Automatic format conversion (OpenAI ↔ Anthropic)
   - Health monitoring and error handling

4. **Configuration Management**:
   - JSON-based configuration files
   - Environment variable overrides
   - Runtime configuration validation

### Request Flow

1. **Client Request**: HTTP request arrives at FastAPI endpoint
2. **Authentication**: API key validation or OAuth token verification
3. **Cache Check**: Look for cached responses (ignoring timestamps)
4. **Expert Classification**: MLX model analyzes query content
5. **Backend Selection**: Route to appropriate backend based on expert/model mapping
6. **Format Conversion**: Convert between API formats if needed
7. **Response Processing**: Handle streaming/non-streaming responses
8. **Caching & Logging**: Store response and log request details

### Extensibility

- **Add New Backends**: Implement `BaseBackend` interface
- **Custom Experts**: Define domain-specific routing logic
- **Plugin System**: Extend functionality through the backend registry
- **Monitoring**: Built-in metrics and health checks

## Use Cases

### Development & Testing
- **Local Development**: Use local models via LM-Studio for development
- **A/B Testing**: Compare different models for the same queries
- **Cost Control**: Route expensive queries to cheaper models during development

### Production Deployment
- **High Availability**: Automatic failover between multiple providers
- **Cost Optimization**: Route simple queries to cheaper models
- **Performance**: Cache responses to reduce latency and costs

### Specialized Applications
- **Domain Experts**: Create expert models for specific use cases (legal, medical, technical)
- **Multi-Modal**: Route vision tasks to vision-capable models
- **Compliance**: Use specific models for regulatory requirements

## Community

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/jedisct1/inferswitch/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/jedisct1/inferswitch/discussions)
- 📚 **Documentation**: [docs/](docs/) directory
- 🤝 **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- MLX support for Apple Silicon
- Compatible with Anthropic, OpenAI, and LM-Studio APIs
