"""
InferSwitch main application module.
"""

from fastapi import FastAPI, Request, Depends, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import uvicorn
import logging

from .config import DEFAULT_HOST, DEFAULT_PORT, CACHE_ENABLED
from .client import AnthropicClient
from .api import count_tokens, get_chat_template, create_message_v2
from .mlx_model import mlx_model_manager
from .expertise_classifier import expert_classifier
from .backends import backend_registry, AnthropicBackend, OpenAIBackend
from .utils.oauth import oauth_manager
from .utils import start_proxy_server, stop_proxy_server

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="InferSwitch",
    description="Anthropic API proxy with logging and chat template support",
    version="0.1.0",
)

# Global client instance (for backward compatibility)
anthropic_client = None


async def get_anthropic_client():
    """Dependency to get the Anthropic client (for backward compatibility)."""
    return anthropic_client


async def get_backend_registry():
    """Dependency to get the backend registry."""
    return backend_registry


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    global anthropic_client

    # Check if OAuth is configured and handle authentication
    if oauth_manager.is_oauth_configured():
        # Check if we have a valid token
        token = await oauth_manager.get_valid_token()
        if not token:
            # No valid token, run interactive OAuth flow
            logger.info(
                "OAuth configured but no valid token found. "
                "Starting interactive authentication..."
            )
            success = await oauth_manager.interactive_oauth_flow()
            if not success:
                logger.warning(
                    "OAuth authentication failed. "
                    "Continuing with API key authentication."
                )
        else:
            logger.info("Using existing OAuth token for Anthropic authentication")

    # Initialize backend registry
    backend_classes = {
        "anthropic": AnthropicBackend,
        "lm-studio": OpenAIBackend,
        "openai": OpenAIBackend,
        "openrouter": OpenAIBackend,
    }
    await backend_registry.initialize(backend_classes)

    # Create legacy client for backward compatibility
    anthropic_client = AnthropicClient()

    # Load MLX model (optional - don't fail if it doesn't work)
    try:
        from .backends.config import BackendConfigManager

        mlx_model = BackendConfigManager.get_mlx_model()
        success, message = mlx_model_manager.load_model(mlx_model)
        logger.info(f"MLX model loading: {message}")
        if not success:
            logger.warning(
                "MLX model failed to load. Difficulty rating will be disabled."
            )
    except Exception as e:
        logger.warning(
            f"MLX model loading error: {e}. Difficulty rating will be disabled."
        )

    # Load expert classifier (optional - don't fail if it doesn't work)
    try:
        from .backends.config import BackendConfigManager

        # Load expert definitions from config
        expert_definitions = BackendConfigManager.get_expert_definitions()
        if expert_definitions:
            expert_classifier.set_expert_definitions(expert_definitions)
            logger.info(
                f"Loaded {len(expert_definitions)} expert definitions: "
                f"{list(expert_definitions.keys())}"
            )

            # Load MLX model for classification
            mlx_model = BackendConfigManager.get_mlx_model()
            success, message = expert_classifier.load_model(mlx_model)
            logger.info(f"Expert classifier loading: {message}")
            if not success:
                logger.warning(
                    "Expert classifier MLX model failed to load. "
                    "Expert routing will be disabled."
                )
        else:
            logger.info(
                "No expert definitions found in config. Expert routing disabled."
            )

    except Exception as e:
        logger.warning(
            f"Expert classifier loading error: {e}. Expert routing will be disabled."
        )

    # Start HTTP/HTTPS proxy server if enabled
    from .backends.config import BackendConfigManager
    proxy_config = BackendConfigManager.get_proxy_config()

    if proxy_config["enabled"]:
        try:
            await start_proxy_server(
                host=proxy_config["host"],
                port=proxy_config["port"],
                timeout=proxy_config["timeout"]
            )
            logger.info(
                f"HTTP/HTTPS proxy server started on "
                f"{proxy_config['host']}:{proxy_config['port']}"
            )
        except Exception as e:
            logger.warning(f"Failed to start proxy server: {e}")
    else:
        logger.info("HTTP/HTTPS proxy server disabled")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    # Stop proxy server
    from .backends.config import BackendConfigManager
    proxy_config = BackendConfigManager.get_proxy_config()

    if proxy_config["enabled"]:
        await stop_proxy_server()
        logger.info("HTTP/HTTPS proxy server stopped")

    # Close all backends
    await backend_registry.close_all()

    # Close legacy client
    if anthropic_client:
        await anthropic_client.close()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed information."""
    return JSONResponse(
        status_code=422, content={"detail": exc.errors(), "body": str(exc.body)}
    )


# API Routes
@app.post("/v1/messages")
async def messages_endpoint(request: Request, registry=Depends(get_backend_registry)):
    """Main messages endpoint with multi-backend support."""
    # Import here to avoid circular imports
    from .models import MessagesRequest

    # Parse request body
    body = await request.json()
    messages_request = MessagesRequest(**body)

    # Get headers
    headers = dict(request.headers)

    # Call the new handler with backend support
    return await create_message_v2(
        request=messages_request,
        x_api_key=headers.get("x-api-key"),
        anthropic_version=headers.get("anthropic-version"),
        anthropic_beta=headers.get("anthropic-beta"),
        x_backend=headers.get("x-backend"),  # New header for backend selection
    )


@app.post("/v1/messages/count_tokens")
@app.post("/v1/messages/count-tokens")  # Also support hyphenated version
async def count_tokens_endpoint(request: Request):
    """Token counting endpoint."""
    from .models import CountTokensRequest

    body = await request.json()
    tokens_request = CountTokensRequest(**body)

    headers = dict(request.headers)

    return await count_tokens(
        request=tokens_request,
        x_api_key=headers.get("x-api-key"),
        anthropic_version=headers.get("anthropic-version"),
        anthropic_beta=headers.get("anthropic-beta"),
    )


@app.post("/v1/chat/completions")
async def chat_completions_endpoint(
    request: Request, registry=Depends(get_backend_registry)
):
    """OpenAI-compatible chat completions endpoint."""
    # This endpoint should automatically use the LM-Studio/OpenAI backend
    # when requests come in OpenAI format

    # Parse request body
    body = await request.json()

    # Convert OpenAI format to Anthropic format for our internal processing
    # Extract messages and other parameters
    messages = body.get("messages", [])
    model = body.get("model", "gpt-3.5-turbo")

    # Convert OpenAI messages to Anthropic format
    anthropic_messages = []
    system_content = None

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            system_content = content
        elif role in ["user", "assistant"]:
            # Convert to Anthropic format with content blocks
            anthropic_messages.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": content}]
                    if isinstance(content, str)
                    else content,
                }
            )

    # Build Anthropic-style request
    from .models import MessagesRequest

    anthropic_request = MessagesRequest(
        model=model,
        messages=anthropic_messages,
        system=system_content,
        max_tokens=body.get("max_tokens", 1024),
        temperature=body.get("temperature"),
        stream=body.get("stream", False),
    )

    # Calculate difficulty rating and log the request
    from .utils import log_request
    from .utils.chat_template import convert_to_chat_template
    from .mlx_model import mlx_model_manager

    # Convert to chat template format for difficulty rating
    try:
        request_dict = anthropic_request.model_dump(exclude_none=True)
        chat_messages = convert_to_chat_template(request_dict)

        # Try MLX model first if available
        if mlx_model_manager.is_loaded():
            difficulty_rating = mlx_model_manager.rate_query_difficulty(chat_messages)
        else:
            # Fallback to simple heuristic rating
            from .utils.simple_difficulty import rate_query_difficulty_simple

            difficulty_rating = rate_query_difficulty_simple(chat_messages)

    except Exception:
        difficulty_rating = None

    # Log the request with difficulty rating
    log_request("/v1/chat/completions", body, difficulty_rating)

    # Get headers
    headers = dict(request.headers)

    # For OpenAI format, extract API key from Authorization header
    auth_header = headers.get("authorization", "")
    api_key = None
    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:]

    # Call the handler with explicit backend selection
    # Since this is OpenAI format, prefer LM-Studio/OpenAI backend
    x_backend = headers.get("x-backend")
    if (
        not x_backend
        and "anthropic" not in model.lower()
        and "claude" not in model.lower()
    ):
        x_backend = "lm-studio"  # Default to LM-Studio for non-Anthropic models

    return await create_message_v2(
        request=anthropic_request,
        x_api_key=api_key or headers.get("x-api-key"),
        anthropic_version=headers.get("anthropic-version", "2023-06-01"),
        anthropic_beta=headers.get("anthropic-beta"),
        x_backend=x_backend,
    )


@app.post("/v1/messages/chat-template")
async def chat_template_endpoint(request: Request):
    """Chat template conversion endpoint."""
    from .models import MessagesRequest

    body = await request.json()
    messages_request = MessagesRequest(**body)

    headers = dict(request.headers)

    return await get_chat_template(
        request=messages_request,
        x_api_key=headers.get("x-api-key"),
        anthropic_version=headers.get("anthropic-version"),
    )


@app.get("/mlx/status")
async def mlx_status():
    """Get MLX model status."""
    info = mlx_model_manager.get_model_info()
    # Test difficulty rating with a simple query
    if mlx_model_manager.is_loaded():
        test_messages = [{"role": "user", "content": "What is 2+2?"}]
        test_rating = mlx_model_manager.rate_query_difficulty(test_messages)
        info["test_rating"] = test_rating
    return info


@app.get("/backends/status")
async def backends_status():
    """Get status of all configured backends."""
    health_results = await backend_registry.health_check_all()
    return {
        "backends": backend_registry.list_backends(),
        "health": health_results,
        "models": backend_registry.get_models_summary(),
        "capabilities": backend_registry.get_capabilities_summary(),
    }


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    if not CACHE_ENABLED:
        return {"enabled": False, "message": "Cache is disabled"}

    from .utils.cache import get_cache

    cache = get_cache()
    return {"enabled": True, **cache.get_stats()}


@app.post("/cache/clear")
async def cache_clear():
    """Clear the cache."""
    if not CACHE_ENABLED:
        return {"enabled": False, "message": "Cache is disabled"}

    from .utils.cache import get_cache

    cache = get_cache()
    cache.clear()
    return {"message": "Cache cleared", "enabled": True}


# OAuth endpoints
@app.get("/oauth/authorize")
async def oauth_authorize():
    """Initiate OAuth authorization flow."""
    auth_url, state, code_verifier = oauth_manager.get_authorization_url()

    # In production, you'd store state and code_verifier in a session
    # For simplicity, we'll return them to the client
    return {
        "auth_url": auth_url,
        "state": state,
        "code_verifier": code_verifier,
        "instructions": (
            "Visit the auth_url and save the state and code_verifier for the callback"
        ),
    }


@app.get("/oauth/callback")
async def oauth_callback(
    code: str = Query(..., description="Authorization code from Anthropic"),
    state: str = Query(..., description="State parameter for CSRF protection"),
    code_verifier: str = Query(..., description="PKCE code verifier"),
):
    """Handle OAuth callback and exchange code for token."""
    try:
        token_info = await oauth_manager.exchange_code_for_token(code, code_verifier)
        return {
            "message": "OAuth authentication successful",
            "token_type": token_info.token_type,
            "expires_in": token_info.expires_in_seconds,
            "has_refresh_token": token_info.refresh_token is not None,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/oauth/status")
async def oauth_status():
    """Check OAuth token status."""
    token_info = oauth_manager.load_token()
    if not token_info:
        return {"authenticated": False, "message": "No OAuth token found"}

    return {
        "authenticated": True,
        "expired": token_info.is_expired,
        "expires_in_seconds": token_info.expires_in_seconds,
        "has_refresh_token": token_info.refresh_token is not None,
    }


@app.post("/oauth/refresh")
async def oauth_refresh():
    """Manually refresh the OAuth token."""
    token_info = oauth_manager.load_token()
    if not token_info:
        raise HTTPException(status_code=401, detail="No OAuth token found")

    if not token_info.refresh_token:
        raise HTTPException(status_code=400, detail="No refresh token available")

    try:
        new_token_info = await oauth_manager.refresh_access_token(
            token_info.refresh_token
        )
        return {
            "message": "Token refreshed successfully",
            "expires_in_seconds": new_token_info.expires_in_seconds,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/oauth/logout")
async def oauth_logout():
    """Clear stored OAuth tokens."""
    oauth_manager.clear_tokens()
    return {"message": "OAuth tokens cleared"}


@app.get("/proxy/status")
async def proxy_status():
    """Get HTTP/HTTPS proxy server status."""
    from .backends.config import BackendConfigManager
    from .utils import get_proxy_server

    proxy_config = BackendConfigManager.get_proxy_config()

    if not proxy_config["enabled"]:
        return {"enabled": False, "message": "Proxy server is disabled"}

    proxy_server = get_proxy_server()
    if proxy_server:
        return {
            "enabled": True,
            "running": proxy_server.running,
            "host": proxy_server.host,
            "port": proxy_server.port,
            "timeout": proxy_server.timeout,
        }
    else:
        return {
            "enabled": True,
            "running": False,
            "message": "Proxy server not started"
        }


def main():
    """Run the application."""
    import logging

    # Configure logging to show INFO level messages
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()  # Console output
        ],
    )

    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT, log_level="info")


if __name__ == "__main__":
    main()
