"""
Messages endpoint handler with multi-backend support.
"""

import json
from typing import Optional

from fastapi import HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse

from ..config import PROXY_MODE, CACHE_ENABLED
from ..backends import backend_registry, BackendError
from ..models import MessagesRequest, MessagesResponse, Usage
from ..utils import log_request, estimate_tokens, generate_sse_events
from ..utils.cache import get_cache
from ..utils.chat_template import convert_to_chat_template
from ..mlx_model import mlx_model_manager
from ..expertise_classifier import expert_classifier


async def create_message_v2(
    request: MessagesRequest,
    x_api_key: Optional[str] = Header(None),
    anthropic_version: Optional[str] = Header(None),
    anthropic_beta: Optional[str] = Header(None),
    x_backend: Optional[str] = Header(None),  # New header for backend selection
):
    """Handle POST /v1/messages requests with multi-backend support."""
    import logging

    logger = logging.getLogger(__name__)

    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing x-api-key header")

    if not anthropic_version:
        raise HTTPException(status_code=400, detail="Missing anthropic-version header")

    request_dict = request.model_dump(exclude_none=True)

    # Check cache first if enabled - avoid MLX computation for cached responses
    cache = get_cache() if CACHE_ENABLED else None
    cached_response = cache.get(request_dict) if cache else None

    # Log cache check for debugging
    if cache:
        logger.debug(
            f"Cache check for model={request.model}, messages={len(request_dict.get('messages', []))}"
        )

    if cached_response is not None:
        logger.info(
            "Cache HIT - Skipping MLX difficulty computation - Returning cached response"
        )

        # Log the request without difficulty rating for cache hits
        log_request("/v1/messages", request_dict, None)

        # Return cached response as-is - router messages already included when needed

        if request.stream:
            # Convert cached response to streaming format
            async def stream_cached_response():
                async for chunk in generate_sse_from_cached_response(
                    cached_response, None
                ):
                    yield chunk

            return StreamingResponse(
                stream_cached_response(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            # Return cached response as-is
            return JSONResponse(content=cached_response)

    # Cache miss - compute routing classification with MLX
    logger.debug("Cache MISS - Computing routing classification with MLX")

    # Get backend router to check routing mode
    router = backend_registry.get_router()
    routing_mode = router.routing_mode

    difficulty_rating = None
    expertise_area = None
    expert_name = None

    if routing_mode == "expert":
        # Check if all expert models are the same - if so, skip MLX classifier
        if router.all_expert_models_are_same():
            logger.debug("All experts use the same model - skipping expert classifier")
            # Use the first expert as default
            expert_definitions = router.expert_models
            if expert_definitions:
                expert_name = list(expert_definitions.keys())[0]
        else:
            # Convert request to chat template format for expert classification
            chat_messages = convert_to_chat_template(request_dict)

            # Classify which expert should handle the query
            expert_name = expert_classifier.classify_expert(chat_messages)
    elif routing_mode == "expertise":
        # Check if all expertise models are the same - if so, skip MLX classifier
        if router.all_expertise_models_are_same():
            logger.debug(
                "All expertise areas use the same model - skipping expertise classifier"
            )
            expertise_area = "general"  # Default area when all models are the same
        else:
            # Convert request to chat template format for expertise classification
            chat_messages = convert_to_chat_template(request_dict)

            # Classify the expertise area of the query (legacy)
            from ..expertise_classifier import ExpertiseClassifier

            legacy_classifier = ExpertiseClassifier()
            expertise_area = legacy_classifier.classify_expertise(chat_messages)
    elif routing_mode == "difficulty":
        # Check if all difficulty models are the same - if so, skip MLX classifier
        if router.all_difficulty_models_are_same():
            logger.debug(
                "All difficulty levels use the same model - skipping MLX classifier"
            )
            difficulty_rating = 0.0  # Default rating when all models are the same
        else:
            # Convert request to chat template format for difficulty rating
            chat_messages = convert_to_chat_template(request_dict)

            # Rate the difficulty of the query
            difficulty_rating = mlx_model_manager.rate_query_difficulty(chat_messages)
    else:
        logger.debug("Normal routing mode - no classification needed")

    # Log the request with difficulty rating
    log_request("/v1/messages", request_dict, difficulty_rating)

    # Log query preview if available
    if request_dict.get("messages"):
        last_msg = request_dict["messages"][-1]
        if isinstance(last_msg.get("content"), str):
            query_preview = last_msg["content"][:50].replace("\n", " ")
            logger.debug(f"Query: {query_preview}...")

    try:
        # Get the actual model to use (may be overridden)
        actual_model = router.get_overridden_model(request.model)
        if actual_model != request.model:
            logger.debug(f"Model override: {request.model} -> {actual_model}")

        # Select backend based on model or explicit header
        backend = router.select_backend(
            model=actual_model,
            explicit_backend=x_backend,
            difficulty_rating=difficulty_rating,
            expertise_area=expertise_area,
            expert_name=expert_name,
        )

        # Get the effective model that will be used
        effective_model = actual_model
        if (
            hasattr(backend, "_expert_selected_model")
            and backend._expert_selected_model
        ):
            effective_model = backend._expert_selected_model
        elif (
            hasattr(backend, "_expertise_selected_model")
            and backend._expertise_selected_model
        ):
            effective_model = backend._expertise_selected_model
        elif (
            hasattr(backend, "_difficulty_selected_model")
            and backend._difficulty_selected_model
        ):
            effective_model = backend._difficulty_selected_model
        elif hasattr(backend, "_fallback_model") and backend._fallback_model:
            effective_model = backend._fallback_model

        # Single line routing summary
        if expert_name:
            logger.info(
                f"Expert: {expert_name} - Routing to {backend.name} {effective_model}"
            )
        elif expertise_area:
            logger.info(
                f"Expertise: {expertise_area} - Routing to {backend.name} {effective_model}"
            )
        elif difficulty_rating is not None:
            logger.info(
                f"Difficulty: {difficulty_rating} - Routing to {backend.name} {effective_model}"
            )
        else:
            logger.info(f"Normal routing to {backend.name} {effective_model}")

        # Extract messages and system from request
        messages = request_dict.get("messages", [])
        system = request_dict.get("system")

        # Non-proxy mode: return OK response
        if not PROXY_MODE:
            return _create_ok_response(request)

        # Proxy mode: forward to backend
        if request.stream:
            # Streaming response
            async def stream_response():
                try:
                    # Remove fields that are explicitly passed to avoid duplicates
                    extra_kwargs = {
                        k: v
                        for k, v in request_dict.items()
                        if k
                        not in [
                            "messages",
                            "model",
                            "system",
                            "max_tokens",
                            "temperature",
                        ]
                    }

                    has_error = False
                    async for event in backend.create_message_stream(
                        messages=messages,
                        model=effective_model,
                        system=system,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        x_api_key=x_api_key,
                        anthropic_version=anthropic_version,
                        anthropic_beta=anthropic_beta,
                        difficulty_rating=difficulty_rating,
                        **extra_kwargs,
                    ):
                        event_type = event.get("type", "")

                        # Forward all events without router injection to prevent duplication
                        # Router messages are already added by the backend when needed
                        yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n".encode(
                            "utf-8"
                        )

                    # Mark success only if we completed without error
                    if not has_error:
                        router.mark_model_success(effective_model)

                except BackendError as e:
                    has_error = True
                    # Check if this is an error that should disable the model
                    should_mark_failed = False

                    # Check for 400 errors that might indicate unsupported model
                    if e.status_code == 400:
                        error_msg = str(e).lower()
                        unsupported_patterns = [
                            "model",
                            "not supported",
                            "not found",
                            "invalid model",
                            "unknown model",
                            "does not exist",
                        ]
                        if any(
                            pattern in error_msg for pattern in unsupported_patterns
                        ):
                            should_mark_failed = True
                            logger.warning(
                                f"Model {effective_model} appears unsupported by {backend.name}: {e}"
                            )

                    # Also check for rate limit and credit errors
                    elif (
                        e.status_code in [429, 402]
                        or "credit" in str(e).lower()
                        or "rate" in str(e).lower()
                    ):
                        should_mark_failed = True
                        logger.warning(
                            f"Model {effective_model} has rate/credit issues: {e}"
                        )

                    if should_mark_failed:
                        router.mark_model_failure(effective_model)
                        logger.warning(
                            f"Marked model {effective_model} as failed due to: {e}"
                        )

                    # Send error as SSE event
                    error_event = {"type": "error", "error": e.to_dict()}
                    yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode(
                        "utf-8"
                    )

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            # Non-streaming response
            # Remove fields that are explicitly passed to avoid duplicates
            extra_kwargs = {
                k: v
                for k, v in request_dict.items()
                if k not in ["messages", "model", "system", "max_tokens", "temperature"]
            }

            response = await backend.create_message(
                messages=messages,
                model=effective_model,
                system=system,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                x_api_key=x_api_key,
                anthropic_version=anthropic_version,
                anthropic_beta=anthropic_beta,
                difficulty_rating=difficulty_rating,
                **extra_kwargs,
            )

            # Convert to API response format - use response content as-is
            # Router messages are already added by the backend when needed
            all_content = response.content

            response_dict = {
                "id": response.raw_response.get("id", "msg_unknown"),
                "type": "message",
                "role": "assistant",
                "content": all_content,
                "model": response.model,
                "stop_reason": response.stop_reason,
                "stop_sequence": response.raw_response.get("stop_sequence"),
                "usage": response.usage,
            }

            # Cache the response
            if cache:
                cache.set(request_dict, response_dict)
                logger.info(f"Cached response for model={request.model}")

            # Mark the model as successful
            router.mark_model_success(effective_model)

            return JSONResponse(content=response_dict)

    except BackendError as e:
        # Check if this is an error that should disable the model
        should_mark_failed = False

        # Check for 400 errors that might indicate unsupported model
        if e.status_code == 400:
            error_msg = str(e).lower()
            # Look for common patterns that indicate model not supported
            unsupported_patterns = [
                "model",
                "not supported",
                "not found",
                "invalid model",
                "unknown model",
                "does not exist",
            ]
            if any(pattern in error_msg for pattern in unsupported_patterns):
                should_mark_failed = True
                logger.warning(
                    f"Model {effective_model} appears unsupported by {backend.name}: {e}"
                )

        # Also check for rate limit and credit errors
        elif (
            e.status_code in [429, 402]
            or "credit" in str(e).lower()
            or "rate" in str(e).lower()
        ):
            should_mark_failed = True
            logger.warning(f"Model {effective_model} has rate/credit issues: {e}")

        # Mark model as failed if applicable
        if should_mark_failed:
            router.mark_model_failure(effective_model)
            logger.warning(f"Marked model {effective_model} as failed due to: {e}")

        raise HTTPException(status_code=e.status_code or 500, detail=e.to_dict())

    except Exception as e:
        # For other errors, also consider marking the model as failed
        if "credit" in str(e).lower() or "insufficient" in str(e).lower():
            router.mark_model_failure(effective_model)
            logger.warning(f"Marked model {effective_model} as failed due to: {e}")

        raise HTTPException(
            status_code=500,
            detail={"error": {"type": "internal_error", "message": str(e)}},
        )


def _create_ok_response(request: MessagesRequest):
    """Create a simple OK response for non-proxy mode."""
    # Estimate tokens
    input_tokens = estimate_tokens(str(request.model_dump()))
    output_tokens = 10  # Fixed small output

    response = MessagesResponse(
        id="msg_ok_response",
        type="message",
        role="assistant",
        content=[{"type": "text", "text": "OK"}],
        model=request.model,
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )

    if request.stream:

        async def generate():
            async for event in generate_sse_events(
                message_id=response.id,
                content=response.content[0].text if response.content else "",
                model=response.model,
                input_tokens=response.usage.input_tokens,
            ):
                yield event

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        return response


async def generate_sse_from_cached_response(
    cached_response: dict, router_text: str = None
):
    """Generate SSE events from a cached response."""
    content_blocks = cached_response.get("content", [])
    message_id = cached_response.get("id", "msg_cached")
    model = cached_response.get("model", "unknown")
    usage = cached_response.get("usage", {})
    stop_reason = cached_response.get("stop_reason", "end_turn")

    # Send message_start event
    message_start = {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": 0},
        },
    }
    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n".encode("utf-8")

    # Send router message first if provided
    if router_text:
        # Send router message as content block 0
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n".encode(
            "utf-8"
        )
        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': router_text}})}\n\n".encode(
            "utf-8"
        )
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n".encode(
            "utf-8"
        )

    # Send content blocks (offset index by 1 if router message was sent)
    for idx, block in enumerate(content_blocks):
        if block.get("type") == "text":
            text = block.get("text", "")
            actual_idx = idx + 1 if router_text else idx

            # Send content_block_start
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': actual_idx, 'content_block': {'type': 'text', 'text': ''}})}\n\n".encode(
                "utf-8"
            )

            # Send content_block_delta
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': actual_idx, 'delta': {'type': 'text_delta', 'text': text}})}\n\n".encode(
                "utf-8"
            )

            # Send content_block_stop
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': actual_idx})}\n\n".encode(
                "utf-8"
            )

    # Send message_delta
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason}, 'usage': {'output_tokens': usage.get('output_tokens', 0)}})}\n\n".encode(
        "utf-8"
    )

    # Send message_stop
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n".encode(
        "utf-8"
    )
