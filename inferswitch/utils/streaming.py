"""
Server-Sent Events (SSE) streaming utilities.
"""

import json
import asyncio
from datetime import datetime
from typing import AsyncGenerator

from ..config import LOG_FILE


async def generate_sse_events(message_id: str, content: str, model: str, input_tokens: int) -> AsyncGenerator[bytes, None]:
    """Generate SSE events for a simple OK response."""
    # Send message start event
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': message_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': input_tokens, 'output_tokens': 0}}})}\n\n".encode('utf-8')
    
    # Send processing message first
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n".encode('utf-8')
    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': '<processing>This request is currently being processed by a local InferSwitch AI gateway.\n</processing>'}})}\n\n".encode('utf-8')
    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n".encode('utf-8')
    
    
    # Send actual content block
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 1, 'content_block': {'type': 'text', 'text': ''}})}\n\n".encode('utf-8')
    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 1, 'delta': {'type': 'text_delta', 'text': content}})}\n\n".encode('utf-8')
    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 1})}\n\n".encode('utf-8')
    
    # Send message delta
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 1}})}\n\n".encode('utf-8')
    
    # Send message stop
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n".encode('utf-8')


async def generate_sse_from_response(response_data: dict, log_events: bool = True) -> AsyncGenerator[bytes, None]:
    """Convert a non-streaming response to SSE format."""
    event_count = 0
    
    # Extract message details
    message_id = response_data.get('id', 'msg_unknown')
    model = response_data.get('model', 'unknown')
    content_blocks = response_data.get('content', [])
    stop_reason = response_data.get('stop_reason', 'end_turn')
    stop_sequence = response_data.get('stop_sequence')
    usage = response_data.get('usage', {})
    
    # 1. Send message_start event
    event_count += 1
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
            "usage": {
                "input_tokens": usage.get('input_tokens', 0),
                "output_tokens": 0
            }
        }
    }
    
    event = f"event: message_start\ndata: {json.dumps(message_start)}\n\n"
    
    # Log the event if requested
    if log_events:
        with open(LOG_FILE, "a") as f:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            f.write(f"\n[SSE EVENT #{event_count}] {timestamp}\n")
            f.write("Event Type: message_start\n")
            f.write(f"Data:\n{json.dumps(message_start, indent=2)[:2000]}\n")
    
    yield event.encode('utf-8')
    
    # 2. Send processing message
    # Send content_block_start for processing message
    event_count += 1
    processing_block_start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "text",
            "text": ""
        }
    }
    
    event = f"event: content_block_start\ndata: {json.dumps(processing_block_start)}\n\n"
    
    if log_events:
        with open(LOG_FILE, "a") as f:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            f.write(f"\n[SSE EVENT #{event_count}] {timestamp}\n")
            f.write("Event Type: content_block_start (processing)\n")
            f.write(f"Data:\n{json.dumps(processing_block_start, indent=2)}\n")
    
    yield event.encode('utf-8')
    
    # Send processing message content
    event_count += 1
    processing_delta = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {
            "type": "text_delta",
            "text": "<processing>This request is currently being processed by the InferSwitch AI Gateway.\n</processing>"
        }
    }
    
    event = f"event: content_block_delta\ndata: {json.dumps(processing_delta)}\n\n"
    
    if log_events:
        with open(LOG_FILE, "a") as f:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            f.write(f"\n[SSE EVENT #{event_count}] {timestamp}\n")
            f.write("Event Type: content_block_delta (processing)\n")
            f.write(f"Data:\n{json.dumps(processing_delta, indent=2)}\n")
    
    yield event.encode('utf-8')
    
    # Send content_block_stop for processing message
    event_count += 1
    processing_block_stop = {
        "type": "content_block_stop",
        "index": 0
    }
    
    event = f"event: content_block_stop\ndata: {json.dumps(processing_block_stop)}\n\n"
    
    if log_events:
        with open(LOG_FILE, "a") as f:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            f.write(f"\n[SSE EVENT #{event_count}] {timestamp}\n")
            f.write("Event Type: content_block_stop (processing)\n")
            f.write(f"Data:\n{json.dumps(processing_block_stop, indent=2)}\n")
    
    yield event.encode('utf-8')
    
    # 3. Send actual content blocks (starting from index 1 since we used 0 for processing)
    for idx, block in enumerate(content_blocks):
        if block.get('type') == 'text':
            text = block.get('text', '')
            actual_idx = idx + 1  # Offset by 1 due to processing message
            
            # Send content_block_start
            event_count += 1
            block_start = {
                "type": "content_block_start",
                "index": actual_idx,
                "content_block": {
                    "type": "text",
                    "text": ""
                }
            }
            
            event = f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"
            
            if log_events:
                with open(LOG_FILE, "a") as f:
                    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                    f.write(f"\n[SSE EVENT #{event_count}] {timestamp}\n")
                    f.write("Event Type: content_block_start\n")
                    f.write(f"Data:\n{json.dumps(block_start, indent=2)}\n")
            
            yield event.encode('utf-8')
            
            # Send content_block_delta with full text
            event_count += 1
            block_delta = {
                "type": "content_block_delta",
                "index": actual_idx,
                "delta": {
                    "type": "text_delta",
                    "text": text
                }
            }
            
            event = f"event: content_block_delta\ndata: {json.dumps(block_delta)}\n\n"
            
            if log_events:
                with open(LOG_FILE, "a") as f:
                    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                    f.write(f"\n[SSE EVENT #{event_count}] {timestamp}\n")
                    f.write("Event Type: content_block_delta\n")
                    f.write(f"Data:\n{json.dumps(block_delta, indent=2)[:2000]}\n")
                    if len(json.dumps(block_delta)) > 2000:
                        f.write("... (truncated)\n")
            
            yield event.encode('utf-8')
            
            # Send content_block_stop
            event_count += 1
            block_stop = {
                "type": "content_block_stop",
                "index": actual_idx
            }
            
            event = f"event: content_block_stop\ndata: {json.dumps(block_stop)}\n\n"
            
            if log_events:
                with open(LOG_FILE, "a") as f:
                    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                    f.write(f"\n[SSE EVENT #{event_count}] {timestamp}\n")
                    f.write("Event Type: content_block_stop\n")
                    f.write(f"Data:\n{json.dumps(block_stop, indent=2)}\n")
            
            yield event.encode('utf-8')
    
    # 4. Send message_delta
    event_count += 1
    message_delta = {
        "type": "message_delta",
        "delta": {
            "stop_reason": stop_reason,
            "stop_sequence": stop_sequence
        },
        "usage": {
            "output_tokens": usage.get('output_tokens', 0)
        }
    }
    
    event = f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"
    
    if log_events:
        with open(LOG_FILE, "a") as f:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            f.write(f"\n[SSE EVENT #{event_count}] {timestamp}\n")
            f.write("Event Type: message_delta\n")
            f.write(f"Data:\n{json.dumps(message_delta, indent=2)}\n")
    
    yield event.encode('utf-8')
    
    # 5. Send message_stop
    event_count += 1
    message_stop = {"type": "message_stop"}
    
    event = f"event: message_stop\ndata: {json.dumps(message_stop)}\n\n"
    
    if log_events:
        with open(LOG_FILE, "a") as f:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            f.write(f"\n[SSE EVENT #{event_count}] {timestamp}\n")
            f.write("Event Type: message_stop\n")
            f.write(f"Data:\n{json.dumps(message_stop, indent=2)}\n")
    
    yield event.encode('utf-8')
    
    # Log streaming completion
    if log_events:
        with open(LOG_FILE, "a") as f:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            f.write(f"\n[STREAM END] {timestamp}\n")
            f.write(f"Total SSE events: {event_count}\n")
