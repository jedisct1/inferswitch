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