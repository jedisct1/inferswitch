"""
MLX backend implementation for using local MLX models as a backend.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List, AsyncIterator
from datetime import datetime

from .base import BaseBackend, BackendConfig, BackendResponse
from .errors import BackendError
from ..mlx_model import mlx_model_manager

logger = logging.getLogger(__name__)


class MLXBackend(BaseBackend):
    """Backend implementation for MLX models."""
    
    async def create_message(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ) -> BackendResponse:
        """Create a chat completion using MLX model."""
        
        if not mlx_model_manager.is_loaded():
            raise BackendError(
                "MLX model not loaded",
                status_code=500,
                backend=self.name
            )
        
        try:
            # Convert messages to a prompt format
            prompt = self._format_messages(messages, system)
            
            # Generate response using MLX
            import mlx_lm
            response_text = mlx_lm.generate(
                model=mlx_model_manager.model,
                tokenizer=mlx_model_manager.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens or 1024,
                temp=temperature or 0.7,
                verbose=False
            )
            
            # Remove the prompt from the response if it's included
            if response_text.startswith(prompt):
                response_text = response_text[len(prompt):].strip()
            
            # Create response in Anthropic format
            content = [{"type": "text", "text": response_text}]
            
            # Estimate token usage (rough approximation)
            input_tokens = len(prompt.split()) * 1.3  # Rough estimate
            output_tokens = len(response_text.split()) * 1.3
            
            return BackendResponse(
                content=content,
                model=mlx_model_manager.model_name or "mlx-model",
                stop_reason="end_turn",
                usage={
                    "input_tokens": int(input_tokens),
                    "output_tokens": int(output_tokens)
                },
                raw_response={
                    "id": f"mlx_{int(time.time()*1000)}",
                    "created": int(time.time()),
                    "model": mlx_model_manager.model_name or "mlx-model"
                }
            )
            
        except Exception as e:
            logger.error(f"MLX generation error: {str(e)}", exc_info=True)
            raise BackendError(
                f"MLX generation failed: {str(e)}",
                status_code=500,
                backend=self.name
            )
    
    async def create_message_stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """Create a streaming chat completion using MLX model."""
        
        if not mlx_model_manager.is_loaded():
            raise BackendError(
                "MLX model not loaded",
                status_code=500,
                backend=self.name
            )
        
        try:
            # Convert messages to a prompt format
            prompt = self._format_messages(messages, system)
            
            # Generate unique message ID
            message_id = f"mlx_{int(time.time()*1000)}"
            
            # Send message_start event
            yield {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": mlx_model_manager.model_name or "mlx-model",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": int(len(prompt.split()) * 1.3),
                        "output_tokens": 0
                    }
                }
            }
            
            # Send content_block_start
            yield {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "text",
                    "text": ""
                }
            }
            
            # Generate response using MLX in a blocking way, then stream it
            # Run the blocking MLX generation in a thread pool
            loop = asyncio.get_event_loop()
            response_text = await loop.run_in_executor(
                None,
                self._generate_sync,
                prompt,
                max_tokens,
                temperature
            )
            
            # Remove the prompt from the response if it's included
            if response_text.startswith(prompt):
                response_text = response_text[len(prompt):].strip()
            
            # Stream the response in chunks
            chunk_size = 20  # Words per chunk
            words = response_text.split()
            
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i+chunk_size]
                chunk_text = " ".join(chunk_words)
                if i + chunk_size < len(words):
                    chunk_text += " "
                
                yield {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "text_delta",
                        "text": chunk_text
                    }
                }
                
                # Small delay to simulate streaming
                await asyncio.sleep(0.01)
            
            # Send content_block_stop
            yield {
                "type": "content_block_stop",
                "index": 0
            }
            
            # Send message_delta with final usage
            output_tokens = int(len(response_text.split()) * 1.3)
            yield {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "end_turn"
                },
                "usage": {
                    "output_tokens": output_tokens
                }
            }
            
            # Send message_stop
            yield {
                "type": "message_stop"
            }
            
        except Exception as e:
            logger.error(f"MLX streaming error: {str(e)}", exc_info=True)
            # Send error event
            yield {
                "type": "error",
                "error": {
                    "type": "internal_error",
                    "message": f"MLX streaming failed: {str(e)}"
                }
            }
    
    def _generate_sync(self, prompt: str, max_tokens: Optional[int], temperature: Optional[float]) -> str:
        """Synchronous generation function to run in thread pool."""
        import mlx_lm
        from mlx_lm.sample_utils import make_sampler
        
        # Create sampler with temperature
        sampler = make_sampler(temp=temperature or 0.7)
        
        return mlx_lm.generate(
            model=mlx_model_manager.model,
            tokenizer=mlx_model_manager.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens or 1024,
            sampler=sampler,
            verbose=False
        )
    
    def _format_messages(self, messages: List[Dict[str, Any]], system: Optional[str]) -> str:
        """Format messages into a prompt for the MLX model."""
        # Use a simple format for now - can be improved based on model's preference
        prompt_parts = []
        
        if system:
            prompt_parts.append(f"System: {system}\n")
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle content that might be a list of content blocks
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                content = "\n".join(text_parts)
            
            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Add the final "Assistant:" to prompt the model
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    async def count_tokens(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        system: Optional[str] = None
    ) -> Dict[str, int]:
        """Count tokens in messages."""
        if not mlx_model_manager.is_loaded():
            # Return rough estimate if model not loaded
            prompt = self._format_messages(messages, system)
            return {"input_tokens": int(len(prompt.split()) * 1.3)}
        
        try:
            prompt = self._format_messages(messages, system)
            
            # Use the tokenizer to get exact count
            tokens = mlx_model_manager.tokenizer.encode(prompt)
            
            return {"input_tokens": len(tokens)}
            
        except Exception as e:
            logger.error(f"Token counting error: {str(e)}")
            # Fallback to rough estimate
            prompt = self._format_messages(messages, system)
            return {"input_tokens": int(len(prompt.split()) * 1.3)}
    
    def supports_model(self, model: str) -> bool:
        """Check if this backend supports a given model."""
        # The MLX backend only supports the special "builtin" model
        return model == "builtin"
    
    async def health_check(self) -> Dict[str, Any]:
        """Check backend health/availability."""
        return {
            "status": "ok" if mlx_model_manager.is_loaded() else "not_loaded",
            "backend": self.name,
            "model_loaded": mlx_model_manager.is_loaded(),
            "model_info": mlx_model_manager.get_model_info()
        }