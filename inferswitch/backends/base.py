"""
Base classes and interfaces for backend implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, AsyncIterator
from dataclasses import dataclass
from pydantic import BaseModel


@dataclass
class BackendConfig:
    """Configuration for a backend instance."""
    name: str
    base_url: str
    api_key: Optional[str] = None
    timeout: int = 600
    max_retries: int = 3
    headers: Optional[Dict[str, str]] = None
    models: Optional[List[str]] = None  # List of supported models


class BackendResponse(BaseModel):
    """Unified response format from backends."""
    content: List[Dict[str, Any]]  # Anthropic-style content blocks
    model: str
    stop_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Dict[str, Any]] = None  # Original response


class BaseBackend(ABC):
    """Abstract base class for all backend implementations."""
    
    def __init__(self, config: BackendConfig):
        self.config = config
        self.name = config.name
        self.base_url = config.base_url
        self.api_key = config.api_key
        # Special attributes set by router
        self._difficulty_selected_model = None
        self._fallback_model = None
        
    @abstractmethod
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
        """
        Create a chat completion.
        
        Args:
            messages: List of message dictionaries
            model: Model identifier
            system: System message (if supported)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional backend-specific parameters
            
        Returns:
            BackendResponse in normalized format
        """
        pass
    
    @abstractmethod
    async def create_message_stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Create a streaming chat completion.
        
        Yields:
            Anthropic-style SSE events
        """
        pass
    
    @abstractmethod
    async def count_tokens(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        system: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Count tokens in messages.
        
        Returns:
            Dictionary with token counts
        """
        pass
    
    @abstractmethod
    def supports_model(self, model: str) -> bool:
        """Check if this backend supports a given model."""
        pass
    
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check backend health/availability.
        
        Returns:
            Health status information
        """
        return {"status": "ok", "backend": self.name}
    
    async def close(self):
        """Clean up resources."""
        pass
    
    def get_effective_model(self, requested_model: str) -> str:
        """
        Get the effective model to use, considering router overrides.
        
        Args:
            requested_model: The model originally requested
            
        Returns:
            The model to actually use
        """
        # Check if router selected a specific model based on difficulty
        if hasattr(self, '_difficulty_selected_model') and self._difficulty_selected_model:
            return self._difficulty_selected_model
        
        # Check if router is using fallback
        if hasattr(self, '_fallback_model') and self._fallback_model:
            return self._fallback_model
        
        # Otherwise use the requested model
        return requested_model