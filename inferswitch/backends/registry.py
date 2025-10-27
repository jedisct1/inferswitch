"""
Backend registry for managing multiple backend instances.
"""

from typing import Dict, Optional, List, Any
import asyncio
import logging
from .base import BaseBackend
from .config import BackendConfigManager
from .router import BackendRouter

logger = logging.getLogger(__name__)


class BackendRegistry:
    """Registry for managing backend instances."""

    def __init__(self):
        """Initialize the registry."""
        self.backends: Dict[str, BaseBackend] = {}
        self.router: Optional[BackendRouter] = None
        self._initialized = False

    async def initialize(self, backend_classes: Dict[str, type]):
        """
        Initialize the registry with backend implementations.

        Args:
            backend_classes: Dictionary mapping backend names to their class types
                           e.g., {"anthropic": AnthropicBackend, "openai": OpenAIBackend}
        """
        if self._initialized:
            return

        # Load configurations
        configs = BackendConfigManager.load_config()

        # Initialize backends
        for name, config in configs.items():
            if name in backend_classes:
                backend_class = backend_classes[name]
                try:
                    backend = backend_class(config)
                    self.backends[name] = backend
                except Exception as e:
                    logger.error(f"Failed to initialize backend '{name}': {e}")

        # Initialize router
        self.router = BackendRouter(self.backends)
        self._initialized = True

    def register(self, name: str, backend: BaseBackend):
        """
        Register a backend instance.

        Args:
            name: Backend name
            backend: Backend instance
        """
        self.backends[name] = backend
        # Reinitialize router
        self.router = BackendRouter(self.backends)

    def unregister(self, name: str):
        """
        Unregister a backend.

        Args:
            name: Backend name to remove
        """
        if name in self.backends:
            del self.backends[name]
            # Reinitialize router
            self.router = BackendRouter(self.backends)

    def get_backend(self, name: str) -> Optional[BaseBackend]:
        """
        Get a specific backend by name.

        Args:
            name: Backend name

        Returns:
            Backend instance or None
        """
        return self.backends.get(name)

    def get_active_backend(self) -> Optional[BaseBackend]:
        """
        Get the currently active backend based on configuration.

        Returns:
            Active backend instance or None
        """
        active_name = BackendConfigManager.get_active_backend()
        return self.backends.get(active_name)

    def list_backends(self) -> List[str]:
        """
        List all registered backend names.

        Returns:
            List of backend names
        """
        return list(self.backends.keys())

    def get_router(self) -> BackendRouter:
        """
        Get the backend router.

        Returns:
            BackendRouter instance

        Raises:
            RuntimeError: If registry not initialized
        """
        if not self.router:
            raise RuntimeError("Registry not initialized")
        return self.router

    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Run health checks on all backends.

        Returns:
            Dictionary mapping backend names to health status
        """
        results = {}

        async def check_backend(name: str, backend: BaseBackend):
            try:
                health = await backend.health_check()
                results[name] = health
            except Exception as e:
                results[name] = {"status": "error", "error": str(e), "backend": name}

        # Run health checks concurrently
        tasks = [
            check_backend(name, backend) for name, backend in self.backends.items()
        ]
        await asyncio.gather(*tasks)

        return results

    async def close_all(self):
        """Close all backend connections."""
        tasks = [backend.close() for backend in self.backends.values()]
        await asyncio.gather(*tasks, return_exceptions=True)

    def get_models_summary(self) -> Dict[str, List[str]]:
        """
        Get supported models for all backends.

        Returns:
            Dictionary mapping backend names to their supported models
        """
        summary = {}
        for name, backend in self.backends.items():
            if hasattr(backend.config, "models") and backend.config.models:
                summary[name] = backend.config.models
            else:
                summary[name] = ["dynamic"]  # For backends with dynamic model lists
        return summary

    def get_capabilities_summary(self) -> Dict[str, Dict[str, bool]]:
        """
        Get capabilities summary for all backends.

        Returns:
            Dictionary mapping backend names to their capabilities
        """
        summary = {}
        for name, backend in self.backends.items():
            summary[name] = {
                "streaming": True,  # All backends support streaming
                "token_counting": True,  # All backends support token counting
                "system_messages": True,  # All backends support system messages
            }
        return summary


# Global registry instance
backend_registry = BackendRegistry()
