"""
Request routing logic for selecting appropriate backends.
"""

from typing import Dict, Optional, Tuple
import logging
from .base import BaseBackend
from .config import BackendConfigManager
from .errors import ModelNotFoundError
from .availability import ModelAvailabilityTracker

logger = logging.getLogger(__name__)


class BackendRouter:
    """Routes requests to appropriate backends based on various criteria."""
    
    def __init__(self, backends: Dict[str, BaseBackend]):
        """
        Initialize router with available backends.
        
        Args:
            backends: Dictionary mapping backend names to backend instances
        """
        self.backends = backends
        self.model_overrides = BackendConfigManager.get_model_overrides()
        self.difficulty_models = BackendConfigManager.get_difficulty_model_mapping()
        self.model_providers = BackendConfigManager.get_model_provider_mapping()
        self.fallback_config = BackendConfigManager.get_fallback_config()
        self.force_difficulty_routing = BackendConfigManager.should_force_difficulty_routing()
        
        # Initialize availability tracker
        availability_config = BackendConfigManager.get_model_availability_config()
        self.availability_tracker = ModelAvailabilityTracker(
            disable_duration_seconds=availability_config["disable_duration_seconds"]
        )
        
    def select_backend(
        self,
        model: str,
        explicit_backend: Optional[str] = None,
        difficulty_rating: Optional[float] = None
    ) -> BaseBackend:
        """
        Select the appropriate backend for a request.
        
        Args:
            model: Model name requested
            explicit_backend: Explicitly requested backend (from header)
            difficulty_rating: Query difficulty rating (0-5)
            
        Returns:
            Selected backend instance
            
        Raises:
            ModelNotFoundError: If no suitable backend is found
        """
        
        logger.debug(f"Backend selection: model={model}, difficulty={difficulty_rating}, explicit={explicit_backend}")
        
        # If force_difficulty_routing is enabled and we have a difficulty rating,
        # skip all other routing logic and go straight to difficulty-based routing
        if self.force_difficulty_routing and difficulty_rating is not None:
            logger.debug(f"Force difficulty routing enabled, using difficulty-based routing for rating {difficulty_rating}")
            result = self._route_by_difficulty(model, difficulty_rating)
            if result:
                backend, selected_model = result
                # Store the selected model for the backend to use
                backend._difficulty_selected_model = selected_model
                logger.debug(f"Selected backend: {backend.name} (forced difficulty-based routing, model: {selected_model})")
                return backend
            else:
                logger.debug(f"No backend found for difficulty {difficulty_rating}, continuing with normal routing")
        
        # Apply model overrides first
        original_model = model
        if self.model_overrides:
            # Check for exact match
            if model in self.model_overrides:
                model = self.model_overrides[model]
                logger.debug(f"Model override: {original_model} -> {model}")
            # Check for wildcard match
            elif "*" in self.model_overrides:
                model = self.model_overrides["*"]
                logger.debug(f"Model override (wildcard): {original_model} -> {model}")
        # 1. Check explicit backend selection (from header)
        if explicit_backend:
            logger.debug(f"Checking explicit backend '{explicit_backend}'")
            if explicit_backend in self.backends:
                backend = self.backends[explicit_backend]
                # For LM-Studio, always allow any model (dynamic model list)
                if backend.name == "lm-studio" or backend.supports_model(model):
                    logger.debug(f"Selected backend: {backend.name} (explicit header)")
                    return backend
                else:
                    raise ModelNotFoundError(
                        f"Model '{model}' not supported by backend '{explicit_backend}'",
                        model=model,
                        backend=explicit_backend
                    )
            else:
                raise ModelNotFoundError(
                    f"Backend '{explicit_backend}' not found",
                    model=model,
                    backend=explicit_backend,
                    available_models=list(self.backends.keys())
                )
        
        # 2. Check if INFERSWITCH_BACKEND is explicitly set to force all traffic
        active_backend_name = BackendConfigManager.get_active_backend()
        force_backend = BackendConfigManager.should_force_backend()
        
        logger.debug(f"Checking INFERSWITCH_BACKEND - active: {active_backend_name}, force: {force_backend}")
        
        if active_backend_name and force_backend:
            # When force_backend is True, send ALL traffic to the specified backend
            if active_backend_name in self.backends:
                backend = self.backends[active_backend_name]
                # For LM-Studio, always allow any model
                if backend.name == "lm-studio" or backend.supports_model(model):
                    logger.debug(f"Selected backend: {backend.name} (forced by INFERSWITCH_BACKEND)")
                    return backend
        
        # 3. Difficulty-based routing (if difficulty rating is provided)
        logger.debug("Checking difficulty-based routing")
        if difficulty_rating is not None:
            result = self._route_by_difficulty(model, difficulty_rating)
            if result:
                backend, selected_model = result
                # Store the selected model for the backend to use
                backend._difficulty_selected_model = selected_model
                logger.debug(f"Selected backend: {backend.name} (difficulty-based routing, model: {selected_model})")
                return backend
            else:
                logger.debug(f"No backend found for difficulty {difficulty_rating}")
        
        # 4. Check model to provider mapping
        logger.debug("Checking model to provider mapping")
        if model in self.model_providers:
            provider_name = self.model_providers[model]
            if provider_name in self.backends:
                logger.debug(f"Selected backend: {provider_name} (model provider mapping)")
                return self.backends[provider_name]
        
        # 5. Use fallback configuration
        logger.debug("Using fallback configuration")
        if self.fallback_config:
            fallback_provider, fallback_model = self.fallback_config
            logger.debug(f"Fallback: provider={fallback_provider}, model={fallback_model}")
            
            if fallback_provider in self.backends:
                backend = self.backends[fallback_provider]
                # Store the fallback model for the backend to use
                backend._fallback_model = fallback_model
                logger.debug(f"Selected backend: {backend.name} (fallback)")
                return backend
        
        # No suitable backend found
        available_models = []
        for backend in self.backends.values():
            if hasattr(backend.config, 'models') and backend.config.models:
                available_models.extend(backend.config.models)
        
        raise ModelNotFoundError(
            f"No backend found for model '{model}'",
            model=model,
            available_models=available_models
        )
    
    def mark_model_failure(self, model: str):
        """
        Mark a model as failed, temporarily disabling it.
        
        Args:
            model: Model name that failed
        """
        self.availability_tracker.mark_failure(model)
    
    def mark_model_success(self, model: str):
        """
        Mark a model as successful.
        
        Args:
            model: Model name that succeeded
        """
        self.availability_tracker.mark_success(model)
    
    
    def _route_by_difficulty(self, model: str, difficulty_rating: float) -> Optional[Tuple[BaseBackend, str]]:
        """
        Route based on difficulty rating using the new configuration system.
        
        Args:
            model: Model name (will be overridden by difficulty mapping)
            difficulty_rating: Query difficulty rating (0-5)
            
        Returns:
            Tuple of (Backend, selected_model) or None if no available model found
        """
        
        logger.debug(f"Checking difficulty routing for rating {difficulty_rating}")
        
        # First, find which models to try based on difficulty
        candidate_models = []
        for (min_diff, max_diff), models in self.difficulty_models.items():
            logger.debug(f"Checking range [{min_diff}, {max_diff}] -> {models}")
            if min_diff <= difficulty_rating <= max_diff:
                candidate_models = models
                logger.debug(f"Difficulty {difficulty_rating} maps to models: {candidate_models}")
                break
        
        if not candidate_models:
            logger.debug(f"No model mapping found for difficulty {difficulty_rating}")
            return None
        
        # Try each model in order until we find one that's available
        for candidate_model in candidate_models:
            # Check if the model is available (not temporarily disabled)
            if not self.availability_tracker.is_available(candidate_model):
                logger.debug(f"Model {candidate_model} is temporarily disabled, skipping")
                continue
            
            # Find the provider for this model
            provider = self.model_providers.get(candidate_model)
            if not provider:
                logger.debug(f"No provider mapping found for model {candidate_model}")
                continue
            
            # Get the backend for this provider
            backend = self.backends.get(provider)
            if backend:
                logger.debug(f"Selected backend: {backend.name} (via model {candidate_model})")
                return (backend, candidate_model)
            else:
                logger.debug(f"Backend {provider} not available")
        
        logger.debug(f"No available models found for difficulty {difficulty_rating}")
        return None
    
    
    def get_backend_for_model(self, model: str) -> Optional[str]:
        """
        Get the backend name for a given model.
        
        Args:
            model: Model name
            
        Returns:
            Backend name or None if not found
        """
        try:
            backend = self.select_backend(model)
            return backend.name
        except ModelNotFoundError:
            return None
    
    def get_overridden_model(self, requested_model: str) -> str:
        """
        Get the overridden model name for a requested model.
        
        Args:
            requested_model: The model name requested by the client
            
        Returns:
            The overridden model name, or the original if no override exists
        """
        if self.model_overrides:
            # Check for exact match
            if requested_model in self.model_overrides:
                return self.model_overrides[requested_model]
            # Check for wildcard match
            elif "*" in self.model_overrides:
                return self.model_overrides["*"]
        return requested_model