"""
Backend configuration management.
"""

import os
import json
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from .base import BackendConfig
from ..utils import get_logger, load_config_file

logger = get_logger(__name__)


class BackendConfigManager:
    """Manages backend configurations from environment and files."""

    @staticmethod
    def load_config() -> Dict[str, BackendConfig]:
        """
        Load backend configurations from environment and config files.

        Priority order:
        1. Environment variables
        2. inferswitch.config.json in current directory
        3. Default configurations

        Returns:
            Dictionary mapping backend names to their configurations
        """
        configs = {}

        # Load defaults
        configs.update(BackendConfigManager._get_default_configs())

        # Load from config file using common utility
        file_config = load_config_file("inferswitch.config.json")
        if file_config:
            # Merge file configs with existing configs instead of replacing
            file_configs = BackendConfigManager._parse_file_config(file_config)
            for name, file_backend_config in file_configs.items():
                if name in configs:
                    # Merge with existing config
                    configs[name] = BackendConfigManager._merge_configs(
                        configs[name], file_backend_config
                    )
                else:
                    # New backend from file
                    configs[name] = file_backend_config

        # Override with environment variables
        configs.update(BackendConfigManager._get_env_configs())

        return configs

    @staticmethod
    def _get_default_configs() -> Dict[str, BackendConfig]:
        """Get default backend configurations."""
        return {
            "anthropic": BackendConfig(
                name="anthropic",
                base_url="https://api.anthropic.com",
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                models=[
                    "claude-3-5-haiku-20241022",
                    "claude-3-5-sonnet-20241022",
                    "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307",
                    "claude-haiku-4-5-20251001",
                ],
            ),
            "lm-studio": BackendConfig(
                name="lm-studio",
                base_url=os.environ.get("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234"),
                api_key=os.environ.get(
                    "LM_STUDIO_API_KEY", "lm-studio"
                ),  # LM-Studio doesn't require real API key
                models=None,  # Will be fetched dynamically
            ),
            "openai": BackendConfig(
                name="openai",
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com"),
                api_key=os.environ.get("OPENAI_API_KEY"),
                models=[
                    "gpt-4-turbo-preview",
                    "gpt-4",
                    "gpt-3.5-turbo",
                    "gpt-4-vision-preview",
                ],
            ),
            "openrouter": BackendConfig(
                name="openrouter",
                base_url=os.environ.get(
                    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
                ),
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                models=None,  # Will be fetched dynamically
            ),
        }

    @staticmethod
    def _get_env_configs() -> Dict[str, BackendConfig]:
        """Load configurations from environment variables."""
        configs = {}

        # Note: INFERSWITCH_BACKEND is handled elsewhere in get_active_backend()

        # Custom backend from environment
        if os.environ.get("CUSTOM_BACKEND_URL"):
            configs["custom"] = BackendConfig(
                name="custom",
                base_url=os.environ["CUSTOM_BACKEND_URL"],
                api_key=os.environ.get("CUSTOM_BACKEND_API_KEY"),
            )

        return configs

    @staticmethod
    def _parse_file_config(file_config: Dict[str, Any]) -> Dict[str, BackendConfig]:
        """Parse backend configurations from JSON file."""
        configs = {}

        backends = file_config.get("backends", {})
        for name, backend_data in backends.items():
            # Get base_url with default for known backends
            base_url = backend_data.get("base_url")
            if not base_url:
                if name == "anthropic":
                    base_url = "https://api.anthropic.com"
                elif name == "openai":
                    base_url = "https://api.openai.com"
                elif name == "openrouter":
                    base_url = "https://openrouter.ai/api/v1"
                else:
                    continue  # Skip backends without base_url

            configs[name] = BackendConfig(
                name=name,
                base_url=base_url,
                api_key=backend_data.get("api_key"),
                timeout=backend_data.get("timeout", 600),
                max_retries=backend_data.get("max_retries", 3),
                headers=backend_data.get("headers"),
                models=backend_data.get("models"),
            )

        return configs

    @staticmethod
    def _merge_configs(
        base_config: BackendConfig, override_config: BackendConfig
    ) -> BackendConfig:
        """Merge two backend configs, with override taking precedence."""
        return BackendConfig(
            name=base_config.name,
            base_url=override_config.base_url or base_config.base_url,
            api_key=override_config.api_key or base_config.api_key,
            timeout=override_config.timeout,
            max_retries=override_config.max_retries,
            headers={**(base_config.headers or {}), **(override_config.headers or {})},
            models=override_config.models or base_config.models,
        )

    @staticmethod
    def get_active_backend() -> str:
        """
        Get the active backend name from configuration.

        Returns:
            Backend name (defaults to "anthropic")
        """
        # Check environment variable
        backend = os.environ.get("INFERSWITCH_BACKEND", "anthropic")

        # Validate backend name
        valid_backends = ["anthropic", "lm-studio", "openai", "openrouter", "custom"]
        if backend not in valid_backends:
            logger.warning(f"Unknown backend '{backend}', falling back to 'anthropic'")
            return "anthropic"

        return backend

    @staticmethod
    def should_force_backend() -> bool:
        """
        Check if the backend should be forced for all requests.

        When INFERSWITCH_BACKEND is explicitly set, we assume the user
        wants ALL traffic to go to that backend, regardless of model.

        Returns:
            True if backend should be forced for all requests
        """
        # If INFERSWITCH_BACKEND is explicitly set, force all traffic to that backend
        return os.environ.get("INFERSWITCH_BACKEND") is not None

    @staticmethod
    def get_model_overrides() -> Dict[str, str]:
        """
        Get model override mappings.

        This allows remapping requested models to different models.
        For example, mapping all Claude requests to a local model.

        Returns:
            Dictionary mapping requested model names to override model names
        """
        overrides = {}

        # Load from environment variable first
        env_override = os.environ.get("INFERSWITCH_MODEL_OVERRIDE")
        if env_override:
            # Format: "requested_model:override_model,requested_model2:override_model2"
            for mapping in env_override.split(","):
                if ":" in mapping:
                    requested, override = mapping.split(":", 1)
                    overrides[requested.strip()] = override.strip()

        # Load from config file
        config_file = Path("inferswitch.config.json")
        if config_file.exists():
            try:
                with open(config_file) as f:
                    file_config = json.load(f)
                    file_overrides = file_config.get("model_overrides", {})
                    # File config takes precedence over env var
                    overrides.update(file_overrides)
            except Exception:
                pass

        # Check for a catch-all override
        default_model = os.environ.get("INFERSWITCH_DEFAULT_MODEL")
        if default_model and "*" not in overrides:
            overrides["*"] = default_model

        return overrides

    @staticmethod
    def get_difficulty_model_mapping() -> Dict[Tuple[float, float], List[str]]:
        """
        Get mapping from difficulty ranges to model names.

        Returns:
            Dictionary mapping (min_difficulty, max_difficulty) tuples to lists of model names
        """
        mappings = {}

        # Load from config file
        config_file = Path("inferswitch.config.json")
        if config_file.exists():
            try:
                with open(config_file) as f:
                    file_config = json.load(f)
                    difficulty_models = file_config.get("difficulty_models", {})

                    # Convert from JSON format to tuple keys
                    for range_str, models in difficulty_models.items():
                        # Parse range string like "0.0-0.3" or "[0.0,0.3]" or single values like "3"
                        range_str = range_str.strip("[]")
                        if "-" in range_str:
                            parts = range_str.split("-")
                        elif "," in range_str:
                            parts = range_str.split(",")
                        else:
                            # Single value - treat as exact match range (e.g., "3" -> (3.0, 3.0))
                            try:
                                value = float(range_str.strip())
                                # Ensure models is always a list
                                if isinstance(models, str):
                                    models = [models]
                                mappings[(value, value)] = models
                                continue
                            except ValueError:
                                logger.warning(f"Invalid difficulty value: {range_str}")
                                continue

                        if len(parts) == 2:
                            try:
                                min_d = float(parts[0].strip())
                                max_d = float(parts[1].strip())
                                # Ensure models is always a list
                                if isinstance(models, str):
                                    models = [models]
                                mappings[(min_d, max_d)] = models
                            except ValueError:
                                logger.warning(
                                    f"Invalid difficulty range format: {range_str}"
                                )
            except Exception as e:
                logger.warning(f"Failed to load difficulty mappings: {e}")

        return mappings

    @staticmethod
    def get_model_provider_mapping() -> Dict[str, str]:
        """
        Get mapping from model names to provider names.

        Returns:
            Dictionary mapping model names to provider/backend names
        """
        # Default mappings
        mappings = {
            # Anthropic models
            "claude-3-5-haiku-20241022": "anthropic",
            "claude-3-5-sonnet-20241022": "anthropic",
            "claude-3-opus-20240229": "anthropic",
            "claude-3-sonnet-20240229": "anthropic",
            "claude-3-haiku-20240307": "anthropic",
            "claude-haiku-4-5-20251001": "anthropic",
            # OpenAI models
            "gpt-4-turbo-preview": "openai",
            "gpt-4": "openai",
            "gpt-3.5-turbo": "openai",
            "gpt-4-vision-preview": "openai",
        }

        # Load custom mappings from config file
        config_file = Path("inferswitch.config.json")
        if config_file.exists():
            try:
                with open(config_file) as f:
                    file_config = json.load(f)
                    custom_mappings = file_config.get("model_providers", {})
                    mappings.update(custom_mappings)
            except Exception:
                pass

        return mappings

    @staticmethod
    def get_fallback_config() -> Optional[Tuple[str, str]]:
        """
        Get fallback provider and model configuration.

        Returns:
            Tuple of (provider_name, model_name) or None if not configured
        """
        # Check environment variables first
        env_provider = os.environ.get("INFERSWITCH_FALLBACK_PROVIDER")
        env_model = os.environ.get("INFERSWITCH_FALLBACK_MODEL")

        if env_provider and env_model:
            return (env_provider, env_model)

        # Load from config file
        config_file = Path("inferswitch.config.json")
        if config_file.exists():
            try:
                with open(config_file) as f:
                    file_config = json.load(f)
                    fallback = file_config.get("fallback")
                    if fallback and isinstance(fallback, dict):
                        provider = fallback.get("provider")
                        model = fallback.get("model")
                        if provider and model:
                            return (provider, model)
            except Exception:
                pass

        # Default fallback
        return ("anthropic", "claude-3-haiku-20240307")

    @staticmethod
    def get_oauth_config(provider: str = "anthropic") -> Dict[str, Any]:
        """
        Get OAuth configuration for a specific provider from config file.

        Args:
            provider: The provider name (default: "anthropic")

        Returns:
            Dictionary with OAuth configuration or empty dict if not configured
        """
        oauth_config = {}

        # Load from config file
        config_file = Path("inferswitch.config.json")
        if config_file.exists():
            try:
                with open(config_file) as f:
                    file_config = json.load(f)

                    # Check the providers_auth structure
                    providers_auth = file_config.get("providers_auth", {})
                    if provider in providers_auth:
                        provider_auth = providers_auth[provider]
                        oauth_config = provider_auth.get("oauth", {})
            except Exception as e:
                logger.warning(f"Failed to load OAuth config: {e}")

        # Environment variables can override
        if os.environ.get("OAUTH_CLIENT_ID"):
            oauth_config["client_id"] = os.environ["OAUTH_CLIENT_ID"]

        return oauth_config

    @staticmethod
    def get_model_availability_config() -> Dict[str, Any]:
        """
        Get model availability configuration (for temporary disabling on failure).

        Returns:
            Dictionary with:
            - disable_duration_seconds: How long to disable a model after failure (default: 300)
            - max_retries: Max number of retries before disabling (default: 1)
        """
        config = {
            "disable_duration_seconds": 300,  # 5 minutes default
            "max_retries": 1,
        }

        # Load from environment
        if os.environ.get("INFERSWITCH_MODEL_DISABLE_DURATION"):
            try:
                config["disable_duration_seconds"] = int(
                    os.environ["INFERSWITCH_MODEL_DISABLE_DURATION"]
                )
            except ValueError:
                pass

        # Load from config file
        config_file = Path("inferswitch.config.json")
        if config_file.exists():
            try:
                with open(config_file) as f:
                    file_config = json.load(f)
                    availability_config = file_config.get("model_availability", {})
                    if "disable_duration_seconds" in availability_config:
                        config["disable_duration_seconds"] = availability_config[
                            "disable_duration_seconds"
                        ]
                    if "max_retries" in availability_config:
                        config["max_retries"] = availability_config["max_retries"]
            except Exception:
                pass

        return config

    @staticmethod
    def should_force_difficulty_routing() -> bool:
        """
        Check if difficulty-based routing should be forced for all requests.

        When enabled, all requests will be routed based on difficulty rating,
        ignoring the model requested by the client.

        Returns:
            True if difficulty routing should be forced for all requests
        """
        # Check environment variable first
        if os.environ.get("INFERSWITCH_FORCE_DIFFICULTY_ROUTING"):
            return os.environ["INFERSWITCH_FORCE_DIFFICULTY_ROUTING"].lower() in [
                "true",
                "1",
                "yes",
                "on",
            ]

        # Load from config file
        config_file = Path("inferswitch.config.json")
        if config_file.exists():
            try:
                with open(config_file) as f:
                    file_config = json.load(f)
                    return file_config.get("force_difficulty_routing", False)
            except Exception:
                pass

        return False

    @staticmethod
    def get_mlx_model() -> str:
        """
        Get the MLX model to use for difficulty rating.

        Returns:
            MLX model name (defaults to "jedisct1/arch-router-1.5b")
        """
        # Check environment variable first
        mlx_model = os.environ.get("INFERSWITCH_MLX_MODEL")
        if mlx_model:
            return mlx_model

        # Load from config file
        config_file = Path("inferswitch.config.json")
        if config_file.exists():
            try:
                with open(config_file) as f:
                    file_config = json.load(f)
                    mlx_model = file_config.get("mlx_model")
                    if mlx_model:
                        return mlx_model
            except Exception:
                pass

        # Default model
        return "jedisct1/arch-router-1.5b"

    @staticmethod
    def get_expert_definitions() -> Dict[str, str]:
        """
        Get user-defined expert definitions from configuration.

        Returns:
            Dictionary mapping expert names to their descriptions
        """
        definitions = {}

        # Load from config file
        config_file = Path("inferswitch.config.json")
        if config_file.exists():
            try:
                with open(config_file) as f:
                    file_config = json.load(f)
                    definitions = file_config.get("expert_definitions", {})

            except Exception as e:
                logger.warning(f"Failed to load expert definitions: {e}")

        return definitions

    @staticmethod
    def get_expert_model_mapping() -> Dict[str, List[str]]:
        """
        Get mapping from expert names to model names.

        Returns:
            Dictionary mapping expert names to lists of model names
        """
        mappings = {}

        # Load from config file
        config_file = Path("inferswitch.config.json")
        if config_file.exists():
            try:
                with open(config_file) as f:
                    file_config = json.load(f)
                    expert_models = file_config.get("expert_models", {})

                    # Ensure models are always lists
                    for expert_name, models in expert_models.items():
                        if isinstance(models, str):
                            models = [models]
                        mappings[expert_name] = models

            except Exception as e:
                logger.warning(f"Failed to load expert model mappings: {e}")

        return mappings

    @staticmethod
    def get_expertise_model_mapping() -> Dict[str, List[str]]:
        """
        Get mapping from expertise areas to model names (legacy support).

        Returns:
            Dictionary mapping expertise areas to lists of model names
        """
        mappings = {}

        # Load from config file
        config_file = Path("inferswitch.config.json")
        if config_file.exists():
            try:
                with open(config_file) as f:
                    file_config = json.load(f)
                    expertise_models = file_config.get("expertise_models", {})

                    # Ensure models are always lists
                    for expertise, models in expertise_models.items():
                        if isinstance(models, str):
                            models = [models]
                        mappings[expertise.lower()] = models

            except Exception as e:
                logger.warning(f"Failed to load expertise mappings: {e}")

        return mappings

    @staticmethod
    def should_force_expert_routing() -> bool:
        """
        Check if expert-based routing should be forced for all requests.

        When enabled, all requests will be routed based on expert classification,
        ignoring the model requested by the client.

        Returns:
            True if expert routing should be forced for all requests
        """
        # Check environment variable first
        if os.environ.get("INFERSWITCH_FORCE_EXPERT_ROUTING"):
            return os.environ["INFERSWITCH_FORCE_EXPERT_ROUTING"].lower() in [
                "true",
                "1",
                "yes",
                "on",
            ]

        # Load from config file
        config_file = Path("inferswitch.config.json")
        if config_file.exists():
            try:
                with open(config_file) as f:
                    file_config = json.load(f)
                    return file_config.get("force_expert_routing", False)
            except Exception:
                pass

        return False

    @staticmethod
    def should_force_expertise_routing() -> bool:
        """
        Check if expertise-based routing should be forced for all requests (legacy support).

        When enabled, all requests will be routed based on expertise classification,
        ignoring the model requested by the client.

        Returns:
            True if expertise routing should be forced for all requests
        """
        # Check environment variable first
        if os.environ.get("INFERSWITCH_FORCE_EXPERTISE_ROUTING"):
            return os.environ["INFERSWITCH_FORCE_EXPERTISE_ROUTING"].lower() in [
                "true",
                "1",
                "yes",
                "on",
            ]

        # Load from config file
        config_file = Path("inferswitch.config.json")
        if config_file.exists():
            try:
                with open(config_file) as f:
                    file_config = json.load(f)
                    return file_config.get("force_expertise_routing", False)
            except Exception:
                pass

        return False

    @staticmethod
    def get_routing_mode() -> str:
        """
        Get the current routing mode: 'expert', 'expertise', 'difficulty', or 'normal'.

        Returns:
            Routing mode string
        """
        # Check for explicit expert routing (new system)
        if BackendConfigManager.should_force_expert_routing():
            return "expert"

        # Check for explicit expertise routing (legacy)
        if BackendConfigManager.should_force_expertise_routing():
            return "expertise"

        # Check for explicit difficulty routing (backward compatibility)
        if BackendConfigManager.should_force_difficulty_routing():
            return "difficulty"

        # Check if we have expert models configured (new system)
        expert_models = BackendConfigManager.get_expert_model_mapping()
        expert_definitions = BackendConfigManager.get_expert_definitions()
        if expert_models and expert_definitions:
            return "expert"

        # Check if we have expertise models configured (legacy)
        expertise_models = BackendConfigManager.get_expertise_model_mapping()
        if expertise_models:
            return "expertise"

        # Check if we have difficulty models configured
        difficulty_models = BackendConfigManager.get_difficulty_model_mapping()
        if difficulty_models:
            return "difficulty"

        # Default to normal routing
        return "normal"
