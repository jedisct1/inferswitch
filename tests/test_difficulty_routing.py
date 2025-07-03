#!/usr/bin/env python3
"""
Test script for difficulty-based routing functionality.
"""

import json
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inferswitch.backends.base import BackendConfig, BaseBackend
from inferswitch.backends.router import BackendRouter
from inferswitch.backends.config import BackendConfigManager


class MockBackend(BaseBackend):
    """Mock backend for testing."""

    def __init__(self, config: BackendConfig, supported_models=None):
        super().__init__(config)
        self.supported_models = supported_models or []

    async def create_message(self, *args, **kwargs):
        return {"response": "mock"}

    async def create_message_stream(self, *args, **kwargs):
        yield {"response": "mock stream"}

    async def count_tokens(self, *args, **kwargs):
        return {"tokens": 100}

    def supports_model(self, model: str) -> bool:
        return model in self.supported_models

    def get_capabilities(self):
        return self.config.capabilities or {}


def test_difficulty_routing():
    """Test difficulty-based routing."""
    print("Testing difficulty-based routing...")

    # Create mock backends with different difficulty ranges
    backends = {
        "easy": MockBackend(
            BackendConfig(
                name="easy",
                base_url="http://easy.local",
                difficulty_range=(0.0, 1.5),
                models=["gpt-3.5-turbo", "claude-3-haiku-20240307"],
            ),
            supported_models=["gpt-3.5-turbo", "claude-3-haiku-20240307"],
        ),
        "medium": MockBackend(
            BackendConfig(
                name="medium",
                base_url="http://medium.local",
                difficulty_range=(1.5, 3.5),
                models=["gpt-4", "claude-3-sonnet-20240229"],
            ),
            supported_models=["gpt-4", "claude-3-sonnet-20240229"],
        ),
        "hard": MockBackend(
            BackendConfig(
                name="hard",
                base_url="http://hard.local",
                difficulty_range=(3.5, 5.0),
                models=["gpt-4-turbo-preview", "claude-3-opus-20240229"],
            ),
            supported_models=["gpt-4-turbo-preview", "claude-3-opus-20240229"],
        ),
        "all": MockBackend(
            BackendConfig(
                name="all",
                base_url="http://all.local",
                difficulty_range=(0.0, 5.0),
                models=["*"],
            ),
            supported_models=[
                "gpt-3.5-turbo",
                "gpt-4",
                "claude-3-haiku-20240307",
                "claude-3-sonnet-20240229",
                "claude-3-opus-20240229",
            ],
        ),
    }

    router = BackendRouter(backends)

    # Test 1: Easy query should route to easy backend
    print("\n1. Testing easy query (difficulty=1.0)...")
    backend = router.select_backend("gpt-3.5-turbo", difficulty_rating=1.0)
    assert backend.name == "easy", f"Expected 'easy', got '{backend.name}'"
    print("✓ Easy query routed correctly")

    # Test 2: Medium query should route to medium backend
    print("\n2. Testing medium query (difficulty=2.5)...")
    backend = router.select_backend("gpt-4", difficulty_rating=2.5)
    assert backend.name == "medium", f"Expected 'medium', got '{backend.name}'"
    print("✓ Medium query routed correctly")

    # Test 3: Hard query should route to hard backend
    print("\n3. Testing hard query (difficulty=4.5)...")
    backend = router.select_backend("claude-3-opus-20240229", difficulty_rating=4.5)
    assert backend.name == "hard", f"Expected 'hard', got '{backend.name}'"
    print("✓ Hard query routed correctly")

    # Test 4: When multiple backends match, prefer narrower range
    print("\n4. Testing preference for narrower range...")
    # Both 'medium' and 'all' backends support this model at difficulty 2.5
    # 'medium' has narrower range (2.0) vs 'all' (5.0)
    backend = router.select_backend("gpt-4", difficulty_rating=2.5)
    assert backend.name == "medium", (
        f"Expected 'medium' (narrower range), got '{backend.name}'"
    )
    print("✓ Narrower range preferred correctly")

    # Test 5: Model not supported by difficulty-appropriate backend
    print("\n5. Testing unsupported model in difficulty range...")
    # claude-3-opus is a 'hard' model, but we're asking for easy difficulty
    # Since 'easy' backend doesn't support this model, it should fall through
    backend = router.select_backend("claude-3-opus-20240229", difficulty_rating=0.5)
    # Should route to 'all' backend which supports all models
    assert backend.name == "all", f"Expected 'all', got '{backend.name}'"
    print("✓ Fallback to broader backend works correctly")

    # Test 6: No difficulty rating provided
    print("\n6. Testing without difficulty rating...")
    backends_no_default = {
        "anthropic": MockBackend(
            BackendConfig(
                name="anthropic",
                base_url="http://anthropic.local",
                models=["claude-3-opus-20240229"],
            ),
            supported_models=["claude-3-opus-20240229"],
        )
    }
    router_no_default = BackendRouter(backends_no_default)
    backend = router_no_default.select_backend(
        "claude-3-opus-20240229", difficulty_rating=None
    )
    assert backend.name == "anthropic", f"Expected 'anthropic', got '{backend.name}'"
    print("✓ Routing without difficulty works correctly")

    print("\n✅ All difficulty routing tests passed!")


def test_config_loading():
    """Test loading difficulty ranges from config file."""
    print("\n\nTesting config file loading...")

    # Create a temporary config file
    config_data = {
        "backends": {
            "local-easy": {
                "base_url": "http://localhost:8080",
                "api_key": "test-key",
                "difficulty_range": [0.0, 2.0],
                "models": ["small-model"],
                "capabilities": {
                    "supports_streaming": True,
                    "supports_system_message": True,
                },
            },
            "cloud-hard": {
                "base_url": "http://cloud.api.com",
                "api_key": "cloud-key",
                "difficulty_range": [3.0, 5.0],
                "models": ["large-model"],
            },
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        config_file = f.name

    try:
        # Load configs directly from the parsed data
        configs = BackendConfigManager._parse_file_config(config_data)

        # Verify difficulty ranges were loaded
        assert "local-easy" in configs
        assert configs["local-easy"].difficulty_range == (0.0, 2.0)
        print("✓ local-easy difficulty range loaded correctly")

        assert "cloud-hard" in configs
        assert configs["cloud-hard"].difficulty_range == (3.0, 5.0)
        print("✓ cloud-hard difficulty range loaded correctly")

        print("\n✅ Config loading tests passed!")

    finally:
        # Clean up temp file
        Path(config_file).unlink()


if __name__ == "__main__":
    test_difficulty_routing()
    test_config_loading()
    print("\n🎉 All tests completed successfully!")
