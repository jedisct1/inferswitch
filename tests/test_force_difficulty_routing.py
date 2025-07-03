#!/usr/bin/env python3
"""Test force difficulty routing configuration."""

import json
import os
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from inferswitch.backends.config import BackendConfigManager
from inferswitch.backends.router import BackendRouter
from inferswitch.backends.base import BaseBackend, BackendConfig


class MockBackend(BaseBackend):
    """Mock backend for testing."""

    def __init__(self, name: str, models: list):
        config = BackendConfig(
            name=name, base_url="http://mock", api_key="mock", models=models
        )
        super().__init__(config)
        self.name = name

    def create_completion(self, *args, **kwargs):
        return {"mock": True}

    def supports_model(self, model: str) -> bool:
        return model in self.config.models if self.config.models else True

    def create_message(self, *args, **kwargs):
        return {"content": [{"text": "mock response"}]}

    def create_message_stream(self, *args, **kwargs):
        yield {"type": "message_start"}
        yield {"type": "content_block_delta", "delta": {"text": "mock"}}
        yield {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}

    def count_tokens(self, *args, **kwargs):
        return {"input_tokens": 10, "output_tokens": 5}


def test_force_difficulty_routing():
    """Test that force_difficulty_routing configuration works correctly."""

    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config = {
            "force_difficulty_routing": True,
            "difficulty_models": {
                "0-2": "easy-model",
                "2-4": "medium-model",
                "4-5": "hard-model",
            },
            "model_providers": {
                "easy-model": "backend1",
                "medium-model": "backend2",
                "hard-model": "backend3",
            },
            "fallback": {"provider": "backend1", "model": "easy-model"},
        }
        json.dump(config, f)
        temp_config_path = f.name

    # Save current directory and change to temp directory
    original_dir = os.getcwd()
    temp_dir = os.path.dirname(temp_config_path)

    try:
        # Copy config to inferswitch.config.json in temp directory
        os.chdir(temp_dir)
        os.rename(os.path.basename(temp_config_path), "inferswitch.config.json")

        # Test 1: Config file loading
        print("Test 1: Loading force_difficulty_routing from config file")
        force_routing = BackendConfigManager.should_force_difficulty_routing()
        assert force_routing is True, f"Expected True, got {force_routing}"
        print("✓ Config loaded correctly")

        # Test 2: Environment variable override
        print("\nTest 2: Environment variable override")
        os.environ["INFERSWITCH_FORCE_DIFFICULTY_ROUTING"] = "false"
        force_routing = BackendConfigManager.should_force_difficulty_routing()
        assert force_routing is False, f"Expected False, got {force_routing}"
        print("✓ Environment variable override works")

        # Test 3: Router behavior with force_difficulty_routing
        print("\nTest 3: Router behavior with force_difficulty_routing enabled")
        os.environ["INFERSWITCH_FORCE_DIFFICULTY_ROUTING"] = "true"

        # Create mock backends
        backends = {
            "backend1": MockBackend("backend1", ["easy-model"]),
            "backend2": MockBackend("backend2", ["medium-model"]),
            "backend3": MockBackend("backend3", ["hard-model"]),
        }

        # Create router
        router = BackendRouter(backends)

        # Test routing with different difficulty ratings
        # Even if client requests a specific model, it should be ignored

        # Easy difficulty (0.5) should go to backend1
        selected = router.select_backend("claude-3-opus", difficulty_rating=0.5)
        assert selected.name == "backend1", f"Expected backend1, got {selected.name}"
        print("✓ Easy query routed to backend1")

        # Medium difficulty (3.0) should go to backend2
        selected = router.select_backend("gpt-4", difficulty_rating=3.0)
        assert selected.name == "backend2", f"Expected backend2, got {selected.name}"
        print("✓ Medium query routed to backend2")

        # Hard difficulty (4.5) should go to backend3
        selected = router.select_backend("claude-3-haiku", difficulty_rating=4.5)
        assert selected.name == "backend3", f"Expected backend3, got {selected.name}"
        print("✓ Hard query routed to backend3")

        # Test 4: Router behavior with force_difficulty_routing disabled
        print("\nTest 4: Router behavior with force_difficulty_routing disabled")
        del os.environ["INFERSWITCH_FORCE_DIFFICULTY_ROUTING"]

        # Recreate router without force routing
        os.rename("inferswitch.config.json", "temp.json")
        with open("inferswitch.config.json", "w") as f:
            config["force_difficulty_routing"] = False
            json.dump(config, f)

        router = BackendRouter(backends)

        # Without force routing, difficulty routing should still work as one of the routing methods
        # But it won't override explicit model requests
        # Let's test that an unknown model with no difficulty falls back
        selected = router.select_backend("unknown-model", difficulty_rating=None)
        assert selected.name == "backend1", (
            f"Expected backend1 (fallback), got {selected.name}"
        )
        print(
            "✓ Without force routing and no difficulty, fallback is used for unknown model"
        )

        # With difficulty but force_routing disabled, it should still consider difficulty
        # as part of normal routing priority
        selected = router.select_backend("unknown-model", difficulty_rating=4.5)
        # This will use difficulty routing as a fallback mechanism
        assert selected.name == "backend3", (
            f"Expected backend3 (from difficulty routing), got {selected.name}"
        )
        print(
            "✓ Without force routing, difficulty is still considered in routing priority"
        )

        print("\nAll tests passed! ✨")

    finally:
        # Cleanup
        os.chdir(original_dir)
        if os.path.exists(os.path.join(temp_dir, "inferswitch.config.json")):
            os.remove(os.path.join(temp_dir, "inferswitch.config.json"))
        if os.path.exists(os.path.join(temp_dir, "temp.json")):
            os.remove(os.path.join(temp_dir, "temp.json"))
        if "INFERSWITCH_FORCE_DIFFICULTY_ROUTING" in os.environ:
            del os.environ["INFERSWITCH_FORCE_DIFFICULTY_ROUTING"]


if __name__ == "__main__":
    test_force_difficulty_routing()
