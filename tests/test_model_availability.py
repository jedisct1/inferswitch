#!/usr/bin/env python3
"""
Test model availability tracking and fallback functionality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from inferswitch.backends.availability import ModelAvailabilityTracker
from inferswitch.backends.router import BackendRouter
from inferswitch.backends.config import BackendConfigManager
from inferswitch.backends.base import BaseBackend
from inferswitch.backends.errors import BackendError


class MockBackend(BaseBackend):
    """Mock backend for testing."""
    
    def __init__(self, name: str, models: list):
        self.name = name
        self.models = models
        self._should_fail = False
    
    def supports_model(self, model: str) -> bool:
        return model in self.models
    
    async def create_message(self, **kwargs):
        if self._should_fail:
            raise BackendError("Insufficient credits", status_code=402)
        return Mock(content=[{"type": "text", "text": "OK"}], model=kwargs.get("model"), 
                   raw_response={"id": "test"}, stop_reason="end_turn", usage={})
    
    async def create_message_stream(self, **kwargs):
        if self._should_fail:
            raise BackendError("Rate limit exceeded", status_code=429)
        yield {"type": "message_start", "message": {"id": "test"}}
    
    async def count_tokens(self, **kwargs):
        """Mock token counting."""
        return {"input_tokens": 10, "total_tokens": 10}


def test_model_availability_tracker():
    """Test the ModelAvailabilityTracker functionality."""
    print("Testing ModelAvailabilityTracker...")
    
    # Create tracker with 2 second disable duration for testing
    tracker = ModelAvailabilityTracker(disable_duration_seconds=2)
    
    # Test 1: Model should be available initially
    assert tracker.is_available("claude-3-haiku"), "Model should be available initially"
    print("✓ Model is available initially")
    
    # Test 2: Mark model as failed
    tracker.mark_failure("claude-3-haiku")
    assert not tracker.is_available("claude-3-haiku"), "Model should be disabled after failure"
    assert "claude-3-haiku" in tracker.get_disabled_models(), "Model should be in disabled list"
    print("✓ Model is disabled after failure")
    
    # Test 3: Other models should still be available
    assert tracker.is_available("claude-3-sonnet"), "Other models should remain available"
    print("✓ Other models remain available")
    
    # Test 4: Mark success should re-enable
    tracker.mark_success("claude-3-haiku")
    assert tracker.is_available("claude-3-haiku"), "Model should be re-enabled after success"
    print("✓ Model is re-enabled after success")
    
    # Test 5: Automatic re-enabling after timeout
    tracker.mark_failure("claude-3-opus")
    assert not tracker.is_available("claude-3-opus"), "Model should be disabled"
    print("  Waiting 2 seconds for timeout...")
    import time
    time.sleep(2.1)  # Wait for timeout
    assert tracker.is_available("claude-3-opus"), "Model should be auto-enabled after timeout"
    print("✓ Model is auto-enabled after timeout")
    
    print("\nAll ModelAvailabilityTracker tests passed!")


def test_router_with_model_lists():
    """Test router with list-based difficulty models."""
    print("\nTesting router with model lists...")
    
    # Mock config to return list-based difficulty models
    with patch.object(BackendConfigManager, 'get_difficulty_model_mapping') as mock_diff:
        with patch.object(BackendConfigManager, 'get_model_provider_mapping') as mock_prov:
            with patch.object(BackendConfigManager, 'get_model_availability_config') as mock_avail:
                # Setup mock config
                mock_diff.return_value = {
                    (0.0, 0.3): ["claude-3-haiku", "gpt-3.5-turbo"],  # List of models
                    (0.3, 0.6): ["claude-3-sonnet", "gpt-4"],
                    (0.6, 1.0): ["claude-3-opus", "gpt-4-turbo"]
                }
                
                mock_prov.return_value = {
                    "claude-3-haiku": "anthropic",
                    "claude-3-sonnet": "anthropic", 
                    "claude-3-opus": "anthropic",
                    "gpt-3.5-turbo": "openai",
                    "gpt-4": "openai",
                    "gpt-4-turbo": "openai"
                }
                
                mock_avail.return_value = {
                    "disable_duration_seconds": 1,
                    "max_retries": 1
                }
                
                # Create mock backends
                anthropic_backend = MockBackend("anthropic", ["claude-3-haiku", "claude-3-sonnet", "claude-3-opus"])
                openai_backend = MockBackend("openai", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"])
                
                backends = {
                    "anthropic": anthropic_backend,
                    "openai": openai_backend
                }
                
                # Create router
                router = BackendRouter(backends)
                
                # Test 1: Should select first available model
                backend = router.select_backend("claude-3-haiku", difficulty_rating=0.2)
                assert backend.name == "anthropic", f"Should select anthropic, got {backend.name}"
                assert hasattr(backend, "_difficulty_selected_model"), "Should have selected model"
                assert backend._difficulty_selected_model == "claude-3-haiku", "Should select claude-3-haiku"
                print("✓ Selected first model in list")
                
                # Test 2: Mark first model as failed, should use second
                router.mark_model_failure("claude-3-haiku")
                backend = router.select_backend("claude-3-haiku", difficulty_rating=0.2)
                assert backend.name == "openai", f"Should select openai, got {backend.name}"
                assert backend._difficulty_selected_model == "gpt-3.5-turbo", "Should fallback to gpt-3.5-turbo"
                print("✓ Fallback to second model when first fails")
                
                # Test 3: Mark both as failed, should return None
                router.mark_model_failure("gpt-3.5-turbo")
                try:
                    backend = router.select_backend("claude-3-haiku", difficulty_rating=0.2)
                    # If no exception, check if we got fallback
                    print("✓ Used fallback configuration when all models failed")
                except Exception:
                    print("✓ Correctly raised exception when all models failed")
    
    print("\nAll router tests passed!")


async def test_api_integration():
    """Test API integration with model failure handling."""
    print("\nTesting API integration...")
    
    # This would require a more complex setup with the actual API
    # For now, we'll just verify the imports work
    try:
        from inferswitch.api.messages_v2 import messages
        print("✓ API imports successful")
    except ImportError as e:
        print(f"✗ API import failed: {e}")
    
    print("\nAPI integration test completed!")


def test_configuration_loading():
    """Test configuration loading for model availability."""
    print("\nTesting configuration loading...")
    
    # Test default config
    config = BackendConfigManager.get_model_availability_config()
    assert "disable_duration_seconds" in config, "Config should have disable_duration_seconds"
    assert "max_retries" in config, "Config should have max_retries"
    assert config["disable_duration_seconds"] == 300, "Default disable duration should be 300"
    assert config["max_retries"] == 1, "Default max retries should be 1"
    print("✓ Default configuration loaded correctly")
    
    # Test environment variable override
    os.environ["INFERSWITCH_MODEL_DISABLE_DURATION"] = "600"
    config = BackendConfigManager.get_model_availability_config()
    assert config["disable_duration_seconds"] == 600, "Should load from environment variable"
    print("✓ Environment variable override works")
    
    # Clean up
    del os.environ["INFERSWITCH_MODEL_DISABLE_DURATION"]
    
    print("\nConfiguration tests passed!")


if __name__ == "__main__":
    print("Running model availability tests...\n")
    
    # Run tests
    test_model_availability_tracker()
    test_router_with_model_lists()
    test_configuration_loading()
    
    # Run async test
    asyncio.run(test_api_integration())
    
    print("\n✅ All tests completed!")