#!/usr/bin/env python3
"""Test that MLX classifier is skipped when all difficulty levels use the same model."""

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inferswitch.backends.router import BackendRouter
from inferswitch.backends.config import BackendConfigManager


def test_all_difficulty_models_are_same():
    """Test the all_difficulty_models_are_same method."""
    
    # Mock backends
    backends = {"anthropic": MagicMock(name="anthropic")}
    
    # Test case 1: All ranges use the same model
    with patch.object(BackendConfigManager, 'get_difficulty_model_mapping') as mock_diff:
        mock_diff.return_value = {
            (0.0, 0.3): ["claude-3-haiku"],
            (0.3, 0.6): ["claude-3-haiku"],
            (0.6, 1.0): ["claude-3-haiku"]
        }
        router = BackendRouter(backends)
        assert router.all_difficulty_models_are_same() == True
        print("✓ Test 1 passed: All ranges with same model returns True")
    
    # Test case 2: Different models for different ranges
    with patch.object(BackendConfigManager, 'get_difficulty_model_mapping') as mock_diff:
        mock_diff.return_value = {
            (0.0, 0.3): ["claude-3-haiku"],
            (0.3, 0.6): ["claude-3-sonnet"],
            (0.6, 1.0): ["claude-3-opus"]
        }
        router = BackendRouter(backends)
        assert router.all_difficulty_models_are_same() == False
        print("✓ Test 2 passed: Different models returns False")
    
    # Test case 3: Multiple models per range, but all the same
    with patch.object(BackendConfigManager, 'get_difficulty_model_mapping') as mock_diff:
        mock_diff.return_value = {
            (0.0, 0.3): ["claude-3-haiku", "claude-3-sonnet"],
            (0.3, 0.6): ["claude-3-haiku", "claude-3-sonnet"],
            (0.6, 1.0): ["claude-3-haiku", "claude-3-sonnet"]
        }
        router = BackendRouter(backends)
        assert router.all_difficulty_models_are_same() == True
        print("✓ Test 3 passed: Multiple models but same list returns True")
    
    # Test case 4: Empty difficulty models
    with patch.object(BackendConfigManager, 'get_difficulty_model_mapping') as mock_diff:
        mock_diff.return_value = {}
        router = BackendRouter(backends)
        assert router.all_difficulty_models_are_same() == False
        print("✓ Test 4 passed: Empty difficulty models returns False")


def test_mlx_skip_in_messages_endpoint():
    """Test that MLX classifier is skipped in messages endpoint when all models are same."""
    
    # Create temporary config file
    config_data = {
        "difficulty_models": {
            "0.0-0.3": ["claude-3-haiku"],
            "0.3-0.6": ["claude-3-haiku"],
            "0.6-1.0": ["claude-3-haiku"]
        },
        "model_providers": {
            "claude-3-haiku": "anthropic"
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        config_path = f.name
    
    try:
        # Set config path
        os.environ['INFERSWITCH_CONFIG_PATH'] = config_path
        
        # This test validates the logic by testing the router directly
        # since we can't easily test the full endpoint without a running server
        
        # Reinitialize to pick up new config
        BackendConfigManager._instance = None
        
        # Create router and verify optimization works
        backends = {"anthropic": MagicMock(name="anthropic")}
        router = BackendRouter(backends)
        
        # Should return True since all models are the same
        assert router.all_difficulty_models_are_same() == True
        
        print("✓ Test 5 passed: MLX skip optimization configured correctly")
            
    finally:
        # Clean up
        os.unlink(config_path)
        if 'INFERSWITCH_CONFIG_PATH' in os.environ:
            del os.environ['INFERSWITCH_CONFIG_PATH']


def test_mlx_not_skipped_with_different_models():
    """Test that MLX classifier is NOT skipped when models differ."""
    
    # Create temporary config file with different models
    config_data = {
        "difficulty_models": {
            "0.0-0.3": ["claude-3-haiku"],
            "0.3-0.6": ["claude-3-sonnet"],
            "0.6-1.0": ["claude-3-opus"]
        },
        "model_providers": {
            "claude-3-haiku": "anthropic",
            "claude-3-sonnet": "anthropic",
            "claude-3-opus": "anthropic"
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        config_path = f.name
    
    try:
        # Set config path
        os.environ['INFERSWITCH_CONFIG_PATH'] = config_path
        
        # Reinitialize to pick up new config
        BackendConfigManager._instance = None
        
        # Create router with mocked config
        backends = {"anthropic": MagicMock(name="anthropic")}
        
        # Mock the config to return different models
        with patch.object(BackendConfigManager, 'get_difficulty_model_mapping') as mock_diff:
            mock_diff.return_value = {
                (0.0, 0.3): ["claude-3-haiku"],
                (0.3, 0.6): ["claude-3-sonnet"],
                (0.6, 1.0): ["claude-3-opus"]
            }
            router = BackendRouter(backends)
            
            assert router.all_difficulty_models_are_same() == False
            print("✓ Test 6 passed: MLX not skipped with different models")
        
    finally:
        # Clean up
        os.unlink(config_path)
        if 'INFERSWITCH_CONFIG_PATH' in os.environ:
            del os.environ['INFERSWITCH_CONFIG_PATH']


if __name__ == "__main__":
    print("Testing MLX skip optimization...")
    test_all_difficulty_models_are_same()
    test_mlx_skip_in_messages_endpoint()
    test_mlx_not_skipped_with_different_models()
    print("\nAll tests passed! ✅")