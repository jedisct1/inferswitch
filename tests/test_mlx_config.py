#!/usr/bin/env python3
"""Test MLX model configuration."""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inferswitch.backends.config import BackendConfigManager


def test_mlx_config():
    """Test MLX model configuration from various sources."""
    
    print("Testing MLX model configuration...")
    
    # Test 1: Default configuration
    print("\n1. Testing default configuration:")
    default_model = BackendConfigManager.get_mlx_model()
    print(f"   Default MLX model: {default_model}")
    assert default_model == "mlx-community/Qwen2.5-Coder-7B-8bit"
    
    # Test 2: Environment variable configuration
    print("\n2. Testing environment variable configuration:")
    os.environ["INFERSWITCH_MLX_MODEL"] = "test-model-from-env"
    env_model = BackendConfigManager.get_mlx_model()
    print(f"   MLX model from env: {env_model}")
    assert env_model == "test-model-from-env"
    del os.environ["INFERSWITCH_MLX_MODEL"]
    
    # Test 3: Config file configuration
    print("\n3. Testing config file configuration:")
    config_data = {
        "mlx_model": "test-model-from-config"
    }
    
    # Save current working directory and temporarily change it
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Create config file
        config_file = Path("inferswitch.config.json")
        with open(config_file, "w") as f:
            json.dump(config_data, f)
        
        # Test loading from config file
        file_model = BackendConfigManager.get_mlx_model()
        print(f"   MLX model from config file: {file_model}")
        assert file_model == "test-model-from-config"
        
        # Test 4: Environment variable takes precedence over config file
        print("\n4. Testing environment variable precedence:")
        os.environ["INFERSWITCH_MLX_MODEL"] = "env-override"
        override_model = BackendConfigManager.get_mlx_model()
        print(f"   MLX model (env overrides config): {override_model}")
        assert override_model == "env-override"
        del os.environ["INFERSWITCH_MLX_MODEL"]
        
        # Restore original working directory
        os.chdir(original_cwd)
    
    print("\n✅ All MLX configuration tests passed!")


if __name__ == "__main__":
    test_mlx_config()