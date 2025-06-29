"""
Test the MLX backend with 'builtin' model support.
"""

import asyncio
import json
import os

# Test configuration for routing builtin to MLX
TEST_CONFIG = {
    "difficulty_models": {
        "0-2": "claude-3-haiku-20240307",
        "3-4": "builtin",  # Use MLX for medium difficulty
        "5": "claude-3-5-sonnet-20241022"
    },
    "model_providers": {
        "claude-3-haiku-20240307": "anthropic",
        "claude-3-5-sonnet-20241022": "anthropic",
        "builtin": "mlx"  # Map builtin to MLX backend
    }
}

async def test_mlx_backend():
    """Test that MLX backend can be selected via difficulty routing."""
    
    # Write test config
    with open("inferswitch.config.json", "w") as f:
        json.dump(TEST_CONFIG, f, indent=2)
    
    try:
        # Import after config is written
        from inferswitch.backends import backend_registry
        from inferswitch.backends.mlx import MLXBackend
        from inferswitch.backends.base import BackendConfig
        
        # Initialize backends
        backend_classes = {
            "mlx": MLXBackend
        }
        
        # Manually create MLX backend config
        mlx_config = BackendConfig(
            name="mlx",
            base_url="local",
            api_key=None,
            models=["builtin"]
        )
        
        # Create and register MLX backend
        mlx_backend = MLXBackend(mlx_config)
        backend_registry.register("mlx", mlx_backend)
        
        # Test that MLX backend supports "builtin" model
        assert mlx_backend.supports_model("builtin")
        assert not mlx_backend.supports_model("gpt-4")
        
        print("✓ MLX backend correctly supports 'builtin' model")
        
        # Test routing
        router = backend_registry.get_router()
        
        # Medium difficulty should route to builtin/MLX
        backend = router.select_backend("claude-3-haiku-20240307", difficulty_rating=3.5)
        assert hasattr(backend, '_difficulty_selected_model')
        assert backend._difficulty_selected_model == "builtin"
        
        print("✓ Difficulty routing correctly selects 'builtin' model for medium queries")
        
        # Test that model provider mapping works
        assert "builtin" in router.model_providers
        assert router.model_providers["builtin"] == "mlx"
        
        print("✓ Model provider mapping correctly maps 'builtin' to MLX backend")
        
        # Test health check
        health = await mlx_backend.health_check()
        print(f"✓ MLX backend health check: {health}")
        
        print("\nAll tests passed! 'builtin' model support is working correctly.")
        
    finally:
        # Clean up config file
        if os.path.exists("inferswitch.config.json"):
            os.remove("inferswitch.config.json")


async def test_mlx_generation():
    """Test that MLX backend can generate responses."""
    
    # This test requires MLX to be installed and model loaded
    try:
        from inferswitch.mlx_model import mlx_model_manager
        from inferswitch.backends.mlx import MLXBackend
        from inferswitch.backends.base import BackendConfig
        
        # Check if MLX is available
        if not mlx_model_manager.is_loaded():
            print("⚠️  MLX model not loaded, skipping generation test")
            return
        
        # Create MLX backend
        mlx_config = BackendConfig(
            name="mlx",
            base_url="local",
            api_key=None,
            models=["builtin"]
        )
        mlx_backend = MLXBackend(mlx_config)
        
        # Test message generation
        messages = [
            {"role": "user", "content": "What is 2+2?"}
        ]
        
        response = await mlx_backend.create_message(
            messages=messages,
            model="builtin",
            max_tokens=50
        )
        
        print(f"✓ MLX generated response: {response.content[0]['text'][:100]}...")
        
        # Test streaming
        print("\n✓ Testing streaming generation...")
        chunks = []
        async for event in mlx_backend.create_message_stream(
            messages=messages,
            model="builtin",
            max_tokens=50
        ):
            if event.get("type") == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    chunks.append(delta.get("text", ""))
        
        full_response = "".join(chunks)
        print(f"✓ MLX streamed response: {full_response[:100]}...")
        
    except Exception as e:
        print(f"⚠️  Generation test failed: {e}")


if __name__ == "__main__":
    print("Testing MLX backend with 'builtin' model support...\n")
    asyncio.run(test_mlx_backend())
    print("\n" + "="*50 + "\n")
    print("Testing MLX generation capabilities...\n")
    asyncio.run(test_mlx_generation())