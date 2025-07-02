#!/usr/bin/env python3
"""
Test script for custom expert-based routing functionality.
"""

import json
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inferswitch.expertise_classifier import ExpertClassifier
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


def test_expert_classifier_with_custom_definitions():
    """Test expert classifier with user-defined experts."""
    print("Testing expert classifier with custom definitions...")
    
    # Define custom experts
    expert_definitions = {
        "python_expert": "A Python programming specialist who excels at writing clean, efficient Python code, debugging issues, and implementing algorithms and data structures in Python.",
        "data_analyst": "A data analysis expert skilled in statistics, data visualization, analyzing datasets, creating reports, and extracting insights from data.",
        "content_writer": "A creative writing specialist focused on creating engaging content, blog posts, marketing copy, and storytelling.",
    }
    
    # Create classifier with custom definitions
    classifier = ExpertClassifier(expert_definitions)
    
    # Test cases mapping queries to expected experts
    test_cases = [
        {
            "query": "Write a Python function to calculate the Fibonacci sequence",
            "expected": "python_expert"
        },
        {
            "query": "Analyze this sales data and create a statistical summary report",
            "expected": "data_analyst"
        },
        {
            "query": "Write a compelling blog post about sustainable living",
            "expected": "content_writer"
        },
        {
            "query": "Debug this Python code that's throwing a TypeError",
            "expected": "python_expert"
        },
        {
            "query": "Create a data visualization showing quarterly revenue trends", 
            "expected": "data_analyst"
        }
    ]
    
    passed = 0
    failed = 0
    
    print(f"Testing with experts: {list(expert_definitions.keys())}")
    
    for i, case in enumerate(test_cases):
        chat_messages = [{"role": "user", "content": case["query"]}]
        
        # Test without MLX (should return None since no MLX model loaded)
        result = classifier.classify_expert(chat_messages)
        
        if result is None:
            print(f"✓ Test {i+1}: No MLX model - correctly returned None")
            print(f"  Query: {case['query'][:50]}...")
            passed += 1
        else:
            print(f"✗ Test {i+1}: Expected None (no MLX), got {result}")
            failed += 1
    
    # Test validation
    validation = classifier.validate_expert_definitions()
    if validation["valid"]:
        print("✓ Expert definitions validation passed")
        passed += 1
    else:
        print(f"✗ Expert definitions validation failed: {validation['issues']}")
        failed += 1
    
    print(f"\nCustom Expert Classification Results: {passed} passed, {failed} failed")
    return failed == 0


def test_expert_routing():
    """Test expert-based routing."""
    print("Testing expert-based routing...")
    
    # Create a temporary config file for testing
    config_data = {
        "expert_definitions": {
            "vision_ai": "An AI specialist focused on computer vision, image analysis, and visual content processing",
            "code_guru": "A programming expert specialized in writing high-quality code and solving complex technical problems",
            "general_assistant": "A general-purpose assistant for answering questions and providing information"
        },
        "expert_models": {
            "vision_ai": ["gpt-4-vision-preview"],
            "code_guru": ["claude-3-5-sonnet-20241022"],
            "general_assistant": ["claude-3-haiku-20240307"]
        },
        "model_providers": {
            "gpt-4-vision-preview": "openai",
            "claude-3-5-sonnet-20241022": "anthropic",
            "claude-3-haiku-20240307": "anthropic"
        }
    }
    
    # Write temporary config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f, indent=2)
        config_path = f.name
    
    try:
        # Temporarily replace the config file path
        original_config_file = Path("inferswitch.config.json")
        temp_config_file = Path(config_path)
        
        # Backup original if it exists
        backup_content = None
        if original_config_file.exists():
            with open(original_config_file) as f:
                backup_content = f.read()
        
        # Copy temp config to the expected location
        with open(temp_config_file) as src, open(original_config_file, 'w') as dst:
            dst.write(src.read())
        
        # Create mock backends
        backends = {
            "openai": MockBackend(
                BackendConfig(
                    name="openai",
                    base_url="https://api.openai.com",
                    api_key="test"
                ),
                supported_models=["gpt-4-vision-preview"]
            ),
            "anthropic": MockBackend(
                BackendConfig(
                    name="anthropic", 
                    base_url="https://api.anthropic.com",
                    api_key="test"
                ),
                supported_models=["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"]
            )
        }
        
        # Create router
        router = BackendRouter(backends)
        
        # Test expert routing
        test_cases = [
            ("vision_ai", "openai"),
            ("code_guru", "anthropic"), 
            ("general_assistant", "anthropic")
        ]
        
        passed = 0
        failed = 0
        
        for expert_name, expected_backend in test_cases:
            try:
                backend = router.select_backend(
                    model="test-model",
                    expert_name=expert_name
                )
                
                if backend.name == expected_backend:
                    print(f"✓ Expert {expert_name} -> {backend.name}")
                    passed += 1
                else:
                    print(f"✗ Expert {expert_name}: Expected {expected_backend}, got {backend.name}")
                    failed += 1
                    
            except Exception as e:
                print(f"✗ Expert {expert_name}: Exception - {e}")
                failed += 1
        
        print(f"\nExpert Routing Results: {passed} passed, {failed} failed")
        return failed == 0
        
    finally:
        # Restore original config
        if backup_content is not None:
            with open(original_config_file, 'w') as f:
                f.write(backup_content)
        elif original_config_file.exists():
            original_config_file.unlink()
        
        # Clean up temp file
        temp_config_file.unlink()


def test_config_loading():
    """Test configuration loading for expert definitions."""
    print("Testing configuration loading...")
    
    # Create a temporary config file
    config_data = {
        "expert_definitions": {
            "ai_researcher": "An AI research specialist with deep knowledge of machine learning, neural networks, and artificial intelligence research",
            "web_developer": "A full-stack web developer expert in modern web technologies, frameworks, and best practices"
        },
        "expert_models": {
            "ai_researcher": ["claude-opus-4-20250514", "claude-3-opus-20240229"],
            "web_developer": ["claude-3-5-sonnet-20241022", "qwen/qwen-2.5-coder-32b"]
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f, indent=2)
        config_path = f.name
    
    try:
        # Temporarily replace the config file
        original_config_file = Path("inferswitch.config.json")
        temp_config_file = Path(config_path)
        
        backup_content = None
        if original_config_file.exists():
            with open(original_config_file) as f:
                backup_content = f.read()
        
        with open(temp_config_file) as src, open(original_config_file, 'w') as dst:
            dst.write(src.read())
        
        # Test config loading
        expert_definitions = BackendConfigManager.get_expert_definitions()
        expert_models = BackendConfigManager.get_expert_model_mapping()
        
        passed = 0
        failed = 0
        
        # Test expert definitions
        if len(expert_definitions) == 2:
            print("✓ Expert definitions loaded correctly")
            passed += 1
        else:
            print(f"✗ Expected 2 expert definitions, got {len(expert_definitions)}")
            failed += 1
        
        if "ai_researcher" in expert_definitions and "web_developer" in expert_definitions:
            print("✓ All expected experts found in definitions")
            passed += 1
        else:
            print("✗ Missing expected experts in definitions")
            failed += 1
        
        # Test expert model mappings
        if len(expert_models) == 2:
            print("✓ Expert model mappings loaded correctly")
            passed += 1
        else:
            print(f"✗ Expected 2 expert model mappings, got {len(expert_models)}")
            failed += 1
        
        # Test routing mode detection
        routing_mode = BackendConfigManager.get_routing_mode()
        if routing_mode == "expert":
            print("✓ Routing mode correctly detected as 'expert'")
            passed += 1
        else:
            print(f"✗ Expected routing mode 'expert', got '{routing_mode}'")
            failed += 1
        
        print(f"\nConfiguration Loading Results: {passed} passed, {failed} failed")
        return failed == 0
        
    finally:
        # Restore original config
        if backup_content is not None:
            with open(original_config_file, 'w') as f:
                f.write(backup_content)
        elif original_config_file.exists():
            original_config_file.unlink()
        
        temp_config_file.unlink()


def main():
    """Run all custom expert tests."""
    print("=" * 60)
    print("CUSTOM EXPERT SYSTEM TESTS")
    print("=" * 60)
    
    all_passed = True
    
    # Test expert classifier
    if not test_expert_classifier_with_custom_definitions():
        all_passed = False
    
    print()
    
    # Test expert routing
    if not test_expert_routing():
        all_passed = False
    
    print()
    
    # Test config loading
    if not test_config_loading():
        all_passed = False
    
    print()
    print("=" * 60)
    if all_passed:
        print("ALL CUSTOM EXPERT TESTS PASSED! ✓")
    else:
        print("SOME CUSTOM EXPERT TESTS FAILED! ✗")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())