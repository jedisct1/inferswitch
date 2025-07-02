#!/usr/bin/env python3
"""
Test script for expertise-based routing functionality.
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
from inferswitch.expertise_classifier import ExpertiseClassifier


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


def test_expertise_classification():
    """Test expertise classification patterns."""
    print("Testing expertise classification...")
    
    classifier = ExpertiseClassifier()
    
    test_cases = [
        # Vision tasks
        {
            "messages": [{"role": "user", "content": "What do you see in this image?"}],
            "expected": "vision"
        },
        {
            "messages": [{"role": "user", "content": "Analyze this screenshot and describe what's happening"}],
            "expected": "vision"
        },
        {
            "messages": [{"role": "user", "content": "Create a matplotlib chart showing sales data"}],
            "expected": "vision"
        },
        
        # Coding tasks
        {
            "messages": [{"role": "user", "content": "Write a Python function to sort a list"}],
            "expected": "coding"
        },
        {
            "messages": [{"role": "user", "content": "Help me debug this JavaScript error: TypeError"}],
            "expected": "coding"
        },
        {
            "messages": [{"role": "user", "content": "How do I create a REST API in Node.js?"}],
            "expected": "coding"
        },
        {
            "messages": [{"role": "user", "content": "Implement a binary search algorithm in C++"}],
            "expected": "coding"
        },
        
        # Math tasks
        {
            "messages": [{"role": "user", "content": "Calculate the derivative of x^2 + 3x + 1"}],
            "expected": "math"
        },
        {
            "messages": [{"role": "user", "content": "Solve this linear regression problem with statistical analysis"}],
            "expected": "math"
        },
        {
            "messages": [{"role": "user", "content": "What's the probability of getting heads twice in three coin flips?"}],
            "expected": "math"
        },
        
        # General tasks
        {
            "messages": [{"role": "user", "content": "Explain the history of the French Revolution"}],
            "expected": "general"
        },
        {
            "messages": [{"role": "user", "content": "What are the advantages and disadvantages of renewable energy?"}],
            "expected": "general"
        },
        {
            "messages": [{"role": "user", "content": "Write a short story about a dragon"}],
            "expected": "general"
        },
        
        # Multimodal tasks
        {
            "messages": [{"role": "user", "content": "Create a Python script to analyze this image data and generate a statistical chart"}],
            "expected": "multimodal"
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, case in enumerate(test_cases):
        try:
            result = classifier.classify_expertise(case["messages"])
            expected = case["expected"]
            
            if result == expected:
                print(f"✓ Test {i+1}: Expected {expected}, got {result}")
                passed += 1
            else:
                print(f"✗ Test {i+1}: Expected {expected}, got {result}")
                print(f"  Query: {case['messages'][0]['content'][:50]}...")
                failed += 1
                
        except Exception as e:
            print(f"✗ Test {i+1}: Exception - {e}")
            failed += 1
    
    print(f"\nExpertise Classification Results: {passed} passed, {failed} failed")
    return failed == 0


def test_expertise_routing():
    """Test expertise-based routing."""
    print("Testing expertise-based routing...")
    
    # Create a temporary config file for testing
    config_data = {
        "expertise_models": {
            "vision": ["gpt-4-vision-preview"],
            "coding": ["claude-3-5-sonnet-20241022"],
            "math": ["claude-3-opus-20240229"],
            "general": ["claude-3-haiku-20240307"],
            "multimodal": ["gpt-4-vision-preview", "claude-3-5-sonnet-20241022"]
        },
        "model_providers": {
            "gpt-4-vision-preview": "openai",
            "claude-3-5-sonnet-20241022": "anthropic",
            "claude-3-opus-20240229": "anthropic",
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
                supported_models=["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
            )
        }
        
        # Create router
        router = BackendRouter(backends)
        
        # Test cases: expertise area -> expected backend name
        test_cases = [
            ("vision", "openai"),
            ("coding", "anthropic"), 
            ("math", "anthropic"),
            ("general", "anthropic"),
            ("multimodal", "openai")  # Should pick first available model
        ]
        
        passed = 0
        failed = 0
        
        for expertise_area, expected_backend in test_cases:
            try:
                backend = router.select_backend(
                    model="test-model",
                    expertise_area=expertise_area
                )
                
                if backend.name == expected_backend:
                    print(f"✓ Expertise {expertise_area} -> {backend.name}")
                    passed += 1
                else:
                    print(f"✗ Expertise {expertise_area}: Expected {expected_backend}, got {backend.name}")
                    failed += 1
                    
            except Exception as e:
                print(f"✗ Expertise {expertise_area}: Exception - {e}")
                failed += 1
        
        print(f"\nExpertise Routing Results: {passed} passed, {failed} failed")
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


def test_expertise_scores():
    """Test detailed expertise scoring."""
    print("Testing expertise scoring...")
    
    classifier = ExpertiseClassifier()
    
    test_cases = [
        {
            "messages": [{"role": "user", "content": "Create a Python script to analyze image data and plot statistical charts"}],
            "should_have_high": ["coding", "vision", "math"]
        },
        {
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "should_have_high": ["general"]
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, case in enumerate(test_cases):
        try:
            scores = classifier.get_expertise_scores(case["messages"])
            high_areas = case["should_have_high"]
            
            print(f"Test {i+1} scores: {scores}")
            
            all_high = True
            for area in high_areas:
                if scores.get(area, 0) < 0.2:  # Threshold for "high" score
                    all_high = False
                    break
            
            if all_high:
                print(f"✓ Test {i+1}: All expected areas scored high")
                passed += 1
            else:
                print(f"✗ Test {i+1}: Not all expected areas scored high")
                failed += 1
                
        except Exception as e:
            print(f"✗ Test {i+1}: Exception - {e}")
            failed += 1
    
    print(f"\nExpertise Scoring Results: {passed} passed, {failed} failed")
    return failed == 0


def main():
    """Run all expertise tests."""
    print("=" * 50)
    print("EXPERTISE-BASED ROUTING TESTS")
    print("=" * 50)
    
    all_passed = True
    
    # Run classification tests
    if not test_expertise_classification():
        all_passed = False
    
    print()
    
    # Run routing tests
    if not test_expertise_routing():
        all_passed = False
    
    print()
    
    # Run scoring tests
    if not test_expertise_scores():
        all_passed = False
    
    print()
    print("=" * 50)
    if all_passed:
        print("ALL EXPERTISE TESTS PASSED! ✓")
    else:
        print("SOME EXPERTISE TESTS FAILED! ✗")
    print("=" * 50)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())