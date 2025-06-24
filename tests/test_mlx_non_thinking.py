#!/usr/bin/env python3
"""
Test MLX classifier in non-thinking mode with more direct prompts.
"""

import time
from inferswitch.mlx_model import MLXModelManager

# Test questions with known difficulties
test_questions = [
    # Trivial (0)
    {"q": "What does API stand for?", "expected": 0},
    {"q": "Can you proofread this README for typos?", "expected": 0},
    
    # Documentation (1-2)
    {"q": "Explain what version control is in simple terms", "expected": 1},
    {"q": "What is the difference between frontend and backend?", "expected": 1},
    
    # Basic programming (3)
    {"q": "How do I print 'Hello World' in Python?", "expected": 3},
    {"q": "Write a for loop that counts from 1 to 10", "expected": 3},
    
    # Production code (4)
    {"q": "Implement user authentication with JWT tokens", "expected": 4},
    {"q": "Create a REST API with CRUD operations", "expected": 4},
    
    # Expert level (5)
    {"q": "Design a microservices architecture with proper communication", "expected": 5},
    {"q": "Implement a distributed caching system", "expected": 5},
]

def test_non_thinking_mode():
    """Test with more direct, completion-style prompts."""
    manager = MLXModelManager()
    
    # Load model
    print("Loading Qwen3-8B-4bit...")
    success, msg = manager.load_model("mlx-community/Qwen3-8B-4bit")
    if not success:
        print(f"Failed to load model: {msg}")
        return
    
    print("\nTesting non-thinking mode with direct prompts...")
    print("=" * 60)
    
    total_error = 0
    correct = 0
    
    for test in test_questions:
        question = test["q"]
        expected = test["expected"]
        
        # Test original method
        messages = [{"role": "user", "content": question}]
        start = time.time()
        predicted = manager.rate_query_difficulty(messages)
        time1 = (time.time() - start) * 1000
        
        error = abs(predicted - expected)
        total_error += error
        if round(predicted) == expected:
            correct += 1
            
        print(f"\nQ: {question[:50]}...")
        print(f"Expected: {expected}, Predicted: {predicted:.1f}")
        print(f"Time: {time1:.0f}ms")
    
    print("\n" + "=" * 60)
    print("Results:")
    print(f"  MAE: {total_error/len(test_questions):.2f}")
    print(f"  Accuracy: {correct}/{len(test_questions)} ({correct/len(test_questions)*100:.0f}%)")

if __name__ == "__main__":
    test_non_thinking_mode()
