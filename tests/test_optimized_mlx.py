#!/usr/bin/env python3
"""
Test the optimized MLX implementation.
"""

import time
from inferswitch.mlx_model import mlx_model_manager

# Test cases covering all difficulty levels
test_cases = [
    # Trivial (0)
    {"q": "What does API stand for?", "exp": 0},
    {"q": "Can you proofread this README for typos?", "exp": 0},
    
    # Simple explanation (1)
    {"q": "Explain what version control is in simple terms", "exp": 1},
    
    # Technical explanation (2)
    {"q": "What is the difference between frontend and backend?", "exp": 2},
    
    # Basic programming (3)
    {"q": "How do I print 'Hello World' in Python?", "exp": 3},
    {"q": "Create a for loop that counts from 1 to 10", "exp": 3},
    
    # Production code (4)
    {"q": "Implement user authentication with JWT tokens", "exp": 4},
    {"q": "Create a REST API with CRUD operations", "exp": 4},
    
    # Expert level (5)
    {"q": "Design a microservices architecture with proper communication", "exp": 5},
    {"q": "Implement a distributed caching system", "exp": 5},
]

print("Loading MLX model...")
success, msg = mlx_model_manager.load_model("mlx-community/Qwen3-8B-4bit")
if not success:
    print(f"Failed to load model: {msg}")
    exit(1)

print(f"Model loaded: {msg}")
print("\nTesting optimized implementation:")
print("=" * 70)
print(f"{'Question':<45} {'Exp':>4} {'Got':>4} {'Time':>10}")
print("=" * 70)

total_error = 0
correct = 0
total_time = 0

for test in test_cases:
    messages = [{"role": "user", "content": test["q"]}]
    
    start = time.time()
    predicted = mlx_model_manager.rate_query_difficulty(messages)
    elapsed = (time.time() - start) * 1000
    
    error = abs(predicted - test["exp"])
    total_error += error
    total_time += elapsed
    
    if round(predicted) == test["exp"]:
        correct += 1
        status = "✓"
    else:
        status = "✗"
    
    print(f"{test['q'][:45]:<45} {test['exp']:>4} {predicted:>4.1f} {elapsed:>9.0f}ms {status}")

n = len(test_cases)
print("\n" + "=" * 70)
print("Results:")
print(f"  MAE: {total_error/n:.2f}")
print(f"  Accuracy: {correct}/{n} ({correct/n*100:.0f}%)")
print(f"  Avg Time: {total_time/n:.0f}ms")
print("\nPerformance: Most queries handled by heuristics (0-1ms)")
print("Expert queries may use model (~500ms)")
