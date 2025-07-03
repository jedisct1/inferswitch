#!/usr/bin/env python3
"""
Quick comparison of approaches.
"""

import time
from inferswitch.mlx_model import MLXModelManager
from inferswitch.mlx_model_optimized import OptimizedMLXClassifier

# Test cases
test_cases = [
    {"q": "What does API stand for?", "exp": 0},
    {"q": "How do I print hello world in Python?", "exp": 3},
    {"q": "Implement JWT authentication", "exp": 4},
    {"q": "Build a distributed cache system", "exp": 5},
]

print("Loading models...")
original = MLXModelManager()
original.load_model("mlx-community/Qwen3-8B-4bit")

optimized = OptimizedMLXClassifier()
optimized.load_model("mlx-community/Qwen3-8B-4bit")

print("\nComparison:")
print("=" * 70)

for test in test_cases:
    messages = [{"role": "user", "content": test["q"]}]

    # Original
    start = time.time()
    orig_pred = original.rate_query_difficulty(messages)
    orig_time = (time.time() - start) * 1000

    # Optimized
    start = time.time()
    opt_pred = optimized.rate_query_difficulty(messages)
    opt_time = (time.time() - start) * 1000

    print(f"\nQ: {test['q'][:40]}...")
    print(f"  Expected: {test['exp']}")
    print(f"  Original: {orig_pred:.1f} ({orig_time:.0f}ms)")
    print(
        f"  Optimized: {opt_pred:.1f} ({opt_time:.0f}ms) - {orig_time / opt_time:.1f}x faster"
    )
