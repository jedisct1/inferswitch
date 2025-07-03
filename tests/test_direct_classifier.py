#!/usr/bin/env python3
"""
Test direct MLX classifier approach.
"""

import time
import json
from inferswitch.mlx_model_direct import DirectMLXClassifier
from inferswitch.mlx_model import MLXModelManager

# Load test dataset
with open("llm_coding_questions.json", "r") as f:
    questions = json.load(f)

# Test first 20 questions
test_questions = questions[:20]


def test_both_approaches():
    """Compare original vs direct approach."""
    # Original approach
    original = MLXModelManager()
    success, msg = original.load_model("mlx-community/Qwen3-8B-4bit")
    if not success:
        print(f"Failed to load original: {msg}")
        return

    # Direct approach
    direct = DirectMLXClassifier()
    success, msg = direct.load_model("mlx-community/Qwen3-8B-4bit")
    if not success:
        print(f"Failed to load direct: {msg}")
        return

    print("Comparing Original vs Direct Approach")
    print("=" * 80)
    print(
        f"{'Question':<50} {'Exp':>4} {'Orig':>6} {'Direct':>8} {'O.Time':>8} {'D.Time':>8}"
    )
    print("=" * 80)

    original_mae = 0
    direct_mae = 0
    original_time = 0
    direct_time = 0

    for q in test_questions:
        question = q["question"][:50]
        expected = q["difficulty"]

        # Test original
        start = time.time()
        messages = [{"role": "user", "content": q["question"]}]
        orig_pred = original.rate_query_difficulty(messages)
        orig_time = (time.time() - start) * 1000

        # Test direct
        start = time.time()
        direct_pred = direct.rate_difficulty_direct(q["question"])
        dir_time = (time.time() - start) * 1000

        original_mae += abs(orig_pred - expected)
        direct_mae += abs(direct_pred - expected)
        original_time += orig_time
        direct_time += dir_time

        print(
            f"{question:<50} {expected:>4} {orig_pred:>6.1f} {direct_pred:>8.1f} {orig_time:>7.0f}ms {dir_time:>7.0f}ms"
        )

    n = len(test_questions)
    print("\n" + "=" * 80)
    print(f"Original MAE: {original_mae / n:.2f}, Avg Time: {original_time / n:.0f}ms")
    print(f"Direct MAE:   {direct_mae / n:.2f}, Avg Time: {direct_time / n:.0f}ms")
    print(f"Speedup: {original_time / direct_time:.1f}x")


if __name__ == "__main__":
    test_both_approaches()
