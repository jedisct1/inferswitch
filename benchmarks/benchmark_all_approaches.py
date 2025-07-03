#!/usr/bin/env python3
"""
Benchmark all MLX classifier approaches.
"""

import json
import time
import sys
from typing import Dict, List

from inferswitch.mlx_model import MLXModelManager
from inferswitch.mlx_model_optimized import OptimizedMLXClassifier


def load_dataset(limit: int = 50) -> List[Dict]:
    """Load test dataset."""
    with open("llm_coding_questions.json", "r") as f:
        questions = json.load(f)
    return questions[:limit]


def evaluate_classifier(classifier, questions: List[Dict], name: str) -> Dict:
    """Evaluate a classifier on the dataset."""
    print(f"\nEvaluating {name}...")

    results = []
    total_time = 0

    for i, q in enumerate(questions):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(questions)}")

        start = time.time()

        # Prepare messages
        messages = [{"role": "user", "content": q["question"]}]

        # Get prediction
        if hasattr(classifier, "rate_query_difficulty"):
            predicted = classifier.rate_query_difficulty(messages)
        else:
            # For simple classifier
            predicted = classifier.rate_difficulty_direct(q["question"])

        elapsed = (time.time() - start) * 1000
        total_time += elapsed

        expected = q["difficulty"]
        error = abs(predicted - expected)

        results.append(
            {
                "question": q["question"],
                "expected": expected,
                "predicted": predicted,
                "error": error,
                "time_ms": elapsed,
                "correct_bucket": round(predicted) == round(expected),
            }
        )

    # Calculate metrics
    mae = sum(r["error"] for r in results) / len(results)
    rmse = (sum(r["error"] ** 2 for r in results) / len(results)) ** 0.5
    bucket_accuracy = sum(r["correct_bucket"] for r in results) / len(results) * 100
    avg_time = total_time / len(results)

    return {
        "name": name,
        "mae": mae,
        "rmse": rmse,
        "bucket_accuracy": bucket_accuracy,
        "avg_time_ms": avg_time,
        "total_time_ms": total_time,
        "results": results,
    }


def print_comparison(evaluations: List[Dict]):
    """Print comparison table."""
    print("\n" + "=" * 90)
    print("MLX Classifier Comparison")
    print("=" * 90)
    print(
        f"{'Approach':<25} {'MAE':>8} {'RMSE':>8} {'Accuracy':>12} {'Avg Time':>12} {'Speedup':>10}"
    )
    print("-" * 90)

    base_time = evaluations[0]["avg_time_ms"]

    for eval in evaluations:
        speedup = base_time / eval["avg_time_ms"]
        print(
            f"{eval['name']:<25} {eval['mae']:>8.3f} {eval['rmse']:>8.3f} "
            f"{eval['bucket_accuracy']:>11.1f}% {eval['avg_time_ms']:>11.0f}ms "
            f"{speedup:>9.1f}x"
        )

    print("\n" + "=" * 90)
    print("Performance by Difficulty Level")
    print("=" * 90)

    # Analyze by difficulty
    for approach in evaluations:
        print(f"\n{approach['name']}:")
        by_difficulty = {}

        for r in approach["results"]:
            level = round(r["expected"])
            if level not in by_difficulty:
                by_difficulty[level] = {"errors": [], "correct": 0, "total": 0}

            by_difficulty[level]["errors"].append(r["error"])
            by_difficulty[level]["total"] += 1
            if r["correct_bucket"]:
                by_difficulty[level]["correct"] += 1

        print(f"  {'Level':<10} {'Count':>8} {'MAE':>8} {'Accuracy':>12}")
        print("  " + "-" * 40)

        for level in sorted(by_difficulty.keys()):
            stats = by_difficulty[level]
            mae = sum(stats["errors"]) / len(stats["errors"])
            acc = stats["correct"] / stats["total"] * 100
            print(f"  {level:<10} {stats['total']:>8} {mae:>8.2f} {acc:>11.0f}%")


def main():
    """Run comprehensive benchmark."""
    # Load dataset
    print("Loading dataset...")
    questions = load_dataset(limit=50)
    print(f"Loaded {len(questions)} questions")

    # Initialize classifiers
    print("\nInitializing classifiers...")

    # Original approach
    original = MLXModelManager()
    success, msg = original.load_model("mlx-community/Qwen3-8B-4bit")
    if not success:
        print(f"Failed to load original: {msg}")
        sys.exit(1)

    # Optimized approach
    optimized = OptimizedMLXClassifier()
    success, msg = optimized.load_model("mlx-community/Qwen3-8B-4bit")
    if not success:
        print(f"Failed to load optimized: {msg}")
        sys.exit(1)

    # Run evaluations
    evaluations = []

    evaluations.append(
        evaluate_classifier(original, questions, "Original (Thinking Mode)")
    )
    evaluations.append(
        evaluate_classifier(optimized, questions, "Optimized (Non-Thinking)")
    )

    # Print results
    print_comparison(evaluations)

    # Save detailed results
    with open("benchmark_comparison.json", "w") as f:
        json.dump(evaluations, f, indent=2)
    print("\nDetailed results saved to benchmark_comparison.json")


if __name__ == "__main__":
    main()
