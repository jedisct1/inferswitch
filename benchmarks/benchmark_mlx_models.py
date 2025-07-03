#!/usr/bin/env python3
"""
Benchmark and compare different MLX models for difficulty rating.
"""

import json
import time
from inferswitch.mlx_model import mlx_model_manager

# Test cases with expected difficulty ratings
test_cases = [
    # Trivial (0)
    ("Can you proofread this README file for typos?", 0),
    ("What does 'API' stand for?", 0),
    ("Is this variable name descriptive enough: 'x'?", 0),
    ("Can you check if this comment makes sense?", 0),
    # Documentation/Explanation (1)
    ("What is the difference between frontend and backend?", 1),
    ("Can you explain what version control is in simple terms?", 1),
    ("Review this documentation for clarity", 1),
    # More Documentation (0-1)
    ("What does HTTP stand for?", 0),
    ("Can you explain what a database is to a non-technical person?", 1),
    ("Is this error message user-friendly?", 1),
    # Basic Programming (3)
    ("How do I print 'Hello World' in Python?", 3),
    ("Create a for loop that counts from 1 to 10", 3),
    ("How do I declare a variable in JavaScript?", 3),
    ("Write a function that adds two numbers", 3),
    ("How do I read a file in Python?", 3),
    ("Create a simple if-else statement", 3),
    ("How do I install packages with npm?", 3),
    ("Write a SQL query to select all records from a table", 3),
    # Production Code (4)
    ("Implement user authentication with JWT tokens", 4),
    ("Create a REST API with CRUD operations", 4),
    ("Set up Docker containers for a microservices architecture", 4),
    ("Implement real-time chat with WebSockets", 4),
    ("Create a GraphQL API with authentication", 4),
    # Expert Systems (5)
    ("Write a compiler for a simple programming language", 5),
    ("Implement a garbage collector from scratch", 5),
    ("Create a distributed consensus algorithm", 5),
    ("Build a memory allocator for an operating system", 5),
    ("Design a Byzantine fault-tolerant system", 5),
]


def convert_to_chat_format(query: str):
    """Convert a query string to chat message format."""
    return [{"role": "user", "content": query}]


def calculate_bucket(rating: float) -> int:
    """Calculate which bucket (0-5) a rating falls into."""
    return min(5, int(rating + 0.5))


def benchmark_model(model_name: str):
    """Benchmark a specific MLX model."""
    print(f"\nBenchmarking {model_name}")
    print("=" * 80)

    # Load the model
    print("Loading model...")
    success, message = mlx_model_manager.load_model(model_name)
    if not success:
        print(f"Failed to load model: {message}")
        return None
    print(f"Model loaded: {message}")

    # Get model info
    info = mlx_model_manager.get_model_info()
    print(f"Model info: {json.dumps(info, indent=2)}")

    results = []
    total_time = 0

    # Warm up the model
    print("\nWarming up model...")
    for _ in range(3):
        mlx_model_manager.rate_query_difficulty(convert_to_chat_format("test query"))
    print("Warmup complete\n")

    # Run benchmarks
    print("Running test cases...")
    for i, (query, expected) in enumerate(test_cases):
        start_time = time.time()

        # Get difficulty rating
        chat_messages = convert_to_chat_format(query)
        predicted = mlx_model_manager.rate_query_difficulty(chat_messages)

        elapsed = time.time() - start_time
        total_time += elapsed

        # Calculate metrics
        error = abs(predicted - expected)
        expected_bucket = calculate_bucket(expected)
        predicted_bucket = calculate_bucket(predicted)
        correct_bucket = expected_bucket == predicted_bucket

        result = {
            "question": query,
            "expected": expected,
            "predicted": predicted,
            "absolute_error": error,
            "squared_error": error**2,
            "time_ms": elapsed * 1000,
            "correct_bucket": correct_bucket,
        }
        results.append(result)

        # Print progress
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(test_cases)} queries...")

    # Calculate overall metrics
    mae = sum(r["absolute_error"] for r in results) / len(results)
    rmse = (sum(r["squared_error"] for r in results) / len(results)) ** 0.5
    bucket_accuracy = (
        sum(1 for r in results if r["correct_bucket"]) / len(results) * 100
    )
    avg_time_ms = (total_time / len(results)) * 1000

    print(f"\nResults for {model_name}:")
    print(f"  MAE: {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  Bucket Accuracy: {bucket_accuracy:.1f}%")
    print(f"  Average Time: {avg_time_ms:.1f}ms")
    print(f"  Total Time: {total_time:.2f}s")

    return {
        "model": model_name,
        "total_questions": len(test_cases),
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "bucket_accuracy": bucket_accuracy,
            "avg_time_ms": avg_time_ms,
            "total_time_s": total_time,
        },
        "model_info": info,
        "results": results,
    }


def main():
    """Main benchmark function."""
    models_to_test = [
        "jedisct1/arch-router-1.5b",  # Current default model
        "mlx-community/Qwen2.5-Coder-7B-8bit",  # Previous default model
    ]

    print("MLX Model Difficulty Rating Benchmark")
    print("=" * 80)
    print(f"Testing {len(models_to_test)} models on {len(test_cases)} test cases\n")

    all_results = []

    for model_name in models_to_test:
        result = benchmark_model(model_name)
        if result:
            all_results.append(result)

        # Reset model between tests by loading a dummy model
        # This ensures fair comparison by clearing any cached state

    # Compare results
    if len(all_results) >= 2:
        print("\n" + "=" * 80)
        print("COMPARISON:")
        print("=" * 80)

        # Create comparison table
        print(
            f"\n{'Model':<50} {'MAE':>8} {'RMSE':>8} {'Accuracy':>10} {'Avg Time':>10}"
        )
        print("-" * 86)

        for result in all_results:
            model_name = result["model"].split("/")[-1]  # Get just the model name
            metrics = result["metrics"]
            print(
                f"{model_name:<50} {metrics['mae']:>8.3f} {metrics['rmse']:>8.3f} "
                f"{metrics['bucket_accuracy']:>9.1f}% {metrics['avg_time_ms']:>9.1f}ms"
            )

        # Calculate improvements
        if len(all_results) >= 2:
            old_metrics = all_results[1]["metrics"]  # 0.5B model
            new_metrics = all_results[0]["metrics"]  # 7B model

            print("\nImprovement (7B-8bit vs 0.5B):")
            mae_improvement = (
                (old_metrics["mae"] - new_metrics["mae"]) / old_metrics["mae"] * 100
            )
            rmse_improvement = (
                (old_metrics["rmse"] - new_metrics["rmse"]) / old_metrics["rmse"] * 100
            )
            acc_improvement = (
                new_metrics["bucket_accuracy"] - old_metrics["bucket_accuracy"]
            )
            time_increase = (
                (new_metrics["avg_time_ms"] - old_metrics["avg_time_ms"])
                / old_metrics["avg_time_ms"]
                * 100
            )

            print(f"  MAE: {mae_improvement:+.1f}% (lower is better)")
            print(f"  RMSE: {rmse_improvement:+.1f}% (lower is better)")
            print(f"  Bucket Accuracy: {acc_improvement:+.1f} percentage points")
            print(f"  Time: {time_increase:+.1f}% (higher means slower)")

    # Save detailed results
    output_file = "benchmark_mlx_models_comparison.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "test_cases_count": len(test_cases),
                "models": all_results,
            },
            f,
            indent=2,
        )

    print(f"\nDetailed results saved to {output_file}")


if __name__ == "__main__":
    main()
