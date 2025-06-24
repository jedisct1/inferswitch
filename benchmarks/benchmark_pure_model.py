#!/usr/bin/env python3
"""
Benchmark the pure model-based difficulty rating approach.
"""

import json
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inferswitch.mlx_model import mlx_model_manager as difficulty_rater

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


def main():
    print("Benchmarking Pure Model-Based Difficulty Rating\n")
    print("=" * 60)
    
    # Get rater info
    info = difficulty_rater.get_info()
    print(f"Strategy: {info['strategy']}")
    print(f"Model loaded: {info['loaded']}")
    print()
    
    results = []
    total_time = 0
    
    # Warm up the model
    print("Warming up model...")
    difficulty_rater.rate_query_difficulty(convert_to_chat_format("test query"))
    print("Warmup complete\n")
    
    # Run benchmarks
    for query, expected in test_cases:
        start_time = time.time()
        
        # Get difficulty rating
        chat_messages = convert_to_chat_format(query)
        predicted = difficulty_rater.rate_query_difficulty(chat_messages)
        
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
            "squared_error": error ** 2,
            "time_ms": elapsed * 1000,
            "correct_bucket": correct_bucket
        }
        results.append(result)
        
        # Print result
        status = "✓" if correct_bucket else "✗"
        print(f"{status} {query[:60]:<60} E:{expected} P:{predicted:.1f} T:{elapsed*1000:.1f}ms")
    
    # Calculate overall metrics
    print("\n" + "=" * 60)
    print("RESULTS:")
    
    mae = sum(r["absolute_error"] for r in results) / len(results)
    rmse = (sum(r["squared_error"] for r in results) / len(results)) ** 0.5
    bucket_accuracy = sum(1 for r in results if r["correct_bucket"]) / len(results) * 100
    avg_time_ms = (total_time / len(results)) * 1000
    
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"Bucket Accuracy: {bucket_accuracy:.1f}%")
    print(f"Average Time: {avg_time_ms:.1f}ms")
    
    # Show cache stats
    cache_stats = info.get('cache_stats', {})
    if cache_stats:
        print("\nCache Stats:")
        print(f"  Hits: {cache_stats.get('cache_hits', 0)}")
        print(f"  Misses: {cache_stats.get('cache_misses', 0)}")
        print(f"  Hit Rate: {cache_stats.get('hit_rate', 0):.1%}")
    
    # Save results
    output = {
        "strategy": "pure_model",
        "total_questions": len(test_cases),
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "bucket_accuracy": bucket_accuracy,
            "avg_time_ms": avg_time_ms
        },
        "cache_stats": cache_stats,
        "results": results
    }
    
    with open("benchmark_pure_model_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to benchmark_pure_model_results.json")


if __name__ == "__main__":
    main()