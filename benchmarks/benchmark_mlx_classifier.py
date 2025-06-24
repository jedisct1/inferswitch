#!/usr/bin/env python3
"""
Benchmark the MLX difficulty classifier against labeled questions.
"""

import json
import time
from typing import List, Dict
import statistics
import argparse

# Import the MLX model manager
from inferswitch.mlx_model import mlx_model_manager, MLX_AVAILABLE


def load_test_data(file_path: str = "llm_coding_questions.json") -> List[Dict]:
    """Load the labeled questions from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []


def evaluate_single_question(question: str, expected_difficulty: float) -> Dict:
    """Evaluate a single question and return metrics."""
    start_time = time.time()
    
    # Create a chat message format for the classifier
    messages = [{"role": "user", "content": question}]
    
    # Get the predicted difficulty
    predicted_difficulty = mlx_model_manager.rate_query_difficulty(messages)
    
    end_time = time.time()
    
    # Calculate error metrics
    absolute_error = abs(predicted_difficulty - expected_difficulty)
    squared_error = (predicted_difficulty - expected_difficulty) ** 2
    
    return {
        "question": question,
        "expected": expected_difficulty,
        "predicted": predicted_difficulty,
        "absolute_error": absolute_error,
        "squared_error": squared_error,
        "time_ms": (end_time - start_time) * 1000,
        "correct_bucket": get_difficulty_bucket(predicted_difficulty) == get_difficulty_bucket(expected_difficulty)
    }


def get_difficulty_bucket(difficulty: float) -> str:
    """Map difficulty score to bucket name."""
    if difficulty < 0.5:
        return "trivial"
    elif difficulty < 1.5:
        return "documentation"
    elif difficulty < 2.5:
        return "explanation"
    elif difficulty < 3.5:
        return "basic_code"
    elif difficulty < 4.5:
        return "production_code"
    else:
        return "expert_code"


def print_results(results: List[Dict], verbose: bool = False):
    """Print benchmark results."""
    total = len(results)
    if total == 0:
        print("No results to display")
        return
    
    # Calculate metrics
    mae = statistics.mean([r["absolute_error"] for r in results])
    rmse = (statistics.mean([r["squared_error"] for r in results])) ** 0.5
    avg_time = statistics.mean([r["time_ms"] for r in results])
    correct_buckets = sum(1 for r in results if r["correct_bucket"])
    bucket_accuracy = (correct_buckets / total) * 100
    
    # Group by difficulty buckets
    bucket_results = {}
    for r in results:
        bucket = get_difficulty_bucket(r["expected"])
        if bucket not in bucket_results:
            bucket_results[bucket] = []
        bucket_results[bucket].append(r)
    
    print("\n" + "="*70)
    print("MLX Difficulty Classifier Benchmark Results")
    print("="*70)
    
    print(f"\nTotal questions evaluated: {total}")
    print(f"Average processing time: {avg_time:.2f} ms per question")
    print("\nOverall Metrics:")
    print(f"  Mean Absolute Error (MAE): {mae:.3f}")
    print(f"  Root Mean Square Error (RMSE): {rmse:.3f}")
    print(f"  Bucket Accuracy: {bucket_accuracy:.1f}% ({correct_buckets}/{total})")
    
    print("\nPer-Bucket Analysis:")
    print("-"*70)
    print(f"{'Bucket':<20} {'Count':<10} {'MAE':<10} {'RMSE':<10} {'Accuracy':<10}")
    print("-"*70)
    
    bucket_order = ["trivial", "documentation", "explanation", "basic_code", "production_code", "expert_code"]
    for bucket in bucket_order:
        if bucket in bucket_results:
            bucket_data = bucket_results[bucket]
            count = len(bucket_data)
            bucket_mae = statistics.mean([r["absolute_error"] for r in bucket_data])
            bucket_rmse = (statistics.mean([r["squared_error"] for r in bucket_data])) ** 0.5
            bucket_correct = sum(1 for r in bucket_data if r["correct_bucket"])
            bucket_acc = (bucket_correct / count) * 100
            
            print(f"{bucket:<20} {count:<10} {bucket_mae:<10.3f} {bucket_rmse:<10.3f} {bucket_acc:<10.1f}%")
    
    # Find worst predictions
    results_sorted = sorted(results, key=lambda x: x["absolute_error"], reverse=True)
    
    print("\n" + "-"*70)
    print("Top 10 Worst Predictions:")
    print("-"*70)
    print(f"{'Question':<50} {'Expected':<10} {'Predicted':<10} {'Error':<10}")
    print("-"*70)
    
    for r in results_sorted[:10]:
        question_short = r["question"][:47] + "..." if len(r["question"]) > 50 else r["question"]
        print(f"{question_short:<50} {r['expected']:<10.1f} {r['predicted']:<10.1f} {r['absolute_error']:<10.2f}")
    
    if verbose:
        print("\n" + "-"*70)
        print("All Results:")
        print("-"*70)
        for r in results:
            print(f"\nQuestion: {r['question']}")
            print(f"Expected: {r['expected']}, Predicted: {r['predicted']:.1f}, Error: {r['absolute_error']:.2f}")
            print(f"Bucket Match: {'✓' if r['correct_bucket'] else '✗'}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MLX difficulty classifier")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-3B-Instruct-4bit", 
                       help="MLX model to use for classification")
    parser.add_argument("--file", default="llm_coding_questions.json",
                       help="Path to the questions JSON file")
    parser.add_argument("--limit", type=int, help="Limit number of questions to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Show detailed results")
    parser.add_argument("--bucket", help="Only test questions from specific difficulty bucket")
    
    args = parser.parse_args()
    
    if not MLX_AVAILABLE:
        print("Error: MLX is not available on this system")
        print("Please install MLX with: pip install mlx-lm")
        return
    
    # Load the model
    print(f"Loading MLX model: {args.model}")
    success, message = mlx_model_manager.load_model(args.model)
    if not success:
        print(f"Failed to load model: {message}")
        return
    
    print(f"Model loaded successfully: {message}")
    
    # Load test data
    print(f"\nLoading test data from: {args.file}")
    test_data = load_test_data(args.file)
    
    if not test_data:
        print("No test data loaded")
        return
    
    # Filter by bucket if requested
    if args.bucket:
        original_count = len(test_data)
        test_data = [q for q in test_data if get_difficulty_bucket(q["difficulty"]) == args.bucket]
        print(f"Filtered to {len(test_data)} questions in bucket '{args.bucket}' (from {original_count} total)")
    
    # Limit if requested
    if args.limit:
        test_data = test_data[:args.limit]
        print(f"Limited to {len(test_data)} questions")
    
    print(f"Total questions to evaluate: {len(test_data)}")
    
    # Evaluate each question
    results = []
    print("\nEvaluating questions...")
    
    for i, item in enumerate(test_data):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(test_data)} ({(i/len(test_data))*100:.1f}%)", end='\r')
        
        result = evaluate_single_question(item["question"], item["difficulty"])
        results.append(result)
    
    print(f"Progress: {len(test_data)}/{len(test_data)} (100.0%)   ")
    
    # Print results
    print_results(results, verbose=args.verbose)
    
    # Save detailed results to file
    output_file = "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "model": args.model,
            "total_questions": len(results),
            "metrics": {
                "mae": statistics.mean([r["absolute_error"] for r in results]),
                "rmse": (statistics.mean([r["squared_error"] for r in results])) ** 0.5,
                "bucket_accuracy": (sum(1 for r in results if r["correct_bucket"]) / len(results)) * 100,
                "avg_time_ms": statistics.mean([r["time_ms"] for r in results])
            },
            "results": results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()