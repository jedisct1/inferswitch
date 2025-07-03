#!/usr/bin/env python3
"""
Visualize benchmark results from the MLX classifier evaluation.
"""

import json
import sys
from typing import Dict, List


def load_results(file_path: str = "benchmark_results.json") -> Dict:
    """Load benchmark results from JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
        print("Please run benchmark_mlx_classifier.py first")
        sys.exit(1)


def create_confusion_matrix(results: List[Dict]) -> Dict[int, Dict[int, int]]:
    """Create a confusion matrix for difficulty predictions."""
    matrix = {}
    for i in range(6):
        matrix[i] = {j: 0 for j in range(6)}

    for r in results:
        expected = int(round(r["expected"]))
        predicted = int(round(r["predicted"]))
        # Clamp to valid range
        expected = max(0, min(5, expected))
        predicted = max(0, min(5, predicted))
        matrix[expected][predicted] += 1

    return matrix


def print_confusion_matrix(matrix: Dict[int, Dict[int, int]]):
    """Print a formatted confusion matrix."""
    print("\nConfusion Matrix (Expected vs Predicted):")
    print("-" * 70)
    print("     ", end="")
    for i in range(6):
        print(f"  {i}  ", end="")
    print("  Total")
    print("-" * 70)

    for expected in range(6):
        print(f"  {expected}  ", end="")
        total = 0
        for predicted in range(6):
            count = matrix[expected][predicted]
            total += count
            if expected == predicted:
                # Highlight correct predictions
                print(f"[{count:3}]", end="")
            else:
                print(f" {count:3} ", end="")
        print(f"   {total:3}")

    print("-" * 70)
    print("Total", end="")
    for predicted in range(6):
        total = sum(matrix[expected][predicted] for expected in range(6))
        print(f" {total:3} ", end="")
    print()


def analyze_errors(results: List[Dict]):
    """Analyze error patterns in the predictions."""
    # Group errors by type
    overestimates = []
    underestimates = []
    correct = []

    for r in results:
        error = r["predicted"] - r["expected"]
        if abs(error) < 0.5:
            correct.append(r)
        elif error > 0:
            overestimates.append((r, error))
        else:
            underestimates.append((r, -error))

    print("\nError Analysis:")
    print("-" * 70)
    print(
        f"Correct predictions (±0.5): {len(correct)} ({len(correct) / len(results) * 100:.1f}%)"
    )
    print(
        f"Overestimates: {len(overestimates)} ({len(overestimates) / len(results) * 100:.1f}%)"
    )
    print(
        f"Underestimates: {len(underestimates)} ({len(underestimates) / len(results) * 100:.1f}%)"
    )

    if overestimates:
        overestimates.sort(key=lambda x: x[1], reverse=True)
        print("\nTop 5 Overestimates:")
        for r, error in overestimates[:5]:
            print(f"  Q: {r['question'][:60]}...")
            print(
                f"     Expected: {r['expected']}, Predicted: {r['predicted']:.1f}, Error: +{error:.1f}"
            )

    if underestimates:
        underestimates.sort(key=lambda x: x[1], reverse=True)
        print("\nTop 5 Underestimates:")
        for r, error in underestimates[:5]:
            print(f"  Q: {r['question'][:60]}...")
            print(
                f"     Expected: {r['expected']}, Predicted: {r['predicted']:.1f}, Error: -{error:.1f}"
            )


def analyze_by_category(results: List[Dict]):
    """Analyze results by question category."""
    # Define categories based on keywords
    categories = {
        "trivial": ["proofread", "typo", "stand for", "acronym", "check"],
        "documentation": ["explain", "describe", "what is", "tell me"],
        "basic_programming": ["print", "hello world", "variable", "loop", "function"],
        "web_development": [
            "api",
            "rest",
            "crud",
            "jwt",
            "auth",
            "react",
            "javascript",
        ],
        "system_design": ["microservice", "distributed", "architect", "scale"],
        "algorithms": ["algorithm", "sort", "search", "tree", "graph"],
        "advanced": ["compiler", "interpreter", "garbage collector", "memory"],
    }

    category_stats = {}

    for r in results:
        question_lower = r["question"].lower()
        matched = False

        for category, keywords in categories.items():
            if any(keyword in question_lower for keyword in keywords):
                if category not in category_stats:
                    category_stats[category] = {"errors": [], "count": 0}

                category_stats[category]["errors"].append(r["absolute_error"])
                category_stats[category]["count"] += 1
                matched = True
                break

        if not matched:
            if "other" not in category_stats:
                category_stats["other"] = {"errors": [], "count": 0}
            category_stats["other"]["errors"].append(r["absolute_error"])
            category_stats["other"]["count"] += 1

    print("\nPerformance by Category:")
    print("-" * 70)
    print(f"{'Category':<20} {'Count':<10} {'Avg Error':<15} {'Max Error':<15}")
    print("-" * 70)

    for category in [
        "trivial",
        "documentation",
        "basic_programming",
        "web_development",
        "system_design",
        "algorithms",
        "advanced",
        "other",
    ]:
        if category in category_stats:
            stats = category_stats[category]
            avg_error = sum(stats["errors"]) / len(stats["errors"])
            max_error = max(stats["errors"])

            print(
                f"{category:<20} {stats['count']:<10} {avg_error:<15.3f} {max_error:<15.3f}"
            )


def main():
    """Main function to run all visualizations."""
    # Load results
    data = load_results()

    print("=" * 70)
    print("MLX Classifier Benchmark Results Analysis")
    print(f"Model: {data['model']}")
    print(f"Total Questions: {data['total_questions']}")
    print("=" * 70)

    print("\nOverall Metrics:")
    print(f"  Mean Absolute Error: {data['metrics']['mae']:.3f}")
    print(f"  Root Mean Square Error: {data['metrics']['rmse']:.3f}")
    print(f"  Bucket Accuracy: {data['metrics']['bucket_accuracy']:.1f}%")
    print(f"  Average Time: {data['metrics']['avg_time_ms']:.2f} ms")

    results = data["results"]

    # Create and print confusion matrix
    matrix = create_confusion_matrix(results)
    print_confusion_matrix(matrix)

    # Analyze errors
    analyze_errors(results)

    # Analyze by category
    analyze_by_category(results)

    # Distribution analysis
    print("\nDifficulty Distribution:")
    print("-" * 70)

    for i in range(6):
        expected_count = sum(1 for r in results if int(round(r["expected"])) == i)
        predicted_count = sum(1 for r in results if int(round(r["predicted"])) == i)

        print(f"Level {i}: Expected {expected_count:3}, Predicted {predicted_count:3}")


if __name__ == "__main__":
    main()
