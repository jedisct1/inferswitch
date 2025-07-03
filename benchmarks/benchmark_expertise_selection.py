#!/usr/bin/env python3
"""
Benchmark MLX models for expertise selection accuracy.

This script compares different MLX models on their ability to correctly classify
queries into expert categories using the ExpertClassifier system.
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inferswitch.expertise_classifier import ExpertClassifier

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class ExpertiseBenchmark:
    """Benchmark expertise selection with different MLX models."""

    def __init__(self):
        self.classifier = ExpertClassifier()
        self.results = {}

    def load_expert_configs(self) -> Dict[str, Dict]:
        """Load expert definitions from config files."""
        configs = {}

        # Load domain experts config
        domain_config_path = (
            Path(__file__).parent.parent / "examples" / "domain_experts_config.json"
        )
        if domain_config_path.exists():
            with open(domain_config_path) as f:
                domain_config = json.load(f)
                configs["domain_experts"] = domain_config.get("expert_definitions", {})

        # Load custom experts config
        custom_config_path = (
            Path(__file__).parent.parent / "examples" / "custom_experts_config.json"
        )
        if custom_config_path.exists():
            with open(custom_config_path) as f:
                custom_config = json.load(f)
                configs["custom_experts"] = custom_config.get("expert_definitions", {})

        return configs

    def create_test_dataset(self) -> List[Dict]:
        """Create comprehensive test dataset for expertise classification."""
        return [
            # Domain Experts - Medical AI
            {
                "query": "What are the potential side effects of combining ACE inhibitors with diuretics?",
                "expected_expert": "medical_ai",
                "config_type": "domain_experts",
                "category": "medical",
            },
            {
                "query": "Interpret this chest X-ray showing bilateral infiltrates",
                "expected_expert": "medical_ai",
                "config_type": "domain_experts",
                "category": "medical",
            },
            {
                "query": "Design a clinical trial protocol for testing a new diabetes medication",
                "expected_expert": "medical_ai",
                "config_type": "domain_experts",
                "category": "medical",
            },
            # Domain Experts - Legal Advisor
            {
                "query": "Review this employment contract for potential legal issues",
                "expected_expert": "legal_advisor",
                "config_type": "domain_experts",
                "category": "legal",
            },
            {
                "query": "What are the GDPR compliance requirements for data processing?",
                "expected_expert": "legal_advisor",
                "config_type": "domain_experts",
                "category": "legal",
            },
            {
                "query": "Analyze the liability implications of this software license agreement",
                "expected_expert": "legal_advisor",
                "config_type": "domain_experts",
                "category": "legal",
            },
            # Domain Experts - Financial Analyst
            {
                "query": "Perform a discounted cash flow analysis for this technology startup",
                "expected_expert": "financial_analyst",
                "config_type": "domain_experts",
                "category": "financial",
            },
            {
                "query": "What's the optimal portfolio allocation for a risk-averse investor?",
                "expected_expert": "financial_analyst",
                "config_type": "domain_experts",
                "category": "financial",
            },
            {
                "query": "Analyze the financial impact of interest rate changes on REIT investments",
                "expected_expert": "financial_analyst",
                "config_type": "domain_experts",
                "category": "financial",
            },
            # Domain Experts - Technical Support
            {
                "query": "Troubleshoot network connectivity issues between VLANs",
                "expected_expert": "technical_support",
                "config_type": "domain_experts",
                "category": "technical",
            },
            {
                "query": "Debug this kernel panic on Ubuntu server",
                "expected_expert": "technical_support",
                "config_type": "domain_experts",
                "category": "technical",
            },
            {
                "query": "Configure SSL certificates for Apache web server",
                "expected_expert": "technical_support",
                "config_type": "domain_experts",
                "category": "technical",
            },
            # Custom Experts - Vision Specialist
            {
                "query": "Analyze this image and describe what you see in detail",
                "expected_expert": "vision_specialist",
                "config_type": "custom_experts",
                "category": "vision",
            },
            {
                "query": "Create a matplotlib visualization showing sales trends over time",
                "expected_expert": "vision_specialist",
                "config_type": "custom_experts",
                "category": "vision",
            },
            {
                "query": "Process this screenshot and extract the text content",
                "expected_expert": "vision_specialist",
                "config_type": "custom_experts",
                "category": "vision",
            },
            # Custom Experts - Code Architect
            {
                "query": "Design a microservices architecture for an e-commerce platform",
                "expected_expert": "code_architect",
                "config_type": "custom_experts",
                "category": "coding",
            },
            {
                "query": "Implement a scalable REST API with proper error handling and authentication",
                "expected_expert": "code_architect",
                "config_type": "custom_experts",
                "category": "coding",
            },
            {
                "query": "Refactor this monolithic application into a clean architecture pattern",
                "expected_expert": "code_architect",
                "config_type": "custom_experts",
                "category": "coding",
            },
            # Custom Experts - Data Scientist
            {
                "query": "Perform statistical significance testing on this A/B test dataset",
                "expected_expert": "data_scientist",
                "config_type": "custom_experts",
                "category": "data_science",
            },
            {
                "query": "Build a machine learning model to predict customer churn",
                "expected_expert": "data_scientist",
                "config_type": "custom_experts",
                "category": "data_science",
            },
            {
                "query": "Analyze correlation patterns in this time series financial data",
                "expected_expert": "data_scientist",
                "config_type": "custom_experts",
                "category": "data_science",
            },
            # Custom Experts - Creative Writer
            {
                "query": "Write an engaging blog post about sustainable technology trends",
                "expected_expert": "creative_writer",
                "config_type": "custom_experts",
                "category": "creative",
            },
            {
                "query": "Create compelling marketing copy for a new mobile app launch",
                "expected_expert": "creative_writer",
                "config_type": "custom_experts",
                "category": "creative",
            },
            {
                "query": "Develop a narrative storyline for a video game character",
                "expected_expert": "creative_writer",
                "config_type": "custom_experts",
                "category": "creative",
            },
            # Custom Experts - Research Assistant
            {
                "query": "Explain the historical context of the Industrial Revolution",
                "expected_expert": "research_assistant",
                "config_type": "custom_experts",
                "category": "research",
            },
            {
                "query": "Summarize recent research on climate change mitigation strategies",
                "expected_expert": "research_assistant",
                "config_type": "custom_experts",
                "category": "research",
            },
            {
                "query": "What are the key differences between quantum and classical computing?",
                "expected_expert": "research_assistant",
                "config_type": "custom_experts",
                "category": "research",
            },
            # Edge Cases - Multi-expert scenarios
            {
                "query": "Analyze this medical device patent for potential legal and technical issues",
                "expected_expert": "legal_advisor",  # Primary expected, but could be medical_ai or technical_support
                "config_type": "domain_experts",
                "category": "multi_expert",
            },
            {
                "query": "Create a Python script to visualize financial data and generate statistical reports",
                "expected_expert": "data_scientist",  # Could be code_architect or vision_specialist too
                "config_type": "custom_experts",
                "category": "multi_expert",
            },
        ]

    def convert_to_chat_format(self, query: str) -> List[Dict[str, str]]:
        """Convert query to chat message format."""
        return [{"role": "user", "content": query}]

    def benchmark_model(
        self, model_name: str, test_cases: List[Dict], expert_configs: Dict[str, Dict]
    ) -> Dict:
        """Benchmark a specific MLX model."""
        print(f"\nBenchmarking {model_name}")
        print("=" * 80)

        # Load the model
        print("Loading model...")
        success, message = self.classifier.load_model(model_name)
        if not success:
            print(f"Failed to load model: {message}")
            return None
        print(f"Model loaded: {message}")

        # Get model info
        info = self.classifier.get_model_info()
        print(f"Model info: {json.dumps(info, indent=2)}")

        results_by_config = {}

        # Test each expert configuration
        for config_name, expert_definitions in expert_configs.items():
            print(f"\nTesting with {config_name} expert definitions...")

            # Set expert definitions for this config
            self.classifier.set_expert_definitions(expert_definitions)

            # Filter test cases for this config
            config_test_cases = [
                tc for tc in test_cases if tc["config_type"] == config_name
            ]

            if not config_test_cases:
                print(f"No test cases found for {config_name}")
                continue

            # Warm up the model
            print("Warming up model...")
            for _ in range(3):
                self.classifier.classify_expert(
                    self.convert_to_chat_format("test query")
                )
            print("Warmup complete")

            results = []
            total_time = 0
            correct_classifications = 0

            print(f"Running {len(config_test_cases)} test cases...")
            for i, test_case in enumerate(config_test_cases):
                start_time = time.time()

                # Get expert classification
                chat_messages = self.convert_to_chat_format(test_case["query"])
                predicted_expert = self.classifier.classify_expert(chat_messages)

                # Get detailed scores
                expert_scores = self.classifier.get_expert_scores(chat_messages)

                elapsed = time.time() - start_time
                total_time += elapsed

                # Check if classification is correct
                is_correct = predicted_expert == test_case["expected_expert"]
                if is_correct:
                    correct_classifications += 1

                # Get confidence score for predicted expert
                confidence = (
                    expert_scores.get(predicted_expert, 0.0)
                    if predicted_expert
                    else 0.0
                )

                result = {
                    "query": test_case["query"],
                    "expected_expert": test_case["expected_expert"],
                    "predicted_expert": predicted_expert,
                    "category": test_case["category"],
                    "is_correct": is_correct,
                    "confidence": confidence,
                    "expert_scores": expert_scores,
                    "time_ms": elapsed * 1000,
                }
                results.append(result)

                # Print progress
                if (i + 1) % 5 == 0:
                    print(f"  Processed {i + 1}/{len(config_test_cases)} queries...")

            # Calculate metrics for this config
            accuracy = (correct_classifications / len(config_test_cases)) * 100
            avg_time_ms = (total_time / len(config_test_cases)) * 1000
            avg_confidence = sum(r["confidence"] for r in results) / len(results)

            # Calculate metrics by category
            category_metrics = {}
            for category in set(tc["category"] for tc in config_test_cases):
                category_results = [r for r in results if r["category"] == category]
                if category_results:
                    category_accuracy = (
                        sum(1 for r in category_results if r["is_correct"])
                        / len(category_results)
                    ) * 100
                    category_confidence = sum(
                        r["confidence"] for r in category_results
                    ) / len(category_results)
                    category_metrics[category] = {
                        "accuracy": category_accuracy,
                        "confidence": category_confidence,
                        "count": len(category_results),
                    }

            config_result = {
                "total_questions": len(config_test_cases),
                "correct_classifications": correct_classifications,
                "accuracy": accuracy,
                "avg_confidence": avg_confidence,
                "avg_time_ms": avg_time_ms,
                "total_time_s": total_time,
                "category_metrics": category_metrics,
                "results": results,
            }

            results_by_config[config_name] = config_result

            print(f"\nResults for {config_name}:")
            print(
                f"  Accuracy: {accuracy:.1f}% ({correct_classifications}/{len(config_test_cases)})"
            )
            print(f"  Average Confidence: {avg_confidence:.3f}")
            print(f"  Average Time: {avg_time_ms:.1f}ms")

            # Print category breakdown
            print("  Category Breakdown:")
            for category, metrics in category_metrics.items():
                print(
                    f"    {category}: {metrics['accuracy']:.1f}% accuracy, {metrics['confidence']:.3f} confidence ({metrics['count']} queries)"
                )

        return {
            "model": model_name,
            "model_info": info,
            "configurations": results_by_config,
        }

    def run_benchmark(self, models_to_test: List[str] = None) -> Dict:
        """Run the full expertise selection benchmark."""
        if models_to_test is None:
            models_to_test = [
                "jedisct1/arch-router-1.5b",  # Current default model
                "mlx-community/Qwen2.5-Coder-7B-8bit",  # Previous default model
            ]

        print("MLX Expertise Selection Benchmark")
        print("=" * 80)

        # Load configurations and test dataset
        expert_configs = self.load_expert_configs()
        test_cases = self.create_test_dataset()

        print(f"Loaded {len(expert_configs)} expert configurations:")
        for config_name, experts in expert_configs.items():
            print(f"  {config_name}: {list(experts.keys())}")

        print(f"Created {len(test_cases)} test cases")
        print(f"Testing {len(models_to_test)} models\n")

        all_results = []

        # Benchmark each model
        for model_name in models_to_test:
            result = self.benchmark_model(model_name, test_cases, expert_configs)
            if result:
                all_results.append(result)

        # Generate comparison
        if len(all_results) >= 2:
            self.print_comparison(all_results)

        # Save results
        output_file = "benchmark_expertise_selection_results.json"
        output_data = {
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "expert_configs": expert_configs,
            "test_cases_count": len(test_cases),
            "models": all_results,
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nDetailed results saved to {output_file}")

        return output_data

    def print_comparison(self, results: List[Dict]):
        """Print comparison between models."""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)

        # Overall comparison table
        print(
            f"\n{'Model':<40} {'Config':<20} {'Accuracy':<10} {'Confidence':<12} {'Avg Time':<10}"
        )
        print("-" * 92)

        for result in results:
            model_name = result["model"].split("/")[-1]  # Get just the model name

            for config_name, config_result in result["configurations"].items():
                print(
                    f"{model_name:<40} {config_name:<20} {config_result['accuracy']:>8.1f}% "
                    f"{config_result['avg_confidence']:>10.3f} {config_result['avg_time_ms']:>9.1f}ms"
                )

        # Detailed comparison if we have exactly 2 models
        if len(results) == 2:
            print(
                f"\nDetailed Comparison ({results[0]['model']} vs {results[1]['model']}):"
            )
            print("-" * 80)

            for config_name in results[0]["configurations"].keys():
                if config_name in results[1]["configurations"]:
                    config1 = results[0]["configurations"][config_name]
                    config2 = results[1]["configurations"][config_name]

                    acc_diff = config1["accuracy"] - config2["accuracy"]
                    conf_diff = config1["avg_confidence"] - config2["avg_confidence"]
                    time_diff = config1["avg_time_ms"] - config2["avg_time_ms"]

                    print(f"\n{config_name}:")
                    print(f"  Accuracy: {acc_diff:+.1f} percentage points")
                    print(f"  Confidence: {conf_diff:+.3f}")
                    print(f"  Time: {time_diff:+.1f}ms")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark MLX models for expertise selection"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to test (default: Qwen2.5-Coder-7B-8bit and arch-router-1.5b)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, force=True)

    benchmark = ExpertiseBenchmark()
    benchmark.run_benchmark(args.models)

    print("\nBenchmark completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
