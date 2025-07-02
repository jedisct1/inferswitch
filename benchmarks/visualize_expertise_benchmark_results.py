#!/usr/bin/env python3
"""
Visualize and analyze expertise selection benchmark results.

This script generates charts and analysis from the benchmark results to help
understand model performance differences in expertise classification.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_benchmark_results(filename: str = "benchmark_expertise_selection_results.json") -> Dict:
    """Load benchmark results from JSON file."""
    results_path = Path(filename)
    if not results_path.exists():
        print(f"Results file not found: {filename}")
        print("Please run the benchmark first: python benchmark_expertise_selection.py")
        return None
    
    with open(results_path) as f:
        return json.load(f)

def create_accuracy_comparison_chart(results: Dict):
    """Create a bar chart comparing accuracy across models and configurations."""
    models = results['models']
    if len(models) < 2:
        print("Need at least 2 models for comparison")
        return
    
    # Collect data for plotting
    config_names = set()
    for model in models:
        config_names.update(model['configurations'].keys())
    config_names = sorted(list(config_names))
    
    model_names = [model['model'].split('/')[-1] for model in models]
    
    # Create data arrays
    accuracies = []
    for model in models:
        model_accuracies = []
        for config_name in config_names:
            if config_name in model['configurations']:
                model_accuracies.append(model['configurations'][config_name]['accuracy'])
            else:
                model_accuracies.append(0)
        accuracies.append(model_accuracies)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(config_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracies[0], width, label=model_names[0], alpha=0.8)
    bars2 = ax.bar(x + width/2, accuracies[1], width, label=model_names[1], alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Expert Configuration')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Expertise Classification Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(config_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('expertise_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_confidence_comparison_chart(results: Dict):
    """Create a bar chart comparing average confidence scores."""
    models = results['models']
    if len(models) < 2:
        return
    
    # Collect data
    config_names = set()
    for model in models:
        config_names.update(model['configurations'].keys())
    config_names = sorted(list(config_names))
    
    model_names = [model['model'].split('/')[-1] for model in models]
    
    confidences = []
    for model in models:
        model_confidences = []
        for config_name in config_names:
            if config_name in model['configurations']:
                model_confidences.append(model['configurations'][config_name]['avg_confidence'])
            else:
                model_confidences.append(0)
        confidences.append(model_confidences)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(config_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, confidences[0], width, label=model_names[0], alpha=0.8)
    bars2 = ax.bar(x + width/2, confidences[1], width, label=model_names[1], alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Expert Configuration')
    ax.set_ylabel('Average Confidence Score')
    ax.set_title('Average Confidence Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(config_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('expertise_confidence_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_category_performance_chart(results: Dict):
    """Create charts showing performance by category."""
    models = results['models']
    if len(models) < 2:
        return
    
    # Collect category data
    all_categories = set()
    for model in models:
        for config_name, config_data in model['configurations'].items():
            all_categories.update(config_data['category_metrics'].keys())
    
    all_categories = sorted(list(all_categories))
    model_names = [model['model'].split('/')[-1] for model in models]
    
    # Create subplots for each configuration
    configs = list(models[0]['configurations'].keys())
    
    fig, axes = plt.subplots(len(configs), 1, figsize=(14, 6*len(configs)))
    if len(configs) == 1:
        axes = [axes]
    
    for i, config_name in enumerate(configs):
        ax = axes[i]
        
        # Get categories for this config
        config_categories = set()
        for model in models:
            if config_name in model['configurations']:
                config_categories.update(model['configurations'][config_name]['category_metrics'].keys())
        config_categories = sorted(list(config_categories))
        
        if not config_categories:
            continue
        
        # Prepare data
        model_accuracies = []
        for model in models:
            accuracies = []
            if config_name in model['configurations']:
                category_metrics = model['configurations'][config_name]['category_metrics']
                for category in config_categories:
                    if category in category_metrics:
                        accuracies.append(category_metrics[category]['accuracy'])
                    else:
                        accuracies.append(0)
            else:
                accuracies = [0] * len(config_categories)
            model_accuracies.append(accuracies)
        
        # Create bars
        x = np.arange(len(config_categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, model_accuracies[0], width, label=model_names[0], alpha=0.8)
        bars2 = ax.bar(x + width/2, model_accuracies[1], width, label=model_names[1], alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.0f}%',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Category')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Category Performance - {config_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(config_categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('expertise_category_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_processing_time_comparison(results: Dict):
    """Create a chart comparing processing times."""
    models = results['models']
    if len(models) < 2:
        return
    
    # Collect timing data
    config_names = set()
    for model in models:
        config_names.update(model['configurations'].keys())
    config_names = sorted(list(config_names))
    
    model_names = [model['model'].split('/')[-1] for model in models]
    
    times = []
    for model in models:
        model_times = []
        for config_name in config_names:
            if config_name in model['configurations']:
                model_times.append(model['configurations'][config_name]['avg_time_ms'])
            else:
                model_times.append(0)
        times.append(model_times)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(config_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, times[0], width, label=model_names[0], alpha=0.8)
    bars2 = ax.bar(x + width/2, times[1], width, label=model_names[1], alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}ms',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Expert Configuration')
    ax.set_ylabel('Average Processing Time (ms)')
    ax.set_title('Processing Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(config_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('expertise_processing_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_analysis(results: Dict):
    """Print detailed textual analysis of the results."""
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    
    models = results['models']
    if len(models) < 2:
        print("Need at least 2 models for comparison")
        return
    
    model1, model2 = models[0], models[1]
    model1_name = model1['model'].split('/')[-1]
    model2_name = model2['model'].split('/')[-1]
    
    print(f"Comparing {model1_name} vs {model2_name}")
    print("-" * 60)
    
    # Overall statistics
    total_accuracy1 = 0
    total_accuracy2 = 0
    total_confidence1 = 0
    total_confidence2 = 0
    total_time1 = 0
    total_time2 = 0
    config_count = 0
    
    for config_name in model1['configurations'].keys():
        if config_name in model2['configurations']:
            config1 = model1['configurations'][config_name]
            config2 = model2['configurations'][config_name]
            
            total_accuracy1 += config1['accuracy']
            total_accuracy2 += config2['accuracy']
            total_confidence1 += config1['avg_confidence']
            total_confidence2 += config2['avg_confidence']
            total_time1 += config1['avg_time_ms']
            total_time2 += config2['avg_time_ms']
            config_count += 1
    
    if config_count > 0:
        avg_accuracy1 = total_accuracy1 / config_count
        avg_accuracy2 = total_accuracy2 / config_count
        avg_confidence1 = total_confidence1 / config_count
        avg_confidence2 = total_confidence2 / config_count
        avg_time1 = total_time1 / config_count
        avg_time2 = total_time2 / config_count
        
        print(f"\nOverall Performance:")
        print(f"  {model1_name}:")
        print(f"    Average Accuracy: {avg_accuracy1:.1f}%")
        print(f"    Average Confidence: {avg_confidence1:.3f}")
        print(f"    Average Time: {avg_time1:.1f}ms")
        print(f"  {model2_name}:")
        print(f"    Average Accuracy: {avg_accuracy2:.1f}%")
        print(f"    Average Confidence: {avg_confidence2:.3f}")
        print(f"    Average Time: {avg_time2:.1f}ms")
        
        print(f"\nDifferences ({model1_name} - {model2_name}):")
        print(f"    Accuracy: {avg_accuracy1 - avg_accuracy2:+.1f} percentage points")
        print(f"    Confidence: {avg_confidence1 - avg_confidence2:+.3f}")
        print(f"    Time: {avg_time1 - avg_time2:+.1f}ms")
    
    # Per-configuration analysis
    print(f"\nPer-Configuration Analysis:")
    for config_name in sorted(model1['configurations'].keys()):
        if config_name in model2['configurations']:
            config1 = model1['configurations'][config_name]
            config2 = model2['configurations'][config_name]
            
            print(f"\n{config_name}:")
            print(f"  Accuracy: {config1['accuracy']:.1f}% vs {config2['accuracy']:.1f}% "
                  f"({config1['accuracy'] - config2['accuracy']:+.1f})")
            print(f"  Confidence: {config1['avg_confidence']:.3f} vs {config2['avg_confidence']:.3f} "
                  f"({config1['avg_confidence'] - config2['avg_confidence']:+.3f})")
            print(f"  Time: {config1['avg_time_ms']:.1f}ms vs {config2['avg_time_ms']:.1f}ms "
                  f"({config1['avg_time_ms'] - config2['avg_time_ms']:+.1f}ms)")
    
    # Find best and worst performing categories
    print(f"\nCategory Performance Analysis:")
    category_diffs = defaultdict(list)
    
    for config_name in model1['configurations'].keys():
        if config_name in model2['configurations']:
            config1 = model1['configurations'][config_name]
            config2 = model2['configurations'][config_name]
            
            for category in config1['category_metrics'].keys():
                if category in config2['category_metrics']:
                    acc1 = config1['category_metrics'][category]['accuracy']
                    acc2 = config2['category_metrics'][category]['accuracy']
                    category_diffs[category].append(acc1 - acc2)
    
    # Calculate average differences by category
    avg_category_diffs = {}
    for category, diffs in category_diffs.items():
        avg_category_diffs[category] = sum(diffs) / len(diffs)
    
    # Sort by performance difference
    sorted_categories = sorted(avg_category_diffs.items(), key=lambda x: x[1], reverse=True)
    
    print(f"  Categories where {model1_name} performs better:")
    for category, diff in sorted_categories:
        if diff > 0:
            print(f"    {category}: +{diff:.1f} percentage points")
    
    print(f"  Categories where {model2_name} performs better:")
    for category, diff in sorted_categories:
        if diff < 0:
            print(f"    {category}: {diff:.1f} percentage points")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize expertise benchmark results")
    parser.add_argument("--results", "-r", default="benchmark_expertise_selection_results.json",
                        help="Results file to analyze")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots (text analysis only)")
    
    args = parser.parse_args()
    
    # Load results
    results = load_benchmark_results(args.results)
    if not results:
        return 1
    
    print(f"Loaded results from {args.results}")
    print(f"Test date: {results['test_date']}")
    print(f"Models tested: {len(results['models'])}")
    
    if not args.no_plots:
        try:
            print("\nGenerating visualization charts...")
            
            create_accuracy_comparison_chart(results)
            print("✓ Accuracy comparison chart saved as 'expertise_accuracy_comparison.png'")
            
            create_confidence_comparison_chart(results)
            print("✓ Confidence comparison chart saved as 'expertise_confidence_comparison.png'")
            
            create_category_performance_chart(results)
            print("✓ Category performance chart saved as 'expertise_category_performance.png'")
            
            create_processing_time_comparison(results)
            print("✓ Processing time chart saved as 'expertise_processing_time_comparison.png'")
            
        except ImportError:
            print("matplotlib not available, skipping plots")
            print("Install with: pip install matplotlib")
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    # Always print detailed analysis
    print_detailed_analysis(results)
    
    print("\nAnalysis complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())