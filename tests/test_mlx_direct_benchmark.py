#!/usr/bin/env python3
"""
Direct test of MLX classifier benchmarking without running the server.
"""

import json
import sys
import os

# Add the parent directory to the path so we can import inferswitch modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inferswitch.mlx_model import MLXModelManager


def test_direct_classification():
    """Test the MLX classifier directly on some sample questions."""
    
    # Create a new model manager instance
    model_manager = MLXModelManager()
    
    # Try to load the model
    print("Loading MLX model...")
    success, message = model_manager.load_model("mlx-community/Qwen2.5-3B-Instruct-4bit")
    
    if not success:
        print(f"Failed to load model: {message}")
        return
    
    print(f"Model loaded: {message}")
    
    # Test questions covering all difficulty levels
    test_questions = [
        ("Can you proofread this README for typos?", 0),
        ("What does API stand for?", 0),
        ("Explain what version control is in simple terms", 1),
        ("What is the difference between frontend and backend?", 1),
        ("Describe how HTTP works", 2),
        ("How do I print 'Hello World' in Python?", 3),
        ("Write a function that adds two numbers", 3),
        ("Implement user authentication with JWT tokens", 4),
        ("Create a REST API with CRUD operations", 4),
        ("Design a microservices architecture", 5),
        ("Build a compiler from scratch", 5),
    ]
    
    print("\nTesting individual questions:")
    print("-" * 70)
    print(f"{'Question':<50} {'Expected':<10} {'Predicted':<10}")
    print("-" * 70)
    
    for question, expected in test_questions:
        messages = [{"role": "user", "content": question}]
        predicted = model_manager.rate_query_difficulty(messages)
        
        question_short = question[:47] + "..." if len(question) > 50 else question
        print(f"{question_short:<50} {expected:<10} {predicted:<10.1f}")
    
    print("\nTesting with full dataset...")
    
    # Load the full dataset
    try:
        with open("llm_coding_questions.json", 'r') as f:
            full_data = json.load(f)
    except FileNotFoundError:
        print("Error: llm_coding_questions.json not found")
        return
    
    # Test a sample of questions from each difficulty level
    difficulty_samples = {}
    for item in full_data:
        diff = int(item["difficulty"])
        if diff not in difficulty_samples:
            difficulty_samples[diff] = []
        if len(difficulty_samples[diff]) < 5:  # Take up to 5 samples per difficulty
            difficulty_samples[diff].append(item)
    
    print("\nSample results by difficulty level:")
    
    for diff in sorted(difficulty_samples.keys()):
        print(f"\n--- Difficulty {diff} ---")
        errors = []
        
        for item in difficulty_samples[diff]:
            messages = [{"role": "user", "content": item["question"]}]
            predicted = model_manager.rate_query_difficulty(messages)
            error = abs(predicted - item["difficulty"])
            errors.append(error)
            
            question_short = item["question"][:60] + "..." if len(item["question"]) > 60 else item["question"]
            print(f"Q: {question_short}")
            print(f"   Expected: {item['difficulty']}, Predicted: {predicted:.1f}, Error: {error:.2f}")
        
        avg_error = sum(errors) / len(errors) if errors else 0
        print(f"   Average error for difficulty {diff}: {avg_error:.2f}")


if __name__ == "__main__":
    test_direct_classification()