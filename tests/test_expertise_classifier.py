#!/usr/bin/env python3
"""
Test script for expertise classifier functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inferswitch.expertise_classifier import ExpertiseClassifier


def test_pattern_classification():
    """Test pattern-based classification without MLX."""
    print("Testing pattern-based expertise classification...")
    
    classifier = ExpertiseClassifier()
    
    # Test vision patterns
    vision_queries = [
        "What do you see in this image?",
        "Analyze this screenshot",
        "Create a chart with matplotlib",
        "Draw a diagram showing the process",
        "Process this PNG file",
        "What's in this photo?",
        "Generate a visualization of the data",
        "Can you see what's happening in this picture?"
    ]
    
    # Test coding patterns  
    coding_queries = [
        "Write a Python function to sort a list",
        "How do I debug this JavaScript error?",
        "Create a REST API in Node.js",
        "Implement binary search in C++",
        "Fix this syntax error in my code",
        "How do I install this npm package?",
        "Create a React component for user authentication",
        "Write a database query in SQL",
        "How to deploy with Docker?"
    ]
    
    # Test math patterns
    math_queries = [
        "Calculate the derivative of x^2 + 3x + 1",
        "Solve this linear regression problem",
        "What's the probability of getting heads twice?",
        "Find the integral of sin(x)",
        "Calculate the mean and standard deviation",
        "Solve this system of equations",
        "What's the correlation coefficient?",
        "Optimize this function using gradient descent"
    ]
    
    # Test general patterns
    general_queries = [
        "Explain the history of the French Revolution",
        "What are the pros and cons of renewable energy?",
        "Write a short story about a dragon",
        "Tell me about quantum physics",
        "What's the difference between socialism and capitalism?",
        "Help me write an email to my boss",
        "Explain how photosynthesis works"
    ]
    
    test_cases = [
        (vision_queries, "vision"),
        (coding_queries, "coding"), 
        (math_queries, "math"),
        (general_queries, "general")
    ]
    
    total_passed = 0
    total_tests = 0
    
    for queries, expected_expertise in test_cases:
        passed = 0
        for query in queries:
            chat_messages = [{"role": "user", "content": query}]
            result = classifier.classify_expertise(chat_messages)
            
            if result == expected_expertise:
                passed += 1
                print(f"✓ {expected_expertise}: '{query[:40]}...' -> {result}")
            else:
                print(f"✗ {expected_expertise}: '{query[:40]}...' -> {result} (expected {expected_expertise})")
            
            total_tests += 1
        
        total_passed += passed
        accuracy = (passed / len(queries)) * 100
        print(f"{expected_expertise.upper()} accuracy: {passed}/{len(queries)} ({accuracy:.1f}%)")
        print()
    
    overall_accuracy = (total_passed / total_tests) * 100
    print(f"OVERALL ACCURACY: {total_passed}/{total_tests} ({overall_accuracy:.1f}%)")
    
    return overall_accuracy >= 70  # 70% accuracy threshold


def test_multimodal_detection():
    """Test multimodal query detection."""
    print("Testing multimodal detection...")
    
    classifier = ExpertiseClassifier()
    
    multimodal_queries = [
        "Create a Python script to analyze this image data and generate charts",
        "Build a dashboard that visualizes machine learning model performance",
        "Write code to process images and create statistical plots",
        "Analyze this photo using computer vision and show the results in a graph"
    ]
    
    passed = 0
    for query in multimodal_queries:
        chat_messages = [{"role": "user", "content": query}]
        result = classifier.classify_expertise(chat_messages)
        
        if result == "multimodal":
            passed += 1
            print(f"✓ Multimodal: '{query[:50]}...' -> {result}")
        else:
            print(f"✗ Multimodal: '{query[:50]}...' -> {result}")
    
    accuracy = (passed / len(multimodal_queries)) * 100
    print(f"Multimodal detection accuracy: {passed}/{len(multimodal_queries)} ({accuracy:.1f}%)")
    
    return accuracy >= 50  # 50% accuracy threshold for multimodal


def test_expertise_scores():
    """Test detailed expertise scoring."""
    print("Testing expertise scoring...")
    
    classifier = ExpertiseClassifier()
    
    # Test a coding query - should have high coding score
    coding_query = [{"role": "user", "content": "Write a Python function to implement quicksort algorithm"}]
    scores = classifier.get_expertise_scores(coding_query)
    
    print(f"Coding query scores: {scores}")
    
    if scores.get('coding', 0) > scores.get('general', 0):
        print("✓ Coding query scored higher for coding than general")
        coding_test_passed = True
    else:
        print("✗ Coding query did not score higher for coding than general")
        coding_test_passed = False
    
    # Test a math query - should have high math score
    math_query = [{"role": "user", "content": "Calculate the integral of x^2 from 0 to 5"}]
    scores = classifier.get_expertise_scores(math_query)
    
    print(f"Math query scores: {scores}")
    
    if scores.get('math', 0) > scores.get('general', 0):
        print("✓ Math query scored higher for math than general")
        math_test_passed = True
    else:
        print("✗ Math query did not score higher for math than general")
        math_test_passed = False
    
    return coding_test_passed and math_test_passed


def main():
    """Run all classifier tests."""
    print("=" * 60)
    print("EXPERTISE CLASSIFIER TESTS")
    print("=" * 60)
    
    all_passed = True
    
    # Test pattern classification
    if not test_pattern_classification():
        all_passed = False
    
    print()
    
    # Test multimodal detection
    if not test_multimodal_detection():
        all_passed = False
    
    print()
    
    # Test scoring
    if not test_expertise_scores():
        all_passed = False
    
    print()
    print("=" * 60)
    if all_passed:
        print("ALL CLASSIFIER TESTS PASSED! ✓")
    else:
        print("SOME CLASSIFIER TESTS FAILED! ✗")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())