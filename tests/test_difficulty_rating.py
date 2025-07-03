#!/usr/bin/env python3
"""Test the difficulty rating functionality."""

import asyncio
import httpx

# Test queries with expected difficulty ranges
test_queries = [
    {
        "query": "Can you proofread this README for typos?",
        "expected_range": (0, 0.5),
        "description": "Trivial - no programming required",
    },
    {
        "query": "What does API stand for?",
        "expected_range": (0, 0.5),
        "description": "Trivial - simple acronym",
    },
    {
        "query": "Explain what version control is in simple terms",
        "expected_range": (0.5, 1.5),
        "description": "Documentation - no code required",
    },
    {
        "query": "What is the difference between frontend and backend?",
        "expected_range": (1, 2),
        "description": "Explanation - conceptual understanding",
    },
    {
        "query": "How do I print 'Hello World' in Python?",
        "expected_range": (2.5, 3.5),
        "description": "Basic programming - simple code",
    },
    {
        "query": "Write a function that adds two numbers",
        "expected_range": (2.5, 3.5),
        "description": "Basic programming - fundamental concept",
    },
    {
        "query": "Implement user authentication with JWT tokens",
        "expected_range": (3.5, 4.5),
        "description": "Real-world pattern - production code",
    },
    {
        "query": "Create a REST API with CRUD operations",
        "expected_range": (3.5, 4.5),
        "description": "Common pattern - standard implementation",
    },
    {
        "query": "Design a microservices architecture with proper communication",
        "expected_range": (4.5, 5),
        "description": "Advanced - distributed systems",
    },
    {
        "query": "Build a compiler from scratch",
        "expected_range": (4.5, 5),
        "description": "Expert level - complex algorithm",
    },
]


async def test_difficulty_rating():
    """Test the difficulty rating feature."""
    print("Testing InferSwitch Difficulty Rating Feature\n")

    async with httpx.AsyncClient() as client:
        for test_case in test_queries:
            # Prepare the request
            request_data = {
                "model": "claude-3-opus-20240229",
                "messages": [{"role": "user", "content": test_case["query"]}],
                "max_tokens": 100,
            }

            headers = {
                "x-api-key": "test-key",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }

            try:
                # Make request to InferSwitch
                response = await client.post(
                    "http://localhost:1235/v1/messages",
                    json=request_data,
                    headers=headers,
                )

                # Check if we got the difficulty rating header
                difficulty_rating = response.headers.get("X-Difficulty-Rating")

                if difficulty_rating:
                    rating = float(difficulty_rating)
                    min_expected, max_expected = test_case["expected_range"]

                    print(f"Query: {test_case['query'][:50]}...")
                    print(f"Description: {test_case['description']}")
                    print(f"Difficulty Rating: {rating}")
                    print(f"Expected Range: {min_expected} - {max_expected}")

                    if min_expected <= rating <= max_expected:
                        print("✓ PASS: Rating within expected range")
                    else:
                        print("✗ FAIL: Rating outside expected range")

                    print("-" * 60)
                else:
                    print(
                        f"✗ FAIL: No difficulty rating header found for query: {test_case['query'][:50]}..."
                    )
                    print("-" * 60)

            except Exception as e:
                print(f"✗ ERROR testing query '{test_case['query'][:50]}...': {e}")
                print("-" * 60)


async def test_streaming_difficulty_rating():
    """Test difficulty rating with streaming responses."""
    print("\nTesting Streaming Response Difficulty Rating\n")

    async with httpx.AsyncClient() as client:
        request_data = {
            "model": "claude-3-opus-20240229",
            "messages": [
                {"role": "user", "content": "Explain how to implement a RESTful API"}
            ],
            "max_tokens": 100,
            "stream": True,
        }

        headers = {
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        try:
            response = await client.post(
                "http://localhost:1235/v1/messages", json=request_data, headers=headers
            )

            difficulty_rating = response.headers.get("X-Difficulty-Rating")

            if difficulty_rating:
                print(f"Streaming response difficulty rating: {difficulty_rating}")
                print("✓ PASS: Difficulty rating header present in streaming response")
            else:
                print("✗ FAIL: No difficulty rating header in streaming response")

        except Exception as e:
            print(f"✗ ERROR testing streaming: {e}")


async def check_request_log():
    """Check if difficulty ratings are being logged."""
    print("\nChecking Request Log for Difficulty Ratings\n")

    try:
        with open("requests.log", "r") as f:
            content = f.read()

        # Look for difficulty rating entries
        if "Difficulty Rating:" in content:
            print("✓ PASS: Difficulty ratings found in request log")

            # Show last few ratings
            lines = content.split("\n")
            rating_lines = [line for line in lines if "Difficulty Rating:" in line]

            print(f"\nLast {min(5, len(rating_lines))} difficulty ratings from log:")
            for line in rating_lines[-5:]:
                print(f"  {line.strip()}")
        else:
            print("✗ FAIL: No difficulty ratings found in request log")

    except FileNotFoundError:
        print("✗ FAIL: requests.log file not found")
    except Exception as e:
        print(f"✗ ERROR reading log: {e}")


async def main():
    """Run all tests."""
    print("=" * 70)
    print("InferSwitch Difficulty Rating Test Suite")
    print("=" * 70)
    print("\nMake sure InferSwitch is running on port 1235 before running tests")
    print("=" * 70 + "\n")

    await test_difficulty_rating()
    await test_streaming_difficulty_rating()
    await check_request_log()

    print("\n" + "=" * 70)
    print("Test suite completed")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
