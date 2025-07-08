#!/usr/bin/env python3
"""
Test script to verify progress logging for long-running streaming responses.
"""

import asyncio
import json
import httpx
import time


async def test_long_streaming_request():
    """Send a request that should take more than 30 seconds to complete."""

    # Create a complex prompt that should generate a long response
    messages = [
        {
            "role": "user",
            "content": """Please write a very detailed, comprehensive tutorial about building a full-stack web application from scratch. 
            Include the following sections with extensive detail:
            
            1. Introduction and prerequisites (detailed explanation of required tools and knowledge)
            2. Setting up the development environment (step-by-step for Windows, Mac, and Linux)
            3. Backend development with Node.js and Express (complete CRUD API with authentication)
            4. Database design with PostgreSQL (schemas, relationships, migrations)
            5. Frontend development with React (components, hooks, state management)
            6. Styling with CSS and Tailwind (responsive design principles)
            7. Testing strategies (unit tests, integration tests, E2E tests)
            8. Deployment process (Docker, CI/CD, cloud platforms)
            9. Performance optimization techniques
            10. Security best practices
            11. Monitoring and logging
            12. Troubleshooting common issues
            
            For each section, provide:
            - Detailed explanations
            - Complete code examples
            - Common pitfalls and how to avoid them
            - Best practices and industry standards
            - Links to further resources
            
            Make this tutorial as comprehensive as possible, suitable for someone who wants to become a professional full-stack developer.""",
        }
    ]

    # Prepare the request
    url = "http://localhost:1235/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": "test-key",  # Use your actual API key
        "anthropic-version": "2023-06-01",
    }

    data = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": messages,
        "max_tokens": 4096,
        "stream": True,
    }

    print("Sending long-running streaming request...")
    print("Watch the requests.log file for progress updates every 30 seconds")
    print("-" * 60)

    start_time = time.time()
    total_tokens = 0

    async with httpx.AsyncClient(timeout=600.0) as client:
        async with client.stream("POST", url, json=data, headers=headers) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        event_data = json.loads(line[6:])
                        event_type = event_data.get("type", "")

                        if event_type == "content_block_delta":
                            delta = event_data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "")
                                total_tokens += len(text.split())
                                print(text, end="", flush=True)
                        elif event_type == "message_stop":
                            print("\n" + "-" * 60)
                            elapsed = time.time() - start_time
                            print("\nStreaming completed!")
                            print(f"Total time: {elapsed:.1f} seconds")
                            print(f"Approximate tokens: {total_tokens}")
                            print("Check requests.log for progress entries")

                    except json.JSONDecodeError:
                        pass


if __name__ == "__main__":
    print("Starting progress logging test...")
    print("Make sure the InferSwitch server is running on port 1235")
    print("This test will generate a very long response to trigger progress logging")
    print("")

    try:
        asyncio.run(test_long_streaming_request())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nError during test: {e}")
