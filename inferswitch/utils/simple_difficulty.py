"""
Simple difficulty rating without MLX.
"""

import re
from typing import List, Dict


def rate_query_difficulty_simple(chat_messages: List[Dict[str, str]]) -> float:
    """
    Simple heuristic-based difficulty rating.

    Args:
        chat_messages: List of message dictionaries

    Returns:
        Difficulty rating from 0 to 5
    """
    # Extract the latest user query
    user_query = ""
    for msg in reversed(chat_messages):
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break

    if not user_query:
        return 2.5

    query_lower = user_query.lower()

    # Trivial patterns (0) - no programming required
    trivial_patterns = [
        (r"proofread", 0.0),
        (r"check.*typo", 0.0),
        (r"what does.*stand for", 0.0),
        (r"is.*name.*descriptive", 0.0),
        (r"review.*comment", 0.0),
    ]

    for pattern, rating in trivial_patterns:
        if re.search(pattern, query_lower):
            return rating

    # Documentation/explanation patterns (1-2) - no programming required
    if any(
        phrase in query_lower
        for phrase in ["explain", "what is", "describe", "tell me about"]
    ):
        if (
            "in simple terms" in query_lower
            or "layman" in query_lower
            or "non-technical" in query_lower
        ):
            return 1.0
        return 2.0

    # Check for "how do I" pattern which often indicates code
    if "how do i" in query_lower or "how to" in query_lower:
        if any(
            word in query_lower
            for word in ["print", "write", "create", "implement", "declare", "make"]
        ):
            # Advanced code tasks (5)
            if any(
                word in query_lower
                for word in [
                    "compiler",
                    "distributed",
                    "consensus",
                    "garbage collector",
                    "memory allocator",
                    "microservice",
                    "architecture",
                ]
            ):
                return 5.0
            # Real-world patterns (4)
            elif any(
                word in query_lower
                for word in [
                    "api",
                    "auth",
                    "jwt",
                    "oauth",
                    "database",
                    "crud",
                    "middleware",
                    "docker",
                    "react",
                ]
            ):
                return 4.0
            # Basic programming (3)
            else:
                return 3.0

    # Code-related patterns (3-5)
    code_indicators = [
        "write",
        "implement",
        "create",
        "build",
        "develop",
        "code",
        "function",
        "class",
        "algorithm",
        "program",
        "script",
    ]

    if any(word in query_lower for word in code_indicators):
        # Advanced code tasks (5)
        if any(
            word in query_lower
            for word in [
                "compiler",
                "distributed",
                "consensus",
                "garbage collector",
                "memory allocator",
                "microservice",
                "architecture",
                "from scratch",
            ]
        ):
            return 5.0
        # Real-world patterns (4)
        elif any(
            word in query_lower
            for word in [
                "api",
                "auth",
                "jwt",
                "oauth",
                "database",
                "crud",
                "middleware",
                "docker",
                "react",
            ]
        ):
            return 4.0
        # Basic programming (3)
        else:
            return 3.0

    # Default medium difficulty
    return 2.5
