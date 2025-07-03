"""
MLX model management for InferSwitch.
"""

from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

# Try to import MLX - make it optional
try:
    import mlx_lm
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mlx_lm = None
    mx = None
    logger.warning("MLX not available. Difficulty rating will be disabled.")


class MLXModelManager:
    """Manages MLX language models."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None

    def load_model(
        self, model_name: str = "jedisct1/arch-router-1.5b"
    ) -> Tuple[bool, str]:
        """
        Load an MLX model and tokenizer.

        Args:
            model_name: HuggingFace model identifier

        Returns:
            Tuple of (success, message)
        """
        if not MLX_AVAILABLE:
            return False, "MLX is not available on this system"

        try:
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(f"Attempting to load MLX model: {model_name}")

            # Load model and tokenizer
            self.model, self.tokenizer = mlx_lm.load(model_name)
            self.model_name = model_name

            logger.debug(f"MLX model {model_name} loaded successfully")
            return True, f"Model {model_name} loaded successfully"

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            error_msg = f"Failed to load MLX model {model_name}: {str(e)}"
            logger.error(error_msg)
            # Don't crash - just disable difficulty rating
            self.model = None
            self.tokenizer = None
            return False, error_msg

    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.model is not None and self.tokenizer is not None

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self.is_loaded():
            return {"loaded": False, "model": None}

        return {
            "loaded": True,
            "model": self.model_name,
            "tokenizer_type": type(self.tokenizer).__name__,
        }

    def rate_query_difficulty(self, chat_messages: List[Dict[str, str]]) -> float:
        """
        Rate the difficulty of a query from 0 (trivial) to 5 (very hard).

        Args:
            chat_messages: List of message dictionaries in chat template format

        Returns:
            Difficulty rating from 0 to 5
        """
        import logging

        logger = logging.getLogger(__name__)

        if not self.is_loaded():
            logger.warning(
                "MLX model not loaded, returning default difficulty rating 2.5"
            )
            return 2.5

        try:
            # Construct a prompt for difficulty rating
            # Extract the latest user query
            user_query = ""
            for msg in reversed(chat_messages):
                if msg.get("role") == "user":
                    user_query = msg.get("content", "")
                    break

            logger.debug(
                f"Extracted user query for difficulty rating: {user_query[:100]}..."
            )

            if not user_query:
                logger.warning(
                    "No user query found in messages, returning default difficulty 2.5"
                )
                return 2.5

            # Clean up the query - remove XML tags and extra whitespace
            import re

            # Remove common XML tags
            cleaned_query = re.sub(
                r"</?(?:task|environment_details|slug|name|model)[^>]*>", "", user_query
            )
            # Remove multiple newlines and extra spaces
            cleaned_query = re.sub(r"\n+", " ", cleaned_query).strip()
            # If we have environment details, just take the first part
            if len(cleaned_query) > 200:
                cleaned_query = cleaned_query[:200]

            logger.debug(f"Original query: {user_query[:100]}...")
            logger.debug(f"Cleaned query: {cleaned_query}")

            # Create a prompt that rates based on AI model capabilities
            # Analyze query characteristics without keywords
            query_lower = cleaned_query.lower()

            # Determine if the query asks for code/implementation
            # First check if it's asking for explanation/information
            info_keywords = [
                "what is",
                "how does",
                "explain",
                "tell me",
                "describe",
                "what are",
            ]
            is_info_query = any(phrase in query_lower for phrase in info_keywords)

            # Check for "how do I" or "how to" which often indicates implementation
            how_to_pattern = r"\bhow\s+(do\s+i|to)\b"
            has_how_to = re.search(how_to_pattern, query_lower) is not None

            # Improved code detection
            if is_info_query and not has_how_to:
                # Pure explanation/information query
                requires_code = False
            else:
                # Check for action verbs that indicate coding tasks
                code_indicators = [
                    r"\b(write|implement|create|build|develop|make|code|program)\s+(a|an|the|some)\s+",
                    r"\b(write|implement|create|build|develop|make)\s+.*(function|program|script|code|app|application|tool|system)",
                    r"\b(write|implement|create|build|develop|make|code|program)\s+\w+\s+(in|using|with)\s+(python|javascript|java|c\+\+|rust|go|ruby|php)",
                    r"\b(implement|create|build|write|develop)\s+[A-Z]\w*",
                    r"\bhow\s+(do\s+i|to)\s+\w*\s*(print|declare|create|write|implement|build|make|code)",  # "How do I print"
                    r"\b(print|output|display|show)\s+.*\s+(in|using|with)\s+(python|javascript|java)",  # "print hello world in Python"
                ]

                requires_code = any(
                    re.search(pattern, query_lower) for pattern in code_indicators
                )

                # Special case: "How do I" + programming verb almost always requires code
                if has_how_to and any(
                    verb in query_lower
                    for verb in [
                        "print",
                        "write",
                        "create",
                        "implement",
                        "declare",
                        "define",
                        "make",
                        "build",
                        "code",
                        "program",
                        "read",
                        "install",
                        "use",
                        "set up",
                        "handle",
                    ]
                ):
                    requires_code = True

                logger.debug(
                    f"Code detection - Query: {query_lower[:50]}... Info: {is_info_query}, How-to: {has_how_to}, Requires code: {requires_code}"
                )

            # Check for expert-level indicators
            expert_keywords = [
                "compiler",
                "interpreter",
                "garbage collector",
                "memory allocator",
                "distributed",
                "consensus",
                "microservice",
                "architecture",
                "design.*system",
                "build.*from scratch",
                "custom.*algorithm",
                "implement.*protocol",
                "crdt",
                "raft",
                "paxos",
                "byzantine",
            ]

            is_expert_level = any(
                re.search(keyword, query_lower) for keyword in expert_keywords
            )

            # If it requires writing code, minimum difficulty is 3
            if requires_code:
                # Let the model determine between 3-5
                min_difficulty = 3
                # But if it has expert keywords, suggest minimum 5
                if is_expert_level:
                    min_difficulty = 4.5  # Allow some flexibility but bias towards 5
                    logger.debug(
                        f"Expert-level task detected -> minimum difficulty: {min_difficulty}"
                    )
                # Check for specific difficulty 4 patterns - expanded list
                production_keywords = [
                    "jwt",
                    "oauth",
                    "api",
                    "crud",
                    "authentication",
                    "docker",
                    "middleware",
                    "websocket",
                    "graphql",
                    "database schema",
                    "pagination",
                    "validation",
                    "error handling",
                    "deployment",
                    "ci/cd",
                    "testing",
                    "webpack",
                    "file upload",
                    "rate limit",
                    "async/await",
                    "promise",
                    "callback",
                    "event",
                    "streaming",
                ]
                if (
                    any(word in query_lower for word in production_keywords)
                    or "rest api" in query_lower
                    or "react component" in query_lower
                ):
                    min_difficulty = 3.5  # Bias towards 4
                    logger.debug(
                        f"Production-level task detected -> minimum difficulty: {min_difficulty}"
                    )
                else:
                    logger.debug(
                        f"Query requires code implementation -> minimum difficulty: {min_difficulty}"
                    )
            else:
                # For non-code queries (explanations, simple questions)
                min_difficulty = 0
                logger.debug(
                    f"Query is informational/simple -> minimum difficulty: {min_difficulty}"
                )

            # Create a prompt that helps the model understand task complexity
            if requires_code:
                # For code tasks, provide better examples from 3-5
                # Add hint about what to look for
                if (
                    "implement" in query_lower
                    or "create" in query_lower
                    or "build" in query_lower
                ):
                    action_hint = "IMPLEMENT/CREATE/BUILD tasks are usually 4 or 5."
                else:
                    action_hint = ""

                prompt = f"""Rate coding difficulty (3, 4, or 5 ONLY):

3 = STUDENT/BEGINNER (first month of coding):
- Print hello world
- Basic loops (for i in range)
- If/else statements
- Define simple function
- Basic math operations

4 = PROFESSIONAL (paid to code):
- Build/implement/create systems
- APIs, databases, authentication
- Real applications
- Production features
- Integration, deployment
- Any "implement X" task

5 = EXPERT (rare specialists):
- Distributed/concurrent systems  
- Compilers/interpreters
- OS/kernel development
- Novel algorithms
- Architecture design

{action_hint}
IMPORTANT: Most "implement/create/build" tasks are 4+
Task: {cleaned_query[:100]}
Answer with ONLY the number (3, 4, or 5):"""
            else:
                # For non-code tasks, clearer examples
                prompt = f"""Rate query difficulty (0-5):

0 = TRIVIAL (one word/phrase answer):
- What does API stand for?
- Check typos
- Is this name good?

1 = SIMPLE (basic explanation):
- Explain X in simple terms
- What is Y?
- Review clarity

2 = DETAILED (technical explanation):
- Compare X vs Y
- Explain complex concept
- Describe architecture

3 = BASIC CODE (simple programming):
- Print hello world
- Basic loops
- Simple functions

4 = PROFESSIONAL CODE (real systems):
- Build APIs
- Implement auth
- Database work

5 = EXPERT CODE (specialized):
- Distributed systems
- Compilers
- Advanced algorithms

Query: {cleaned_query[:100]}
Answer with ONLY the number (0-5):"""

            # Generate just a few tokens for the rating
            try:
                logger.debug(f"MLX prompt: {prompt}")
                response = mlx_lm.generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    max_tokens=3,  # Allow for " 5" or similar responses
                    verbose=False,
                )
                logger.debug(f"MLX generation response: {repr(response)}")

            except Exception as e:
                logger.error(f"Error during MLX generation: {str(e)}", exc_info=True)
                return 2.5

            # Extract the rating from the response
            try:
                import re

                # Clean the response first
                clean_response = response.strip()

                # The response should start with a number 0-5
                # Look for a number at the beginning of the response
                match = re.search(r"^\s*(\d(?:\.\d)?)", clean_response)
                if match:
                    rating = float(match.group(1))
                    # Clamp to valid range
                    if rating > 5:
                        rating = 5.0
                    elif rating < 0:
                        rating = 0.0

                    # Enforce minimum difficulty for code tasks
                    if requires_code and rating < min_difficulty:
                        logger.debug(
                            f"Model rated {rating}, but enforcing minimum {min_difficulty} for code task"
                        )
                        rating = float(min_difficulty)

                    logger.debug(f"Final difficulty rating: {rating}")
                    return rating

                # If no number at start, look for first occurrence of 0-5
                numbers = re.findall(r"[0-5](?:\.\d)?", clean_response)
                if numbers:
                    rating = float(numbers[0])
                    rating = max(0.0, min(5.0, rating))

                    # Enforce minimum difficulty for code tasks
                    if requires_code and rating < min_difficulty:
                        logger.debug(
                            f"Model rated {rating}, but enforcing minimum {min_difficulty} for code task"
                        )
                        rating = float(min_difficulty)

                    logger.debug(f"Final difficulty rating: {rating}")
                    return rating
                else:
                    logger.warning(
                        f"No rating found in response '{clean_response}', using default 2.5"
                    )
                    return 2.5
            except Exception:
                return 2.5

        except Exception as e:
            logger.error(f"Error in rate_query_difficulty: {str(e)}", exc_info=True)
            return 2.5


# Global model manager instance
mlx_model_manager = MLXModelManager()
