"""
Dynamic Expert Classification for intelligent model routing.

This module provides functionality to classify queries based on user-defined expert
descriptions using MLX-based language model classification.
"""

from typing import List, Dict, Optional, Any
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
    logger.warning("MLX not available. Expert classification will be disabled.")


class ExpertClassifier:
    """Classifies queries by user-defined expert areas for targeted model routing."""

    def __init__(self, expert_definitions: Optional[Dict[str, str]] = None):
        """
        Initialize the classifier with expert definitions.

        Args:
            expert_definitions: Dictionary mapping expert names to their descriptions
        """
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.expert_definitions = expert_definitions or {}

    def set_expert_definitions(self, expert_definitions: Dict[str, str]):
        """
        Set or update the expert definitions.

        Args:
            expert_definitions: Dictionary mapping expert names to their descriptions
        """
        self.expert_definitions = expert_definitions
        logger.debug(f"Updated expert definitions: {list(expert_definitions.keys())}")

    def get_expert_definitions(self) -> Dict[str, str]:
        """Get the current expert definitions."""
        return self.expert_definitions.copy()

    def load_model(
        self, model_name: str = "jedisct1/arch-router-1.5b"
    ) -> tuple[bool, str]:
        """
        Load an MLX model for expert classification.

        Args:
            model_name: HuggingFace model identifier

        Returns:
            Tuple of (success, message)
        """
        if not MLX_AVAILABLE:
            return False, "MLX is not available on this system"

        try:
            logger.debug(
                f"Attempting to load MLX model for expert classification: {model_name}"
            )

            # Load model and tokenizer
            self.model, self.tokenizer = mlx_lm.load(model_name)
            self.model_name = model_name

            logger.debug(
                f"MLX model {model_name} loaded successfully for expert classification"
            )
            return True, f"Expert classification model {model_name} loaded successfully"

        except Exception as e:
            error_msg = f"Failed to load MLX model {model_name} for expert classification: {str(e)}"
            logger.error(error_msg)
            # Don't crash - just disable MLX-based classification
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
            "expert_count": len(self.expert_definitions),
        }

    def _extract_user_query(self, chat_messages: List[Dict[str, str]]) -> str:
        """Extract the latest user query from chat messages."""
        for msg in reversed(chat_messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def _clean_query(self, query: str) -> str:
        """Clean and normalize query text for analysis."""
        import re

        # Remove common XML tags
        cleaned = re.sub(
            r"</?(?:task|environment_details|slug|name|model)[^>]*>", "", query
        )
        # Remove multiple newlines and extra spaces
        cleaned = re.sub(r"\n+", " ", cleaned).strip()
        # Truncate if too long for better processing
        if len(cleaned) > 400:
            cleaned = cleaned[:400]
        return cleaned

    def _classify_with_mlx(self, query: str) -> Optional[str]:
        """
        Use MLX model to classify query based on expert definitions.

        Args:
            query: Clean query text to classify

        Returns:
            Expert name or None if classification fails
        """
        if not self.is_loaded():
            logger.warning("MLX model not loaded, cannot classify query")
            return None

        if not self.expert_definitions:
            logger.warning("No expert definitions provided, cannot classify query")
            return None

        try:
            # Build the classification prompt
            expert_descriptions = []
            for expert_name, description in self.expert_definitions.items():
                expert_descriptions.append(f"- {expert_name}: {description}")

            experts_text = "\n".join(expert_descriptions)

            prompt = f"""You are an expert classification system. Based on the query below, determine which expert would be most appropriate to handle this request.

AVAILABLE EXPERTS:
{experts_text}

QUERY: {query[:300]}

Analyze the query and respond with ONLY the name of the most appropriate expert from the list above. Do not provide any explanation, just the expert name.

Expert:"""

            try:
                response = mlx_lm.generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    max_tokens=20,  # Should be enough for expert name
                    verbose=False,
                )

                logger.debug(f"MLX classification response: {repr(response)}")

                # Clean and validate the response
                expert_name = response.strip().lower()

                # Find matching expert (case-insensitive)
                for defined_expert in self.expert_definitions.keys():
                    if defined_expert.lower() == expert_name:
                        return defined_expert

                # If exact match not found, try partial matching
                for defined_expert in self.expert_definitions.keys():
                    if (
                        defined_expert.lower() in expert_name
                        or expert_name in defined_expert.lower()
                    ):
                        logger.debug(
                            f"Partial match: '{expert_name}' -> '{defined_expert}'"
                        )
                        return defined_expert

                logger.warning(
                    f"MLX returned unrecognized expert: '{expert_name}'. Available: {list(self.expert_definitions.keys())}"
                )
                return None

            except Exception as e:
                logger.error(f"Error during MLX generation: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Error in MLX classification: {str(e)}")
            return None

    def classify_expert(self, chat_messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Classify which expert should handle the query.

        Args:
            chat_messages: List of message dictionaries in chat template format

        Returns:
            Expert name or None if classification fails
        """
        try:
            # Extract and clean the user query
            user_query = self._extract_user_query(chat_messages)
            if not user_query:
                logger.warning("No user query found in messages")
                return None

            cleaned_query = self._clean_query(user_query)
            logger.debug(f"Classifying expert for query: {cleaned_query[:100]}...")

            # Use MLX classification based on expert definitions
            expert = self._classify_with_mlx(cleaned_query)

            if expert:
                logger.debug(f"Query classified as requiring expert: {expert}")
            else:
                logger.warning("Could not classify query to any expert")

            return expert

        except Exception as e:
            logger.error(f"Error in classify_expert: {str(e)}", exc_info=True)
            return None

    def get_expert_scores(
        self, chat_messages: List[Dict[str, str]]
    ) -> Dict[str, float]:
        """
        Get detailed expert scores for a query using MLX.

        Args:
            chat_messages: List of message dictionaries in chat template format

        Returns:
            Dict mapping expert names to confidence score (0-1)
        """
        if not self.is_loaded() or not self.expert_definitions:
            return {}

        try:
            user_query = self._extract_user_query(chat_messages)
            if not user_query:
                return {}

            cleaned_query = self._clean_query(user_query)

            # Build scoring prompt
            expert_descriptions = []
            for expert_name, description in self.expert_definitions.items():
                expert_descriptions.append(f"- {expert_name}: {description}")

            experts_text = "\n".join(expert_descriptions)

            prompt = f"""Rate how well each expert matches this query. Rate from 0 (not relevant) to 5 (highly relevant).

AVAILABLE EXPERTS:
{experts_text}

QUERY: {cleaned_query[:300]}

Respond with ONLY this format:
{chr(10).join([f"{name}: X" for name in self.expert_definitions.keys()])}

Where X is a number 0-5:"""

            try:
                response = mlx_lm.generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    max_tokens=100,
                    verbose=False,
                )

                logger.debug(f"MLX expert scoring response: {repr(response)}")

                # Parse the response
                scores = {}
                lines = response.strip().split("\n")

                for line in lines:
                    if ":" in line:
                        try:
                            expert_name, score_str = line.split(":", 1)
                            expert_name = expert_name.strip()
                            score = float(score_str.strip())

                            # Find matching expert name (case-insensitive)
                            for defined_expert in self.expert_definitions.keys():
                                if defined_expert.lower() == expert_name.lower():
                                    # Normalize to 0-1 range
                                    scores[defined_expert] = min(
                                        1.0, max(0.0, score / 5.0)
                                    )
                                    break

                        except (ValueError, IndexError):
                            continue

                # Ensure all experts have scores
                for expert_name in self.expert_definitions.keys():
                    if expert_name not in scores:
                        scores[expert_name] = 0.0

                return scores

            except Exception as e:
                logger.error(f"Error during MLX expert scoring: {str(e)}")
                return {}

        except Exception as e:
            logger.error(f"Error in get_expert_scores: {str(e)}", exc_info=True)
            return {}

    def validate_expert_definitions(self) -> Dict[str, Any]:
        """
        Validate the current expert definitions.

        Returns:
            Dict with validation results
        """
        results = {
            "valid": True,
            "issues": [],
            "expert_count": len(self.expert_definitions),
            "experts": list(self.expert_definitions.keys()),
        }

        if not self.expert_definitions:
            results["valid"] = False
            results["issues"].append("No expert definitions provided")
            return results

        for expert_name, description in self.expert_definitions.items():
            if not expert_name or not expert_name.strip():
                results["valid"] = False
                results["issues"].append("Empty expert name found")

            if not description or not description.strip():
                results["valid"] = False
                results["issues"].append(
                    f"Empty description for expert '{expert_name}'"
                )
            elif len(description.strip()) < 10:
                results["issues"].append(
                    f"Very short description for expert '{expert_name}' (may affect classification quality)"
                )

        return results


# Global expert classifier instance
expert_classifier = ExpertClassifier()
