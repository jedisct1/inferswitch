"""
Intelligent message compression using MLX models.

This module provides context-aware compression for messages that exceed model limits.
It uses MLX models to score message importance and applies intelligent compression
strategies to maintain conversation quality while reducing token count.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..config import MODEL_CONTEXT_SIZES
from ..mlx_model import mlx_model_manager
from .chat_template import truncate_chat_template_to_fit

logger = logging.getLogger(__name__)


class CompressionStrategy(Enum):
    """Compression strategies for different scenarios."""

    TRUNCATE = "truncate"  # Simple truncation of old messages
    SUMMARIZE = "summarize"  # Summarize old messages
    HYBRID = "hybrid"  # Combine truncation and summarization
    SMART_TRUNCATE = "smart_truncate"  # MLX-guided intelligent truncation


@dataclass
class CompressionResult:
    """Result of compression operation."""

    messages: List[Dict[str, Any]]
    original_count: int
    compressed_count: int
    original_tokens: int
    compressed_tokens: int
    strategy_used: CompressionStrategy
    compression_notice: str


class MessageCompressor:
    """Intelligent message compression using MLX models."""

    def __init__(self):
        self.mlx_available = mlx_model_manager.is_loaded()
        self.chars_per_token = 4  # Rough estimate

    def compress_messages(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        target_ratio: float = 0.7,
        strategy: Optional[CompressionStrategy] = None,
    ) -> CompressionResult:
        """
        Compress messages to fit within model context window.

        Args:
            messages: List of message dictionaries
            model: Model name to determine context size
            target_ratio: Target compression ratio (0.7 = 70% of max context)
            strategy: Specific strategy to use (auto-select if None)

        Returns:
            CompressionResult with compressed messages
        """
        if not messages:
            return CompressionResult(
                messages=[],
                original_count=0,
                compressed_count=0,
                original_tokens=0,
                compressed_tokens=0,
                strategy_used=CompressionStrategy.TRUNCATE,
                compression_notice="",
            )

        # Calculate original size
        original_tokens = self._estimate_tokens(messages)
        original_count = len(messages)

        # Get model context size
        max_context = MODEL_CONTEXT_SIZES.get(model, 200000)
        target_tokens = int(max_context * target_ratio)

        # Check if compression is needed
        if original_tokens <= target_tokens:
            logger.info(
                f"No compression needed: {original_tokens} tokens < {target_tokens} target"
            )
            return CompressionResult(
                messages=messages,
                original_count=original_count,
                compressed_count=original_count,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                strategy_used=CompressionStrategy.TRUNCATE,
                compression_notice="",
            )

        # Select strategy
        if strategy is None:
            strategy = self._select_strategy(messages, original_tokens, target_tokens)

        logger.info(
            f"Compressing {original_tokens} tokens to {target_tokens} using {strategy.value}"
        )

        # Apply compression
        if strategy == CompressionStrategy.SMART_TRUNCATE and self.mlx_available:
            compressed_messages, notice = self._smart_truncate(
                messages, target_tokens, model
            )
        elif strategy == CompressionStrategy.SUMMARIZE and self.mlx_available:
            compressed_messages, notice = self._summarize_messages(
                messages, target_tokens, model
            )
        elif strategy == CompressionStrategy.HYBRID and self.mlx_available:
            compressed_messages, notice = self._hybrid_compression(
                messages, target_tokens, model
            )
        else:
            # Fallback to simple truncation
            compressed_messages, notice = self._simple_truncate(
                messages, target_tokens, model
            )
            strategy = CompressionStrategy.TRUNCATE

        # Calculate final size
        compressed_tokens = self._estimate_tokens(compressed_messages)
        compressed_count = len(compressed_messages)

        return CompressionResult(
            messages=compressed_messages,
            original_count=original_count,
            compressed_count=compressed_count,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            strategy_used=strategy,
            compression_notice=notice,
        )

    def _estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate token count from messages."""
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # Handle multimodal messages
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        total_chars += len(item["text"])
                    elif isinstance(item, str):
                        total_chars += len(item)
        return total_chars // self.chars_per_token

    def _select_strategy(
        self, messages: List[Dict[str, Any]], original_tokens: int, target_tokens: int
    ) -> CompressionStrategy:
        """Select best compression strategy based on content."""
        compression_ratio = target_tokens / original_tokens

        # High compression needed
        if compression_ratio < 0.3:
            return (
                CompressionStrategy.HYBRID
                if self.mlx_available
                else CompressionStrategy.TRUNCATE
            )

        # Moderate compression
        if compression_ratio < 0.6:
            return (
                CompressionStrategy.SMART_TRUNCATE
                if self.mlx_available
                else CompressionStrategy.TRUNCATE
            )

        # Light compression
        return CompressionStrategy.TRUNCATE

    def _simple_truncate(
        self, messages: List[Dict[str, Any]], target_tokens: int, model: str
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Simple truncation using existing utilities."""
        # Use existing truncation utility
        truncated = truncate_chat_template_to_fit(messages, model)

        # Count removed messages
        removed_count = len(messages) - len(truncated)

        if removed_count > 0:
            notice = self._create_compression_notice(
                "truncation", removed_count, len(messages)
            )
            # Prepend notice as system message
            truncated = self._add_compression_notice(truncated, notice)
        else:
            notice = ""

        return truncated, notice

    def _smart_truncate(
        self, messages: List[Dict[str, Any]], target_tokens: int, model: str
    ) -> Tuple[List[Dict[str, Any]], str]:
        """MLX-guided intelligent truncation."""
        if not self.mlx_available or not messages:
            return self._simple_truncate(messages, target_tokens, model)

        try:
            # Score each message for importance
            importance_scores = self._score_message_importance(messages)

            # Sort messages by importance (keeping order for same importance)
            indexed_messages = list(enumerate(messages))
            indexed_messages.sort(
                key=lambda x: (importance_scores[x[0]], -x[0]), reverse=True
            )

            # Keep most important messages within token budget
            kept_messages = []
            current_tokens = 0

            for idx, msg in indexed_messages:
                msg_tokens = self._estimate_tokens([msg])
                if current_tokens + msg_tokens <= target_tokens:
                    kept_messages.append((idx, msg))
                    current_tokens += msg_tokens

            # Sort back to original order
            kept_messages.sort(key=lambda x: x[0])
            result_messages = [msg for _, msg in kept_messages]

            # Create notice
            removed_count = len(messages) - len(result_messages)
            notice = self._create_compression_notice(
                "intelligent selection", removed_count, len(messages)
            )

            if removed_count > 0:
                result_messages = self._add_compression_notice(result_messages, notice)

            return result_messages, notice

        except Exception as e:
            logger.warning(f"Smart truncation failed: {e}, falling back to simple")
            return self._simple_truncate(messages, target_tokens, model)

    def _summarize_messages(
        self, messages: List[Dict[str, Any]], target_tokens: int, model: str
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Summarize older messages to reduce token count."""
        if not self.mlx_available or len(messages) < 3:
            return self._simple_truncate(messages, target_tokens, model)

        try:
            # Keep recent messages intact
            recent_count = min(3, len(messages) // 2)
            recent_messages = messages[-recent_count:]
            older_messages = messages[:-recent_count]

            if not older_messages:
                return messages, ""

            # Create summary of older messages
            summary = self._create_summary(older_messages)

            # Build result with summary
            result_messages = [
                {
                    "role": "system",
                    "content": f"[Summary of {len(older_messages)} previous messages]\n{summary}",
                }
            ] + recent_messages

            # Check if within budget
            if self._estimate_tokens(result_messages) > target_tokens:
                # Need more aggressive compression
                return self._simple_truncate(result_messages, target_tokens, model)

            notice = self._create_compression_notice(
                "summarization", len(older_messages), len(messages)
            )
            return result_messages, notice

        except Exception as e:
            logger.warning(f"Summarization failed: {e}, falling back to truncation")
            return self._simple_truncate(messages, target_tokens, model)

    def _hybrid_compression(
        self, messages: List[Dict[str, Any]], target_tokens: int, model: str
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Combine summarization and smart truncation."""
        if not self.mlx_available or len(messages) < 5:
            return self._smart_truncate(messages, target_tokens, model)

        try:
            # Divide messages into groups
            total_msgs = len(messages)
            keep_recent = max(2, total_msgs // 4)
            summarize_count = max(2, total_msgs // 2)

            recent = messages[-keep_recent:]
            to_summarize = (
                messages[-summarize_count:-keep_recent]
                if keep_recent < summarize_count
                else []
            )
            to_truncate = (
                messages[:-summarize_count] if summarize_count < total_msgs else []
            )

            # Create summary of middle section
            summary_content = ""
            if to_summarize:
                summary = self._create_summary(to_summarize)
                summary_content = f"[Summary of messages {len(to_truncate) + 1}-{len(to_truncate) + len(to_summarize)}]\n{summary}"

            # Smart truncate the oldest messages
            truncated = []
            if to_truncate:
                truncated_msgs, _ = self._smart_truncate(
                    to_truncate, target_tokens // 3, model
                )
                truncated = truncated_msgs

            # Combine all parts
            result_messages = truncated
            if summary_content:
                result_messages.append({"role": "system", "content": summary_content})
            result_messages.extend(recent)

            # Final check and adjustment
            if self._estimate_tokens(result_messages) > target_tokens:
                result_messages, _ = self._simple_truncate(
                    result_messages, target_tokens, model
                )

            notice = self._create_compression_notice(
                "hybrid compression",
                len(messages) - len(result_messages),
                len(messages),
            )
            return result_messages, notice

        except Exception as e:
            logger.warning(f"Hybrid compression failed: {e}, falling back")
            return self._smart_truncate(messages, target_tokens, model)

    def _score_message_importance(self, messages: List[Dict[str, Any]]) -> List[float]:
        """Score message importance using MLX model."""
        scores = []

        for i, msg in enumerate(messages):
            # Base score by position (recent = more important)
            position_score = (i + 1) / len(messages)

            # Role-based scoring
            role_score = 0.5
            if msg.get("role") == "system":
                role_score = 0.9
            elif msg.get("role") == "user":
                role_score = 0.7
            elif msg.get("role") == "assistant":
                role_score = 0.6

            # Content-based scoring (using MLX if available)
            content_score = 0.5
            if self.mlx_available:
                try:
                    # Use MLX to assess content importance
                    difficulty = mlx_model_manager.rate_query_difficulty([msg])
                    content_score = difficulty / 5.0  # Normalize to 0-1
                except Exception:
                    pass

            # Length penalty (very long messages less important)
            content = msg.get("content", "")
            length = len(content) if isinstance(content, str) else 100
            length_score = min(1.0, 500 / max(length, 1))

            # Combined score
            final_score = (
                position_score * 0.3
                + role_score * 0.3
                + content_score * 0.2
                + length_score * 0.2
            )
            scores.append(final_score)

        return scores

    def _create_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Create a concise summary of messages."""
        # Simple summary for now - can be enhanced with MLX
        summary_parts = []

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if isinstance(content, str):
                # Take first 100 chars
                preview = content[:100].strip()
                if len(content) > 100:
                    preview += "..."
            else:
                preview = "[multimodal content]"

            summary_parts.append(f"{role}: {preview}")

        # Join and limit total summary
        summary = "\n".join(summary_parts)
        if len(summary) > 500:
            summary = summary[:497] + "..."

        return summary

    def _create_compression_notice(self, method: str, removed: int, total: int) -> str:
        """Create a notice about compression."""
        if removed == 0:
            return ""

        percentage = (removed / total) * 100
        return (
            f"[NOTICE: Request compressed using {method}. "
            f"Removed {removed}/{total} messages ({percentage:.0f}%) to fit context window. "
            f"This is an automatically compressed version of the original request.]"
        )

    def _add_compression_notice(
        self, messages: List[Dict[str, Any]], notice: str
    ) -> List[Dict[str, Any]]:
        """Add compression notice to messages."""
        if not notice or not messages:
            return messages

        # Check if first message is system
        if messages and messages[0].get("role") == "system":
            # Prepend to existing system message
            original_content = messages[0].get("content", "")
            messages[0]["content"] = f"{notice}\n\n{original_content}"
        else:
            # Add as new system message
            messages = [{"role": "system", "content": notice}] + messages

        return messages


# Global compressor instance
message_compressor = MessageCompressor()
