"""
Cache implementation for InferSwitch.
"""

import hashlib
import json
import time
import logging
from typing import Dict, Optional, Any, Tuple
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)


class RequestCache:
    """Thread-safe LRU cache for API requests."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _remove_processing_tag(self, text: str) -> str:
        """Remove processing tags from text content."""
        # Remove <processing>...</processing> tags and their content
        import re

        text = re.sub(r"<processing>.*?</processing>\s*", "", text, flags=re.DOTALL)
        return text.strip()

    def _extract_cache_key_fields(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract only the fields that should be used for cache key computation.

        Args:
            request_data: The full request dictionary

        Returns:
            Dictionary with only cache-relevant fields
        """
        # Fields that affect the response
        cache_fields = {}

        # Handle system prompt
        if "system" in request_data and request_data["system"] is not None:
            system_content = request_data["system"]
            # If system is a string, check for and remove environment details
            if isinstance(system_content, str):
                # Remove environment_details blocks from system prompt
                import re

                # Pattern to match <environment_details>...</environment_details> blocks
                pattern = r"<environment_details>.*?</environment_details>\s*"
                system_content = re.sub(pattern, "", system_content, flags=re.DOTALL)
                # Also remove any remaining timestamp patterns like "Current Time: ..."
                system_content = re.sub(
                    r"(Current Time|Timestamp|Date):\s*[^\n]+\n?",
                    "",
                    system_content,
                    flags=re.IGNORECASE,
                )
                system_content = system_content.strip()
            elif isinstance(system_content, list):
                # Handle system as array of objects
                cleaned_system = []
                for item in system_content:
                    if isinstance(item, dict) and "text" in item:
                        text = item["text"]
                        # Remove environment_details blocks and timestamps from text
                        import re

                        text = re.sub(
                            r"<environment_details>.*?</environment_details>\s*",
                            "",
                            text,
                            flags=re.DOTALL,
                        )
                        text = re.sub(
                            r"(Current Time|Timestamp|Date):\s*[^\n]+\n?",
                            "",
                            text,
                            flags=re.IGNORECASE,
                        )
                        text = text.strip()
                        if text:  # Only add if there's content after cleaning
                            cleaned_system.append({"text": text})
                    else:
                        # Keep non-text items as-is
                        cleaned_system.append(item)
                # Sort system prompts by text content to ensure consistent ordering
                system_content = sorted(
                    cleaned_system,
                    key=lambda x: x.get("text", "") if isinstance(x, dict) else str(x),
                )
            cache_fields["system"] = system_content

        # Handle messages - extract only the actual content, not metadata
        if "messages" in request_data:
            cleaned_messages = []
            for msg in request_data["messages"]:
                cleaned_msg = {"role": msg["role"]}

                # Handle content
                if "content" in msg:
                    if isinstance(msg["content"], str):
                        # Remove processing tags from string content
                        content = msg["content"]
                        content = self._remove_processing_tag(content)
                        # Also remove environment details and timestamps
                        import re

                        content = re.sub(
                            r"<environment_details>.*?</environment_details>\s*",
                            "",
                            content,
                            flags=re.DOTALL,
                        )
                        content = re.sub(
                            r"(Current Time|Timestamp|Date):\s*[^\n]+\n?",
                            "",
                            content,
                            flags=re.IGNORECASE,
                        )
                        content = content.strip()
                        cleaned_msg["content"] = content
                    elif isinstance(msg["content"], list):
                        # Extract text content, ignoring cache_control and environment details
                        cleaned_content = []
                        for content_item in msg["content"]:
                            if (
                                isinstance(content_item, dict)
                                and content_item.get("type") == "text"
                            ):
                                # Skip content with ephemeral cache_control
                                if (
                                    content_item.get("cache_control", {}).get("type")
                                    == "ephemeral"
                                ):
                                    continue

                                text = content_item.get("text", "")
                                # Skip environment_details blocks which contain timestamps
                                if not text.startswith("<environment_details>"):
                                    # Remove processing tags
                                    text = self._remove_processing_tag(text)
                                    # Also remove environment_details blocks and timestamps from text content
                                    import re

                                    text = re.sub(
                                        r"<environment_details>.*?</environment_details>\s*",
                                        "",
                                        text,
                                        flags=re.DOTALL,
                                    )
                                    text = re.sub(
                                        r"(Current Time|Timestamp|Date):\s*[^\n]+\n?",
                                        "",
                                        text,
                                        flags=re.IGNORECASE,
                                    )
                                    text = text.strip()
                                    if (
                                        text
                                    ):  # Only add if there's content after cleaning
                                        cleaned_content.append(
                                            {"type": "text", "text": text}
                                        )
                        cleaned_msg["content"] = cleaned_content

                cleaned_messages.append(cleaned_msg)

            cache_fields["messages"] = cleaned_messages

        return cache_fields

    def _compute_hash(self, request_data: Dict[str, Any]) -> str:
        """
        Compute hash of request data using only cache-relevant fields.

        Args:
            request_data: The request dictionary

        Returns:
            Hash string
        """
        # Extract only cache-relevant fields
        cache_fields = self._extract_cache_key_fields(request_data)

        # Create a stable string representation of the request
        # Sort keys to ensure consistent hashing
        stable_json = json.dumps(cache_fields, sort_keys=True)

        # Use SHA256 for hash computation
        hash_obj = hashlib.sha256(stable_json.encode("utf-8"))
        hash_value = hash_obj.hexdigest()
        return hash_value

    def get(self, request_data: Dict[str, Any]) -> Optional[Any]:
        """
        Get cached response for request.

        Args:
            request_data: The request dictionary

        Returns:
            Cached response or None if not found/expired
        """
        cache_key = self._compute_hash(request_data)

        with self.lock:
            if cache_key in self.cache:
                response, timestamp = self.cache[cache_key]

                # Check if entry has expired
                if time.time() - timestamp > self.ttl_seconds:
                    # Remove expired entry
                    del self.cache[cache_key]
                    self.misses += 1
                    return None

                # Move to end (most recently used)
                self.cache.move_to_end(cache_key)
                self.hits += 1
                return response

            self.misses += 1
            return None

    def set(self, request_data: Dict[str, Any], response: Any) -> None:
        """
        Store response in cache.

        Args:
            request_data: The request dictionary
            response: The response to cache
        """
        cache_key = self._compute_hash(request_data)

        with self.lock:
            # If cache is full, remove oldest entry
            if len(self.cache) >= self.max_size and cache_key not in self.cache:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]

            # Add/update entry
            self.cache[cache_key] = (response, time.time())
            self.cache.move_to_end(cache_key)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "total_requests": total_requests,
            }


# Global cache instance
_cache: Optional[RequestCache] = None


def get_cache() -> RequestCache:
    """Get the global cache instance."""
    global _cache
    if _cache is None:
        from ..config import CACHE_MAX_SIZE, CACHE_TTL_SECONDS

        _cache = RequestCache(max_size=CACHE_MAX_SIZE, ttl_seconds=CACHE_TTL_SECONDS)
    return _cache
