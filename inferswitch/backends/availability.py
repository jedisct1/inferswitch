"""
Model availability tracking for temporary disabling of failed models.
"""

from datetime import datetime, timedelta
from typing import Dict, Set
import threading
import logging

logger = logging.getLogger(__name__)


class ModelAvailabilityTracker:
    """
    Tracks model availability and manages temporary disabling of models that fail.
    """

    def __init__(self, disable_duration_seconds: int = 300):
        """
        Initialize the availability tracker.

        Args:
            disable_duration_seconds: How long to disable a model after failure
        """
        self.disable_duration_seconds = disable_duration_seconds
        self._disabled_models: Dict[str, datetime] = {}
        self._lock = threading.Lock()

    def is_available(self, model: str) -> bool:
        """
        Check if a model is currently available.

        Args:
            model: Model name to check

        Returns:
            True if the model is available, False if temporarily disabled
        """
        with self._lock:
            if model not in self._disabled_models:
                return True

            disabled_until = self._disabled_models[model]
            if datetime.now() >= disabled_until:
                # Re-enable the model
                del self._disabled_models[model]
                logger.info(f"Model {model} has been re-enabled")
                return True

            remaining = (disabled_until - datetime.now()).total_seconds()
            logger.debug(f"Model {model} is disabled for {remaining:.0f} more seconds")
            return False

    def mark_failure(self, model: str):
        """
        Mark a model as failed and temporarily disable it.

        Args:
            model: Model name that failed
        """
        with self._lock:
            disabled_until = datetime.now() + timedelta(
                seconds=self.disable_duration_seconds
            )
            self._disabled_models[model] = disabled_until
            logger.warning(
                f"Model {model} has been temporarily disabled until {disabled_until.isoformat()} "
                f"({self.disable_duration_seconds} seconds)"
            )

    def mark_success(self, model: str):
        """
        Mark a model as successful, ensuring it's not disabled.

        Args:
            model: Model name that succeeded
        """
        with self._lock:
            if model in self._disabled_models:
                del self._disabled_models[model]
                logger.info(
                    f"Model {model} has been re-enabled due to successful request"
                )

    def get_disabled_models(self) -> Set[str]:
        """
        Get the set of currently disabled models.

        Returns:
            Set of model names that are currently disabled
        """
        with self._lock:
            # Clean up expired disables
            now = datetime.now()
            expired = [
                model for model, until in self._disabled_models.items() if now >= until
            ]
            for model in expired:
                del self._disabled_models[model]

            return set(self._disabled_models.keys())

    def clear_all_disabled(self):
        """Clear all disabled models, re-enabling everything."""
        with self._lock:
            count = len(self._disabled_models)
            self._disabled_models.clear()
            if count > 0:
                logger.info(f"Cleared {count} disabled models")
