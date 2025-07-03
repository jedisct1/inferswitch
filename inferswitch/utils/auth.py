"""
Common authentication utilities for InferSwitch.
"""

import logging
from typing import Optional, Tuple


logger = logging.getLogger(__name__)


async def get_auth_credentials(
    x_api_key: Optional[str] = None, use_oauth: bool = True, oauth_manager=None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Get authentication credentials, prioritizing OAuth when available.

    Args:
        x_api_key: API key from request header
        use_oauth: Whether to attempt OAuth authentication
        oauth_manager: OAuth manager instance (to avoid circular imports)

    Returns:
        Tuple of (api_key, oauth_token) where one will be None
    """
    oauth_token = None
    api_key = x_api_key

    if use_oauth and oauth_manager:
        try:
            oauth_token = await oauth_manager.get_valid_token()
            if oauth_token:
                logger.debug("Using OAuth token for authentication")
                return None, oauth_token
        except Exception as e:
            logger.debug(f"OAuth token unavailable: {e}")

    if api_key:
        logger.debug("Using API key for authentication")
        return api_key, None

    logger.warning("No authentication credentials available")
    return None, None


def should_use_oauth(
    x_api_key: Optional[str], default_api_key: Optional[str] = None
) -> bool:
    """
    Determine if OAuth should be used based on API key presence.

    Args:
        x_api_key: API key from request header
        default_api_key: Default API key from environment

    Returns:
        True if OAuth should be used, False otherwise
    """
    # Use OAuth if no API key provided or if API key matches default
    if not x_api_key:
        return True

    if default_api_key and x_api_key == default_api_key:
        return True

    return False


async def get_anthropic_auth_headers(
    x_api_key: Optional[str] = None,
    anthropic_version: Optional[str] = None,
    use_oauth: bool = True,
    oauth_manager=None,
) -> dict:
    """
    Get authentication headers for Anthropic API requests.

    Args:
        x_api_key: API key from request header
        anthropic_version: Anthropic version header
        use_oauth: Whether to attempt OAuth authentication
        oauth_manager: OAuth manager instance (to avoid circular imports)

    Returns:
        Dictionary of headers for Anthropic API
    """
    headers = {}

    if anthropic_version:
        headers["anthropic-version"] = anthropic_version

    api_key, oauth_token = await get_auth_credentials(
        x_api_key, use_oauth, oauth_manager
    )

    if oauth_token:
        headers["authorization"] = f"Bearer {oauth_token}"
    elif api_key:
        headers["x-api-key"] = api_key
    else:
        logger.warning("No authentication credentials available for Anthropic API")

    return headers


async def get_openai_auth_headers(
    x_api_key: Optional[str] = None, use_oauth: bool = False
) -> dict:
    """
    Get authentication headers for OpenAI API requests.

    Args:
        x_api_key: API key from request header
        use_oauth: Whether to attempt OAuth authentication (not supported for OpenAI)

    Returns:
        Dictionary of headers for OpenAI API
    """
    headers = {}

    if x_api_key:
        headers["Authorization"] = f"Bearer {x_api_key}"
    else:
        logger.warning("No API key available for OpenAI API")

    return headers


class AuthenticationError(Exception):
    """Exception raised when authentication fails."""

    def __init__(self, message: str, auth_type: str = "unknown"):
        self.message = message
        self.auth_type = auth_type
        super().__init__(self.message)


def validate_authentication(
    api_key: Optional[str] = None,
    oauth_token: Optional[str] = None,
    require_auth: bool = True,
) -> None:
    """
    Validate that authentication credentials are available.

    Args:
        api_key: API key to validate
        oauth_token: OAuth token to validate
        require_auth: Whether authentication is required

    Raises:
        AuthenticationError: If authentication is required but not available
    """
    if not require_auth:
        return

    if not api_key and not oauth_token:
        raise AuthenticationError(
            "No authentication credentials available", auth_type="none"
        )

    if api_key and len(api_key.strip()) == 0:
        raise AuthenticationError("Empty API key provided", auth_type="api_key")

    if oauth_token and len(oauth_token.strip()) == 0:
        raise AuthenticationError("Empty OAuth token provided", auth_type="oauth")
