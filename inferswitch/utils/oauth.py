"""OAuth authentication utilities for Anthropic."""

import base64
import hashlib
import json
import logging
import os
import secrets
import time
from typing import Optional, Tuple
from urllib.parse import quote

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class OAuthConfig(BaseModel):
    """OAuth configuration for Anthropic."""

    client_id: str = (
        "9d1c250a-e61b-44d9-88ed-5944d1962f5e"  # Default, can be overridden
    )
    redirect_uri: str = "https://console.anthropic.com/oauth/code/callback"
    auth_url: str = "https://claude.ai/oauth/authorize"
    token_url: str = "https://console.anthropic.com/v1/oauth/token"
    scopes: str = "org:create_api_key user:profile user:inference"


class TokenInfo(BaseModel):
    """OAuth token information."""

    access_token: str
    refresh_token: Optional[str] = None
    expires_at: float
    token_type: str = "Bearer"

    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return time.time() >= self.expires_at

    @property
    def expires_in_seconds(self) -> float:
        """Get seconds until expiration."""
        return max(0, self.expires_at - time.time())


class OAuthManager:
    """Manages OAuth authentication flow for Anthropic."""

    def __init__(self, config: Optional[OAuthConfig] = None):
        if config:
            self.config = config
        else:
            # Load config from inferswitch configuration
            self.config = self._load_oauth_config()
        self.token_storage_path = os.path.expanduser("~/.inferswitch/oauth_tokens.json")
        self._ensure_storage_dir()

    def _load_oauth_config(self) -> OAuthConfig:
        """Load OAuth configuration from inferswitch config file."""
        # Import here to avoid circular imports
        from ..backends.config import BackendConfigManager

        # Get OAuth config from the config manager for Anthropic provider
        oauth_settings = BackendConfigManager.get_oauth_config("anthropic")

        # Create OAuthConfig with defaults, overridden by config file values
        config_dict = {
            "client_id": oauth_settings.get(
                "client_id", "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
            ),
            "redirect_uri": oauth_settings.get(
                "redirect_uri", "https://console.anthropic.com/oauth/code/callback"
            ),
            "auth_url": oauth_settings.get(
                "auth_url", "https://claude.ai/oauth/authorize"
            ),
            "token_url": oauth_settings.get(
                "token_url", "https://console.anthropic.com/v1/oauth/token"
            ),
            "scopes": oauth_settings.get(
                "scopes", "org:create_api_key user:profile user:inference"
            ),
        }

        return OAuthConfig(**config_dict)

    def _ensure_storage_dir(self):
        """Ensure token storage directory exists."""
        os.makedirs(os.path.dirname(self.token_storage_path), exist_ok=True)

    def generate_pkce_pair(self) -> Tuple[str, str]:
        """Generate PKCE code verifier and challenge."""
        code_verifier = (
            base64.urlsafe_b64encode(secrets.token_bytes(32))
            .decode("utf-8")
            .rstrip("=")
        )
        code_challenge = (
            base64.urlsafe_b64encode(
                hashlib.sha256(code_verifier.encode("utf-8")).digest()
            )
            .decode("utf-8")
            .rstrip("=")
        )
        return code_verifier, code_challenge

    def get_authorization_url(
        self, state: Optional[str] = None
    ) -> Tuple[str, str, str]:
        """Generate authorization URL with PKCE."""
        if not state:
            state = secrets.token_urlsafe(32)

        code_verifier, code_challenge = self.generate_pkce_pair()

        # Build URL with proper encoding
        auth_url = (
            f"{self.config.auth_url}?"
            f"response_type=code&"
            f"client_id={self.config.client_id}&"
            f"redirect_uri={quote(self.config.redirect_uri, safe='')}&"
            f"scope={quote(self.config.scopes, safe='')}&"
            f"state={state}&"
            f"code_challenge={code_challenge}&"
            f"code_challenge_method=S256"
        )

        return auth_url, state, code_verifier

    async def exchange_code_for_token(
        self, code: str, code_verifier: str, state: Optional[str] = None
    ) -> TokenInfo:
        """Exchange authorization code for access token."""
        async with httpx.AsyncClient() as client:
            # Based on the gist, 'state' in the token exchange is the PKCE verifier!
            data = {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.config.redirect_uri,
                "client_id": self.config.client_id,
                "code_verifier": code_verifier,
                "state": code_verifier,  # Non-standard: Anthropic expects verifier here too!
            }

            # Send as JSON with required beta header
            headers = {
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "oauth-2025-04-20",
            }

            response = await client.post(
                self.config.token_url,
                json=data,  # Send as JSON
                headers=headers,
            )

            if response.status_code != 200:
                logger.error(
                    f"Token exchange failed: {response.status_code} - {response.text}"
                )
                raise Exception(f"Token exchange failed: {response.text}")

            token_data = response.json()

            # Calculate expiration time
            expires_in = token_data.get("expires_in", 3600)
            expires_at = time.time() + expires_in

            token_info = TokenInfo(
                access_token=token_data["access_token"],
                refresh_token=token_data.get("refresh_token"),
                expires_at=expires_at,
                token_type=token_data.get("token_type", "Bearer"),
            )

            # Store token
            self.store_token(token_info)

            return token_info

    async def refresh_access_token(self, refresh_token: str) -> TokenInfo:
        """Refresh access token using refresh token."""
        async with httpx.AsyncClient() as client:
            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self.config.client_id,
            }

            # Send as JSON with required beta header
            headers = {
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "oauth-2025-04-20",
            }

            response = await client.post(
                self.config.token_url,
                json=data,  # Send as JSON
                headers=headers,
            )

            if response.status_code != 200:
                logger.error(
                    f"Token refresh failed: {response.status_code} - {response.text}"
                )
                raise Exception(f"Token refresh failed: {response.text}")

            token_data = response.json()

            # Calculate expiration time
            expires_in = token_data.get("expires_in", 3600)
            expires_at = time.time() + expires_in

            token_info = TokenInfo(
                access_token=token_data["access_token"],
                refresh_token=token_data.get("refresh_token", refresh_token),
                expires_at=expires_at,
                token_type=token_data.get("token_type", "Bearer"),
            )

            # Store updated token
            self.store_token(token_info)

            return token_info

    def store_token(self, token_info: TokenInfo):
        """Store token info to disk."""
        try:
            with open(self.token_storage_path, "w") as f:
                json.dump(token_info.model_dump(), f)
            logger.info("OAuth token stored successfully")
        except Exception as e:
            logger.error(f"Failed to store token: {e}")

    def load_token(self) -> Optional[TokenInfo]:
        """Load token info from disk."""
        if not os.path.exists(self.token_storage_path):
            return None

        try:
            with open(self.token_storage_path, "r") as f:
                data = json.load(f)
            return TokenInfo(**data)
        except Exception as e:
            logger.error(f"Failed to load token: {e}")
            return None

    async def get_valid_token(self) -> Optional[str]:
        """Get a valid access token, refreshing if necessary."""
        token_info = self.load_token()
        if not token_info:
            return None

        # Check if token is expired or about to expire (5 min buffer)
        if token_info.expires_in_seconds <= 300:
            if token_info.refresh_token:
                try:
                    logger.info("Token expired or expiring soon, refreshing...")
                    token_info = await self.refresh_access_token(
                        token_info.refresh_token
                    )
                except Exception as e:
                    logger.error(f"Failed to refresh token: {e}")
                    return None
            else:
                logger.error("Token expired and no refresh token available")
                return None

        return token_info.access_token

    def clear_tokens(self):
        """Clear stored tokens."""
        if os.path.exists(self.token_storage_path):
            os.remove(self.token_storage_path)
            logger.info("OAuth tokens cleared")

    async def interactive_oauth_flow(self) -> bool:
        """
        Run interactive OAuth flow from command line.

        Returns:
            True if authentication successful, False otherwise
        """
        logger.info("\n" + "=" * 60)
        logger.info("OAuth Authentication Required")
        logger.info("=" * 60)

        # Generate authorization URL
        auth_url, state, code_verifier = self.get_authorization_url()

        logger.info("\nPlease visit this URL to authenticate:")
        logger.info(f"\n{auth_url}\n")
        logger.info("After authorizing, you'll be redirected to a URL that looks like:")
        logger.info(
            "https://console.anthropic.com/oauth/code/callback?code=XXX&state=YYY"
        )

        # Ask for the authorization code
        logger.info("\nAfter authorizing, you will be redirected to a callback page.")
        logger.info("\n" + "=" * 60)
        logger.info("IMPORTANT: Look for an authorization code displayed ON THE PAGE")
        logger.info("It will look like: XXXXXXXXXX#YYYYYYYYYY")
        logger.info("(long string with # in the middle)")
        logger.info("=" * 60)
        logger.info("\nPlease paste the authorization code from the page:")
        user_input = input("Authorization code: ").strip()

        if not user_input:
            logger.info("No input provided. OAuth setup cancelled.")
            return False

        # Parse the input
        auth_code = None
        returned_state = None

        # Check if input contains # separator (format: code#state)
        if "#" in user_input and not user_input.startswith("http"):
            # This is the authorization code from the page in format: code#state
            parts = user_input.split("#")
            auth_code = parts[0]
            returned_state = parts[1] if len(parts) > 1 else None
            logger.info("\nParsed authorization code from page display")
            logger.info(f"Code: {auth_code[:10]}...")
            logger.info(
                f"State: {returned_state[:10]}..." if returned_state else "State: None"
            )
        else:
            # Try to parse as URL
            try:
                from urllib.parse import urlparse, parse_qs

                parsed = urlparse(user_input)
                params = parse_qs(parsed.query)

                # Extract code and state from URL parameters
                url_code = params.get("code", [None])[0]
                url_state = params.get("state", [None])[0]

                if url_code:
                    # URL parsing successful, but this might not be the right code
                    # The actual auth code might be displayed on the page
                    logger.info(
                        "\nFound URL parameters - but the actual authorization code might be different!"
                    )
                    logger.info(f"URL Code: {url_code[:10]}...")
                    logger.info(
                        f"URL State: {url_state[:10]}..."
                        if url_state
                        else "URL State: None"
                    )
                    logger.info(
                        "\nIf you see a different authorization code on the page itself,"
                    )
                    logger.info(
                        "please run the OAuth flow again and provide that code instead."
                    )

                    # For now, use what we found
                    auth_code = url_code
                    returned_state = url_state

            except Exception:
                # Not a valid URL, treat as plain code
                auth_code = user_input
                returned_state = state

        if not auth_code:
            logger.info("No authorization code found. OAuth setup cancelled.")
            return False

        # Verify state matches (for security)
        if returned_state and returned_state != state:
            logger.warning("\nWarning: State mismatch!")
            logger.warning(f"Expected: {state}")
            logger.warning(f"Received: {returned_state}")
            # Continue anyway, as the state might be handled differently by Anthropic

        try:
            # Exchange code for token
            logger.info("\nExchanging authorization code for access token...")
            # Use the state from the redirect URL, not the one we generated
            token_info = await self.exchange_code_for_token(
                auth_code, code_verifier, returned_state
            )

            logger.info("✓ OAuth authentication successful!")
            logger.info(
                f"✓ Token expires in {token_info.expires_in_seconds:.0f} seconds"
            )
            if token_info.refresh_token:
                logger.info("✓ Refresh token obtained (automatic renewal enabled)")

            return True

        except Exception as e:
            logger.error(f"\n✗ OAuth authentication failed: {e}")
            return False

    def is_oauth_configured(self) -> bool:
        """Check if OAuth is configured (has client_id)."""
        return bool(self.config.client_id)


# Global OAuth manager instance
oauth_manager = OAuthManager()
