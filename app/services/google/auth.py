"""Google OAuth 2.0 authentication handler."""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID
import json

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.config import settings
from app.db.models import User


class GoogleAuthService:
    """Handles Google OAuth 2.0 flow and token management."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.client_config = {
            "web": {
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uris": [settings.google_redirect_uri],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        }

    def get_authorization_url(self, state: Optional[str] = None) -> tuple[str, str]:
        """Generate the Google OAuth authorization URL.

        Args:
            state: Optional state parameter for CSRF protection

        Returns:
            Tuple of (authorization_url, state)
        """
        flow = Flow.from_client_config(
            self.client_config,
            scopes=settings.google_scopes,
            redirect_uri=settings.google_redirect_uri,
        )

        authorization_url, state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
            state=state,
        )

        return authorization_url, state

    async def handle_callback(self, code: str, state: Optional[str] = None) -> dict:
        """Handle the OAuth callback and exchange code for tokens.

        Args:
            code: Authorization code from Google
            state: State parameter for verification

        Returns:
            Dict with user info and tokens
        """
        flow = Flow.from_client_config(
            self.client_config,
            scopes=settings.google_scopes,
            redirect_uri=settings.google_redirect_uri,
        )

        # Exchange code for tokens
        flow.fetch_token(code=code)
        credentials = flow.credentials

        # Get user info from Google
        service = build("oauth2", "v2", credentials=credentials)
        user_info = service.userinfo().get().execute()

        # Store or update user in database
        user = await self._get_or_create_user(
            email=user_info["email"],
            access_token=credentials.token,
            refresh_token=credentials.refresh_token,
            token_expiry=credentials.expiry,
        )

        return {
            "user_id": str(user.id),
            "email": user.email,
            "access_token": credentials.token,
            "expires_at": credentials.expiry.isoformat() if credentials.expiry else None,
        }

    async def _get_or_create_user(
        self,
        email: str,
        access_token: str,
        refresh_token: Optional[str],
        token_expiry: Optional[datetime],
    ) -> User:
        """Get existing user or create new one.

        Args:
            email: User's email
            access_token: OAuth access token
            refresh_token: OAuth refresh token
            token_expiry: Token expiration time

        Returns:
            User model instance
        """
        # Check for existing user
        result = await self.db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

        if user:
            # Update existing user's tokens
            user.google_access_token = access_token
            if refresh_token:
                user.google_refresh_token = refresh_token
            user.token_expires_at = token_expiry
            await self.db.commit()
            await self.db.refresh(user)
        else:
            # Create new user
            user = User(
                email=email,
                google_access_token=access_token,
                google_refresh_token=refresh_token,
                token_expires_at=token_expiry,
            )
            self.db.add(user)
            await self.db.commit()
            await self.db.refresh(user)

        return user

    async def get_credentials(self, user_id: str) -> Optional[Credentials]:
        """Get valid Google credentials for a user.

        Automatically refreshes expired tokens.

        Args:
            user_id: The user's ID

        Returns:
            Valid Credentials object or None
        """
        result = await self.db.execute(
            select(User).where(User.id == UUID(user_id))
        )
        user = result.scalar_one_or_none()

        if not user or not user.google_access_token:
            return None

        credentials = Credentials(
            token=user.google_access_token,
            refresh_token=user.google_refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret,
            scopes=settings.google_scopes,
        )

        # Check if token needs refresh
        if credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())

                # Update stored tokens
                user.google_access_token = credentials.token
                user.token_expires_at = credentials.expiry
                await self.db.commit()
            except Exception as e:
                # Token refresh failed - user needs to re-authenticate
                return None

        return credentials

    async def revoke_tokens(self, user_id: str) -> bool:
        """Revoke user's Google tokens.

        Args:
            user_id: The user's ID

        Returns:
            True if successful
        """
        result = await self.db.execute(
            select(User).where(User.id == UUID(user_id))
        )
        user = result.scalar_one_or_none()

        if user:
            user.google_access_token = None
            user.google_refresh_token = None
            user.token_expires_at = None
            await self.db.commit()
            return True

        return False
