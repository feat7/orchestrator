"""FastAPI dependencies for dependency injection."""

from functools import lru_cache
from typing import Optional
from uuid import UUID
from dataclasses import dataclass

from fastapi import Depends, Request, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from google.oauth2.credentials import Credentials

from app.db.database import get_db
from app.db.models import User
from app.core.llm import get_llm, LLMProvider
from app.core.intent import IntentClassifier
from app.core.planner import QueryPlanner
from app.core.orchestrator import Orchestrator
from app.core.synthesizer import ResponseSynthesizer
from app.services.embedding import EmbeddingService
from app.services.cache import CacheService
from app.services.google.gmail import GmailService
from app.services.google.calendar import CalendarService
from app.services.google.drive import DriveService
from app.agents.gmail import GmailAgent
from app.agents.gcal import GcalAgent
from app.agents.gdrive import GdriveAgent
from app.schemas.intent import ServiceType
from app.config import settings


# Singleton cache service
_cache_service: Optional[CacheService] = None


@dataclass
class CurrentUser:
    """Current authenticated user."""
    user_id: str
    email: Optional[str] = None


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> CurrentUser:
    """Get the current authenticated user from session.

    Args:
        request: The FastAPI request object
        db: Database session

    Returns:
        CurrentUser object

    Raises:
        HTTPException: If user is not authenticated
    """
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated. Please login at /api/v1/auth/login",
        )

    # Optionally fetch user email from DB
    try:
        result = await db.execute(
            select(User).where(User.id == UUID(user_id))
        )
        user = result.scalar_one_or_none()
        email = user.email if user else None
    except Exception:
        email = None

    return CurrentUser(user_id=user_id, email=email)


async def get_current_user_optional(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Optional[CurrentUser]:
    """Get the current user if authenticated, otherwise None.

    Args:
        request: The FastAPI request object
        db: Database session

    Returns:
        CurrentUser object or None
    """
    user_id = request.session.get("user_id")
    if not user_id:
        return None

    try:
        result = await db.execute(
            select(User).where(User.id == UUID(user_id))
        )
        user = result.scalar_one_or_none()
        email = user.email if user else None
    except Exception:
        email = None

    return CurrentUser(user_id=user_id, email=email)


def get_cache_service() -> CacheService:
    """Get singleton cache service instance.

    Returns:
        CacheService instance
    """
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service


def get_llm_provider() -> LLMProvider:
    """Get LLM provider instance.

    Returns:
        Configured LLM provider
    """
    return get_llm()


def get_embedding_service(
    cache: CacheService = Depends(get_cache_service),
) -> EmbeddingService:
    """Get embedding service with cache.

    Args:
        cache: Cache service instance

    Returns:
        EmbeddingService instance
    """
    return EmbeddingService(cache=cache)


async def get_user_credentials(
    user_id: str,
    db: AsyncSession,
) -> Optional[Credentials]:
    """Get Google credentials for a user.

    Args:
        user_id: The user's ID
        db: Database session

    Returns:
        Google Credentials object or None
    """
    if settings.use_mock_google:
        return None

    try:
        result = await db.execute(
            select(User).where(User.id == UUID(user_id))
        )
        user = result.scalar_one_or_none()

        if user and user.google_access_token:
            return Credentials(
                token=user.google_access_token,
                refresh_token=user.google_refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=settings.google_client_id,
                client_secret=settings.google_client_secret,
                scopes=settings.google_scopes,
            )
    except Exception as e:
        print(f"Error getting credentials: {e}")

    return None


async def get_orchestrator_for_user(
    user_id: str,
    db: AsyncSession,
    embedding_service: EmbeddingService,
) -> Orchestrator:
    """Get fully configured orchestrator for a specific user.

    Args:
        user_id: The user's ID
        db: Database session
        embedding_service: Embedding service

    Returns:
        Configured Orchestrator instance
    """
    llm = get_llm()

    # Get user credentials for real API mode
    credentials = await get_user_credentials(user_id, db)

    # Initialize services with credentials
    gmail_service = GmailService(db, credentials)
    calendar_service = CalendarService(db, credentials)
    drive_service = DriveService(db, credentials)

    # Initialize agents
    gmail_agent = GmailAgent(gmail_service, embedding_service)
    gcal_agent = GcalAgent(calendar_service, embedding_service)
    gdrive_agent = GdriveAgent(drive_service, embedding_service)

    # Map agents to service types
    agents = {
        ServiceType.GMAIL: gmail_agent,
        ServiceType.GCAL: gcal_agent,
        ServiceType.GDRIVE: gdrive_agent,
    }

    # Initialize core components
    classifier = IntentClassifier(llm)
    planner = QueryPlanner()

    return Orchestrator(classifier, planner, agents)


def get_synthesizer() -> ResponseSynthesizer:
    """Get response synthesizer.

    Returns:
        ResponseSynthesizer instance
    """
    return ResponseSynthesizer(get_llm())


async def get_gmail_service(db: AsyncSession = Depends(get_db)) -> GmailService:
    """Get Gmail service.

    Args:
        db: Database session

    Returns:
        GmailService instance
    """
    return GmailService(db)


async def get_calendar_service(db: AsyncSession = Depends(get_db)) -> CalendarService:
    """Get Calendar service.

    Args:
        db: Database session

    Returns:
        CalendarService instance
    """
    return CalendarService(db)


async def get_drive_service(db: AsyncSession = Depends(get_db)) -> DriveService:
    """Get Drive service.

    Args:
        db: Database session

    Returns:
        DriveService instance
    """
    return DriveService(db)
