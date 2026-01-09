"""API routes for the Google Workspace Orchestrator."""

from uuid import UUID, uuid4
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db
from app.db.models import User, Conversation, Message, SyncStatus
from app.schemas.query import (
    QueryRequest,
    QueryResponse,
    ActionTaken,
    SyncTriggerRequest,
    SyncStatusResponse,
    HealthResponse,
)
from app.schemas.intent import ParsedIntent, StepResult
from app.api.dependencies import (
    get_orchestrator_for_user,
    get_synthesizer,
    get_cache_service,
    get_embedding_service,
    get_user_credentials,
    get_current_user,
    get_current_user_optional,
    CurrentUser,
)
from app.core.orchestrator import Orchestrator
from app.services.google.gmail import GmailService
from app.services.google.calendar import CalendarService
from app.services.google.drive import DriveService
from app.core.synthesizer import ResponseSynthesizer
from app.services.cache import CacheService
from app.services.google.auth import GoogleAuthService
from app.config import settings


router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    embedding_service=Depends(get_embedding_service),
    synthesizer: ResponseSynthesizer = Depends(get_synthesizer),
    cache: CacheService = Depends(get_cache_service),
):
    """Process a natural language query.

    This is the main endpoint for the orchestrator. It:
    1. Classifies the user's intent
    2. Creates an execution plan
    3. Executes the plan across services
    4. Synthesizes a natural language response

    Requires authentication via session cookie.

    Args:
        request: The query request with user's natural language input
        current_user: The authenticated user
        db: Database session
        embedding_service: Embedding service
        synthesizer: The response synthesizer
        cache: Cache service for conversation context

    Returns:
        QueryResponse with the natural language response and actions taken
    """
    # Get or create conversation ID
    conversation_id = request.conversation_id or uuid4()

    # Get conversation context from cache
    conversation_context = []
    if request.conversation_id:
        conversation_context = await cache.get_conversation_context(
            str(request.conversation_id)
        )

    try:
        # Get orchestrator for current user
        orchestrator = await get_orchestrator_for_user(
            current_user.user_id, db, embedding_service
        )

        # Execute the query
        result = await orchestrator.execute_query(
            query=request.query,
            user_id=current_user.user_id,
            conversation_context=conversation_context,
        )

        # Extract intent and results
        intent = ParsedIntent(**result["intent"])
        step_results = [
            r if isinstance(r, StepResult) else StepResult(**r.model_dump() if hasattr(r, 'model_dump') else r)
            for r in result["results"]
        ]

        # Synthesize response
        response_text = await synthesizer.synthesize(
            query=request.query,
            intent=intent,
            results=step_results,
        )

        # Convert results to ActionTaken format
        actions_taken = [
            ActionTaken(
                step=r.step.value if hasattr(r.step, 'value') else r.step,
                success=r.success,
                data=r.data,
                error=r.error,
            )
            for r in step_results
        ]

        # Save to conversation context
        await cache.add_to_conversation(
            str(conversation_id),
            {"query": request.query, "intent": result["intent"]},
        )

        # Create conversation if new
        if not request.conversation_id:
            from app.db.models import Conversation
            conversation = Conversation(
                id=conversation_id,
                user_id=UUID(current_user.user_id),
            )
            db.add(conversation)

        # Save message to database
        message = Message(
            conversation_id=conversation_id,
            query=request.query,
            intent=result["intent"],
            response=response_text,
            actions_taken=result["plan"],
        )
        db.add(message)
        await db.commit()

        return QueryResponse(
            response=response_text,
            actions_taken=actions_taken,
            conversation_id=conversation_id,
            intent=result["intent"],
        )

    except Exception as e:
        # Generate error response
        error_response = await synthesizer.synthesize_error(request.query, str(e))
        return QueryResponse(
            response=error_response,
            actions_taken=[],
            conversation_id=conversation_id,
            intent=None,
        )


@router.post("/sync/trigger")
async def trigger_sync(
    request: SyncTriggerRequest = SyncTriggerRequest(),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Trigger a manual sync of Google services.

    Kicks off Celery background tasks to sync data from Google APIs
    and generate embeddings for semantic search.

    Args:
        request: Sync trigger request with optional service filter
        current_user: Authenticated user
        db: Database session

    Returns:
        Status with task IDs
    """
    from app.tasks.sync_tasks import (
        sync_gmail_task,
        sync_calendar_task,
        sync_drive_task,
        sync_all_task,
    )

    user_id = current_user.user_id
    task_ids = {}

    if request.service == "all" or request.service is None:
        # Trigger all syncs
        task = sync_all_task.delay(user_id)
        task_ids["all"] = task.id
    elif request.service == "gmail":
        task = sync_gmail_task.delay(user_id)
        task_ids["gmail"] = task.id
    elif request.service == "gcal":
        task = sync_calendar_task.delay(user_id)
        task_ids["gcal"] = task.id
    elif request.service == "gdrive":
        task = sync_drive_task.delay(user_id)
        task_ids["gdrive"] = task.id
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown service: {request.service}. Use 'gmail', 'gcal', 'gdrive', or 'all'.",
        )

    return {
        "status": "sync_triggered",
        "service": request.service or "all",
        "task_ids": task_ids,
        "message": f"Sync triggered for {request.service or 'all'}. Check /sync/status for progress.",
    }


@router.get("/sync/status", response_model=SyncStatusResponse)
async def get_sync_status(
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the current sync status for all services.

    Requires authentication.

    Returns:
        SyncStatusResponse with last sync times and status
    """
    # Query sync status from database
    result = await db.execute(
        select(SyncStatus).where(SyncStatus.user_id == UUID(current_user.user_id))
    )
    statuses = {s.service: s for s in result.scalars().all()}

    gmail_status = statuses.get("gmail")
    gcal_status = statuses.get("gcal")
    gdrive_status = statuses.get("gdrive")

    return SyncStatusResponse(
        gmail_last_sync=gmail_status.last_sync_at if gmail_status else None,
        gmail_status=gmail_status.status if gmail_status else "never_synced",
        gcal_last_sync=gcal_status.last_sync_at if gcal_status else None,
        gcal_status=gcal_status.status if gcal_status else "never_synced",
        gdrive_last_sync=gdrive_status.last_sync_at if gdrive_status else None,
        gdrive_status=gdrive_status.status if gdrive_status else "never_synced",
    )


@router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    """Health check endpoint.

    Verifies database and Redis connectivity.

    Returns:
        HealthResponse with service statuses
    """
    # Check database
    db_status = "connected"
    try:
        await db.execute(select(1))
    except Exception:
        db_status = "disconnected"

    # Check Redis
    redis_status = "connected"
    try:
        cache = get_cache_service()
        await cache.set("health_check", "ok", ttl=10)
    except Exception:
        redis_status = "disconnected"

    return HealthResponse(
        status="healthy" if db_status == "connected" else "degraded",
        database=db_status,
        redis=redis_status,
    )


@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: UUID,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
):
    """Get messages from a conversation.

    Args:
        conversation_id: The conversation UUID
        limit: Max messages to return
        db: Database session

    Returns:
        List of messages
    """
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    messages = result.scalars().all()

    return [
        {
            "id": str(m.id),
            "query": m.query,
            "response": m.response,
            "intent": m.intent,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in messages
    ]


# =============================================================================
# Authentication Routes
# =============================================================================


@router.get("/auth/login")
async def google_login(db: AsyncSession = Depends(get_db)):
    """Initiate Google OAuth login flow.

    Redirects user to Google's consent screen.

    Returns:
        Redirect to Google OAuth consent page
    """
    if settings.use_mock_google:
        raise HTTPException(
            status_code=400,
            detail="Google OAuth is disabled in mock mode. Set USE_MOCK_GOOGLE=false in .env",
        )

    auth_service = GoogleAuthService(db)
    authorization_url, state = auth_service.get_authorization_url()

    return RedirectResponse(url=authorization_url)


@router.get("/auth/callback")
async def google_callback(
    request: Request,
    code: str,
    state: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Handle Google OAuth callback.

    Exchanges authorization code for tokens and creates/updates user.
    Sets session cookie for subsequent authenticated requests.

    Args:
        request: FastAPI request object (for session access)
        code: Authorization code from Google
        state: State parameter for CSRF protection
        db: Database session

    Returns:
        Redirect to home page with session set
    """
    if settings.use_mock_google:
        raise HTTPException(
            status_code=400,
            detail="Google OAuth is disabled in mock mode",
        )

    try:
        auth_service = GoogleAuthService(db)
        result = await auth_service.handle_callback(code, state)

        # Set user_id in session
        request.session["user_id"] = result["user_id"]
        request.session["email"] = result["email"]

        # Redirect to home page instead of returning JSON
        return RedirectResponse(url="/?auth=success", status_code=302)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"OAuth callback failed: {str(e)}",
        )


@router.post("/auth/logout")
async def google_logout(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Revoke Google OAuth tokens and clear session.

    Args:
        request: FastAPI request object (for session access)
        db: Database session

    Returns:
        Logout status
    """
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not logged in")

    auth_service = GoogleAuthService(db)
    success = await auth_service.revoke_tokens(user_id)

    # Clear session
    request.session.clear()

    if success:
        return {"status": "logged_out", "message": "Google tokens revoked and session cleared"}
    else:
        return {"status": "logged_out", "message": "Session cleared (user not found in DB)"}


@router.get("/auth/status")
async def auth_status(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Check authentication status from session.

    Args:
        request: FastAPI request object (for session access)
        db: Database session

    Returns:
        Authentication status and connected services
    """
    user_id = request.session.get("user_id")
    email = request.session.get("email")

    if settings.use_mock_google:
        return {
            "mode": "mock",
            "authenticated": user_id is not None,
            "user_id": user_id,
            "email": email or "mock@example.com",
            "services": {
                "gmail": "mock",
                "calendar": "mock",
                "drive": "mock",
            },
        }

    if not user_id:
        return {
            "mode": "real",
            "authenticated": False,
            "message": "Not logged in. Visit /api/v1/auth/login to authenticate.",
        }

    # Check if user has valid tokens
    auth_service = GoogleAuthService(db)
    credentials = await auth_service.get_credentials(user_id)

    if credentials:
        return {
            "mode": "real",
            "authenticated": True,
            "user_id": user_id,
            "email": email,
            "services": {
                "gmail": "connected",
                "calendar": "connected",
                "drive": "connected",
            },
        }
    else:
        # Clear invalid session
        request.session.clear()
        return {
            "mode": "real",
            "authenticated": False,
            "message": "Token expired or invalid. Visit /api/v1/auth/login to re-authenticate.",
        }


# =============================================================================
# Direct API Test Routes
# =============================================================================


@router.get("/test/gmail")
async def test_gmail_api(
    current_user: CurrentUser = Depends(get_current_user),
    q: Optional[str] = None,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
):
    """Test Gmail API directly.

    Requires authentication via session.

    Args:
        current_user: Authenticated user from session
        q: Optional search query
        limit: Max results

    Returns:
        List of emails from Gmail API
    """
    credentials = await get_user_credentials(current_user.user_id, db)
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="No valid credentials. Please authenticate first.",
        )

    gmail_service = GmailService(db, credentials)

    try:
        # Search emails with optional filters
        filters = {"subject": q} if q else None
        # Use a dummy embedding for now (real search would use semantic search)
        dummy_embedding = [0.0] * 1536

        emails = await gmail_service.search_emails(
            user_id=current_user.user_id,
            embedding=dummy_embedding,
            filters=filters,
            limit=limit,
        )
        return {"status": "success", "count": len(emails), "emails": emails}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/test/calendar")
async def test_calendar_api(
    current_user: CurrentUser = Depends(get_current_user),
    time_range: Optional[str] = "next_week",
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
):
    """Test Calendar API directly.

    Requires authentication via session.

    Args:
        current_user: Authenticated user from session
        time_range: Time range filter (today, tomorrow, next_week)
        limit: Max results

    Returns:
        List of calendar events
    """
    credentials = await get_user_credentials(current_user.user_id, db)
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="No valid credentials. Please authenticate first.",
        )

    calendar_service = CalendarService(db, credentials)

    try:
        # Search events
        filters = {"time_range": time_range}
        dummy_embedding = [0.0] * 1536

        events = await calendar_service.search_events(
            user_id=current_user.user_id,
            embedding=dummy_embedding,
            filters=filters,
            limit=limit,
        )
        return {"status": "success", "count": len(events), "events": events}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/test/drive")
async def test_drive_api(
    current_user: CurrentUser = Depends(get_current_user),
    name: Optional[str] = None,
    mime_type: Optional[str] = None,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
):
    """Test Drive API directly.

    Requires authentication via session.

    Args:
        current_user: Authenticated user from session
        name: Optional file name filter
        mime_type: Optional MIME type filter
        limit: Max results

    Returns:
        List of Drive files
    """
    credentials = await get_user_credentials(current_user.user_id, db)
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="No valid credentials. Please authenticate first.",
        )

    drive_service = DriveService(db, credentials)

    try:
        # Search files
        filters = {}
        if name:
            filters["name"] = name
        if mime_type:
            filters["mime_type"] = mime_type

        dummy_embedding = [0.0] * 1536

        files = await drive_service.search_files(
            user_id=current_user.user_id,
            embedding=dummy_embedding,
            filters=filters if filters else None,
            limit=limit,
        )
        return {"status": "success", "count": len(files), "files": files}
    except Exception as e:
        return {"status": "error", "error": str(e)}
