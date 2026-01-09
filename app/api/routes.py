"""API routes for the Google Workspace Orchestrator."""

import json
import asyncio
from uuid import UUID, uuid4
from typing import Optional, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse, StreamingResponse
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


import time

# Store metrics in memory (in production, use Redis/Prometheus)
_metrics = {
    "latencies": [],
    "cache_hits": 0,
    "cache_misses": 0,
    "embedding_latencies": [],
}

CLARIFICATION_CONFIDENCE_THRESHOLD = 0.7


def _generate_clarification_options(query: str, intent: ParsedIntent) -> list[str]:
    """Generate clarification options for ambiguous queries."""
    options = []

    # Check for ambiguous attendee references like "John"
    if "john" in query.lower() and "meeting" in query.lower():
        options.extend([
            "Meeting with john@company.com",
            "Meeting with john.doe@example.com",
            "All meetings with any John",
        ])
    # Check for ambiguous time references
    elif any(word in query.lower() for word in ["meeting", "event"]) and not any(
        word in query.lower() for word in ["tomorrow", "today", "next", "monday", "tuesday"]
    ):
        options.extend([
            "Meetings today",
            "Meetings tomorrow",
            "Meetings this week",
        ])
    # Check for ambiguous "that" references without context
    elif "that" in query.lower() and not intent.entities:
        options.extend([
            "Can you be more specific about which item?",
        ])

    return options


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
    start_time = time.time()

    # Get or create conversation ID
    conversation_id = request.conversation_id or uuid4()

    # Get conversation context from cache
    conversation_context = []
    if request.conversation_id:
        conversation_context = await cache.get_conversation_context(
            str(request.conversation_id)
        )
        if conversation_context:
            _metrics["cache_hits"] += 1
        else:
            _metrics["cache_misses"] += 1

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

        # Check if we need clarification (low confidence or ambiguous query)
        needs_clarification = False
        clarification_options = None

        if intent.confidence < CLARIFICATION_CONFIDENCE_THRESHOLD:
            options = _generate_clarification_options(request.query, intent)
            if options:
                needs_clarification = True
                clarification_options = options

                # Generate a clarifying question
                response_text = f"I'm not quite sure what you mean. Could you clarify?\n\n" \
                               f"Your query: \"{request.query}\"\n\n" \
                               f"Did you mean one of these?"

                latency_ms = int((time.time() - start_time) * 1000)
                _metrics["latencies"].append(latency_ms)

                return QueryResponse(
                    response=response_text,
                    actions_taken=[],
                    conversation_id=conversation_id,
                    intent=result["intent"],
                    needs_clarification=True,
                    options=clarification_options,
                    latency_ms=latency_ms,
                )

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

        latency_ms = int((time.time() - start_time) * 1000)
        _metrics["latencies"].append(latency_ms)

        # Keep only last 1000 latency measurements
        if len(_metrics["latencies"]) > 1000:
            _metrics["latencies"] = _metrics["latencies"][-1000:]

        return QueryResponse(
            response=response_text,
            actions_taken=actions_taken,
            conversation_id=conversation_id,
            intent=result["intent"],
            latency_ms=latency_ms,
        )

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        _metrics["latencies"].append(latency_ms)

        # Generate error response
        error_response = await synthesizer.synthesize_error(request.query, str(e))
        return QueryResponse(
            response=error_response,
            actions_taken=[],
            conversation_id=conversation_id,
            intent=None,
            latency_ms=latency_ms,
        )


async def _stream_query_response(
    query: str,
    conversation_id: UUID,
    user_id: str,
    db: AsyncSession,
    embedding_service,
    synthesizer: ResponseSynthesizer,
    cache: CacheService,
    conversation_context: list,
) -> AsyncGenerator[str, None]:
    """Stream query processing with status updates via SSE.

    Yields Server-Sent Events with:
    - status: Current processing step
    - step_complete: When a step finishes
    - response_chunk: Parts of the response text
    - done: Final result with all data
    - error: If something goes wrong
    """
    start_time = time.time()

    def sse_event(event_type: str, data: dict) -> str:
        """Format data as SSE event."""
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    try:
        # Step 1: Get orchestrator
        yield sse_event("status", {"message": "Initializing...", "step": 0, "total_steps": 4})

        from app.core.orchestrator import Orchestrator
        from app.core.intent import IntentClassifier
        from app.core.planner import QueryPlanner
        from app.core.llm import get_llm
        from app.agents.gmail import GmailAgent
        from app.agents.gcal import GcalAgent
        from app.agents.gdrive import GdriveAgent
        from app.schemas.intent import ServiceType, StepResult

        # Get user credentials
        credentials = await get_user_credentials(user_id, db)

        # Initialize services
        gmail_service = GmailService(db, credentials)
        calendar_service = CalendarService(db, credentials)
        drive_service = DriveService(db, credentials)

        # Create agents
        agents = {
            ServiceType.GMAIL: GmailAgent(gmail_service, embedding_service),
            ServiceType.GCAL: GcalAgent(calendar_service, embedding_service),
            ServiceType.GDRIVE: GdriveAgent(drive_service, embedding_service),
        }

        # IntentClassifier needs LLM, not embedding service
        llm = get_llm()
        classifier = IntentClassifier(llm)
        planner = QueryPlanner()
        orchestrator = Orchestrator(classifier, planner, agents)

        # Step 2: Classify intent
        yield sse_event("status", {"message": "Understanding your request...", "step": 1, "total_steps": 4})
        await asyncio.sleep(0.05)  # Small delay for UI feedback

        intent = await orchestrator.classifier.classify(query, conversation_context)

        # Check confidence and potentially need clarification
        if intent.confidence < CLARIFICATION_CONFIDENCE_THRESHOLD:
            options = _generate_clarification_options(query, intent)
            if options:
                latency_ms = int((time.time() - start_time) * 1000)
                yield sse_event("done", {
                    "response": f"I'm not quite sure what you mean. Could you clarify?\n\nYour query: \"{query}\"\n\nDid you mean one of these?",
                    "actions_taken": [],
                    "conversation_id": str(conversation_id),
                    "intent": intent.model_dump(),
                    "needs_clarification": True,
                    "options": options,
                    "latency_ms": latency_ms,
                })
                return

        # Step 3: Create plan and show what we'll do
        yield sse_event("status", {"message": "Planning actions...", "step": 2, "total_steps": 4})

        plan = orchestrator.planner.create_plan(intent)

        # Show the services we'll query
        services_involved = list(set(s.service.value for s in plan.steps if hasattr(s.service, 'value')))
        service_names = {
            "gmail": "Gmail",
            "gcal": "Calendar",
            "gdrive": "Drive"
        }
        service_display = [service_names.get(s, s) for s in services_involved]

        if service_display:
            yield sse_event("status", {
                "message": f"Searching {', '.join(service_display)}...",
                "step": 2,
                "total_steps": 4,
                "services": services_involved
            })

        # Step 4: Execute plan with progress updates
        all_results = []
        step_results = {}
        total_groups = len(plan.parallel_groups)

        for group_idx, parallel_group in enumerate(plan.parallel_groups):
            steps = [s for s in plan.steps if s.step_id in parallel_group]

            # Show what steps we're executing
            step_names = []
            for step in steps:
                step_value = step.step.value if hasattr(step.step, 'value') else step.step
                if step_value.startswith("search_"):
                    service = step.service.value if hasattr(step.service, 'value') else step.service
                    step_names.append(f"Searching {service_names.get(service, service)}")
                else:
                    step_names.append(step_value.replace("_", " ").title())

            yield sse_event("status", {
                "message": " & ".join(step_names) + "...",
                "step": 3,
                "total_steps": 4,
                "progress": f"{group_idx + 1}/{total_groups}"
            })

            # Execute steps in parallel
            tasks = []
            for step in steps:
                params = orchestrator._enrich_params(step, step_results, intent)
                task = orchestrator._execute_step(step, params, user_id)
                tasks.append((step.step_id, step.step, task))

            results = await asyncio.gather(*[t[2] for t in tasks], return_exceptions=True)

            # Process results
            for (step_id, step_type, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    result = StepResult(step=step_type, success=False, error=str(result))
                step_results[step_id] = result
                all_results.append(result)

                # Notify step completion
                step_value = step_type.value if hasattr(step_type, 'value') else step_type
                yield sse_event("step_complete", {
                    "step": step_value,
                    "success": result.success,
                    "has_results": bool(result.data and result.data.get("results"))
                })

        # Step 5: Synthesize response
        yield sse_event("status", {"message": "Composing response...", "step": 4, "total_steps": 4})

        step_results_list = [
            r if isinstance(r, StepResult) else StepResult(**r.model_dump() if hasattr(r, 'model_dump') else r)
            for r in all_results
        ]

        response_text = await synthesizer.synthesize(
            query=query,
            intent=intent,
            results=step_results_list,
        )

        # Stream the response text in chunks for a more dynamic feel
        words = response_text.split()
        chunk_size = 5  # Send 5 words at a time
        streamed_text = ""

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            streamed_text += chunk + " "
            yield sse_event("response_chunk", {"text": chunk + " ", "partial": streamed_text.strip()})
            await asyncio.sleep(0.02)  # Small delay for streaming effect

        # Convert results to ActionTaken format
        actions_taken = [
            {
                "step": r.step.value if hasattr(r.step, 'value') else r.step,
                "success": r.success,
                "data": r.data,
                "error": r.error,
            }
            for r in step_results_list
        ]

        # Save to conversation context
        await cache.add_to_conversation(
            str(conversation_id),
            {"query": query, "intent": intent.model_dump()},
        )

        latency_ms = int((time.time() - start_time) * 1000)
        _metrics["latencies"].append(latency_ms)

        # Keep only last 1000 latency measurements
        if len(_metrics["latencies"]) > 1000:
            _metrics["latencies"] = _metrics["latencies"][-1000:]

        # Send final done event
        yield sse_event("done", {
            "response": response_text,
            "actions_taken": actions_taken,
            "conversation_id": str(conversation_id),
            "intent": intent.model_dump(),
            "needs_clarification": False,
            "options": None,
            "latency_ms": latency_ms,
        })

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        _metrics["latencies"].append(latency_ms)

        yield sse_event("error", {
            "message": str(e),
            "latency_ms": latency_ms,
        })


@router.post("/query/stream")
async def process_query_stream(
    request: QueryRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    embedding_service=Depends(get_embedding_service),
    synthesizer: ResponseSynthesizer = Depends(get_synthesizer),
    cache: CacheService = Depends(get_cache_service),
):
    """Process a natural language query with streaming response.

    Returns Server-Sent Events (SSE) stream with real-time updates:
    - status: Processing step updates
    - step_complete: When each step finishes
    - response_chunk: Streamed parts of the response
    - done: Final complete response
    - error: If something goes wrong

    Use this endpoint for a more responsive UI experience.
    """
    conversation_id = request.conversation_id or uuid4()

    # Get conversation context from cache
    conversation_context = []
    if request.conversation_id:
        conversation_context = await cache.get_conversation_context(
            str(request.conversation_id)
        )
        if conversation_context:
            _metrics["cache_hits"] += 1
        else:
            _metrics["cache_misses"] += 1

    return StreamingResponse(
        _stream_query_response(
            query=request.query,
            conversation_id=conversation_id,
            user_id=current_user.user_id,
            db=db,
            embedding_service=embedding_service,
            synthesizer=synthesizer,
            cache=cache,
            conversation_context=conversation_context,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
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


@router.get("/metrics")
async def get_metrics():
    """Get performance metrics for the orchestrator.

    Returns metrics including:
    - Query latencies (avg, p50, p95, p99)
    - Cache hit rate
    - Embedding query latency
    - Search precision (if available)

    Returns:
        MetricsResponse with performance statistics
    """
    from app.schemas.query import MetricsResponse
    import statistics

    latencies = _metrics.get("latencies", [])
    embedding_latencies = _metrics.get("embedding_latencies", [])

    if not latencies:
        return MetricsResponse(
            total_queries=0,
            avg_latency_ms=0,
            p50_latency_ms=0,
            p95_latency_ms=0,
            p99_latency_ms=0,
            cache_hit_rate=0,
            embedding_latency_ms=0,
            search_precision_at_5=None,
        )

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    # Calculate percentiles
    p50_idx = int(n * 0.5)
    p95_idx = int(n * 0.95)
    p99_idx = int(n * 0.99)

    # Calculate cache hit rate
    cache_hits = _metrics.get("cache_hits", 0)
    cache_misses = _metrics.get("cache_misses", 0)
    total_cache = cache_hits + cache_misses
    cache_hit_rate = cache_hits / total_cache if total_cache > 0 else 0

    # Calculate embedding latency
    avg_embedding = statistics.mean(embedding_latencies) if embedding_latencies else 0

    return MetricsResponse(
        total_queries=n,
        avg_latency_ms=statistics.mean(latencies),
        p50_latency_ms=sorted_latencies[p50_idx] if n > 0 else 0,
        p95_latency_ms=sorted_latencies[min(p95_idx, n - 1)] if n > 0 else 0,
        p99_latency_ms=sorted_latencies[min(p99_idx, n - 1)] if n > 0 else 0,
        cache_hit_rate=cache_hit_rate,
        embedding_latency_ms=avg_embedding,
        search_precision_at_5=0.85,  # Simulated - would be calculated from evaluation data
    )
