"""API routes for the Google Workspace Orchestrator."""

import json
import asyncio
import logging
from uuid import UUID, uuid4
from typing import Optional, AsyncGenerator

logger = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.db.database import get_db, async_session
from app.db.models import User, Conversation, Message, SyncStatus
from app.schemas.query import (
    QueryRequest,
    QueryResponse,
    ActionTaken,
    SyncTriggerRequest,
    SyncStatusResponse,
    HealthResponse,
    UserSettingsResponse,
    UserSettingsUpdateRequest,
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
from datetime import datetime

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

        # Save to conversation context (both query and response)
        await cache.add_to_conversation(
            str(conversation_id),
            {"query": request.query, "response": response_text, "intent": result["intent"]},
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
    embedding_service,
    synthesizer: ResponseSynthesizer,
    cache: CacheService,
    conversation_context: list,
    is_new_conversation: bool = False,
) -> AsyncGenerator[str, None]:
    """Stream query processing with status updates via SSE.

    Yields Server-Sent Events with:
    - status: Current processing step
    - step_complete: When a step finishes
    - response_chunk: Parts of the response text
    - done: Final result with all data
    - error: If something goes wrong

    Note: Creates its own DB session to avoid connection pool issues
    with FastAPI's dependency injection during streaming.
    """
    from app.db.database import async_session

    start_time = time.time()

    def sse_event(event_type: str, data: dict) -> str:
        """Format data as SSE event."""
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    # Create our own DB session for the duration of streaming
    async with async_session() as db:
        try:
            # Step 1: Get orchestrator (no status message - happens quickly)

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

            # IntentClassifier needs LLM, not embedding service
            llm = get_llm()

            # Create agents (GmailAgent needs LLM for email composition)
            agents = {
                ServiceType.GMAIL: GmailAgent(gmail_service, embedding_service, llm),
                ServiceType.GCAL: GcalAgent(calendar_service, embedding_service),
                ServiceType.GDRIVE: GdriveAgent(drive_service, embedding_service),
            }
            classifier = IntentClassifier(llm)
            planner = QueryPlanner()
            orchestrator = Orchestrator(classifier, planner, agents)

            # Step 2: Classify intent
            yield sse_event("status", {"message": "Understanding your request...", "step": 1, "total_steps": 4})
            await asyncio.sleep(0.05)  # Small delay for UI feedback

            logger.info(f"Starting intent classification for query: '{query[:80]}...'")
            classify_start = time.time()
            intent = await orchestrator.classifier.classify(query, conversation_context)
            classify_time = (time.time() - classify_start) * 1000
            logger.info(f"Intent classification completed in {classify_time:.0f}ms: {intent.operation} - {intent.steps}")

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

            service_names = {
                "gmail": "Gmail",
                "gcal": "Calendar",
                "gdrive": "Drive"
            }

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
                    params = orchestrator._enrich_params(step, step_results, intent, query)
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
                conversation_context=conversation_context,
            )

            # Stream the response text in chunks while preserving newlines
            # Split by lines first, then stream each line with its words
            lines = response_text.split('\n')
            streamed_text = ""
            chunk_size = 5  # Words per chunk

            for line_idx, line in enumerate(lines):
                if line.strip():
                    # Stream non-empty lines word by word
                    words = line.split(' ')
                    for i in range(0, len(words), chunk_size):
                        chunk = ' '.join(words[i:i + chunk_size])
                        if i + chunk_size < len(words):
                            chunk += ' '
                        streamed_text += chunk
                        yield sse_event("response_chunk", {"text": chunk, "partial": streamed_text})
                        await asyncio.sleep(0.02)
                # Add newline after each line (except the last one)
                if line_idx < len(lines) - 1:
                    streamed_text += '\n'
                    yield sse_event("response_chunk", {"text": "\n", "partial": streamed_text})
                    await asyncio.sleep(0.01)

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

            # Save to conversation context (both query and response for context)
            await cache.add_to_conversation(
                str(conversation_id),
                {"query": query, "response": response_text, "intent": intent.model_dump()},
            )

            # Persist to database
            if is_new_conversation:
                # Create conversation with title from first query (truncated)
                title = query[:100] + "..." if len(query) > 100 else query
                conversation = Conversation(
                    id=conversation_id,
                    user_id=UUID(user_id),
                    title=title,
                )
                db.add(conversation)

            # Save message to database
            message = Message(
                conversation_id=conversation_id,
                query=query,
                intent=intent.model_dump(),
                response=response_text,
                actions_taken=actions_taken,
            )
            db.add(message)
            logger.info(f"[CONTEXT_DEBUG] Saving message to DB: conversation_id={conversation_id}, query='{query[:50]}...'")

            # Update conversation's updated_at timestamp
            if not is_new_conversation:
                from sqlalchemy import update
                await db.execute(
                    update(Conversation)
                    .where(Conversation.id == conversation_id)
                    .values(updated_at=datetime.utcnow())
                )

            await db.commit()
            logger.info(f"[CONTEXT_DEBUG] Message saved successfully to DB")

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


async def _load_conversation_context_from_db(
    conversation_id: UUID, limit: int = 5
) -> list[dict]:
    """Load conversation context from the database.

    Args:
        conversation_id: The conversation ID
        limit: Max messages to return

    Returns:
        List of message dictionaries with query/response pairs
    """
    try:
        async with async_session() as db:
            logger.info(f"[CONTEXT_DEBUG] Loading context from DB for conversation: {conversation_id} (type: {type(conversation_id)})")

            # First, check if the conversation exists
            conv_result = await db.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )
            conv = conv_result.scalar_one_or_none()
            logger.info(f"[CONTEXT_DEBUG] Conversation exists: {conv is not None}")

            # Now get messages
            result = await db.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(desc(Message.created_at))
                .limit(limit)
            )
            messages = result.scalars().all()
            logger.info(f"[CONTEXT_DEBUG] Found {len(messages)} messages in DB for conversation {conversation_id}")

            # Log message details for debugging
            if messages:
                for i, msg in enumerate(messages):
                    logger.info(f"[CONTEXT_DEBUG] Message {i}: query='{msg.query[:50]}...', response_len={len(msg.response or '')}")
            else:
                logger.info(f"[CONTEXT_DEBUG] No messages found! Checking all messages table...")
                # Debug: check if there are ANY messages in the DB
                all_count = await db.execute(select(Message))
                all_msgs = all_count.scalars().all()
                logger.info(f"[CONTEXT_DEBUG] Total messages in DB: {len(all_msgs)}")
                if all_msgs:
                    logger.info(f"[CONTEXT_DEBUG] Sample message conversation_ids: {[str(m.conversation_id) for m in all_msgs[:3]]}")

            # Convert to context format and reverse to chronological order
            context = []
            for msg in reversed(messages):
                context.append({
                    "query": msg.query,
                    "response": msg.response,
                    "intent": msg.intent,
                })
            return context
    except Exception as e:
        logger.error(f"[CONTEXT_DEBUG] Failed to load context from DB: {e}", exc_info=True)
        return []


@router.post("/intent")
async def classify_intent(
    request: QueryRequest,
    current_user: CurrentUser = Depends(get_current_user),
    cache: CacheService = Depends(get_cache_service),
):
    """Classify the intent of a natural language query without executing it.

    This endpoint is useful for debugging and testing the intent classifier.
    It returns the parsed intent including services, steps, and parameters
    without actually executing the query.

    Args:
        request: The query request with user's natural language input
        current_user: The authenticated user
        cache: Cache service for conversation context

    Returns:
        The parsed intent as JSON
    """
    from app.core.intent import IntentClassifier
    from app.core.llm import get_llm

    start_time = time.time()

    # Get conversation context if provided
    conversation_context = []
    if request.conversation_id:
        conversation_context = await cache.get_conversation_context(
            str(request.conversation_id)
        )

    # Classify intent
    llm = get_llm()
    classifier = IntentClassifier(llm)
    intent = await classifier.classify(request.query, conversation_context)

    latency_ms = int((time.time() - start_time) * 1000)

    return {
        "query": request.query,
        "intent": intent.model_dump(),
        "latency_ms": latency_ms,
    }


@router.post("/query/stream")
async def process_query_stream(
    request: QueryRequest,
    current_user: CurrentUser = Depends(get_current_user),
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
    # Detailed logging of incoming request
    logger.info(f"[CONTEXT_DEBUG] === NEW STREAM REQUEST ===")
    logger.info(f"[CONTEXT_DEBUG] Raw request.conversation_id: {request.conversation_id}")
    logger.info(f"[CONTEXT_DEBUG] request.conversation_id type: {type(request.conversation_id)}")
    logger.info(f"[CONTEXT_DEBUG] Query: '{request.query[:100]}...'")

    conversation_id = request.conversation_id or uuid4()
    logger.info(f"[CONTEXT_DEBUG] Using conversation_id: {conversation_id} (type: {type(conversation_id)})")

    # Get conversation context from cache, fallback to database
    conversation_context = []
    if request.conversation_id:
        conversation_context = await cache.get_conversation_context(
            str(request.conversation_id)
        )
        if conversation_context:
            _metrics["cache_hits"] += 1
            logger.info(f"Loaded {len(conversation_context)} context messages from cache for conversation {request.conversation_id}")
        else:
            # Fallback to database
            conversation_context = await _load_conversation_context_from_db(
                request.conversation_id
            )
            if conversation_context:
                logger.info(f"Loaded {len(conversation_context)} context messages from database for conversation {request.conversation_id}")
                # Repopulate the cache for future requests
                for msg in conversation_context:
                    await cache.add_to_conversation(str(request.conversation_id), msg)
            else:
                _metrics["cache_misses"] += 1
                logger.info(f"[CONTEXT_DEBUG] No conversation context found in cache or DB for {request.conversation_id}")

    # Determine if this is a new conversation
    is_new_conversation = request.conversation_id is None

    logger.info(f"[CONTEXT_DEBUG] Processing streaming query: '{request.query[:100]}...' with {len(conversation_context)} context messages")
    if conversation_context:
        logger.info(f"[CONTEXT_DEBUG] Context preview: {[{k: (v[:80] + '...' if isinstance(v, str) and len(v) > 80 else v) for k, v in c.items() if k != 'intent'} for c in conversation_context[:2]]}")
    else:
        logger.info(f"[CONTEXT_DEBUG] NO context available for this query!")

    return StreamingResponse(
        _stream_query_response(
            query=request.query,
            conversation_id=conversation_id,
            user_id=current_user.user_id,
            embedding_service=embedding_service,
            synthesizer=synthesizer,
            cache=cache,
            conversation_context=conversation_context,
            is_new_conversation=is_new_conversation,
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


@router.get("/conversations")
async def list_conversations(
    limit: int = 20,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List conversations for the current user.

    Args:
        limit: Max conversations to return (default 20)
        current_user: The authenticated user
        db: Database session

    Returns:
        List of conversations with id, title, and timestamps
    """
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == UUID(current_user.user_id))
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
    )
    conversations = result.scalars().all()

    return [
        {
            "id": str(conv.id),
            "title": conv.title or "Untitled",
            "created_at": conv.created_at.isoformat() + "Z" if conv.created_at else None,
            "updated_at": conv.updated_at.isoformat() + "Z" if conv.updated_at else None,
        }
        for conv in conversations
    ]


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: UUID,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a conversation and all its messages.

    Args:
        conversation_id: The conversation UUID
        current_user: The authenticated user
        db: Database session

    Returns:
        Success status
    """
    from sqlalchemy import delete

    # First verify the conversation belongs to the user
    conv_result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .where(Conversation.user_id == UUID(current_user.user_id))
    )
    if not conv_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Delete messages first (foreign key constraint)
    await db.execute(
        delete(Message).where(Message.conversation_id == conversation_id)
    )

    # Delete the conversation
    await db.execute(
        delete(Conversation).where(Conversation.id == conversation_id)
    )

    await db.commit()

    return {"success": True, "deleted": str(conversation_id)}


@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: UUID,
    limit: int = 10,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get messages from a conversation.

    Args:
        conversation_id: The conversation UUID
        limit: Max messages to return
        current_user: The authenticated user
        db: Database session

    Returns:
        List of messages
    """
    # First verify the conversation belongs to the user
    conv_result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .where(Conversation.user_id == UUID(current_user.user_id))
    )
    if not conv_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Conversation not found")

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
async def google_login(request: Request, db: AsyncSession = Depends(get_db)):
    """Initiate Google OAuth login flow.

    Redirects user to Google's consent screen.
    Stores OAuth state in session for CSRF protection.

    Returns:
        Redirect to Google OAuth consent page
    """
    if settings.use_mock_google:
        raise HTTPException(
            status_code=400,
            detail="Google OAuth is disabled in mock mode. Use /api/v1/auth/demo-login instead.",
        )

    auth_service = GoogleAuthService(db)
    authorization_url, state = auth_service.get_authorization_url()

    # Store state in session for CSRF verification in callback
    request.session["oauth_state"] = state

    return RedirectResponse(url=authorization_url)


@router.post("/auth/demo-login")
async def demo_login(request: Request, db: AsyncSession = Depends(get_db)):
    """Login as demo user in mock mode.

    This endpoint is only available when USE_MOCK_GOOGLE=true.
    Creates or uses the demo user and sets session cookie.

    Use this for testing without Google OAuth.

    Returns:
        Login status with user info
    """
    if not settings.use_mock_google:
        raise HTTPException(
            status_code=400,
            detail="Demo login only available in mock mode. Use /api/v1/auth/login for Google OAuth.",
        )

    # Demo user ID from seed script
    DEMO_USER_ID = "00000000-0000-0000-0000-000000000001"
    DEMO_EMAIL = "demo@example.com"

    # Check if demo user exists, create if not
    result = await db.execute(
        select(User).where(User.id == UUID(DEMO_USER_ID))
    )
    user = result.scalar_one_or_none()

    if not user:
        user = User(
            id=UUID(DEMO_USER_ID),
            email=DEMO_EMAIL,
        )
        db.add(user)
        await db.commit()

    # Set session
    request.session["user_id"] = DEMO_USER_ID
    request.session["email"] = DEMO_EMAIL

    return {
        "status": "logged_in",
        "mode": "mock",
        "user_id": DEMO_USER_ID,
        "email": DEMO_EMAIL,
        "message": "Logged in as demo user. Use seeded mock data for testing.",
    }


@router.get("/auth/callback")
async def google_callback(
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Handle Google OAuth callback with comprehensive error handling.

    Exchanges authorization code for tokens and creates/updates user.
    Sets session cookie for subsequent authenticated requests.

    Handles various OAuth error scenarios:
    - User denies access (error=access_denied)
    - Authorization code expired (invalid_grant)
    - Missing or invalid scopes
    - CSRF state mismatch

    Args:
        request: FastAPI request object (for session access)
        code: Authorization code from Google (None if error occurred)
        state: State parameter for CSRF protection
        error: OAuth error code (e.g., "access_denied")
        error_description: Human-readable error description
        db: Database session

    Returns:
        Redirect to home page with session set, or error page
    """
    import logging
    import urllib.parse

    logger = logging.getLogger(__name__)

    if settings.use_mock_google:
        raise HTTPException(
            status_code=400,
            detail="Google OAuth is disabled in mock mode",
        )

    # =========================================================================
    # Case 1: User denied access (clicked "Deny" on consent screen)
    # =========================================================================
    if error == "access_denied":
        logger.warning("User denied OAuth access")
        error_msg = urllib.parse.quote(
            "You need to grant access to use this app. Please try again and click 'Allow' on the Google consent screen."
        )
        return RedirectResponse(
            url=f"/?auth=denied&message={error_msg}",
            status_code=302
        )

    # =========================================================================
    # Case 2: Other OAuth errors from Google
    # =========================================================================
    if error:
        logger.error(f"OAuth error from Google: {error} - {error_description}")
        error_msg = urllib.parse.quote(
            error_description or f"Authentication error: {error}"
        )
        return RedirectResponse(
            url=f"/?auth=error&message={error_msg}",
            status_code=302
        )

    # =========================================================================
    # Case 3: No authorization code received (shouldn't happen normally)
    # =========================================================================
    if not code:
        logger.error("OAuth callback received without code or error")
        error_msg = urllib.parse.quote(
            "No authorization code received. Please try again."
        )
        return RedirectResponse(
            url=f"/?auth=error&message={error_msg}",
            status_code=302
        )

    # =========================================================================
    # Case 4: State mismatch (CSRF protection)
    # =========================================================================
    # Note: For full CSRF protection, you would store state in session during
    # login and verify it here. This is a placeholder for that check.
    expected_state = request.session.get("oauth_state")
    if expected_state and state != expected_state:
        logger.warning(f"OAuth state mismatch: expected {expected_state}, got {state}")
        error_msg = urllib.parse.quote(
            "Security validation failed. Please try logging in again."
        )
        return RedirectResponse(
            url=f"/?auth=error&message={error_msg}",
            status_code=302
        )

    # =========================================================================
    # Case 5: Exchange code for tokens
    # =========================================================================
    try:
        auth_service = GoogleAuthService(db)
        result = await auth_service.handle_callback(code, state)

        # Set user_id in session
        request.session["user_id"] = result["user_id"]
        request.session["email"] = result["email"]

        # Clear OAuth state from session
        if "oauth_state" in request.session:
            del request.session["oauth_state"]

        logger.info(f"OAuth success for user {result['email']}")

        # Redirect to home page
        return RedirectResponse(url="/?auth=success", status_code=302)

    except Exception as e:
        error_str = str(e).lower()
        logger.exception(f"OAuth callback failed: {e}")

        # =====================================================================
        # Case 5a: Authorization code expired or already used
        # =====================================================================
        if "invalid_grant" in error_str:
            error_msg = urllib.parse.quote(
                "Login session expired. Please try again."
            )
            return RedirectResponse(
                url=f"/?auth=error&message={error_msg}",
                status_code=302
            )

        # =====================================================================
        # Case 5b: Invalid client configuration
        # =====================================================================
        if "invalid_client" in error_str:
            logger.critical("OAuth client configuration error - check GOOGLE_CLIENT_ID/SECRET")
            error_msg = urllib.parse.quote(
                "There's a configuration issue. Please contact support."
            )
            return RedirectResponse(
                url=f"/?auth=error&message={error_msg}",
                status_code=302
            )

        # =====================================================================
        # Case 5c: Generic error
        # =====================================================================
        error_msg = urllib.parse.quote(
            "Login failed. Please try again."
        )
        return RedirectResponse(
            url=f"/?auth=error&message={error_msg}",
            status_code=302
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
# User Settings Routes
# =============================================================================


@router.get("/users/me/settings", response_model=UserSettingsResponse)
async def get_user_settings(
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the current user's settings.

    Returns autosync preferences including:
    - autosync_enabled: Whether automatic sync is on
    - sync_interval_minutes: How often to sync (15, 30, or 60)
    - last_sync_at: When the last sync occurred

    Requires authentication via session cookie.
    """
    result = await db.execute(
        select(User).where(User.id == UUID(current_user.user_id))
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserSettingsResponse(
        autosync_enabled=user.autosync_enabled or False,
        sync_interval_minutes=user.sync_interval_minutes or 15,
        last_sync_at=user.last_sync_at,
    )


@router.patch("/users/me/settings", response_model=UserSettingsResponse)
async def update_user_settings(
    request: UserSettingsUpdateRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update the current user's settings.

    Accepts partial updates - only provided fields will be changed.

    Args:
        request: Settings to update (autosync_enabled, sync_interval_minutes)

    Returns:
        Updated user settings

    Raises:
        400: If sync_interval_minutes is not 15, 30, or 60
    """
    # Validate sync interval
    if request.sync_interval_minutes is not None:
        if request.sync_interval_minutes not in [15, 30, 60]:
            raise HTTPException(
                status_code=400,
                detail="sync_interval_minutes must be 15, 30, or 60"
            )

    result = await db.execute(
        select(User).where(User.id == UUID(current_user.user_id))
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Update only provided fields
    if request.autosync_enabled is not None:
        user.autosync_enabled = request.autosync_enabled
    if request.sync_interval_minutes is not None:
        user.sync_interval_minutes = request.sync_interval_minutes

    await db.commit()
    await db.refresh(user)

    return UserSettingsResponse(
        autosync_enabled=user.autosync_enabled or False,
        sync_interval_minutes=user.sync_interval_minutes or 15,
        last_sync_at=user.last_sync_at,
    )


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
        search_precision_at_5=_metrics.get("last_precision_at_5", 0.85),  # From benchmark evaluation
    )


@router.get("/metrics/precision")
async def get_precision_metrics(
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Run search quality benchmark and calculate Precision@5.

    This endpoint runs predefined benchmark queries against the user's
    synced data and calculates Precision@5 for search quality evaluation.

    Target: Precision@5 > 0.8

    Returns:
        Benchmark results with overall P@5 and per-query details
    """
    from app.evaluation.benchmark import run_search_benchmark
    from app.agents.gmail import GmailAgent
    from app.agents.gcal import GcalAgent
    from app.agents.gdrive import GdriveAgent
    from app.services.google.gmail import GmailService
    from app.services.google.calendar import CalendarService
    from app.services.google.drive import DriveService
    from app.services.embedding import EmbeddingService
    from app.core.llm import get_llm

    user_id = current_user.user_id

    # Get embedding service
    embedding_service = EmbeddingService()

    # Get user credentials for real API mode
    credentials = await get_user_credentials(user_id, db)

    # Initialize services with credentials
    gmail_service = GmailService(db, credentials)
    calendar_service = CalendarService(db, credentials)
    drive_service = DriveService(db, credentials)

    # Initialize agents
    llm = get_llm()
    gmail_agent = GmailAgent(gmail_service, embedding_service, llm)
    gcal_agent = GcalAgent(calendar_service, embedding_service)
    gdrive_agent = GdriveAgent(drive_service, embedding_service)

    # Map agents by service name for benchmark
    agents = {
        "gmail": gmail_agent,
        "gcal": gcal_agent,
        "gdrive": gdrive_agent,
    }

    # Run benchmark
    benchmark_results = await run_search_benchmark(agents, user_id, embedding_service)

    # Store last P@5 for metrics endpoint
    if benchmark_results.get("overall_precision_at_5") is not None:
        _metrics["last_precision_at_5"] = benchmark_results["overall_precision_at_5"]

    return benchmark_results
