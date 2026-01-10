"""Celery tasks for syncing Google services.

Uses Celery's native retry mechanism for non-blocking retries.
When a task fails with a retryable error, it goes back to the queue
with a delay, freeing the worker to process other tasks immediately.
"""

import logging
from app.celery_app import celery_app
from app.services.sync import SyncService
from app.utils.resilience import (
    RateLimitError,
    QuotaExceededError,
    TokenExpiredError,
    ServiceUnavailableError,
)

logger = logging.getLogger(__name__)

# Retry configuration
RETRY_BACKOFF = 2  # Exponential base
RETRY_BACKOFF_MAX = 300  # Max 5 minutes between retries
RETRY_JITTER = True  # Add randomness to prevent thundering herd


@celery_app.task(
    bind=True,
    name="sync.gmail",
    max_retries=5,
    autoretry_for=(RateLimitError, ServiceUnavailableError, ConnectionError),
    retry_backoff=RETRY_BACKOFF,
    retry_backoff_max=RETRY_BACKOFF_MAX,
    retry_jitter=RETRY_JITTER,
)
def sync_gmail_task(self, user_id: str, max_results: int = 100):
    """Sync Gmail messages for a user.

    Uses Celery's autoretry for rate limits and transient errors.
    Worker is freed immediately on retry - task goes back to queue.
    Also backfills search vectors for existing emails that don't have them.

    Args:
        user_id: The user's UUID string
        max_results: Maximum messages to sync

    Returns:
        Sync result dict
    """
    try:
        sync_service = SyncService(user_id)
        result = sync_service.sync_gmail(max_results=max_results)

        # Backfill search vectors for any existing emails that don't have them
        # This enables BM25 keyword search for older cached emails
        backfill_result = sync_service.backfill_gmail_search_vectors()
        if backfill_result.get("updated", 0) > 0:
            logger.info(f"Backfilled {backfill_result['updated']} search vectors")
            result["backfilled"] = backfill_result["updated"]

        return result
    except (QuotaExceededError, TokenExpiredError) as e:
        # Don't retry these - they need user action
        logger.error(f"Gmail sync failed (non-retryable): {e}")
        return {"service": "gmail", "status": "error", "error": str(e)}


@celery_app.task(
    bind=True,
    name="sync.calendar",
    max_retries=5,
    autoretry_for=(RateLimitError, ServiceUnavailableError, ConnectionError),
    retry_backoff=RETRY_BACKOFF,
    retry_backoff_max=RETRY_BACKOFF_MAX,
    retry_jitter=RETRY_JITTER,
)
def sync_calendar_task(self, user_id: str, days_ahead: int = 30, days_back: int = 7):
    """Sync Google Calendar events for a user.

    Uses Celery's autoretry for rate limits and transient errors.
    Worker is freed immediately on retry - task goes back to queue.

    Args:
        user_id: The user's UUID string
        days_ahead: Days ahead to sync
        days_back: Days back to sync

    Returns:
        Sync result dict
    """
    try:
        sync_service = SyncService(user_id)
        return sync_service.sync_calendar(days_ahead=days_ahead, days_back=days_back)
    except (QuotaExceededError, TokenExpiredError) as e:
        logger.error(f"Calendar sync failed (non-retryable): {e}")
        return {"service": "gcal", "status": "error", "error": str(e)}


@celery_app.task(
    bind=True,
    name="sync.drive",
    max_retries=5,
    autoretry_for=(RateLimitError, ServiceUnavailableError, ConnectionError),
    retry_backoff=RETRY_BACKOFF,
    retry_backoff_max=RETRY_BACKOFF_MAX,
    retry_jitter=RETRY_JITTER,
)
def sync_drive_task(self, user_id: str, max_results: int = 100):
    """Sync Google Drive files for a user.

    Uses Celery's autoretry for rate limits and transient errors.
    Worker is freed immediately on retry - task goes back to queue.

    Args:
        user_id: The user's UUID string
        max_results: Maximum files to sync

    Returns:
        Sync result dict
    """
    try:
        sync_service = SyncService(user_id)
        return sync_service.sync_drive(max_results=max_results)
    except (QuotaExceededError, TokenExpiredError) as e:
        logger.error(f"Drive sync failed (non-retryable): {e}")
        return {"service": "gdrive", "status": "error", "error": str(e)}


@celery_app.task(bind=True, name="sync.all")
def sync_all_task(self, user_id: str):
    """Sync all Google services for a user using parallel task execution.

    Dispatches individual sync tasks (gmail, calendar, drive) as a Celery group.
    Each task runs independently with its own retry logic.

    Args:
        user_id: The user's UUID string

    Returns:
        Task group ID for tracking
    """
    from celery import group

    # Create a group of tasks to run in parallel
    # Each task has its own retry logic and doesn't block others
    job = group(
        sync_gmail_task.s(user_id),
        sync_calendar_task.s(user_id),
        sync_drive_task.s(user_id),
    )

    # Apply the group (dispatch to workers)
    result = job.apply_async()

    return {
        "status": "dispatched",
        "user_id": user_id,
        "group_id": str(result.id),
        "tasks": ["gmail", "calendar", "drive"],
    }


@celery_app.task(name="sync.all_users")
def sync_all_users():
    """Sync services for users who have autosync enabled and are due for sync.

    This is designed for periodic background sync via Celery Beat (runs every 5 mins).
    Only syncs users who:
    1. Have autosync_enabled = True
    2. Have valid Google credentials
    3. Are due for sync (last_sync_at + sync_interval has passed)

    Returns:
        Dict with dispatched and skipped counts
    """
    from datetime import datetime, timedelta
    from sqlalchemy import create_engine, select, and_, or_
    from sqlalchemy.orm import Session
    from app.db.models import User
    from app.config import settings

    sync_db_url = settings.database_url.replace("+asyncpg", "")
    engine = create_engine(sync_db_url)

    dispatched = 0
    skipped = 0
    now = datetime.utcnow()

    with Session(engine) as session:
        # Get users with autosync enabled and valid tokens
        users = session.execute(
            select(User).where(
                and_(
                    User.google_access_token.isnot(None),
                    User.autosync_enabled == True,  # noqa: E712
                )
            )
        ).scalars().all()

        for user in users:
            # Check if user is due for sync
            sync_interval = timedelta(minutes=user.sync_interval_minutes or 15)

            if user.last_sync_at is not None:
                next_sync_at = user.last_sync_at + sync_interval
                if now < next_sync_at:
                    # Not due yet, skip
                    skipped += 1
                    continue

            user_id = str(user.id)

            # Update last_sync_at before dispatching
            user.last_sync_at = now
            session.commit()

            # Dispatch sync task - don't wait for completion
            sync_all_task.delay(user_id)
            dispatched += 1

    logger.info(f"Sync dispatch: {dispatched} synced, {skipped} skipped (not due yet)")
    return {"dispatched": dispatched, "skipped": skipped, "status": "ok"}


@celery_app.task(name="sync.backfill_search_vectors")
def backfill_search_vectors_task(user_id: str):
    """Backfill search_vector for existing Gmail cache entries.

    This updates emails that have NULL search_vector to enable
    BM25 keyword search. Run this after updating the sync code
    to populate search_vector for new emails.

    Args:
        user_id: The user's UUID string

    Returns:
        Backfill result dict
    """
    sync_service = SyncService(user_id)
    return sync_service.backfill_gmail_search_vectors()
