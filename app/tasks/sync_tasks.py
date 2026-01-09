"""Celery tasks for syncing Google services."""

from app.celery_app import celery_app
from app.services.sync import SyncService


@celery_app.task(bind=True, name="sync.gmail")
def sync_gmail_task(self, user_id: str, max_results: int = 100):
    """Sync Gmail messages for a user.

    Args:
        user_id: The user's UUID string
        max_results: Maximum messages to sync

    Returns:
        Sync result dict
    """
    sync_service = SyncService(user_id)
    return sync_service.sync_gmail(max_results=max_results)


@celery_app.task(bind=True, name="sync.calendar")
def sync_calendar_task(self, user_id: str, days_ahead: int = 30, days_back: int = 7):
    """Sync Google Calendar events for a user.

    Args:
        user_id: The user's UUID string
        days_ahead: Days ahead to sync
        days_back: Days back to sync

    Returns:
        Sync result dict
    """
    sync_service = SyncService(user_id)
    return sync_service.sync_calendar(days_ahead=days_ahead, days_back=days_back)


@celery_app.task(bind=True, name="sync.drive")
def sync_drive_task(self, user_id: str, max_results: int = 100):
    """Sync Google Drive files for a user.

    Args:
        user_id: The user's UUID string
        max_results: Maximum files to sync

    Returns:
        Sync result dict
    """
    sync_service = SyncService(user_id)
    return sync_service.sync_drive(max_results=max_results)


@celery_app.task(bind=True, name="sync.all")
def sync_all_task(self, user_id: str):
    """Sync all Google services for a user.

    Args:
        user_id: The user's UUID string

    Returns:
        Combined sync results
    """
    sync_service = SyncService(user_id)
    return sync_service.sync_all()


@celery_app.task(name="sync.all_users")
def sync_all_users():
    """Sync all services for all users with valid credentials.

    This is designed for periodic background sync via Celery Beat.

    Returns:
        Dict of user_id -> sync results
    """
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session
    from app.db.models import User
    from app.config import settings

    sync_db_url = settings.database_url.replace("+asyncpg", "")
    engine = create_engine(sync_db_url)

    results = {}

    with Session(engine) as session:
        # Get all users with tokens
        users = session.execute(
            select(User).where(User.google_access_token.isnot(None))
        ).scalars().all()

        for user in users:
            user_id = str(user.id)
            try:
                sync_service = SyncService(user_id)
                results[user_id] = sync_service.sync_all()
            except Exception as e:
                results[user_id] = {"error": str(e)}

    return results
