"""Celery application configuration."""

from celery import Celery

from app.config import settings

# Create Celery app
celery_app = Celery(
    "orchestrator",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks.sync_tasks"],
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes max per task
    worker_prefetch_multiplier=1,  # One task at a time for better control
    result_expires=3600,  # Results expire after 1 hour
)

# Configure periodic tasks (beat schedule)
# Runs every 5 minutes to check for users with autosync enabled
# The sync_all_users task respects user's sync_interval_minutes setting
celery_app.conf.beat_schedule = {
    "periodic-sync-check": {
        "task": "sync.all_users",
        "schedule": 300.0,  # Check every 5 minutes
    },
}
