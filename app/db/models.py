from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Index, Boolean, Computed, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB, TSVECTOR
from pgvector.sqlalchemy import Vector
from datetime import datetime
import uuid

from app.db.database import Base


class User(Base):
    """User model with Google OAuth tokens and sync preferences."""

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    google_access_token = Column(Text)
    google_refresh_token = Column(Text)
    token_expires_at = Column(DateTime)

    # Sync preferences
    autosync_enabled = Column(Boolean, default=False)  # User must opt-in
    sync_interval_minutes = Column(Integer, default=15)  # 15, 30, 60 mins
    last_sync_at = Column(DateTime)  # Track when last synced

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Conversation(Base):
    """Conversation session for tracking multi-turn interactions."""

    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    title = Column(String(255), nullable=True)  # Generated from first message
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Message(Base):
    """Individual message within a conversation."""

    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    query = Column(Text, nullable=False)
    intent = Column(JSONB)
    response = Column(Text)
    actions_taken = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)


class GmailCache(Base):
    """Cached Gmail messages with embeddings for semantic search."""

    __tablename__ = "gmail_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    email_id = Column(String(255), nullable=False)
    thread_id = Column(String(255))
    subject = Column(Text)
    sender = Column(String(255))
    recipients = Column(JSONB)  # List of recipient emails
    body_preview = Column(Text)  # First ~500 chars
    body_full = Column(Text)  # Full body for context
    embedding = Column(Vector(1536))
    received_at = Column(DateTime)
    labels = Column(JSONB)  # Gmail labels
    is_read = Column(Boolean, default=False)
    has_attachments = Column(Boolean, default=False)
    synced_at = Column(DateTime, default=datetime.utcnow)
    # Full-text search vector (PostgreSQL generated column for BM25)
    # Automatically computed from subject, sender, and body_preview
    search_vector = Column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', coalesce(subject, '') || ' ' || coalesce(sender, '') || ' ' || coalesce(body_preview, ''))",
            persisted=True
        )
    )

    __table_args__ = (
        Index(
            "ix_gmail_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_gmail_user_email", "user_id", "email_id", unique=True),
        Index("ix_gmail_user_received", "user_id", "received_at"),
        Index("ix_gmail_user_sender", "user_id", "sender"),
        Index("ix_gmail_search_vector", "search_vector", postgresql_using="gin"),
    )


class GcalCache(Base):
    """Cached Google Calendar events with embeddings for semantic search."""

    __tablename__ = "gcal_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    event_id = Column(String(255), nullable=False)
    calendar_id = Column(String(255), default="primary")
    title = Column(Text)
    description = Column(Text)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    attendees = Column(JSONB)  # List of attendee emails
    location = Column(Text)
    meeting_link = Column(Text)
    status = Column(String(50))  # confirmed, tentative, cancelled
    embedding = Column(Vector(1536))
    synced_at = Column(DateTime, default=datetime.utcnow)
    # Full-text search vector (PostgreSQL generated column for BM25)
    # Automatically computed from title, description, and location
    search_vector = Column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', coalesce(title, '') || ' ' || coalesce(description, '') || ' ' || coalesce(location, ''))",
            persisted=True
        )
    )

    __table_args__ = (
        Index(
            "ix_gcal_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_gcal_user_event", "user_id", "event_id", unique=True),
        Index("ix_gcal_user_time", "user_id", "start_time", "end_time"),
        Index("ix_gcal_search_vector", "search_vector", postgresql_using="gin"),
    )


class GdriveCache(Base):
    """Cached Google Drive files with embeddings for semantic search."""

    __tablename__ = "gdrive_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    file_id = Column(String(255), nullable=False)
    name = Column(Text)
    mime_type = Column(String(255))
    content_preview = Column(Text)  # First ~1000 chars of content
    parent_folder = Column(String(255))
    web_link = Column(Text)
    owners = Column(JSONB)  # List of owner emails
    shared_with = Column(JSONB)  # List of shared user emails
    embedding = Column(Vector(1536))
    created_at = Column(DateTime)
    modified_at = Column(DateTime)
    synced_at = Column(DateTime, default=datetime.utcnow)
    # Full-text search vector (PostgreSQL generated column for BM25)
    # Automatically computed from name and content_preview
    search_vector = Column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', coalesce(name, '') || ' ' || coalesce(content_preview, ''))",
            persisted=True
        )
    )

    __table_args__ = (
        Index(
            "ix_gdrive_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_gdrive_user_file", "user_id", "file_id", unique=True),
        Index("ix_gdrive_user_modified", "user_id", "modified_at"),
        Index("ix_gdrive_user_mime", "user_id", "mime_type"),
        Index("ix_gdrive_search_vector", "search_vector", postgresql_using="gin"),
    )


class SyncStatus(Base):
    """Track sync status for each service per user."""

    __tablename__ = "sync_status"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    service = Column(String(50), nullable=False)  # gmail, gcal, gdrive
    last_sync_at = Column(DateTime)
    last_sync_token = Column(Text)  # For incremental sync
    status = Column(String(50), default="idle")  # idle, syncing, error
    error_message = Column(Text)

    __table_args__ = (Index("ix_sync_user_service", "user_id", "service", unique=True),)
