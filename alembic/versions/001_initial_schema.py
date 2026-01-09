"""Initial schema with pgvector support

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create users table
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("google_access_token", sa.Text()),
        sa.Column("google_refresh_token", sa.Text()),
        sa.Column("token_expires_at", sa.DateTime()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # Create conversations table
    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # Create messages table
    op.create_table(
        "messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("conversations.id"), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("intent", postgresql.JSONB()),
        sa.Column("response", sa.Text()),
        sa.Column("actions_taken", postgresql.JSONB()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # Create gmail_cache table
    op.create_table(
        "gmail_cache",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("email_id", sa.String(255), nullable=False),
        sa.Column("thread_id", sa.String(255)),
        sa.Column("subject", sa.Text()),
        sa.Column("sender", sa.String(255)),
        sa.Column("recipients", postgresql.JSONB()),
        sa.Column("body_preview", sa.Text()),
        sa.Column("body_full", sa.Text()),
        sa.Column("embedding", Vector(1536)),
        sa.Column("received_at", sa.DateTime()),
        sa.Column("labels", postgresql.JSONB()),
        sa.Column("is_read", sa.Boolean(), default=False),
        sa.Column("has_attachments", sa.Boolean(), default=False),
        sa.Column("synced_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_gmail_user_email", "gmail_cache", ["user_id", "email_id"], unique=True)
    op.create_index("ix_gmail_user_received", "gmail_cache", ["user_id", "received_at"])
    op.create_index("ix_gmail_user_sender", "gmail_cache", ["user_id", "sender"])

    # Create gcal_cache table
    op.create_table(
        "gcal_cache",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("event_id", sa.String(255), nullable=False),
        sa.Column("calendar_id", sa.String(255), default="primary"),
        sa.Column("title", sa.Text()),
        sa.Column("description", sa.Text()),
        sa.Column("start_time", sa.DateTime()),
        sa.Column("end_time", sa.DateTime()),
        sa.Column("attendees", postgresql.JSONB()),
        sa.Column("location", sa.Text()),
        sa.Column("meeting_link", sa.Text()),
        sa.Column("status", sa.String(50)),
        sa.Column("embedding", Vector(1536)),
        sa.Column("synced_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_gcal_user_event", "gcal_cache", ["user_id", "event_id"], unique=True)
    op.create_index("ix_gcal_user_time", "gcal_cache", ["user_id", "start_time", "end_time"])

    # Create gdrive_cache table
    op.create_table(
        "gdrive_cache",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("file_id", sa.String(255), nullable=False),
        sa.Column("name", sa.Text()),
        sa.Column("mime_type", sa.String(255)),
        sa.Column("content_preview", sa.Text()),
        sa.Column("parent_folder", sa.String(255)),
        sa.Column("web_link", sa.Text()),
        sa.Column("owners", postgresql.JSONB()),
        sa.Column("shared_with", postgresql.JSONB()),
        sa.Column("embedding", Vector(1536)),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("modified_at", sa.DateTime()),
        sa.Column("synced_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_gdrive_user_file", "gdrive_cache", ["user_id", "file_id"], unique=True)
    op.create_index("ix_gdrive_user_modified", "gdrive_cache", ["user_id", "modified_at"])
    op.create_index("ix_gdrive_user_mime", "gdrive_cache", ["user_id", "mime_type"])

    # Create sync_status table
    op.create_table(
        "sync_status",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("service", sa.String(50), nullable=False),
        sa.Column("last_sync_at", sa.DateTime()),
        sa.Column("last_sync_token", sa.Text()),
        sa.Column("status", sa.String(50), default="idle"),
        sa.Column("error_message", sa.Text()),
    )
    op.create_index("ix_sync_user_service", "sync_status", ["user_id", "service"], unique=True)

    # Create vector indexes (after data is populated, these can be created)
    # Note: IVFFlat indexes require data to build, so we create them separately
    # For small datasets, we can use hnsw instead which doesn't require data
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_gmail_embedding
        ON gmail_cache USING hnsw (embedding vector_cosine_ops)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_gcal_embedding
        ON gcal_cache USING hnsw (embedding vector_cosine_ops)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_gdrive_embedding
        ON gdrive_cache USING hnsw (embedding vector_cosine_ops)
        """
    )


def downgrade() -> None:
    op.drop_table("sync_status")
    op.drop_table("gdrive_cache")
    op.drop_table("gcal_cache")
    op.drop_table("gmail_cache")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_table("users")
    op.execute("DROP EXTENSION IF EXISTS vector")
