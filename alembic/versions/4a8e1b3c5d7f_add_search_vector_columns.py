"""Add search_vector columns to cache tables

Revision ID: 4a8e1b3c5d7f
Revises: 889e18cccf5d
Create Date: 2026-01-13 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "4a8e1b3c5d7f"
down_revision: Union[str, None] = "889e18cccf5d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add search_vector column to gmail_cache
    op.execute("""
        ALTER TABLE gmail_cache
        ADD COLUMN IF NOT EXISTS search_vector tsvector
        GENERATED ALWAYS AS (
            to_tsvector('english', coalesce(subject, '') || ' ' || coalesce(sender, '') || ' ' || coalesce(body_preview, ''))
        ) STORED
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_gmail_search_vector
        ON gmail_cache USING gin (search_vector)
    """)

    # Add search_vector column to gcal_cache
    op.execute("""
        ALTER TABLE gcal_cache
        ADD COLUMN IF NOT EXISTS search_vector tsvector
        GENERATED ALWAYS AS (
            to_tsvector('english', coalesce(title, '') || ' ' || coalesce(description, '') || ' ' || coalesce(location, ''))
        ) STORED
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_gcal_search_vector
        ON gcal_cache USING gin (search_vector)
    """)

    # Add search_vector column to gdrive_cache
    op.execute("""
        ALTER TABLE gdrive_cache
        ADD COLUMN IF NOT EXISTS search_vector tsvector
        GENERATED ALWAYS AS (
            to_tsvector('english', coalesce(name, '') || ' ' || coalesce(content_preview, ''))
        ) STORED
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_gdrive_search_vector
        ON gdrive_cache USING gin (search_vector)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_gmail_search_vector")
    op.execute("ALTER TABLE gmail_cache DROP COLUMN IF EXISTS search_vector")

    op.execute("DROP INDEX IF EXISTS ix_gcal_search_vector")
    op.execute("ALTER TABLE gcal_cache DROP COLUMN IF EXISTS search_vector")

    op.execute("DROP INDEX IF EXISTS ix_gdrive_search_vector")
    op.execute("ALTER TABLE gdrive_cache DROP COLUMN IF EXISTS search_vector")
