"""Gmail service with mock and real implementations."""

from typing import Optional
import uuid
import base64
from datetime import datetime, timedelta
from email.mime.text import MIMEText

from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from app.db.models import GmailCache
from app.config import settings


class GmailService:
    """Service for Gmail operations with mock and real implementations."""

    def __init__(self, db: AsyncSession, credentials: Optional[Credentials] = None):
        """Initialize the Gmail service.

        Args:
            db: Database session
            credentials: Google OAuth credentials (None for mock mode)
        """
        self.db = db
        self.credentials = credentials
        self._service = None

    @property
    def service(self):
        """Lazy-load Gmail API service."""
        if self._service is None and self.credentials:
            self._service = build("gmail", "v1", credentials=self.credentials)
        return self._service

    async def search_emails(
        self,
        user_id: str,
        embedding: list[float],
        filters: Optional[dict] = None,
        limit: int = 10,
        similarity_threshold: float = 0.35,
    ) -> list[dict]:
        """Search emails using local cache with vector similarity.

        Always searches the local pgvector cache first (fast semantic search).
        This uses embeddings generated during sync for semantic similarity ranking.

        Args:
            user_id: The user's ID
            embedding: Query embedding vector
            filters: Optional filters (sender, date_from, date_to, labels)
            limit: Max results to return
            similarity_threshold: Minimum similarity score (0-1) to include results.
                                  Results below this threshold are filtered out.

        Returns:
            List of matching emails with similarity scores
        """
        from sqlalchemy import func

        # Calculate cosine distance and convert to similarity (1 - distance)
        distance_expr = GmailCache.embedding.cosine_distance(embedding)

        # Always use local pgvector search for speed and semantic matching
        query = select(
            GmailCache,
            (1 - distance_expr).label("similarity")
        ).where(
            GmailCache.user_id == uuid.UUID(user_id)
        )

        # Apply metadata filters
        if filters:
            if filters.get("sender"):
                query = query.where(
                    GmailCache.sender.ilike(f"%{filters['sender']}%")
                )
            # Support both date_from/date_to and after_date/before_date
            date_from = filters.get("date_from") or filters.get("after_date")
            date_to = filters.get("date_to") or filters.get("before_date")
            if date_from:
                if isinstance(date_from, str):
                    date_from = datetime.fromisoformat(date_from.replace("Z", "+00:00").replace("T", " ").split("+")[0])
                query = query.where(GmailCache.received_at >= date_from)
            if date_to:
                if isinstance(date_to, str):
                    date_to = datetime.fromisoformat(date_to.replace("Z", "+00:00").replace("T", " ").split("+")[0])
                query = query.where(GmailCache.received_at <= date_to)
            if filters.get("subject"):
                query = query.where(
                    GmailCache.subject.ilike(f"%{filters['subject']}%")
                )
            # Filter by label (e.g., "IMPORTANT", "INBOX")
            if filters.get("label"):
                query = query.where(
                    GmailCache.labels.op("?")(filters["label"])
                )

            # Handle time_range strings (convert to dates)
            now = datetime.utcnow()
            time_range = filters.get("time_range")
            if time_range == "last_week" or time_range == "last week":
                query = query.where(
                    GmailCache.received_at >= now - timedelta(days=7)
                )
            elif time_range == "this_week" or time_range == "this week":
                start_of_week = now - timedelta(days=now.weekday())
                start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
                query = query.where(
                    GmailCache.received_at >= start_of_week
                )
            elif time_range == "today":
                start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
                query = query.where(
                    GmailCache.received_at >= start_of_day
                )
            elif time_range == "yesterday":
                start_of_yesterday = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                end_of_yesterday = now.replace(hour=0, minute=0, second=0, microsecond=0)
                query = query.where(
                    GmailCache.received_at >= start_of_yesterday,
                    GmailCache.received_at < end_of_yesterday
                )
            elif time_range == "recent":
                query = query.where(
                    GmailCache.received_at >= now - timedelta(days=30)
                )

        # Order by similarity (descending) and limit
        query = query.order_by(distance_expr).limit(limit * 2)  # Fetch extra to filter

        result = await self.db.execute(query)
        rows = result.all()

        # Filter by similarity threshold and format results
        emails = []
        for row in rows:
            email = row[0]
            similarity = float(row[1]) if row[1] is not None else 0

            # Skip results below threshold
            if similarity < similarity_threshold:
                continue

            emails.append({
                "id": str(email.email_id),
                "subject": email.subject,
                "sender": email.sender,
                "body_preview": email.body_preview,
                "received_at": email.received_at.isoformat() if email.received_at else None,
                "labels": email.labels,
                "has_attachments": email.has_attachments,
                "similarity": round(similarity, 3),
            })

            if len(emails) >= limit:
                break

        return emails

    async def search_emails_bm25(
        self,
        user_id: str,
        query: str,
        filters: Optional[dict] = None,
        limit: int = 20,
    ) -> list[dict]:
        """Search emails using BM25/full-text search (keyword matching).

        Uses PostgreSQL's tsvector and ts_rank for fast keyword-based search.
        This complements semantic search by finding exact keyword matches.

        Args:
            user_id: The user's ID
            query: Search query string
            filters: Optional filters (sender, date_from, date_to, subject)
            limit: Max results to return

        Returns:
            List of matching emails with BM25 rank scores
        """
        # Convert query to tsquery format
        # Split into words and join with & for AND matching
        words = query.lower().split()
        if not words:
            return []

        # Create tsquery - use | for OR matching (more lenient)
        tsquery_str = " | ".join(words)

        # Use plainto_tsquery for simpler matching
        tsquery = func.plainto_tsquery("english", query)

        # Query with ts_rank for relevance scoring
        query_stmt = (
            select(
                GmailCache,
                func.ts_rank(GmailCache.search_vector, tsquery).label("rank")
            )
            .where(GmailCache.user_id == uuid.UUID(user_id))
            .where(GmailCache.search_vector.op("@@")(tsquery))
        )

        # Apply metadata filters (same as vector search)
        if filters:
            if filters.get("sender"):
                query_stmt = query_stmt.where(
                    GmailCache.sender.ilike(f"%{filters['sender']}%")
                )
            # Support both date_from/date_to and after_date/before_date
            date_from = filters.get("date_from") or filters.get("after_date")
            date_to = filters.get("date_to") or filters.get("before_date")
            if date_from:
                if isinstance(date_from, str):
                    date_from = datetime.fromisoformat(date_from.replace("Z", "+00:00").replace("T", " ").split("+")[0])
                query_stmt = query_stmt.where(GmailCache.received_at >= date_from)
            if date_to:
                if isinstance(date_to, str):
                    date_to = datetime.fromisoformat(date_to.replace("Z", "+00:00").replace("T", " ").split("+")[0])
                query_stmt = query_stmt.where(GmailCache.received_at <= date_to)
            if filters.get("subject"):
                query_stmt = query_stmt.where(
                    GmailCache.subject.ilike(f"%{filters['subject']}%")
                )
            # Filter by label (e.g., "IMPORTANT", "INBOX")
            if filters.get("label"):
                query_stmt = query_stmt.where(
                    GmailCache.labels.op("?")(filters["label"])
                )

            # Handle time_range strings (convert to dates)
            now = datetime.utcnow()
            time_range = filters.get("time_range")
            if time_range == "last_week" or time_range == "last week":
                query_stmt = query_stmt.where(
                    GmailCache.received_at >= now - timedelta(days=7)
                )
            elif time_range == "this_week" or time_range == "this week":
                start_of_week = now - timedelta(days=now.weekday())
                start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
                query_stmt = query_stmt.where(
                    GmailCache.received_at >= start_of_week
                )
            elif time_range == "today":
                start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
                query_stmt = query_stmt.where(
                    GmailCache.received_at >= start_of_day
                )
            elif time_range == "yesterday":
                start_of_yesterday = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                end_of_yesterday = now.replace(hour=0, minute=0, second=0, microsecond=0)
                query_stmt = query_stmt.where(
                    GmailCache.received_at >= start_of_yesterday,
                    GmailCache.received_at < end_of_yesterday
                )
            elif time_range == "recent":
                query_stmt = query_stmt.where(
                    GmailCache.received_at >= now - timedelta(days=30)
                )

        query_stmt = (
            query_stmt
            .order_by(func.ts_rank(GmailCache.search_vector, tsquery).desc())
            .limit(limit)
        )

        result = await self.db.execute(query_stmt)
        rows = result.all()

        emails = []
        for row in rows:
            email = row[0]
            rank = float(row[1]) if row[1] is not None else 0

            emails.append({
                "id": str(email.email_id),
                "subject": email.subject,
                "sender": email.sender,
                "body_preview": email.body_preview,
                "received_at": email.received_at.isoformat() if email.received_at else None,
                "labels": email.labels,
                "has_attachments": email.has_attachments,
                "bm25_rank": round(rank, 4),
            })

        return emails

    async def search_emails_filter_only(
        self,
        user_id: str,
        filters: dict,
        limit: int = 20,
    ) -> list[dict]:
        """Search emails using only metadata filters (no semantic search).

        Used when there's no search query but filters exist (e.g., "emails from last week").
        Returns emails sorted by received date descending.

        Args:
            user_id: The user's ID
            filters: Filters (sender, time_range, subject, date_from, date_to)
            limit: Max results to return

        Returns:
            List of matching emails sorted by recency
        """
        query = select(GmailCache).where(
            GmailCache.user_id == uuid.UUID(user_id)
        )

        # Apply filters
        if filters.get("sender"):
            query = query.where(
                GmailCache.sender.ilike(f"%{filters['sender']}%")
            )
        # Support both date_from/date_to and after_date/before_date
        date_from = filters.get("date_from") or filters.get("after_date")
        date_to = filters.get("date_to") or filters.get("before_date")
        if date_from:
            if isinstance(date_from, str):
                date_from = datetime.fromisoformat(date_from.replace("Z", "+00:00").replace("T", " ").split("+")[0])
            query = query.where(GmailCache.received_at >= date_from)
        if date_to:
            if isinstance(date_to, str):
                date_to = datetime.fromisoformat(date_to.replace("Z", "+00:00").replace("T", " ").split("+")[0])
            query = query.where(GmailCache.received_at <= date_to)
        if filters.get("subject"):
            query = query.where(
                GmailCache.subject.ilike(f"%{filters['subject']}%")
            )
        # Filter by label (e.g., "IMPORTANT", "INBOX")
        if filters.get("label"):
            query = query.where(
                GmailCache.labels.op("?")(filters["label"])
            )

        # Handle time_range strings (convert to dates)
        now = datetime.utcnow()
        time_range = filters.get("time_range")
        if time_range == "last_week" or time_range == "last week":
            query = query.where(
                GmailCache.received_at >= now - timedelta(days=7)
            )
        elif time_range == "this_week" or time_range == "this week":
            start_of_week = now - timedelta(days=now.weekday())
            start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
            query = query.where(
                GmailCache.received_at >= start_of_week
            )
        elif time_range == "today":
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            query = query.where(
                GmailCache.received_at >= start_of_day
            )
        elif time_range == "yesterday":
            start_of_yesterday = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_yesterday = now.replace(hour=0, minute=0, second=0, microsecond=0)
            query = query.where(
                GmailCache.received_at >= start_of_yesterday,
                GmailCache.received_at < end_of_yesterday
            )
        elif time_range == "recent":
            query = query.where(
                GmailCache.received_at >= now - timedelta(days=30)
            )

        # Sort by received date descending (most recent first)
        query = query.order_by(GmailCache.received_at.desc()).limit(limit)

        result = await self.db.execute(query)
        rows = result.scalars().all()

        emails = []
        for email in rows:
            emails.append({
                "id": str(email.email_id),
                "subject": email.subject,
                "sender": email.sender,
                "body_preview": email.body_preview,
                "received_at": email.received_at.isoformat() if email.received_at else None,
                "labels": email.labels,
                "has_attachments": email.has_attachments,
            })

        return emails

    async def get_email(self, user_id: str, email_id: str) -> Optional[dict]:
        """Get full email content from local cache.

        Always uses the local cache (populated by sync) for read operations.
        This avoids unnecessary API calls and provides consistent fast responses.

        Args:
            user_id: The user's ID
            email_id: The email ID

        Returns:
            Email data dictionary or None
        """
        # Always use local cache for read operations
        result = await self.db.execute(
            select(GmailCache).where(
                GmailCache.user_id == uuid.UUID(user_id),
                GmailCache.email_id == email_id,
            )
        )
        email = result.scalar_one_or_none()

        if email:
            return {
                "id": email.email_id,
                "thread_id": email.thread_id,
                "subject": email.subject,
                "sender": email.sender,
                "recipients": email.recipients,
                "body": email.body_full or email.body_preview,
                "received_at": (
                    email.received_at.isoformat() if email.received_at else None
                ),
                "labels": email.labels,
                "has_attachments": email.has_attachments,
            }
        return None

    async def create_draft(
        self, user_id: str, to: str, subject: str, body: str
    ) -> dict:
        """Create an email draft.

        Args:
            user_id: The user's ID
            to: Recipient email
            subject: Email subject
            body: Email body

        Returns:
            Draft data with ID
        """
        if not settings.is_gmail_mock and self.service:
            try:
                message = MIMEText(body)
                message["to"] = to
                message["subject"] = subject

                encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

                draft = self.service.users().drafts().create(
                    userId="me",
                    body={"message": {"raw": encoded_message}},
                ).execute()

                return {
                    "id": draft["id"],
                    "message_id": draft["message"]["id"],
                    "to": to,
                    "subject": subject,
                    "body_preview": body[:100] if body else "",
                    "status": "draft",
                }
            except Exception as e:
                print(f"Gmail API error creating draft: {e}")
                raise

        # Mock mode
        return {
            "id": f"draft_{uuid.uuid4().hex[:8]}",
            "to": to,
            "subject": subject,
            "body_preview": body[:100] if body else "",
            "status": "draft",
        }

    async def send_email(
        self, user_id: str, to: str, subject: str, body: str
    ) -> dict:
        """Send an email.

        Args:
            user_id: The user's ID
            to: Recipient email
            subject: Email subject
            body: Email body

        Returns:
            Sent message data
        """
        if not settings.is_gmail_mock and self.service:
            try:
                message = MIMEText(body)
                message["to"] = to
                message["subject"] = subject

                encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

                sent = self.service.users().messages().send(
                    userId="me",
                    body={"raw": encoded_message},
                ).execute()

                return {
                    "id": sent["id"],
                    "thread_id": sent.get("threadId"),
                    "to": to,
                    "subject": subject,
                    "sent_at": datetime.utcnow().isoformat(),
                    "status": "sent",
                }
            except Exception as e:
                print(f"Gmail API error sending email: {e}")
                raise

        # Mock mode
        return {
            "id": f"sent_{uuid.uuid4().hex[:8]}",
            "to": to,
            "subject": subject,
            "sent_at": datetime.utcnow().isoformat(),
            "status": "sent",
        }

    async def send_draft(self, user_id: str, draft_id: str) -> dict:
        """Send an existing draft email.

        This sends the draft and automatically removes it from the drafts folder.

        Args:
            user_id: The user's ID
            draft_id: The draft ID to send

        Returns:
            Sent message data
        """
        if not settings.is_gmail_mock and self.service:
            try:
                sent = self.service.users().drafts().send(
                    userId="me",
                    body={"id": draft_id},
                ).execute()

                return {
                    "id": sent.get("id", ""),
                    "thread_id": sent.get("threadId"),
                    "sent_at": datetime.utcnow().isoformat(),
                    "status": "sent",
                    "draft_id": draft_id,
                }
            except Exception as e:
                print(f"Gmail API error sending draft: {e}")
                raise

        # Mock mode - simulate sending the draft
        return {
            "id": f"sent_{uuid.uuid4().hex[:8]}",
            "sent_at": datetime.utcnow().isoformat(),
            "status": "sent",
            "draft_id": draft_id,
        }

    async def update_labels(
        self, user_id: str, email_id: str, add_labels: list[str], remove_labels: list[str]
    ) -> dict:
        """Update email labels.

        Args:
            user_id: The user's ID
            email_id: The email ID
            add_labels: Labels to add
            remove_labels: Labels to remove

        Returns:
            Updated email data
        """
        if not settings.is_gmail_mock and self.service:
            try:
                result = self.service.users().messages().modify(
                    userId="me",
                    id=email_id,
                    body={
                        "addLabelIds": add_labels,
                        "removeLabelIds": remove_labels,
                    },
                ).execute()

                return {
                    "id": email_id,
                    "labels": result.get("labelIds", []),
                    "labels_added": add_labels,
                    "labels_removed": remove_labels,
                }
            except Exception as e:
                print(f"Gmail API error updating labels: {e}")
                raise

        # Mock mode
        return {
            "id": email_id,
            "labels_added": add_labels,
            "labels_removed": remove_labels,
        }
