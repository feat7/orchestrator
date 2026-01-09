"""Gmail service with mock and real implementations."""

from typing import Optional
import uuid
import base64
from datetime import datetime
from email.mime.text import MIMEText

from sqlalchemy import select
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
    ) -> list[dict]:
        """Search emails using local cache with vector similarity.

        Always searches the local pgvector cache first (fast semantic search).
        This uses embeddings generated during sync for semantic similarity ranking.

        Args:
            user_id: The user's ID
            embedding: Query embedding vector
            filters: Optional filters (sender, date_from, date_to, labels)
            limit: Max results to return

        Returns:
            List of matching emails
        """
        # Always use local pgvector search for speed and semantic matching
        query = select(GmailCache).where(
            GmailCache.user_id == uuid.UUID(user_id)
        )

        # Apply metadata filters
        if filters:
            if filters.get("sender"):
                query = query.where(
                    GmailCache.sender.ilike(f"%{filters['sender']}%")
                )
            if filters.get("date_from"):
                query = query.where(GmailCache.received_at >= filters["date_from"])
            if filters.get("date_to"):
                query = query.where(GmailCache.received_at <= filters["date_to"])
            if filters.get("subject"):
                query = query.where(
                    GmailCache.subject.ilike(f"%{filters['subject']}%")
                )

        # Order by vector similarity (cosine distance)
        query = query.order_by(
            GmailCache.embedding.cosine_distance(embedding)
        ).limit(limit)

        result = await self.db.execute(query)
        emails = result.scalars().all()

        return [
            {
                "id": str(e.email_id),
                "subject": e.subject,
                "sender": e.sender,
                "body_preview": e.body_preview,
                "received_at": e.received_at.isoformat() if e.received_at else None,
                "labels": e.labels,
                "has_attachments": e.has_attachments,
            }
            for e in emails
        ]

    async def get_email(self, user_id: str, email_id: str) -> Optional[dict]:
        """Get full email content.

        Args:
            user_id: The user's ID
            email_id: The email ID

        Returns:
            Email data dictionary or None
        """
        if not settings.use_mock_google and self.service:
            try:
                message = self.service.users().messages().get(
                    userId="me",
                    id=email_id,
                    format="full",
                ).execute()

                headers = {h["name"]: h["value"] for h in message.get("payload", {}).get("headers", [])}

                # Extract body
                body = ""
                payload = message.get("payload", {})
                if "body" in payload and payload["body"].get("data"):
                    body = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8")
                elif "parts" in payload:
                    for part in payload["parts"]:
                        if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
                            body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
                            break

                return {
                    "id": email_id,
                    "thread_id": message.get("threadId"),
                    "subject": headers.get("Subject", "No Subject"),
                    "sender": headers.get("From", "Unknown"),
                    "recipients": headers.get("To", "").split(","),
                    "body": body,
                    "received_at": headers.get("Date"),
                    "labels": message.get("labelIds", []),
                    "has_attachments": any(
                        part.get("filename") for part in payload.get("parts", [])
                    ),
                }
            except Exception as e:
                print(f"Gmail API error: {e}")
                return None

        # Mock mode: return from cache
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
        if not settings.use_mock_google and self.service:
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
        if not settings.use_mock_google and self.service:
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
        if not settings.use_mock_google and self.service:
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
