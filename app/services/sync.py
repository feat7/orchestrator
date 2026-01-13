"""Synchronization service for pulling data from Google APIs into local cache."""

import logging
import base64
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from openai import OpenAI

from app.db.models import User, GmailCache, GcalCache, GdriveCache, SyncStatus
from app.config import settings
from app.utils.resilience import (
    google_api_call_with_retry,
    TokenExpiredError,
    QuotaExceededError,
)

logger = logging.getLogger(__name__)


class SyncService:
    """Service for syncing Google data to local cache with embeddings."""

    def __init__(self, user_id: str):
        """Initialize sync service for a user.

        Args:
            user_id: The user's UUID string
        """
        self.user_id = UUID(user_id)
        # Use synchronous engine for Celery compatibility
        sync_db_url = settings.database_url.replace("+asyncpg", "")
        self.engine = create_engine(sync_db_url)
        self.openai = OpenAI(api_key=settings.openai_api_key)

    def _get_credentials(self, session: Session) -> Optional[Credentials]:
        """Get Google credentials for the user.

        Args:
            session: Database session

        Returns:
            Credentials object or None
        """
        user = session.execute(
            select(User).where(User.id == self.user_id)
        ).scalar_one_or_none()

        if not user or not user.google_access_token:
            return None

        return Credentials(
            token=user.google_access_token,
            refresh_token=user.google_refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret,
            scopes=settings.google_scopes,
        )

    def _extract_file_content(self, service, file_id: str, mime_type: str) -> str:
        """Extract text content from a Google Workspace file.

        Uses export_media for Google Docs/Sheets/Slides, returns empty for others.

        Args:
            service: Google Drive API service
            file_id: The file ID
            mime_type: The file's MIME type

        Returns:
            Extracted text content or empty string
        """
        # Map Google Workspace MIME types to export formats
        export_formats = {
            "application/vnd.google-apps.document": "text/plain",
            "application/vnd.google-apps.spreadsheet": "text/csv",
            "application/vnd.google-apps.presentation": "text/plain",
        }

        export_mime = export_formats.get(mime_type)
        if not export_mime:
            # Not a Google Workspace file we can export
            return ""

        try:
            # Export the file as text
            content = google_api_call_with_retry(
                lambda: service.files().export(
                    fileId=file_id,
                    mimeType=export_mime
                ).execute()
            )

            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="ignore")

            # Truncate if too long (for storage and embedding)
            max_content_chars = 10000
            if len(content) > max_content_chars:
                content = content[:max_content_chars] + "..."

            return content.strip()

        except Exception as e:
            logger.warning(f"Could not extract content from file {file_id}: {e}")
            return ""

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text using OpenAI.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Truncate text if too long (OpenAI has token limits)
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars]

        response = self.openai.embeddings.create(
            model=settings.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def _update_sync_status(
        self,
        session: Session,
        service: str,
        status: str,
        error: Optional[str] = None,
    ):
        """Update sync status for a service.

        Args:
            session: Database session
            service: Service name (gmail, gcal, gdrive)
            status: Status (syncing, completed, error)
            error: Optional error message
        """
        stmt = insert(SyncStatus).values(
            user_id=self.user_id,
            service=service,
            status=status,
            last_sync_at=datetime.utcnow() if status == "completed" else None,
            error_message=error,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["user_id", "service"],
            set_={
                "status": status,
                "last_sync_at": datetime.utcnow() if status == "completed" else SyncStatus.last_sync_at,
                "error_message": error,
            },
        )
        session.execute(stmt)
        session.commit()

    def sync_gmail(self, max_results: int = 500) -> dict:
        """Sync Gmail messages to local cache (incremental).

        Only generates embeddings for NEW emails not already in cache.
        Uses pagination to fetch all emails up to max_results.

        Args:
            max_results: Maximum number of messages to sync (default 500)

        Returns:
            Sync result with counts
        """
        with Session(self.engine) as session:
            self._update_sync_status(session, "gmail", "syncing")

            try:
                credentials = self._get_credentials(session)
                if not credentials:
                    raise ValueError("No valid credentials found")

                service = build("gmail", "v1", credentials=credentials)

                # Get existing email IDs from cache to skip re-processing
                existing_ids_result = session.execute(
                    select(GmailCache.email_id).where(
                        GmailCache.user_id == self.user_id
                    )
                )
                existing_ids = set(row[0] for row in existing_ids_result)
                logger.info(f"Found {len(existing_ids)} existing emails in cache")

                # Paginate through messages
                all_messages = []
                page_token = None

                while len(all_messages) < max_results:
                    page_size = min(100, max_results - len(all_messages))
                    results = google_api_call_with_retry(
                        lambda pt=page_token, ps=page_size: service.users().messages().list(
                            userId="me",
                            maxResults=ps,
                            pageToken=pt,
                            q="in:inbox OR in:sent",
                        ).execute()
                    )

                    messages = results.get("messages", [])
                    if not messages:
                        break

                    all_messages.extend(messages)
                    page_token = results.get("nextPageToken")

                    if not page_token:
                        break

                logger.info(f"Fetched {len(all_messages)} message refs from Gmail API")

                synced_count = 0
                skipped_count = 0
                error_count = 0

                for msg_ref in all_messages:
                    msg_id = msg_ref["id"]

                    # Skip if already in cache (incremental sync)
                    if msg_id in existing_ids:
                        skipped_count += 1
                        continue

                    try:
                        # Get full message with retry
                        msg = google_api_call_with_retry(
                            lambda mid=msg_id: service.users().messages().get(
                                userId="me",
                                id=mid,
                                format="full",
                            ).execute()
                        )

                        headers = {
                            h["name"]: h["value"]
                            for h in msg.get("payload", {}).get("headers", [])
                        }

                        # Extract body
                        body = ""
                        payload = msg.get("payload", {})
                        if "body" in payload and payload["body"].get("data"):
                            body = base64.urlsafe_b64decode(
                                payload["body"]["data"]
                            ).decode("utf-8", errors="ignore")
                        elif "parts" in payload:
                            for part in payload["parts"]:
                                if part.get("mimeType") == "text/plain":
                                    if part.get("body", {}).get("data"):
                                        body = base64.urlsafe_b64decode(
                                            part["body"]["data"]
                                        ).decode("utf-8", errors="ignore")
                                        break

                        # Create text for embedding
                        embed_text = f"Subject: {headers.get('Subject', '')}\n"
                        embed_text += f"From: {headers.get('From', '')}\n"
                        embed_text += f"To: {headers.get('To', '')}\n"
                        embed_text += f"Body: {body[:2000]}"

                        embedding = self._generate_embedding(embed_text)

                        # Parse date
                        received_at = None
                        date_str = headers.get("Date")
                        if date_str:
                            try:
                                from email.utils import parsedate_to_datetime
                                received_at = parsedate_to_datetime(date_str)
                            except Exception:
                                pass

                        # Insert new email (not upsert since we skip existing)
                        # Note: search_vector is auto-generated by PostgreSQL (GENERATED ALWAYS column)
                        stmt = insert(GmailCache).values(
                            user_id=self.user_id,
                            email_id=msg["id"],
                            thread_id=msg.get("threadId"),
                            subject=headers.get("Subject", ""),
                            sender=headers.get("From", ""),
                            recipients=[headers.get("To", "")],
                            body_preview=body[:500] if body else "",
                            body_full=body,
                            embedding=embedding,
                            received_at=received_at,
                            labels=msg.get("labelIds", []),
                            is_read="UNREAD" not in msg.get("labelIds", []),
                            has_attachments=any(
                                p.get("filename")
                                for p in payload.get("parts", [])
                            ),
                            synced_at=datetime.utcnow(),
                        )
                        # Use on_conflict_do_nothing in case of race conditions
                        stmt = stmt.on_conflict_do_nothing(
                            index_elements=["user_id", "email_id"]
                        )
                        session.execute(stmt)
                        synced_count += 1

                    except Exception as e:
                        logger.error(f"Error syncing message {msg_id}: {e}")
                        error_count += 1

                session.commit()
                self._update_sync_status(session, "gmail", "completed")

                logger.info(
                    f"Gmail sync complete: {synced_count} new, "
                    f"{skipped_count} skipped (already cached), {error_count} errors"
                )

                return {
                    "service": "gmail",
                    "synced": synced_count,
                    "skipped": skipped_count,
                    "errors": error_count,
                    "status": "completed",
                }

            except Exception as e:
                self._update_sync_status(session, "gmail", "error", str(e))
                return {
                    "service": "gmail",
                    "synced": 0,
                    "errors": 1,
                    "status": "error",
                    "error": str(e),
                }

    def sync_calendar(self, days_ahead: int = 30, days_back: int = 7) -> dict:
        """Sync Google Calendar events to local cache.

        Args:
            days_ahead: Number of days ahead to sync
            days_back: Number of days back to sync

        Returns:
            Sync result with counts
        """
        with Session(self.engine) as session:
            self._update_sync_status(session, "gcal", "syncing")

            try:
                credentials = self._get_credentials(session)
                if not credentials:
                    raise ValueError("No valid credentials found")

                service = build("calendar", "v3", credentials=credentials)

                # Calculate time bounds
                now = datetime.utcnow()
                time_min = (now - timedelta(days=days_back)).isoformat() + "Z"
                time_max = (now + timedelta(days=days_ahead)).isoformat() + "Z"

                # Get events with retry
                events_result = google_api_call_with_retry(
                    lambda: service.events().list(
                        calendarId="primary",
                        timeMin=time_min,
                        timeMax=time_max,
                        maxResults=250,
                        singleEvents=True,
                        orderBy="startTime",
                    ).execute()
                )

                events = events_result.get("items", [])
                synced_count = 0
                error_count = 0

                for event in events:
                    try:
                        # Create text for embedding
                        embed_text = f"Event: {event.get('summary', 'Untitled')}\n"
                        if event.get("description"):
                            embed_text += f"Description: {event['description']}\n"
                        if event.get("location"):
                            embed_text += f"Location: {event['location']}\n"
                        attendees = event.get("attendees", [])
                        if attendees:
                            attendee_emails = [a.get("email", "") for a in attendees]
                            embed_text += f"Attendees: {', '.join(attendee_emails)}\n"

                        embedding = self._generate_embedding(embed_text)

                        # Parse times
                        start = event["start"]
                        end = event["end"]
                        start_time = start.get("dateTime") or start.get("date")
                        end_time = end.get("dateTime") or end.get("date")

                        # Convert to datetime
                        if "T" in str(start_time):
                            start_dt = datetime.fromisoformat(
                                start_time.replace("Z", "+00:00")
                            )
                        else:
                            start_dt = datetime.strptime(start_time, "%Y-%m-%d")

                        if "T" in str(end_time):
                            end_dt = datetime.fromisoformat(
                                end_time.replace("Z", "+00:00")
                            )
                        else:
                            end_dt = datetime.strptime(end_time, "%Y-%m-%d")

                        # Upsert into cache
                        stmt = insert(GcalCache).values(
                            user_id=self.user_id,
                            event_id=event["id"],
                            calendar_id="primary",
                            title=event.get("summary", "Untitled"),
                            description=event.get("description", ""),
                            start_time=start_dt,
                            end_time=end_dt,
                            attendees=[a.get("email") for a in attendees],
                            location=event.get("location", ""),
                            meeting_link=event.get("hangoutLink", ""),
                            status=event.get("status", "confirmed"),
                            embedding=embedding,
                            synced_at=datetime.utcnow(),
                        )
                        stmt = stmt.on_conflict_do_update(
                            index_elements=["user_id", "event_id"],
                            set_={
                                "title": stmt.excluded.title,
                                "description": stmt.excluded.description,
                                "start_time": stmt.excluded.start_time,
                                "end_time": stmt.excluded.end_time,
                                "attendees": stmt.excluded.attendees,
                                "location": stmt.excluded.location,
                                "meeting_link": stmt.excluded.meeting_link,
                                "status": stmt.excluded.status,
                                "embedding": stmt.excluded.embedding,
                                "synced_at": datetime.utcnow(),
                            },
                        )
                        session.execute(stmt)
                        synced_count += 1

                    except Exception as e:
                        print(f"Error syncing event {event.get('id')}: {e}")
                        error_count += 1

                session.commit()
                self._update_sync_status(session, "gcal", "completed")

                return {
                    "service": "gcal",
                    "synced": synced_count,
                    "errors": error_count,
                    "status": "completed",
                }

            except Exception as e:
                self._update_sync_status(session, "gcal", "error", str(e))
                return {
                    "service": "gcal",
                    "synced": 0,
                    "errors": 1,
                    "status": "error",
                    "error": str(e),
                }

    def sync_drive(self, max_results: int = 500) -> dict:
        """Sync Google Drive files to local cache (incremental).

        Only generates embeddings for NEW files not already in cache.
        Uses pagination to fetch all files up to max_results.

        Args:
            max_results: Maximum number of files to sync (default 500)

        Returns:
            Sync result with counts
        """
        with Session(self.engine) as session:
            self._update_sync_status(session, "gdrive", "syncing")

            try:
                credentials = self._get_credentials(session)
                if not credentials:
                    raise ValueError("No valid credentials found")

                service = build("drive", "v3", credentials=credentials)

                # Get existing file IDs from cache to skip re-processing
                existing_ids_result = session.execute(
                    select(GdriveCache.file_id).where(
                        GdriveCache.user_id == self.user_id
                    )
                )
                existing_ids = set(row[0] for row in existing_ids_result)
                logger.info(f"Found {len(existing_ids)} existing files in cache")

                # Paginate through files
                all_files = []
                page_token = None

                while len(all_files) < max_results:
                    page_size = min(100, max_results - len(all_files))
                    results = google_api_call_with_retry(
                        lambda pt=page_token, ps=page_size: service.files().list(
                            pageSize=ps,
                            pageToken=pt,
                            fields="nextPageToken, files(id, name, mimeType, webViewLink, modifiedTime, createdTime, owners, parents, description)",
                            orderBy="modifiedTime desc",
                            q="trashed = false",
                        ).execute()
                    )

                    files = results.get("files", [])
                    if not files:
                        break

                    all_files.extend(files)
                    page_token = results.get("nextPageToken")

                    if not page_token:
                        break

                logger.info(f"Fetched {len(all_files)} files from Drive API")

                synced_count = 0
                skipped_count = 0
                error_count = 0

                for file in all_files:
                    file_id = file["id"]

                    # Skip if already in cache (incremental sync)
                    if file_id in existing_ids:
                        skipped_count += 1
                        continue

                    try:
                        # Extract file content for Google Workspace files
                        mime_type = file.get("mimeType", "")
                        content_preview = self._extract_file_content(
                            service, file_id, mime_type
                        )

                        # Fall back to description if no content extracted
                        if not content_preview:
                            content_preview = file.get("description", "")

                        # Create text for embedding (include content for better search)
                        embed_text = f"File: {file.get('name', 'Untitled')}\n"
                        embed_text += f"Type: {mime_type}\n"
                        if file.get("description"):
                            embed_text += f"Description: {file['description']}\n"
                        if content_preview:
                            embed_text += f"Content: {content_preview[:5000]}\n"

                        embedding = self._generate_embedding(embed_text)

                        # Parse dates
                        created_at = None
                        modified_at = None
                        if file.get("createdTime"):
                            created_at = datetime.fromisoformat(
                                file["createdTime"].replace("Z", "+00:00")
                            )
                        if file.get("modifiedTime"):
                            modified_at = datetime.fromisoformat(
                                file["modifiedTime"].replace("Z", "+00:00")
                            )

                        owners = [o.get("emailAddress", "") for o in file.get("owners", [])]

                        # Insert new file (not upsert since we skip existing)
                        # Note: search_vector is auto-generated from name + content_preview
                        stmt = insert(GdriveCache).values(
                            user_id=self.user_id,
                            file_id=file_id,
                            name=file.get("name", "Untitled"),
                            mime_type=mime_type,
                            content_preview=content_preview,
                            parent_folder=file.get("parents", [None])[0] if file.get("parents") else None,
                            web_link=file.get("webViewLink", ""),
                            owners=owners,
                            shared_with=[],  # Would need separate API call
                            embedding=embedding,
                            created_at=created_at,
                            modified_at=modified_at,
                            synced_at=datetime.utcnow(),
                        )
                        # Use on_conflict_do_nothing in case of race conditions
                        stmt = stmt.on_conflict_do_nothing(
                            index_elements=["user_id", "file_id"]
                        )
                        session.execute(stmt)
                        synced_count += 1

                    except Exception as e:
                        logger.error(f"Error syncing file {file_id}: {e}")
                        error_count += 1

                session.commit()
                self._update_sync_status(session, "gdrive", "completed")

                logger.info(
                    f"Drive sync complete: {synced_count} new, "
                    f"{skipped_count} skipped (already cached), {error_count} errors"
                )

                return {
                    "service": "gdrive",
                    "synced": synced_count,
                    "skipped": skipped_count,
                    "errors": error_count,
                    "status": "completed",
                }

            except Exception as e:
                self._update_sync_status(session, "gdrive", "error", str(e))
                return {
                    "service": "gdrive",
                    "synced": 0,
                    "errors": 1,
                    "status": "error",
                    "error": str(e),
                }

    def sync_all(self) -> dict:
        """Sync all services.

        Returns:
            Combined sync results
        """
        results = {
            "gmail": self.sync_gmail(),
            "calendar": self.sync_calendar(),
            "drive": self.sync_drive(),
        }
        return results

    def backfill_gmail_search_vectors(self) -> dict:
        """Backfill search_vector for existing Gmail cache entries.

        This updates emails that have NULL search_vector to enable
        BM25 keyword search. Run this after updating the sync code
        to populate search_vector for new emails.

        Returns:
            Dict with count of updated emails
        """
        with Session(self.engine) as session:
            try:
                # Update all emails with NULL search_vector for this user
                from sqlalchemy import update, text

                # Use raw SQL for the update since we need to reference columns
                result = session.execute(
                    text("""
                        UPDATE gmail_cache
                        SET search_vector = to_tsvector('english',
                            COALESCE(subject, '') || ' ' || COALESCE(LEFT(body_full, 5000), '')
                        )
                        WHERE user_id = :user_id
                        AND search_vector IS NULL
                    """),
                    {"user_id": str(self.user_id)},
                )
                session.commit()

                updated_count = result.rowcount
                logger.info(f"Backfilled search_vector for {updated_count} emails")

                return {
                    "service": "gmail_backfill",
                    "updated": updated_count,
                    "status": "completed",
                }

            except Exception as e:
                logger.error(f"Error backfilling search vectors: {e}")
                return {
                    "service": "gmail_backfill",
                    "updated": 0,
                    "status": "error",
                    "error": str(e),
                }
