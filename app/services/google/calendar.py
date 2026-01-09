"""Google Calendar service with mock and real implementations."""

from typing import Optional
import uuid
from datetime import datetime, timedelta

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from app.db.models import GcalCache
from app.config import settings


class CalendarService:
    """Service for Google Calendar operations with mock and real implementations."""

    def __init__(self, db: AsyncSession, credentials: Optional[Credentials] = None):
        """Initialize the Calendar service.

        Args:
            db: Database session
            credentials: Google OAuth credentials (None for mock mode)
        """
        self.db = db
        self.credentials = credentials
        self._service = None

    @property
    def service(self):
        """Lazy-load Calendar API service."""
        if self._service is None and self.credentials:
            self._service = build("calendar", "v3", credentials=self.credentials)
        return self._service

    async def search_events(
        self,
        user_id: str,
        embedding: list[float],
        filters: Optional[dict] = None,
        limit: int = 10,
        similarity_threshold: float = 0.25,
    ) -> list[dict]:
        """Search events using local cache with vector similarity.

        Always searches the local pgvector cache first (fast semantic search).
        This uses embeddings generated during sync for semantic similarity ranking.

        Args:
            user_id: The user's ID
            embedding: Query embedding vector
            filters: Optional filters (time_range, attendees)
            limit: Max results to return
            similarity_threshold: Minimum similarity score (0-1) to include results

        Returns:
            List of matching events with similarity scores
        """
        # Calculate cosine distance and convert to similarity (1 - distance)
        distance_expr = GcalCache.embedding.cosine_distance(embedding)

        # Always use local pgvector search for speed and semantic matching
        query = select(
            GcalCache,
            (1 - distance_expr).label("similarity")
        ).where(
            GcalCache.user_id == uuid.UUID(user_id)
        )

        # Apply metadata filters
        if filters:
            if filters.get("time_range") == "next_week":
                now = datetime.utcnow()
                query = query.where(
                    GcalCache.start_time >= now,
                    GcalCache.start_time <= now + timedelta(days=7),
                )
            elif filters.get("time_range") == "this_week":
                now = datetime.utcnow()
                query = query.where(
                    GcalCache.start_time >= now,
                    GcalCache.start_time <= now + timedelta(days=7),
                )
            elif filters.get("time") == "tomorrow" or filters.get("time_range") == "tomorrow":
                now = datetime.utcnow()
                tomorrow_start = now.replace(hour=0, minute=0, second=0) + timedelta(days=1)
                tomorrow_end = tomorrow_start + timedelta(days=1)
                query = query.where(
                    GcalCache.start_time >= tomorrow_start,
                    GcalCache.start_time < tomorrow_end,
                )
            elif filters.get("time_range") == "today":
                now = datetime.utcnow()
                today_start = now.replace(hour=0, minute=0, second=0)
                today_end = today_start + timedelta(days=1)
                query = query.where(
                    GcalCache.start_time >= today_start,
                    GcalCache.start_time < today_end,
                )
            if filters.get("date_from"):
                query = query.where(GcalCache.start_time >= filters["date_from"])
            if filters.get("date_to"):
                query = query.where(GcalCache.end_time <= filters["date_to"])

        # Order by similarity (descending) and limit
        query = query.order_by(distance_expr).limit(limit * 2)

        result = await self.db.execute(query)
        rows = result.all()

        # Filter by similarity threshold and format results
        events = []
        for row in rows:
            event = row[0]
            similarity = float(row[1]) if row[1] is not None else 0

            # Skip results below threshold
            if similarity < similarity_threshold:
                continue

            events.append({
                "id": str(event.event_id),
                "title": event.title,
                "description": event.description,
                "start_time": event.start_time.isoformat() if event.start_time else None,
                "end_time": event.end_time.isoformat() if event.end_time else None,
                "attendees": event.attendees,
                "location": event.location,
                "meeting_link": event.meeting_link,
                "status": event.status,
                "similarity": round(similarity, 3),
            })

            if len(events) >= limit:
                break

        return events

    async def search_events_bm25(
        self,
        user_id: str,
        query: str,
        filters: Optional[dict] = None,
        limit: int = 20,
    ) -> list[dict]:
        """Search events using BM25/full-text search (keyword matching).

        Uses PostgreSQL's tsvector and ts_rank for fast keyword-based search.

        Args:
            user_id: The user's ID
            query: Search query string
            filters: Optional filters (time_range, date_from, date_to)
            limit: Max results to return

        Returns:
            List of matching events with BM25 rank scores
        """
        words = query.lower().split()
        if not words:
            return []

        tsquery = func.plainto_tsquery("english", query)

        query_stmt = (
            select(
                GcalCache,
                func.ts_rank(GcalCache.search_vector, tsquery).label("rank")
            )
            .where(GcalCache.user_id == uuid.UUID(user_id))
            .where(GcalCache.search_vector.op("@@")(tsquery))
        )

        # Apply metadata filters (same as vector search)
        if filters:
            if filters.get("time_range") == "next_week":
                now = datetime.utcnow()
                query_stmt = query_stmt.where(
                    GcalCache.start_time >= now,
                    GcalCache.start_time <= now + timedelta(days=7),
                )
            elif filters.get("time_range") == "this_week":
                now = datetime.utcnow()
                query_stmt = query_stmt.where(
                    GcalCache.start_time >= now,
                    GcalCache.start_time <= now + timedelta(days=7),
                )
            elif filters.get("time") == "tomorrow" or filters.get("time_range") == "tomorrow":
                now = datetime.utcnow()
                tomorrow_start = now.replace(hour=0, minute=0, second=0) + timedelta(days=1)
                tomorrow_end = tomorrow_start + timedelta(days=1)
                query_stmt = query_stmt.where(
                    GcalCache.start_time >= tomorrow_start,
                    GcalCache.start_time < tomorrow_end,
                )
            elif filters.get("time_range") == "today":
                now = datetime.utcnow()
                today_start = now.replace(hour=0, minute=0, second=0)
                today_end = today_start + timedelta(days=1)
                query_stmt = query_stmt.where(
                    GcalCache.start_time >= today_start,
                    GcalCache.start_time < today_end,
                )
            if filters.get("date_from"):
                query_stmt = query_stmt.where(GcalCache.start_time >= filters["date_from"])
            if filters.get("date_to"):
                query_stmt = query_stmt.where(GcalCache.end_time <= filters["date_to"])

        query_stmt = (
            query_stmt
            .order_by(func.ts_rank(GcalCache.search_vector, tsquery).desc())
            .limit(limit)
        )

        result = await self.db.execute(query_stmt)
        rows = result.all()

        events = []
        for row in rows:
            event = row[0]
            rank = float(row[1]) if row[1] is not None else 0

            events.append({
                "id": str(event.event_id),
                "title": event.title,
                "description": event.description,
                "start_time": event.start_time.isoformat() if event.start_time else None,
                "end_time": event.end_time.isoformat() if event.end_time else None,
                "attendees": event.attendees,
                "location": event.location,
                "meeting_link": event.meeting_link,
                "status": event.status,
                "bm25_rank": round(rank, 4),
            })

        return events

    async def get_event(self, user_id: str, event_id: str) -> Optional[dict]:
        """Get full event details.

        Args:
            user_id: The user's ID
            event_id: The event ID

        Returns:
            Event data dictionary or None
        """
        if not settings.use_mock_google and self.service:
            try:
                event = self.service.events().get(
                    calendarId="primary",
                    eventId=event_id,
                ).execute()

                return {
                    "id": event["id"],
                    "calendar_id": "primary",
                    "title": event.get("summary", "Untitled Event"),
                    "description": event.get("description", ""),
                    "start_time": event["start"].get("dateTime", event["start"].get("date")),
                    "end_time": event["end"].get("dateTime", event["end"].get("date")),
                    "attendees": [a["email"] for a in event.get("attendees", [])],
                    "location": event.get("location", ""),
                    "meeting_link": event.get("hangoutLink", ""),
                    "status": event.get("status", "confirmed"),
                }
            except Exception as e:
                print(f"Calendar API error: {e}")
                return None

        # Mock mode: use local cache
        result = await self.db.execute(
            select(GcalCache).where(
                GcalCache.user_id == uuid.UUID(user_id),
                GcalCache.event_id == event_id,
            )
        )
        event = result.scalar_one_or_none()

        if event:
            return {
                "id": event.event_id,
                "calendar_id": event.calendar_id,
                "title": event.title,
                "description": event.description,
                "start_time": (
                    event.start_time.isoformat() if event.start_time else None
                ),
                "end_time": event.end_time.isoformat() if event.end_time else None,
                "attendees": event.attendees,
                "location": event.location,
                "meeting_link": event.meeting_link,
                "status": event.status,
            }
        return None

    async def create_event(
        self,
        user_id: str,
        title: str,
        start_time: datetime,
        end_time: datetime,
        attendees: Optional[list[str]] = None,
        description: str = "",
        location: str = "",
    ) -> dict:
        """Create a new calendar event.

        Args:
            user_id: The user's ID
            title: Event title
            start_time: Event start time
            end_time: Event end time
            attendees: List of attendee emails
            description: Event description
            location: Event location

        Returns:
            Created event data
        """
        if not settings.use_mock_google and self.service:
            try:
                event_body = {
                    "summary": title,
                    "description": description,
                    "location": location,
                    "start": {
                        "dateTime": start_time.isoformat(),
                        "timeZone": "UTC",
                    },
                    "end": {
                        "dateTime": end_time.isoformat(),
                        "timeZone": "UTC",
                    },
                }

                if attendees:
                    event_body["attendees"] = [{"email": a} for a in attendees]

                event = self.service.events().insert(
                    calendarId="primary",
                    body=event_body,
                    sendUpdates="all" if attendees else "none",
                ).execute()

                return {
                    "id": event["id"],
                    "title": event.get("summary"),
                    "start_time": event["start"].get("dateTime"),
                    "end_time": event["end"].get("dateTime"),
                    "attendees": [a["email"] for a in event.get("attendees", [])],
                    "description": event.get("description", ""),
                    "location": event.get("location", ""),
                    "status": "confirmed",
                    "html_link": event.get("htmlLink"),
                }
            except Exception as e:
                print(f"Calendar API error creating event: {e}")
                raise

        # Mock mode
        event_id = f"event_{uuid.uuid4().hex[:8]}"
        return {
            "id": event_id,
            "title": title,
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "attendees": attendees or [],
            "description": description,
            "location": location,
            "status": "confirmed",
        }

    async def update_event(
        self, user_id: str, event_id: str, updates: dict
    ) -> dict:
        """Update an existing event.

        Args:
            user_id: The user's ID
            event_id: The event ID
            updates: Dictionary of fields to update

        Returns:
            Updated event data
        """
        if not settings.use_mock_google and self.service:
            try:
                # Get existing event
                event = self.service.events().get(
                    calendarId="primary",
                    eventId=event_id,
                ).execute()

                # Apply updates
                if "title" in updates:
                    event["summary"] = updates["title"]
                if "description" in updates:
                    event["description"] = updates["description"]
                if "location" in updates:
                    event["location"] = updates["location"]
                if "start_time" in updates:
                    event["start"]["dateTime"] = updates["start_time"].isoformat()
                if "end_time" in updates:
                    event["end"]["dateTime"] = updates["end_time"].isoformat()
                if "attendees" in updates:
                    event["attendees"] = [{"email": a} for a in updates["attendees"]]

                updated = self.service.events().update(
                    calendarId="primary",
                    eventId=event_id,
                    body=event,
                ).execute()

                return {
                    "id": updated["id"],
                    "updated": True,
                    "title": updated.get("summary"),
                    "start_time": updated["start"].get("dateTime"),
                    "end_time": updated["end"].get("dateTime"),
                }
            except Exception as e:
                print(f"Calendar API error updating event: {e}")
                raise

        # Mock mode
        return {
            "id": event_id,
            "updated": True,
            "updates_applied": list(updates.keys()),
        }

    async def delete_event(self, user_id: str, event_id: str) -> None:
        """Delete a calendar event.

        Args:
            user_id: The user's ID
            event_id: The event ID
        """
        if not settings.use_mock_google and self.service:
            try:
                self.service.events().delete(
                    calendarId="primary",
                    eventId=event_id,
                ).execute()
                return
            except Exception as e:
                print(f"Calendar API error deleting event: {e}")
                raise

        # Mock mode: just return success
        return
