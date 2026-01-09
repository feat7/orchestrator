"""Google Calendar agent for event operations."""

from typing import Optional

from app.agents.base import BaseAgent
from app.schemas.intent import StepType, StepResult
from app.services.google.calendar import CalendarService
from app.services.embedding import EmbeddingService


class GcalAgent(BaseAgent):
    """Agent for Google Calendar operations: search, create, update, delete events."""

    def __init__(
        self, calendar_service: CalendarService, embedding_service: EmbeddingService
    ):
        """Initialize the Calendar agent.

        Args:
            calendar_service: The Calendar service instance
            embedding_service: The embedding service for semantic search
        """
        self.calendar = calendar_service
        self.embeddings = embedding_service

    async def search(
        self, query: str, user_id: str, filters: Optional[dict] = None
    ) -> list[dict]:
        """Search calendar events using semantic similarity.

        Args:
            query: The search query
            user_id: The user's ID
            filters: Optional filters (date_range, attendees)

        Returns:
            List of matching events
        """
        # Generate embedding for query
        query_embedding = await self.embeddings.embed(query)

        # Search using vector similarity + metadata filters
        results = await self.calendar.search_events(
            user_id=user_id,
            embedding=query_embedding,
            filters=filters,
            limit=10,
        )
        return results

    async def execute(
        self, step: StepType, params: dict, user_id: str
    ) -> StepResult:
        """Execute a Calendar operation.

        Args:
            step: The step to execute
            params: Operation parameters
            user_id: The user's ID

        Returns:
            StepResult with success status and data
        """
        try:
            if step == StepType.GET_EVENT:
                event = await self.calendar.get_event(
                    user_id=user_id, event_id=params.get("event_id", "")
                )
                if event:
                    return StepResult(step=step, success=True, data=event)
                return StepResult(step=step, success=False, error="Event not found")

            elif step == StepType.CREATE_EVENT:
                event = await self.calendar.create_event(
                    user_id=user_id,
                    title=params.get("title", ""),
                    start_time=params.get("start_time"),
                    end_time=params.get("end_time"),
                    attendees=params.get("attendees", []),
                    description=params.get("description", ""),
                    location=params.get("location", ""),
                )
                return StepResult(step=step, success=True, data=event)

            elif step == StepType.UPDATE_EVENT:
                event = await self.calendar.update_event(
                    user_id=user_id,
                    event_id=params.get("event_id", ""),
                    updates=params.get("updates", {}),
                )
                return StepResult(step=step, success=True, data=event)

            elif step == StepType.DELETE_EVENT:
                await self.calendar.delete_event(
                    user_id=user_id, event_id=params.get("event_id", "")
                )
                return StepResult(step=step, success=True, data={"deleted": True})

            else:
                return StepResult(
                    step=step, success=False, error=f"Unsupported step: {step}"
                )

        except Exception as e:
            return StepResult(step=step, success=False, error=str(e))

    async def get_context(self, item_id: str, user_id: str) -> Optional[dict]:
        """Get full event content for LLM context.

        Args:
            item_id: The event ID
            user_id: The user's ID

        Returns:
            Full event data or None
        """
        return await self.calendar.get_event(user_id=user_id, event_id=item_id)
