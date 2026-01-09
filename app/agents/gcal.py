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
        """Search events using 3-way hybrid search with RRF fusion.

        Combines three retrieval methods for best results:
        1. BM25/Full-text search (keyword matching)
        2. Vector search (semantic similarity)
        3. Filtered vector search (if filters provided)

        Results are fused using Reciprocal Rank Fusion (RRF).

        Args:
            query: The search query
            user_id: The user's ID
            filters: Optional filters (date_range, attendees)

        Returns:
            List of matching events, ranked by RRF score
        """
        query_embedding = await self.embeddings.embed(query)

        # 1. BM25 search - also apply filters
        bm25_results = await self.calendar.search_events_bm25(
            user_id=user_id,
            query=query,
            filters=filters,
            limit=20,
        )

        # 2. Vector search
        semantic_results = await self.calendar.search_events(
            user_id=user_id,
            embedding=query_embedding,
            filters=None,
            limit=20,
        )

        # 3. Filtered search (if filters provided)
        filtered_results = []
        if filters:
            filtered_results = await self.calendar.search_events(
                user_id=user_id,
                embedding=query_embedding,
                filters=filters,
                limit=20,
            )

        # Combine using RRF
        results = self._rrf_fusion(bm25_results, semantic_results, filtered_results)
        return results

    def _rrf_fusion(
        self,
        bm25_results: list[dict],
        semantic_results: list[dict],
        filtered_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """Combine results using Reciprocal Rank Fusion (RRF)."""
        rrf_scores: dict[str, float] = {}
        result_data: dict[str, dict] = {}
        match_sources: dict[str, list[str]] = {}

        for rank, result in enumerate(bm25_results, start=1):
            item_id = result.get("id")
            if item_id:
                rrf_scores[item_id] = rrf_scores.get(item_id, 0) + (1 / (k + rank))
                if item_id not in result_data:
                    result_data[item_id] = result
                    match_sources[item_id] = []
                match_sources[item_id].append("bm25")

        for rank, result in enumerate(semantic_results, start=1):
            item_id = result.get("id")
            if item_id:
                rrf_scores[item_id] = rrf_scores.get(item_id, 0) + (1 / (k + rank))
                if item_id not in result_data:
                    result_data[item_id] = result
                    match_sources[item_id] = []
                match_sources[item_id].append("semantic")

        for rank, result in enumerate(filtered_results, start=1):
            item_id = result.get("id")
            if item_id:
                rrf_scores[item_id] = rrf_scores.get(item_id, 0) + (1.5 / (k + rank))
                if item_id not in result_data:
                    result_data[item_id] = result
                    match_sources[item_id] = []
                match_sources[item_id].append("filter")

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for item_id in sorted_ids[:10]:
            result = result_data[item_id].copy()
            result["rrf_score"] = round(rrf_scores[item_id], 4)
            result["match_sources"] = match_sources[item_id]
            results.append(result)

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
