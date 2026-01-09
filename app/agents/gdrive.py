"""Google Drive agent for file operations."""

from typing import Optional

from app.agents.base import BaseAgent
from app.schemas.intent import StepType, StepResult
from app.services.google.drive import DriveService
from app.services.embedding import EmbeddingService


class GdriveAgent(BaseAgent):
    """Agent for Google Drive operations: search, read, share files."""

    def __init__(self, drive_service: DriveService, embedding_service: EmbeddingService):
        """Initialize the Drive agent.

        Args:
            drive_service: The Drive service instance
            embedding_service: The embedding service for semantic search
        """
        self.drive = drive_service
        self.embeddings = embedding_service

    async def search(
        self, query: str, user_id: str, filters: Optional[dict] = None
    ) -> list[dict]:
        """Search files using 3-way hybrid search with RRF fusion.

        Combines three retrieval methods for best results:
        1. BM25/Full-text search (keyword matching)
        2. Vector search (semantic similarity)
        3. Filtered vector search (if filters provided)

        Results are fused using Reciprocal Rank Fusion (RRF).

        Args:
            query: The search query
            user_id: The user's ID
            filters: Optional filters (mime_type, modified_date)

        Returns:
            List of matching files, ranked by RRF score
        """
        query_embedding = await self.embeddings.embed(query)

        # 1. BM25 search - also apply filters
        bm25_results = await self.drive.search_files_bm25(
            user_id=user_id,
            query=query,
            filters=filters,
            limit=20,
        )

        # 2. Vector search
        semantic_results = await self.drive.search_files(
            user_id=user_id,
            embedding=query_embedding,
            filters=None,
            limit=20,
        )

        # 3. Filtered search (if filters provided)
        filtered_results = []
        if filters:
            filtered_results = await self.drive.search_files(
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
        """Execute a Drive operation.

        Args:
            step: The step to execute
            params: Operation parameters
            user_id: The user's ID

        Returns:
            StepResult with success status and data
        """
        try:
            if step == StepType.GET_FILE:
                file_data = await self.drive.get_file(
                    user_id=user_id, file_id=params.get("file_id", "")
                )
                if file_data:
                    return StepResult(step=step, success=True, data=file_data)
                return StepResult(step=step, success=False, error="File not found")

            elif step == StepType.SHARE_FILE:
                result = await self.drive.share_file(
                    user_id=user_id,
                    file_id=params.get("file_id", ""),
                    email=params.get("email", ""),
                    role=params.get("role", "reader"),
                )
                return StepResult(step=step, success=True, data=result)

            else:
                return StepResult(
                    step=step, success=False, error=f"Unsupported step: {step}"
                )

        except Exception as e:
            return StepResult(step=step, success=False, error=str(e))

    async def get_context(self, item_id: str, user_id: str) -> Optional[dict]:
        """Get full file content for LLM context.

        Args:
            item_id: The file ID
            user_id: The user's ID

        Returns:
            Full file data or None
        """
        return await self.drive.get_file(user_id=user_id, file_id=item_id)
