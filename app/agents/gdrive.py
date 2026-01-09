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
        """Search Drive files using semantic similarity.

        Args:
            query: The search query
            user_id: The user's ID
            filters: Optional filters (mime_type, modified_date)

        Returns:
            List of matching files
        """
        # Generate embedding for query
        query_embedding = await self.embeddings.embed(query)

        # Search using vector similarity + metadata filters
        results = await self.drive.search_files(
            user_id=user_id,
            embedding=query_embedding,
            filters=filters,
            limit=10,
        )
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
