"""Base agent interface for Google Workspace services."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from app.schemas.intent import StepType, StepResult


class BaseAgent(ABC):
    """Abstract base class for all service agents (Gmail, GCal, GDrive).

    Each agent implements three core operations:
    - search: Semantic search using embeddings
    - execute: Perform write operations (send, create, delete, etc.)
    - get_context: Retrieve full content for LLM reasoning
    """

    @abstractmethod
    async def search(
        self, query: str, user_id: str, filters: Optional[dict] = None
    ) -> list[dict]:
        """Semantic search using embeddings.

        Args:
            query: The search query
            user_id: The user's ID
            filters: Optional metadata filters (date range, sender, etc.)

        Returns:
            List of matching items as dictionaries
        """
        pass

    @abstractmethod
    async def execute(
        self, step: StepType, params: dict, user_id: str
    ) -> StepResult:
        """Execute a write operation.

        Args:
            step: The step type to execute
            params: Parameters for the operation
            user_id: The user's ID

        Returns:
            A StepResult indicating success/failure and any data
        """
        pass

    @abstractmethod
    async def get_context(self, item_id: str, user_id: str) -> Optional[dict]:
        """Get full content for LLM reasoning.

        Args:
            item_id: The ID of the item to retrieve
            user_id: The user's ID

        Returns:
            Full item data as dictionary, or None if not found
        """
        pass
