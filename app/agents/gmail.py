"""Gmail agent for email operations."""

from typing import Optional

from app.agents.base import BaseAgent
from app.schemas.intent import StepType, StepResult
from app.services.google.gmail import GmailService
from app.services.embedding import EmbeddingService


class GmailAgent(BaseAgent):
    """Agent for Gmail operations: search, read, draft, send emails."""

    def __init__(self, gmail_service: GmailService, embedding_service: EmbeddingService):
        """Initialize the Gmail agent.

        Args:
            gmail_service: The Gmail service instance
            embedding_service: The embedding service for semantic search
        """
        self.gmail = gmail_service
        self.embeddings = embedding_service

    async def search(
        self, query: str, user_id: str, filters: Optional[dict] = None
    ) -> list[dict]:
        """Search emails using semantic similarity.

        Args:
            query: The search query
            user_id: The user's ID
            filters: Optional filters (sender, date_range, labels)

        Returns:
            List of matching emails
        """
        # Generate embedding for query
        query_embedding = await self.embeddings.embed(query)

        # Search using vector similarity + metadata filters
        results = await self.gmail.search_emails(
            user_id=user_id,
            embedding=query_embedding,
            filters=filters,
            limit=10,
        )
        return results

    async def execute(
        self, step: StepType, params: dict, user_id: str
    ) -> StepResult:
        """Execute a Gmail operation.

        Args:
            step: The step to execute (get_email, draft_email, send_email)
            params: Operation parameters
            user_id: The user's ID

        Returns:
            StepResult with success status and data
        """
        try:
            if step == StepType.GET_EMAIL:
                email = await self.gmail.get_email(
                    user_id=user_id, email_id=params.get("email_id", "")
                )
                if email:
                    return StepResult(step=step, success=True, data=email)
                return StepResult(step=step, success=False, error="Email not found")

            elif step == StepType.DRAFT_EMAIL:
                # Build draft from params and source data
                to = params.get("to", "")
                subject = params.get("subject", "")
                body = params.get("body", "")

                # If source_data exists, use it to build the draft
                source = params.get("source_data", {})
                if source and not to:
                    to = source.get("sender", "")
                if source and not subject:
                    subject = f"Re: {source.get('subject', '')}"

                draft = await self.gmail.create_draft(
                    user_id=user_id, to=to, subject=subject, body=body
                )
                return StepResult(step=step, success=True, data=draft)

            elif step == StepType.SEND_EMAIL:
                to = params.get("to", "")
                subject = params.get("subject", "")
                body = params.get("body", "")

                sent = await self.gmail.send_email(
                    user_id=user_id, to=to, subject=subject, body=body
                )
                return StepResult(step=step, success=True, data=sent)

            else:
                return StepResult(
                    step=step, success=False, error=f"Unsupported step: {step}"
                )

        except Exception as e:
            return StepResult(step=step, success=False, error=str(e))

    async def get_context(self, item_id: str, user_id: str) -> Optional[dict]:
        """Get full email content for LLM context.

        Args:
            item_id: The email ID
            user_id: The user's ID

        Returns:
            Full email data or None
        """
        return await self.gmail.get_email(user_id=user_id, email_id=item_id)
