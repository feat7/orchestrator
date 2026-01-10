"""Gmail agent for email operations."""

import json
import re
from typing import Optional, TYPE_CHECKING

from app.agents.base import BaseAgent
from app.schemas.intent import StepType, StepResult
from app.services.google.gmail import GmailService
from app.services.embedding import EmbeddingService

if TYPE_CHECKING:
    from app.core.llm import LLMProvider


EMAIL_COMPOSER_SYSTEM = """You are an email composer assistant. Given the context, compose a professional email.

RULES:
1. Generate an appropriate subject line based on the message intent
2. Start with a friendly greeting (use recipient's first name if available, otherwise use "Hi")
3. Write the body in a professional but friendly tone
4. Keep it concise and to the point
5. End with an appropriate closing (e.g., "Best regards,", "Thanks,")
6. Do NOT include a signature line (it will be added automatically)

Return ONLY valid JSON with this exact structure:
{
  "subject": "Clear, concise subject line",
  "body": "Full email body with greeting and closing"
}"""


class GmailAgent(BaseAgent):
    """Agent for Gmail operations: search, read, draft, send emails."""

    def __init__(
        self,
        gmail_service: GmailService,
        embedding_service: EmbeddingService,
        llm: Optional["LLMProvider"] = None,
    ):
        """Initialize the Gmail agent.

        Args:
            gmail_service: The Gmail service instance
            embedding_service: The embedding service for semantic search
            llm: Optional LLM provider for email composition
        """
        self.gmail = gmail_service
        self.embeddings = embedding_service
        self.llm = llm

    async def search(
        self, query: str, user_id: str, filters: Optional[dict] = None
    ) -> list[dict]:
        """Search emails using 3-way hybrid search with RRF fusion.

        Combines three retrieval methods for best results:
        1. BM25/Full-text search (keyword matching) - fast, ~5ms
        2. Vector search (semantic similarity) - ~20ms
        3. Filtered vector search (if filters provided) - ~20ms

        Results are fused using Reciprocal Rank Fusion (RRF) - no neural models.

        When query is empty but filters exist (e.g., "emails from last week"),
        falls back to filter-only search without semantic matching.

        Args:
            query: The search query
            user_id: The user's ID
            filters: Optional filters (sender, time_range, date_range, labels)

        Returns:
            List of matching emails, ranked by RRF score
        """
        # Handle empty/minimal queries with filters (e.g., "emails from last week")
        # In this case, do a filter-only search without semantic matching
        if not query or not query.strip():
            if filters:
                return await self.gmail.search_emails_filter_only(
                    user_id=user_id,
                    filters=filters,
                    limit=20,
                )
            # No query and no filters - return empty
            return []

        # Generate embedding for semantic search
        query_embedding = await self.embeddings.embed(query)

        # Run searches sequentially (SQLAlchemy session limitation)
        # 1. BM25/Full-text search (keyword matching) - also apply filters
        bm25_results = await self.gmail.search_emails_bm25(
            user_id=user_id,
            query=query,
            filters=filters,
            limit=20,
        )

        # 2. Vector search (semantic)
        semantic_results = await self.gmail.search_emails(
            user_id=user_id,
            embedding=query_embedding,
            filters=None,
            limit=20,
        )

        # 3. Filtered vector search (if filters provided)
        filtered_results = []
        if filters:
            filtered_results = await self.gmail.search_emails(
                user_id=user_id,
                embedding=query_embedding,
                filters=filters,
                limit=20,
            )

        # Combine using RRF fusion
        results = self._rrf_fusion(
            bm25_results=bm25_results,
            semantic_results=semantic_results,
            filtered_results=filtered_results,
        )

        return results

    def _rrf_fusion(
        self,
        bm25_results: list[dict],
        semantic_results: list[dict],
        filtered_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """Combine results using Reciprocal Rank Fusion (RRF).

        RRF is parameter-free and treats all methods fairly by converting
        scores to ranks, then fusing: score = Σ (1 / (k + rank))

        Args:
            bm25_results: Results from BM25/full-text search
            semantic_results: Results from vector/semantic search
            filtered_results: Results from filtered vector search
            k: RRF constant (default 60, standard value)

        Returns:
            Fused and deduplicated results sorted by RRF score
        """
        # Track RRF scores and result data by email ID
        rrf_scores: dict[str, float] = {}
        result_data: dict[str, dict] = {}
        match_sources: dict[str, list[str]] = {}

        # Process BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            email_id = result.get("id")
            if email_id:
                rrf_scores[email_id] = rrf_scores.get(email_id, 0) + (1 / (k + rank))
                if email_id not in result_data:
                    result_data[email_id] = result
                    match_sources[email_id] = []
                match_sources[email_id].append("bm25")

        # Process semantic results
        for rank, result in enumerate(semantic_results, start=1):
            email_id = result.get("id")
            if email_id:
                rrf_scores[email_id] = rrf_scores.get(email_id, 0) + (1 / (k + rank))
                if email_id not in result_data:
                    result_data[email_id] = result
                    match_sources[email_id] = []
                match_sources[email_id].append("semantic")

        # Process filtered results (with bonus weight for filter match)
        for rank, result in enumerate(filtered_results, start=1):
            email_id = result.get("id")
            if email_id:
                # Give filter matches a 1.5x boost in RRF score
                rrf_scores[email_id] = rrf_scores.get(email_id, 0) + (1.5 / (k + rank))
                if email_id not in result_data:
                    result_data[email_id] = result
                    match_sources[email_id] = []
                match_sources[email_id].append("filter")

        # Sort by RRF score and build final results
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for email_id in sorted_ids[:10]:  # Top 10
            result = result_data[email_id].copy()
            result["rrf_score"] = round(rrf_scores[email_id], 4)
            result["match_sources"] = match_sources[email_id]
            results.append(result)

        return results

    async def _compose_email(
        self, to: str, message_intent: str, topic: Optional[str] = None, recipient_name: Optional[str] = None, original_query: Optional[str] = None
    ) -> dict:
        """Use LLM to compose a proper email with subject and body.

        Args:
            to: Recipient email address
            message_intent: The user's intent/message to convey
            topic: Optional topic for the email
            recipient_name: Optional recipient name for greeting (preferred over extracting from email)
            original_query: Original user query for additional context

        Returns:
            Dict with 'subject' and 'body' keys
        """
        if not self.llm:
            # Fallback: use raw message as body with basic subject
            subject = topic if topic else message_intent[:50]
            greeting_name = recipient_name or "there"
            body = f"Hi {greeting_name},\n\n{message_intent}\n\nBest regards"
            return {"subject": subject, "body": body}

        # Use provided recipient_name, or extract from email as fallback
        if not recipient_name:
            if "<" in to:
                recipient_name = to.split("<")[0].strip()
            elif "@" in to:
                recipient_name = to.split("@")[0].replace(".", " ").title()

        prompt = f"""Compose an email with the following details:
- Recipient: {to}
- Recipient name (for greeting): {recipient_name or "Unknown"}
- Message intent: {message_intent}
- Topic: {topic or "N/A"}
- Original user request: {original_query or "N/A"}

Generate the email subject and body."""

        try:
            response = await self.llm.complete(
                prompt=prompt,
                system=EMAIL_COMPOSER_SYSTEM,
                json_mode=True,
            )
            data = self._parse_json_response(response)
            return {
                "subject": data.get("subject", topic or message_intent[:50]),
                "body": data.get("body", message_intent),
            }
        except Exception as e:
            # Fallback on error
            print(f"Email composition error: {e}")
            subject = topic if topic else message_intent[:50]
            body = f"Hi,\n\n{message_intent}\n\nBest regards"
            return {"subject": subject, "body": body}

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        patterns = [
            r"```json\s*(.*?)\s*```",
            r"```\s*(.*?)\s*```",
            r"\{.*\}",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1) if "```" in pattern else match.group(0))
                except json.JSONDecodeError:
                    continue
        return {}

    async def _apply_modification(self, body: str, instruction: str) -> str:
        """Apply a modification to the email body using LLM.

        Args:
            body: The original email body
            instruction: The modification instruction (e.g., "add 'John' after 'Thanks'")

        Returns:
            The modified email body
        """
        if not self.llm:
            # Fallback: just return original body
            return body

        prompt = f"""Apply this modification to the email body.

MODIFICATION REQUEST: {instruction}

ORIGINAL EMAIL BODY:
{body}

Apply the modification and return ONLY the modified email body text. Do not add any explanation or formatting - just the modified email content."""

        try:
            response = await self.llm.complete(prompt=prompt)
            return response.strip()
        except Exception as e:
            print(f"Modification error: {e}")
            return body

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
                message_intent = params.get("message", "")
                topic = params.get("topic", "")
                recipient_name = params.get("recipient_name", "")

                # If source_data exists, use it to build the draft
                source = params.get("source_data", {})
                if source and not to:
                    to = source.get("sender", "")
                if source and not subject:
                    subject = f"Re: {source.get('subject', '')}"

                # Validate required parameters
                if not to:
                    return StepResult(
                        step=step,
                        success=False,
                        error="Recipient email address is required. Please provide the recipient's email address.",
                    )

                # Use LLM to compose proper email if we have message intent but no body
                if message_intent and not body:
                    original_query = params.get("original_query", "")
                    composed = await self._compose_email(
                        to=to,
                        message_intent=message_intent,
                        topic=topic or None,
                        recipient_name=recipient_name or None,
                        original_query=original_query or None,
                    )
                    subject = subject or composed["subject"]
                    body = composed["body"]

                draft = await self.gmail.create_draft(
                    user_id=user_id, to=to, subject=subject, body=body
                )
                # Include email content in result for display
                # Use "draft_id" key so synthesizer can find it
                draft["draft_id"] = draft.get("id", draft.get("message_id", ""))
                draft["to"] = to
                draft["subject"] = subject
                draft["body"] = body
                return StepResult(step=step, success=True, data=draft)

            elif step == StepType.SEND_EMAIL:
                draft_id = params.get("draft_id", "")
                to = params.get("to", "")
                subject = params.get("subject", "")
                body = params.get("body", "")
                modification = params.get("modification_instruction", "")

                # Check if there's a modification request
                if modification and body:
                    # Apply modification using LLM
                    body = await self._apply_modification(body, modification)
                    # Send modified email directly (can't use send_draft since content changed)
                    # Note: The old draft will remain in Gmail - user can delete it manually
                    if not to:
                        return StepResult(
                            step=step,
                            success=False,
                            error="Recipient email address is required.",
                        )
                    sent = await self.gmail.send_email(
                        user_id=user_id, to=to, subject=subject, body=body
                    )
                    sent["to"] = to
                    sent["subject"] = subject
                    return StepResult(step=step, success=True, data=sent)

                # If we have a draft_id (no modification), send the draft directly (removes it from drafts)
                if draft_id:
                    sent = await self.gmail.send_draft(user_id=user_id, draft_id=draft_id)
                    sent["to"] = to
                    sent["subject"] = subject
                    return StepResult(step=step, success=True, data=sent)

                # Validate required parameters for new email
                if not to:
                    return StepResult(
                        step=step,
                        success=False,
                        error="Recipient email address is required. Please provide the recipient's email address.",
                    )

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
