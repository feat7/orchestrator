"""Response synthesizer for generating natural language responses."""

from typing import Optional

from app.core.llm import LLMProvider
from app.schemas.intent import ParsedIntent, StepResult


SYNTHESIS_SYSTEM = """You are a helpful assistant that synthesizes results from Google Workspace operations.

Given the user's original query, the intent classification, and the results from executed steps,
generate a natural, conversational response that:

1. SUMMARIZES what was found or done clearly and concisely
2. HIGHLIGHTS key information (dates, names, important details)
3. ASKS for confirmation if actions are pending (like sending emails)
4. HANDLES failures gracefully with helpful suggestions

FORMATTING GUIDELINES:
- Use checkmarks for completed actions
- Use bullet points for lists of items
- Format dates in human-readable way (e.g., "Monday, January 15 at 2:00 PM")
- Keep responses concise but complete
- If multiple items found, summarize the most relevant ones

EXAMPLES:

For a successful flight search:
"I found your Turkish Airlines booking (TK1234) in an email from Oct 15.
- Calendar event: Istanbul - NYC Flight on Nov 5 at 10:30 AM
- Drafted a cancellation email to support@turkishairlines.com

Would you like me to send the cancellation request?"

For a calendar search:
"Here's what's on your calendar next week:
- Monday 9 AM: Team standup
- Tuesday 2 PM: Client call with Acme Corp
- Thursday 10 AM: 1:1 with Sarah

Would you like more details about any of these events?"

For a failure:
"I couldn't find any emails matching your search. Try:
- Checking if the sender's email is correct
- Broadening your search terms
- Looking in a different date range"

Be helpful, accurate, and conversational."""


class ResponseSynthesizer:
    """Synthesizes natural language responses from execution results."""

    def __init__(self, llm: LLMProvider):
        """Initialize the synthesizer.

        Args:
            llm: The LLM provider to use
        """
        self.llm = llm

    async def synthesize(
        self,
        query: str,
        intent: ParsedIntent,
        results: list[StepResult],
    ) -> str:
        """Generate a natural language response from execution results.

        Args:
            query: The original user query
            intent: The parsed intent
            results: List of step results

        Returns:
            A natural language response string
        """
        # Build context for LLM
        results_summary = self._format_results(results)
        services_used = ", ".join(s.value if hasattr(s, 'value') else s for s in intent.services)

        prompt = f"""User query: "{query}"

Intent: {intent.operation} operation on {services_used}
Entities extracted: {intent.entities}

Execution results:
{results_summary}

Generate a helpful, natural response for the user. Be concise but informative."""

        response = await self.llm.complete(prompt, system=SYNTHESIS_SYSTEM)
        return response.strip()

    def _format_results(self, results: list[StepResult]) -> str:
        """Format step results for LLM context.

        Args:
            results: List of step results

        Returns:
            Formatted string of results
        """
        lines = []

        for r in results:
            status = "SUCCESS" if r.success else f"FAILED: {r.error}"
            lines.append(f"- {r.step}: {status}")

            if r.success and r.data:
                # Format search results
                if "results" in r.data:
                    items = r.data["results"][:5]  # Limit for context
                    if not items:
                        lines.append("  (No results found)")
                    else:
                        lines.append(f"  Found {len(r.data['results'])} items:")
                        for item in items:
                            lines.append(self._format_item(item))

                # Format action results
                elif "draft_id" in r.data:
                    lines.append(f"  Draft created: {r.data['draft_id']}")
                elif "message_id" in r.data:
                    lines.append(f"  Message sent: {r.data['message_id']}")
                elif "event_id" in r.data:
                    lines.append(f"  Event: {r.data['event_id']}")
                elif "deleted" in r.data:
                    lines.append("  Item deleted successfully")

        return "\n".join(lines)

    def _format_item(self, item: dict) -> str:
        """Format a single result item.

        Args:
            item: The item dictionary

        Returns:
            Formatted string
        """
        # Email format
        if "subject" in item:
            sender = item.get("sender", "Unknown")
            subject = item.get("subject", "No subject")
            date = item.get("received_at", "")
            return f"    - Email: \"{subject}\" from {sender} ({date})"

        # Event format
        if "title" in item and "start_time" in item:
            title = item.get("title", "Untitled")
            start = item.get("start_time", "")
            location = item.get("location", "")
            loc_str = f" at {location}" if location else ""
            return f"    - Event: \"{title}\" on {start}{loc_str}"

        # File format
        if "name" in item:
            name = item.get("name", "Unnamed")
            mime = item.get("mime_type", "")
            return f"    - File: \"{name}\" ({mime})"

        # Generic format
        return f"    - {item}"

    async def synthesize_error(self, query: str, error: str) -> str:
        """Generate an error response.

        Args:
            query: The original query
            error: The error message

        Returns:
            A helpful error response
        """
        prompt = f"""The user asked: "{query}"

But an error occurred: {error}

Generate a helpful, apologetic response explaining what went wrong
and suggesting what the user could try instead."""

        response = await self.llm.complete(prompt, system=SYNTHESIS_SYSTEM)
        return response.strip()
