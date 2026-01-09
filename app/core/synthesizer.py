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

IMPORTANT - EMAIL DRAFTS:
When a draft email is created, ALWAYS:
1. Mention that the draft has been SAVED to their Gmail Drafts folder
2. Show the draft content clearly with PROPER LINE BREAKS (To, Subject, Body on SEPARATE lines)
3. Let them know they can edit it here or directly in Gmail
4. Ask for explicit confirmation before sending

CRITICAL: Use actual line breaks between sections. DO NOT put everything on one line.

Use EXACTLY this format (with real newlines between each line):

"I've drafted an email for you and saved it to your Gmail Drafts:

**To:** recipient@example.com
**Subject:** Subject line here

Body of the email goes here.
It can have multiple lines.

Best regards

---

You can edit this draft here or directly in your Gmail. Would you like me to send it? Reply 'yes' to send, or tell me what changes you'd like."

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

For clarification needed (multiple recipients):
"I found multiple email addresses for [name]. Which one should I use?
1. john@company1.com
2. john@company2.com

Please let me know which one, or provide a different email address."

For recipient not found:
"I couldn't find an email address for [name] in your recent emails. Could you provide their email address so I can send the message?"

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
        conversation_context: list[dict] = None,
    ) -> str:
        """Generate a natural language response from execution results.

        Args:
            query: The original user query
            intent: The parsed intent
            results: List of step results
            conversation_context: Optional conversation history

        Returns:
            A natural language response string
        """
        # Check if this is a draft email result - format it ourselves for consistency
        draft_result = self._extract_draft_result(results)
        if draft_result:
            return self._format_draft_response(draft_result)

        # Build context for LLM
        results_summary = self._format_results(results)
        services_used = ", ".join(s.value if hasattr(s, 'value') else s for s in intent.services)

        # Include conversation context if available
        context_str = ""
        if conversation_context:
            context_parts = []
            for msg in conversation_context[-3:]:
                user_q = msg.get('query', '')
                ai_resp = msg.get('response', '')
                if user_q:
                    context_parts.append(f"User: {user_q}")
                if ai_resp:
                    truncated = ai_resp[:300] + "..." if len(ai_resp) > 300 else ai_resp
                    context_parts.append(f"Assistant: {truncated}")
            if context_parts:
                context_str = f"Previous conversation:\n{chr(10).join(context_parts)}\n\n"

        prompt = f"""{context_str}Current user query: "{query}"

Intent: {intent.operation} operation on {services_used}
Entities extracted: {intent.entities}

Execution results:
{results_summary}

Generate a helpful, natural response. If the user is asking about previous results or seems confused, acknowledge their question and clarify. Be concise but informative."""

        response = await self.llm.complete(prompt, system=SYNTHESIS_SYSTEM)
        return response.strip()

    def _extract_draft_result(self, results: list[StepResult]) -> dict | None:
        """Extract draft email data from results if present.

        Args:
            results: List of step results

        Returns:
            Draft data dict if found, None otherwise
        """
        for r in results:
            if r.success and r.data and "draft_id" in r.data:
                return r.data
        return None

    def _format_draft_response(self, draft: dict) -> str:
        """Format a draft email response with proper line breaks.

        Args:
            draft: Draft email data with to, subject, body, draft_id

        Returns:
            Formatted response string with proper newlines
        """
        to = draft.get("to", "")
        subject = draft.get("subject", "")
        body = draft.get("body", "")
        draft_id = draft.get("draft_id", "")

        response = f"""I've drafted an email for you and saved it to your Gmail Drafts:

**To:** {to}
**Subject:** {subject}
**Draft ID:** {draft_id}

{body}

---

This draft is saved in your Gmail. You can edit it here or directly in your mailbox.

Would you like me to send it? Reply 'yes' to send, or let me know what changes you'd like."""

        return response

    def _format_results(self, results: list[StepResult]) -> str:
        """Format step results for LLM context.

        Args:
            results: List of step results

        Returns:
            Formatted string of results
        """
        lines = []

        # Check if this is a recipient resolution scenario that failed
        # In that case, don't show the search results since they're not relevant
        is_recipient_resolution_failure = any(
            not r.success and r.data and r.data.get("type") in ("clarification_needed", "recipient_not_found")
            for r in results
        )

        for r in results:
            status = "SUCCESS" if r.success else f"FAILED: {r.error}"

            # Skip showing search results if this was for recipient resolution that failed
            step_str = r.step.value if hasattr(r.step, 'value') else r.step
            if is_recipient_resolution_failure and step_str == "search_gmail":
                # Don't show the search results - they were just for finding the recipient's email
                lines.append(f"- {r.step}: (searched for recipient's email)")
                continue

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

                # Format draft email results - include full content for display
                elif "draft_id" in r.data:
                    lines.append(f"  Draft saved to Gmail Drafts folder (ID: {r.data['draft_id']})")
                    lines.append(f"  EMAIL DETAILS:")
                    if r.data.get("to"):
                        lines.append(f"    To: {r.data['to']}")
                    if r.data.get("subject"):
                        lines.append(f"    Subject: {r.data['subject']}")
                    if r.data.get("body"):
                        # Format body with clear line breaks
                        body = r.data['body'].replace('\n', '\n    ')
                        lines.append(f"    Body:")
                        lines.append(f"    {body}")
                    lines.append("  STATUS: Draft saved. User can edit here or in Gmail. Ask for confirmation to send.")
                elif "message_id" in r.data:
                    lines.append(f"  Message sent: {r.data['message_id']}")
                elif "event_id" in r.data:
                    lines.append(f"  Event: {r.data['event_id']}")
                elif "deleted" in r.data:
                    lines.append("  Item deleted successfully")

            # Handle special clarification/resolution cases
            elif not r.success and r.data:
                if r.data.get("type") == "clarification_needed":
                    recipient = r.data.get("recipient_name", "")
                    options = r.data.get("email_options", [])
                    lines.append(f"  CLARIFICATION NEEDED: Found multiple emails for '{recipient}'")
                    for i, email in enumerate(options, 1):
                        lines.append(f"    {i}. {email}")
                    lines.append("  Ask user which email to use.")

                elif r.data.get("type") == "recipient_not_found":
                    recipient = r.data.get("recipient_name", "")
                    lines.append(f"  RECIPIENT NOT FOUND: Could not find email for '{recipient}'")
                    lines.append("  Ask user to provide the email address. DO NOT show any email search results.")

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
