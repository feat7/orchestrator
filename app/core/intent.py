"""Intent classification using LLM."""

import json
import re
from typing import Optional

from app.core.llm import LLMProvider
from app.schemas.intent import ParsedIntent, ServiceType, StepType


INTENT_CLASSIFIER_SYSTEM = """You are an intent classifier for a Google Workspace assistant.
Given a user query, classify it into structured JSON format.

AVAILABLE SERVICES:
- gmail: Email operations (search, read, send, draft emails)
- gcal: Calendar operations (search, create, update, delete events)
- gdrive: File operations (search, read, share files)

AVAILABLE STEPS (use ONLY these exact values):
- search_gmail: Search emails by keyword, sender, date, etc.
- get_email: Get full email content by ID
- draft_email: Create an email draft
- send_email: Send an email
- search_calendar: Search calendar events
- get_event: Get event details by ID
- create_event: Create a new calendar event
- update_event: Modify an existing event
- delete_event: Delete a calendar event
- search_drive: Search files in Google Drive
- get_file: Get file content/metadata
- share_file: Share a file with someone

RULES:
1. Identify ALL services needed (can be multiple for complex queries)
2. Extract entities: names, dates, companies, email addresses, times, etc.
3. Order steps logically:
   - Search steps come first (to find relevant items)
   - Action steps depend on search results
   - Multiple searches can run in parallel
4. For temporal references:
   - "next week" = upcoming 7 days
   - "tomorrow" = next day
   - "today" = current day
5. For ambiguous queries, provide your best interpretation and set confidence lower

IMPORTANT - EMAIL SENDING REQUIRES CONFIRMATION:
When the user wants to send an email, ALWAYS use "draft_email" step FIRST (never "send_email" directly).
The system will show the draft to the user for review and ask for confirmation before sending.
Only use "send_email" step when user explicitly confirms (e.g., "yes send it", "looks good, send", "confirmed").

IMPORTANT - RECIPIENT RESOLUTION:
When the user wants to send/draft an email to a PERSON'S NAME (not an email address with @):
- ALWAYS add "search_gmail" as the FIRST step to find their email address from past emails
- Set "recipient_needs_resolution": true in entities
- Set "recipient_name": "<the person's name>" in entities
- The system will search for emails from/to that person and resolve the email address

EXAMPLES:

Query: "What's on my calendar next week?"
{
  "services": ["gcal"],
  "operation": "search",
  "entities": {"time_range": "next_week"},
  "steps": ["search_calendar"],
  "confidence": 0.95
}

Query: "Cancel my Turkish Airlines flight"
{
  "services": ["gmail", "gcal"],
  "operation": "update",
  "entities": {"airline": "Turkish Airlines", "action": "cancel"},
  "steps": ["search_gmail", "search_calendar", "draft_email"],
  "confidence": 0.9
}

Query: "Prepare for tomorrow's meeting with Acme Corp"
{
  "services": ["gcal", "gmail", "gdrive"],
  "operation": "search",
  "entities": {"company": "Acme Corp", "time": "tomorrow", "event_type": "meeting"},
  "steps": ["search_calendar", "search_gmail", "search_drive"],
  "confidence": 0.85
}

Query: "Find emails from sarah@company.com about the budget"
{
  "services": ["gmail"],
  "operation": "search",
  "entities": {"sender": "sarah@company.com", "topic": "budget"},
  "steps": ["search_gmail"],
  "confidence": 0.95
}

Query: "Send an email to Vinay telling him I'm waiting for the product launch"
{
  "services": ["gmail"],
  "operation": "draft",
  "entities": {
    "recipient_name": "Vinay",
    "recipient_needs_resolution": true,
    "message": "waiting for the product launch"
  },
  "steps": ["search_gmail", "draft_email"],
  "confidence": 0.9
}

Query: "Draft email to john@acme.com about the project update"
{
  "services": ["gmail"],
  "operation": "draft",
  "entities": {"recipient": "john@acme.com", "topic": "project update"},
  "steps": ["draft_email"],
  "confidence": 0.95
}

IMPORTANT - FOLLOW-UP EMAIL ADDRESS:
When the conversation context shows the assistant asked for an email address, and the user responds with just an email address:
- This is the user providing the recipient email for the previous send/draft request
- Set "recipient": "<the email address>" in entities
- Use "draft_email" step (create draft for review, NOT send directly)
- The message content should be inferred from the previous conversation context

Example conversation context:
User: "send email to vinay about the product launch"
Assistant: "I couldn't find an email for Vinay. Could you provide their email address?"
User: "vinay@example.com"

For the follow-up "vinay@example.com":
{
  "services": ["gmail"],
  "operation": "draft",
  "entities": {"recipient": "vinay@example.com", "message": "product launch"},
  "steps": ["draft_email"],
  "confidence": 0.95
}

IMPORTANT - SEND CONFIRMATION:
When user confirms they want to send a draft (e.g., "yes", "send it", "looks good", "confirmed"):
- Check conversation context for a pending draft email
- Use "send_email" step
- Set "confirmed_send": true in entities
- CRITICAL: Extract the draft details from the assistant's previous response and include them in entities:
  - "draft_id": the draft ID (shown as **Draft ID:** in the response) - THIS IS REQUIRED
  - "to": the recipient email address
  - "subject": the email subject line
  - "body": the full email body content

Example conversation context:
User: "send email to vinay about product launch"
Assistant: "I've drafted an email... **To:** vinay@example.com **Subject:** Product Launch **Draft ID:** draft_abc123 Hi Vinay, I wanted to reach out... Best regards ---"
User: "yes send it"

For the confirmation "yes send it", extract details from the draft shown above:
{
  "services": ["gmail"],
  "operation": "send",
  "entities": {
    "confirmed_send": true,
    "draft_id": "draft_abc123",
    "to": "vinay@example.com",
    "subject": "Product Launch",
    "body": "Hi Vinay,\n\nI wanted to reach out...\n\nBest regards"
  },
  "steps": ["send_email"],
  "confidence": 0.95
}

Return ONLY valid JSON with this exact structure:
{
  "services": ["gmail", "gcal", "gdrive"],
  "operation": "search|create|update|delete|send|draft",
  "entities": {"key": "value"},
  "steps": ["step1", "step2"],
  "confidence": 0.9
}"""


class IntentClassifier:
    """Classifies user queries into structured intents using LLM."""

    def __init__(self, llm: LLMProvider):
        """Initialize the classifier with an LLM provider.

        Args:
            llm: The LLM provider to use for classification
        """
        self.llm = llm

    async def classify(self, query: str, conversation_context: Optional[list[dict]] = None) -> ParsedIntent:
        """Classify a user query into a structured intent.

        Args:
            query: The natural language query to classify
            conversation_context: Optional list of previous messages for context

        Returns:
            A ParsedIntent object with services, operation, entities, and steps
        """
        # Build prompt with optional context
        prompt = f"User query: {query}"

        if conversation_context:
            context_parts = []
            # Check if this might be a send confirmation (short query like "yes", "send", etc.)
            is_potential_confirmation = len(query.split()) <= 5 and any(
                word in query.lower() for word in ["yes", "send", "confirm", "ok", "sure", "go ahead", "do it"]
            )
            for msg in conversation_context[-3:]:
                user_q = msg.get('query', '')
                ai_resp = msg.get('response', '')
                if user_q:
                    context_parts.append(f"User: {user_q}")
                if ai_resp:
                    # Don't truncate if this might be a send confirmation - we need the full draft
                    if is_potential_confirmation:
                        context_parts.append(f"Assistant: {ai_resp}")
                    else:
                        # Truncate long responses for other cases
                        truncated = ai_resp[:200] + "..." if len(ai_resp) > 200 else ai_resp
                        context_parts.append(f"Assistant: {truncated}")
            if context_parts:
                prompt = f"Recent conversation:\n{chr(10).join(context_parts)}\n\nCurrent {prompt}"

        prompt += "\n\nClassify this query and return JSON:"

        # Get LLM response
        response = await self.llm.complete(
            prompt=prompt, system=INTENT_CLASSIFIER_SYSTEM, json_mode=True
        )

        # Parse JSON from response
        data = self._parse_json_response(response)

        # Validate and construct ParsedIntent
        return self._build_intent(data)

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response, handling markdown code blocks.

        Args:
            response: The raw LLM response

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If JSON cannot be parsed
        """
        # Try direct JSON parse first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract from markdown code blocks
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

        raise ValueError(f"Failed to parse intent JSON from response: {response[:200]}")

    def _build_intent(self, data: dict) -> ParsedIntent:
        """Build a ParsedIntent from parsed JSON data.

        Args:
            data: The parsed JSON dictionary

        Returns:
            A validated ParsedIntent object
        """
        entities = data.get("entities", {})

        # Map service strings to enum values
        services = []
        for s in data.get("services", []):
            try:
                services.append(ServiceType(s))
            except ValueError:
                # Skip invalid services
                pass

        # Map step strings to enum values
        steps = []
        for s in data.get("steps", []):
            try:
                steps.append(StepType(s))
            except ValueError:
                # Skip invalid steps
                pass

        # Ensure we have at least one service and step
        if not services:
            services = [ServiceType.GMAIL]
        if not steps:
            steps = [StepType.SEARCH_GMAIL]

        # Post-processing: Ensure recipient resolution works correctly
        # If recipient_needs_resolution is set, ensure search_gmail is first
        if entities.get("recipient_needs_resolution"):
            if StepType.SEARCH_GMAIL not in steps:
                steps.insert(0, StepType.SEARCH_GMAIL)
            elif steps[0] != StepType.SEARCH_GMAIL:
                # Move search_gmail to front
                steps.remove(StepType.SEARCH_GMAIL)
                steps.insert(0, StepType.SEARCH_GMAIL)

            # Ensure gmail service is included
            if ServiceType.GMAIL not in services:
                services.append(ServiceType.GMAIL)

        return ParsedIntent(
            services=services,
            operation=data.get("operation", "search"),
            entities=entities,
            steps=steps,
            confidence=data.get("confidence", 0.8),
        )
