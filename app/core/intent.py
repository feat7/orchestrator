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
            context_str = "\n".join(
                f"- Previous query: {msg.get('query', '')}" for msg in conversation_context[-3:]
            )
            prompt = f"Recent conversation context:\n{context_str}\n\n{prompt}"

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

        return ParsedIntent(
            services=services,
            operation=data.get("operation", "search"),
            entities=data.get("entities", {}),
            steps=steps,
            confidence=data.get("confidence", 0.8),
        )
