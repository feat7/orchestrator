"""Intent classification using LLM with tool-based parameter extraction."""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Optional

from app.core.llm import LLMProvider
from app.schemas.intent import ParsedIntent, ServiceType, StepType, ExecutionStep

logger = logging.getLogger(__name__)

# Timeout for LLM classification calls (30 seconds)
CLASSIFICATION_TIMEOUT = 30.0


def get_intent_classifier_prompt(current_datetime: datetime) -> str:
    """Generate the intent classifier system prompt with current date context."""

    # Format dates for the prompt
    date_str = current_datetime.strftime("%A, %B %d, %Y")
    time_str = current_datetime.strftime("%I:%M %p")
    iso_today = current_datetime.strftime("%Y-%m-%d")
    iso_yesterday = (current_datetime - timedelta(days=1)).strftime("%Y-%m-%d")
    iso_week_ago = (current_datetime - timedelta(days=7)).strftime("%Y-%m-%d")
    iso_month_ago = (current_datetime - timedelta(days=30)).strftime("%Y-%m-%d")
    iso_tomorrow = (current_datetime + timedelta(days=1)).strftime("%Y-%m-%d")
    iso_day_after = (current_datetime + timedelta(days=2)).strftime("%Y-%m-%d")

    # Calculate start of this week (Monday)
    days_since_monday = current_datetime.weekday()
    start_of_week = (current_datetime - timedelta(days=days_since_monday)).strftime("%Y-%m-%d")

    return f"""You are an intent classifier for a Google Workspace assistant.
Given a user query, classify it and extract the EXACT parameters needed for execution.

CURRENT DATE/TIME: {date_str} at {time_str}
- Today's date (ISO): {iso_today}
- Yesterday (ISO): {iso_yesterday}
- Tomorrow (ISO): {iso_tomorrow}
- Start of this week (Monday): {start_of_week}
- 7 days ago: {iso_week_ago}
- 30 days ago: {iso_month_ago}

AVAILABLE TOOLS AND THEIR PARAMETERS:

1. search_drive - Search Google Drive files
   params:
   - search_query: string (semantic search for file content/name, use "" if only filtering by date)
   - modified_after: ISO date string (e.g., "{iso_week_ago}" for files from last week)
   - modified_before: ISO date string
   - mime_type: string ("application/pdf" for PDFs, "application/vnd.google-apps.document" for Docs)
   - file_name: string (partial name match)

2. search_gmail - Search emails
   params:
   - search_query: string (semantic search for email content)
   - sender: string (email or name)
   - recipient: string (email or name)
   - subject: string
   - after_date: ISO date string
   - before_date: ISO date string

3. get_email - Get full email content (REQUIRES search_gmail first to find the email)
   params:
   - email_id: string (from search results)
   - subject: string (for reference/fallback search)
   IMPORTANT: Always add search_gmail step BEFORE get_email with depends_on: [search_step_index]
   Use this when user wants to see the full content of an email found in a previous search.

4. search_calendar - Search calendar events
   params:
   - search_query: string (semantic search for event content)
   - start_after: ISO datetime (e.g., "{iso_today}T00:00:00")
   - start_before: ISO datetime
   - attendee: string (email or name)

5. get_file - Get file details (REQUIRES search_drive first to get file_id)
   params:
   - file_id: string (from search results)
   - file_name: string (for reference)
   IMPORTANT: Always add search_drive step BEFORE get_file with depends_on: [search_step_index]

6. draft_email - Create email draft (ALWAYS use this before send_email)
   params:
   - to: string (email address if known)
   - to_name: string (recipient name, for resolving email from search)
   - subject: string (optional)
   - message: string (the message intent/content)

7. send_email - Send email (only after user confirms draft)
   params:
   - draft_id: string (from previous draft result)
   - to: string
   - subject: string
   - body: string

8. create_event - Create calendar event
   params:
   - title: string (required)
   - start_time: ISO datetime (required)
   - end_time: ISO datetime (required)
   - attendees: array of emails
   - description: string
   - location: string

9. share_file - Share a file
   params:
   - file_id: string (from search)
   - email: string (who to share with)
   - role: string ("reader", "writer", "commenter")

LIMITATIONS - THINGS YOU CANNOT DO:
- CANNOT edit/modify Google Drive files (only read, search, share)
- CANNOT delete files or emails
- CANNOT modify calendar events (only create new ones)

If user asks to edit a file, respond with operation: "chat" and explain they need to edit in Google Drive directly. Provide the file's web link if available.

CRITICAL RULES:
1. For temporal queries, CALCULATE the actual ISO date - don't use words like "recent":
   - "recent files" → modified_after: "{iso_month_ago}"
   - "last week" / "past week" → modified_after: "{iso_week_ago}"
   - "this week" → modified_after: "{start_of_week}"
   - "yesterday" → modified_after: "{iso_yesterday}", modified_before: "{iso_today}"
   - "today" → modified_after: "{iso_today}"
   - "tomorrow" → start_after: "{iso_tomorrow}T00:00:00", start_before: "{iso_day_after}T00:00:00"

2. If a step needs data from a previous step, use depends_on: [step_index]

3. For email sending: ALWAYS use draft_email first, NEVER send_email directly

4. If recipient is a NAME (not email), set to_name AND add search_gmail step first with depends_on

5. CONVERSATIONAL RESPONSES: If the user message is just an acknowledgment or reaction (e.g., "interesting", "cool", "nice", "okay", "thanks", "got it", "I see", "makes sense"), use operation: "chat" with empty steps - these are NOT search queries!

OUTPUT FORMAT (JSON):
{{
  "services": ["gdrive"],
  "operation": "search",
  "steps": [
    {{
      "step": "search_drive",
      "params": {{
        "search_query": "",
        "modified_after": "{iso_month_ago}"
      }}
    }}
  ],
  "confidence": 0.95
}}

EXAMPLES:

Query: "show me recent files"
{{
  "services": ["gdrive"],
  "operation": "search",
  "steps": [
    {{"step": "search_drive", "params": {{"search_query": "", "modified_after": "{iso_month_ago}"}}}}
  ],
  "confidence": 0.95
}}

Query: "files from this week"
{{
  "services": ["gdrive"],
  "operation": "search",
  "steps": [
    {{"step": "search_drive", "params": {{"search_query": "", "modified_after": "{start_of_week}"}}}}
  ],
  "confidence": 0.95
}}

Query: "recent PDFs"
{{
  "services": ["gdrive"],
  "operation": "search",
  "steps": [
    {{"step": "search_drive", "params": {{"search_query": "", "modified_after": "{iso_month_ago}", "mime_type": "application/pdf"}}}}
  ],
  "confidence": 0.95
}}

Query: "when was the project proposal last modified?"
{{
  "services": ["gdrive"],
  "operation": "search",
  "steps": [
    {{"step": "search_drive", "params": {{"search_query": "project proposal", "file_name": "project proposal"}}}}
  ],
  "confidence": 0.9
}}

Query: "what's in the Orchestrator file?" or "tell me the contents of that doc"
{{
  "services": ["gdrive"],
  "operation": "search",
  "steps": [
    {{"step": "search_drive", "params": {{"search_query": "Orchestrator", "file_name": "Orchestrator"}}}},
    {{"step": "get_file", "params": {{"file_name": "Orchestrator"}}, "depends_on": [0]}}
  ],
  "confidence": 0.9
}}

Query: "send email to John about the meeting"
{{
  "services": ["gmail"],
  "operation": "draft",
  "steps": [
    {{"step": "search_gmail", "params": {{"search_query": "John", "sender": "John"}}}},
    {{"step": "draft_email", "params": {{"to_name": "John", "message": "about the meeting"}}, "depends_on": [0]}}
  ],
  "confidence": 0.9
}}

Query: "my meetings tomorrow"
{{
  "services": ["gcal"],
  "operation": "search",
  "steps": [
    {{"step": "search_calendar", "params": {{"search_query": "", "start_after": "{iso_tomorrow}T00:00:00", "start_before": "{iso_day_after}T00:00:00"}}}}
  ],
  "confidence": 0.95
}}

Query: "emails from Sarah last week"
{{
  "services": ["gmail"],
  "operation": "search",
  "steps": [
    {{"step": "search_gmail", "params": {{"search_query": "", "sender": "Sarah", "after_date": "{iso_week_ago}"}}}}
  ],
  "confidence": 0.95
}}

FOLLOW-UP QUERIES ABOUT PREVIOUS RESULTS:
When user asks about something from a previous search/response (e.g., "tell me more about that", "show me the full email", "open that file", "what's in it"):
- Look at the conversation context to identify what "that" or "it" refers to
- Extract relevant identifiers (email subject, file name, etc.) from the previous response
- Create appropriate search/get steps to fetch the full details

Example 1: If previous response found "email about HeyGen training", and user says "tell me more about that" or "show me the full email":
{{
  "services": ["gmail"],
  "operation": "search",
  "steps": [
    {{"step": "search_gmail", "params": {{"search_query": "HeyGen training"}}}},
    {{"step": "get_email", "params": {{"subject": "HeyGen training"}}, "depends_on": [0]}}
  ],
  "confidence": 0.9
}}

Example 2: If previous response found files, and user says "show me the first one" or "open that":
{{
  "services": ["gdrive"],
  "operation": "search",
  "steps": [
    {{"step": "search_drive", "params": {{"search_query": "[file name from context]"}}}},
    {{"step": "get_file", "params": {{"file_name": "[file name from context]"}}, "depends_on": [0]}}
  ],
  "confidence": 0.9
}}

Example 3: "what does it say" or "read it to me" after finding an email:
{{
  "services": ["gmail"],
  "operation": "search",
  "steps": [
    {{"step": "search_gmail", "params": {{"search_query": "[topic from context]"}}}},
    {{"step": "get_email", "params": {{"subject": "[subject from context]"}}, "depends_on": [0]}}
  ],
  "confidence": 0.9
}}

IMPORTANT: Do NOT ask for clarification if the context makes it clear what the user is referring to. Use the conversation history to resolve ambiguous references like "that", "it", "the email", "the file", etc. When only ONE item was found/mentioned in the previous response, assume the user is referring to that item.

SEND CONFIRMATION:
When user confirms a draft (e.g., "yes", "send it", "looks good"):
- Check conversation for draft details (Draft ID, to, subject, body)
- Use send_email with those extracted details
{{
  "services": ["gmail"],
  "operation": "send",
  "steps": [
    {{"step": "send_email", "params": {{"draft_id": "draft_xxx", "to": "user@email.com", "subject": "...", "body": "..."}}}}
  ],
  "confidence": 0.95
}}

CONVERSATIONAL RESPONSES (no action needed):
When user just acknowledges or reacts (e.g., "interesting", "cool", "nice", "okay", "I see"):
{{
  "services": [],
  "operation": "chat",
  "steps": [],
  "confidence": 1.0,
  "response": "Is there anything else you'd like to know?"
}}

UNSUPPORTED ACTIONS - Edit/delete requests:
When user asks to edit, modify, update, or delete a file:
{{
  "services": [],
  "operation": "chat",
  "steps": [],
  "confidence": 1.0,
  "response": "I can't edit files directly, but I can help you find and open them. You'll need to edit the file in Google Drive. Would you like me to show you the file so you can open it in Drive?"
}}

Return ONLY valid JSON."""


class IntentClassifier:
    """Classifies user queries into structured intents with typed parameters."""

    def __init__(self, llm: LLMProvider):
        """Initialize the intent classifier.

        Args:
            llm: The LLM provider instance
        """
        self.llm = llm

    async def classify(
        self, query: str, conversation_context: Optional[list[dict]] = None
    ) -> ParsedIntent:
        """Classify a user query into a structured intent with typed step parameters.

        Args:
            query: The user's natural language query
            conversation_context: Optional previous messages for context

        Returns:
            A ParsedIntent object with services, operation, steps with params
        """
        # Get current datetime for the prompt
        now = datetime.now()
        system_prompt = get_intent_classifier_prompt(now)

        # Build user prompt
        prompt = f"User query: {query}"

        if conversation_context:
            context_parts = []
            # Check if this might be a send confirmation
            query_lower = query.lower()
            is_potential_confirmation = any(
                word in query_lower for word in ["yes", "send", "confirm", "ok", "sure", "go ahead"]
            )

            for ctx in conversation_context[-3:]:
                if ctx.get("query"):
                    context_parts.append(f"User: {ctx['query']}")
                if ctx.get("response"):
                    resp = ctx["response"]
                    # Keep full response for send confirmations
                    if not is_potential_confirmation and len(resp) > 300:
                        resp = resp[:300] + "..."
                    context_parts.append(f"Assistant: {resp}")

            if context_parts:
                prompt = f"Recent conversation:\n{chr(10).join(context_parts)}\n\nCurrent query: {query}"

        prompt += "\n\nClassify this query and return JSON with exact parameters:"

        logger.info(f"[CONTEXT_DEBUG] Classifying query: {query[:100]}...")
        if conversation_context:
            logger.info(f"[CONTEXT_DEBUG] With conversation context: {len(conversation_context)} messages")
            logger.info(f"[CONTEXT_DEBUG] Full prompt being sent to LLM:\n{prompt[:1000]}...")

        # Get LLM response with timeout
        try:
            response = await asyncio.wait_for(
                self.llm.complete(
                    prompt=prompt, system=system_prompt, json_mode=True
                ),
                timeout=CLASSIFICATION_TIMEOUT,
            )
            logger.info(f"LLM classification response received ({len(response)} chars): {response[:400]}...")
        except asyncio.TimeoutError:
            logger.error(f"Intent classification timed out after {CLASSIFICATION_TIMEOUT}s for query: {query[:100]}")
            return self._fallback_intent(query)
        except Exception as e:
            logger.error(f"Intent classification failed for query '{query[:100]}': {e}", exc_info=True)
            return self._fallback_intent(query)

        # Parse JSON from response
        try:
            data = self._parse_json_response(response)
            logger.debug(f"Parsed intent data: {data}")
        except ValueError as e:
            logger.error(f"Failed to parse intent JSON: {e}")
            return self._fallback_intent(query)

        # Build the ParsedIntent
        try:
            return self._build_intent(data, query)
        except Exception as e:
            logger.error(f"Failed to build intent from data: {e}", exc_info=True)
            return self._fallback_intent(query)

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        text = response.strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # Remove first line (```json or ```)
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        # Try to extract JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Failed to parse JSON from: {response[:200]}")

    def _build_intent(self, data: dict, query: str) -> ParsedIntent:
        """Build a ParsedIntent from parsed JSON data."""
        # Extract services
        services = []
        for s in data.get("services", []):
            try:
                services.append(ServiceType(s))
            except ValueError:
                logger.warning(f"Unknown service: {s}")

        if not services:
            # Infer from steps
            steps_data = data.get("steps", [])
            for step_data in steps_data:
                step_name = step_data.get("step", "") if isinstance(step_data, dict) else ""
                if "gmail" in step_name:
                    services.append(ServiceType.GMAIL)
                elif "calendar" in step_name:
                    services.append(ServiceType.GCAL)
                elif "drive" in step_name or "file" in step_name:
                    services.append(ServiceType.GDRIVE)
            services = list(set(services)) or [ServiceType.GMAIL]

        # Extract operation
        operation = data.get("operation", "search")
        if operation not in ["search", "create", "update", "delete", "send", "draft", "chat"]:
            operation = "search"

        # Handle chat operation (no steps needed)
        if operation == "chat":
            response = data.get("response", "Is there anything else you'd like to know?")
            return ParsedIntent(
                services=[],
                operation="chat",
                steps=[],
                confidence=data.get("confidence", 1.0),
                response=response,
                entities={}
            )

        # Build steps with params
        steps = []
        for step_data in data.get("steps", []):
            if not isinstance(step_data, dict):
                continue

            step_name = step_data.get("step", "")
            try:
                step_type = StepType(step_name)
            except ValueError:
                logger.warning(f"Unknown step type: {step_name}")
                continue

            params = step_data.get("params", {})
            depends_on = step_data.get("depends_on")

            steps.append(ExecutionStep(
                step=step_type,
                params=params,
                depends_on=depends_on
            ))

        # If no valid steps, create a default search step
        if not steps:
            if ServiceType.GDRIVE in services:
                steps.append(ExecutionStep(
                    step=StepType.SEARCH_DRIVE,
                    params={"search_query": query}
                ))
            elif ServiceType.GCAL in services:
                steps.append(ExecutionStep(
                    step=StepType.SEARCH_CALENDAR,
                    params={"search_query": query}
                ))
            else:
                steps.append(ExecutionStep(
                    step=StepType.SEARCH_GMAIL,
                    params={"search_query": query}
                ))

        # Post-process: Ensure get_file has a search_drive step before it
        steps = self._ensure_search_before_get(steps, query)

        confidence = data.get("confidence", 0.8)

        return ParsedIntent(
            services=services,
            operation=operation,
            steps=steps,
            confidence=confidence,
            entities={}  # Deprecated
        )

    def _ensure_search_before_get(self, steps: list[ExecutionStep], query: str) -> list[ExecutionStep]:
        """Ensure get_file/get_email/get_event have search steps before them.

        If a get_ step exists without a corresponding search step, insert one before it.
        """
        new_steps = []
        has_search_drive = any(s.step == StepType.SEARCH_DRIVE for s in steps)
        has_search_gmail = any(s.step == StepType.SEARCH_GMAIL for s in steps)
        has_search_calendar = any(s.step == StepType.SEARCH_CALENDAR for s in steps)

        for i, step in enumerate(steps):
            # If get_file without preceding search_drive, insert search first
            if step.step == StepType.GET_FILE and not has_search_drive:
                file_name = step.params.get("file_name", "")
                search_step = ExecutionStep(
                    step=StepType.SEARCH_DRIVE,
                    params={"search_query": file_name, "file_name": file_name} if file_name else {"search_query": query}
                )
                new_steps.append(search_step)
                has_search_drive = True
                # Update depends_on to point to the search step we just added
                step.depends_on = [len(new_steps) - 1]

            # If get_email without preceding search_gmail, insert search first
            if step.step == StepType.GET_EMAIL and not has_search_gmail:
                search_step = ExecutionStep(
                    step=StepType.SEARCH_GMAIL,
                    params={"search_query": query}
                )
                new_steps.append(search_step)
                has_search_gmail = True
                step.depends_on = [len(new_steps) - 1]

            # If get_event without preceding search_calendar, insert search first
            if step.step == StepType.GET_EVENT and not has_search_calendar:
                search_step = ExecutionStep(
                    step=StepType.SEARCH_CALENDAR,
                    params={"search_query": query}
                )
                new_steps.append(search_step)
                has_search_calendar = True
                step.depends_on = [len(new_steps) - 1]

            new_steps.append(step)

        return new_steps

    def _fallback_intent(self, query: str) -> ParsedIntent:
        """Create a fallback intent when classification fails."""
        query_lower = query.lower()

        if any(word in query_lower for word in ["file", "document", "drive", "pdf", "folder"]):
            return ParsedIntent(
                services=[ServiceType.GDRIVE],
                operation="search",
                steps=[ExecutionStep(
                    step=StepType.SEARCH_DRIVE,
                    params={"search_query": query}
                )],
                confidence=0.5,
                entities={}
            )
        elif any(word in query_lower for word in ["meeting", "calendar", "event", "schedule"]):
            return ParsedIntent(
                services=[ServiceType.GCAL],
                operation="search",
                steps=[ExecutionStep(
                    step=StepType.SEARCH_CALENDAR,
                    params={"search_query": query}
                )],
                confidence=0.5,
                entities={}
            )
        else:
            return ParsedIntent(
                services=[ServiceType.GMAIL],
                operation="search",
                steps=[ExecutionStep(
                    step=StepType.SEARCH_GMAIL,
                    params={"search_query": query}
                )],
                confidence=0.5,
                entities={}
            )
