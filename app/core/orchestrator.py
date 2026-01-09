"""Service orchestrator for parallel execution of query plans."""

import asyncio
from typing import Optional, Any

from app.core.planner import QueryPlanner, ExecutionPlan, ExecutionStep
from app.core.intent import IntentClassifier
from app.schemas.intent import ParsedIntent, StepType, StepResult, ServiceType


class Orchestrator:
    """Orchestrates execution of multi-service queries."""

    def __init__(
        self,
        intent_classifier: IntentClassifier,
        planner: QueryPlanner,
        agents: dict,  # ServiceType -> Agent
    ):
        """Initialize the orchestrator.

        Args:
            intent_classifier: The intent classifier instance
            planner: The query planner instance
            agents: Dictionary mapping ServiceType to agent instances
        """
        self.classifier = intent_classifier
        self.planner = planner
        self.agents = agents

    async def execute_query(
        self,
        query: str,
        user_id: str,
        conversation_context: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """Execute a natural language query.

        Main entry point: classifies intent, creates plan, executes, returns results.

        Args:
            query: The natural language query
            user_id: The user's ID
            conversation_context: Optional previous messages for context

        Returns:
            Dictionary with intent, plan, and results
        """
        # 1. Classify intent (LLM extracts draft details for confirmed_send)
        intent = await self.classifier.classify(query, conversation_context)

        # 2. Create execution plan
        plan = self.planner.create_plan(intent)

        # 3. Execute plan (pass original query for search fallback)
        results = await self._execute_plan(plan, user_id, intent, query)

        return {
            "intent": intent.model_dump(),
            "plan": {
                "steps": [s.step.value if hasattr(s.step, 'value') else s.step for s in plan.steps],
                "parallel_groups": plan.parallel_groups,
            },
            "results": results,
        }

    async def _execute_plan(
        self, plan: ExecutionPlan, user_id: str, intent: ParsedIntent, original_query: str = ""
    ) -> list[StepResult]:
        """Execute the plan respecting dependencies, parallelizing where possible.

        Args:
            plan: The execution plan
            user_id: The user's ID
            intent: The parsed intent
            original_query: The original user query for search fallback

        Returns:
            List of step results
        """
        all_results = []
        step_results = {}  # step_id -> result for dependency passing

        for parallel_group in plan.parallel_groups:
            # Get steps in this group
            steps = [s for s in plan.steps if s.step_id in parallel_group]

            # Execute all steps in the group in parallel
            tasks = []
            for step in steps:
                # Enrich params with results from dependencies
                params = self._enrich_params(step, step_results, intent, original_query)
                task = self._execute_step(step, params, user_id)
                tasks.append((step.step_id, step.step, task))

            # Wait for all steps in group to complete
            results = await asyncio.gather(
                *[t[2] for t in tasks], return_exceptions=True
            )

            # Process results
            for (step_id, step_type, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    result = StepResult(
                        step=step_type, success=False, error=str(result)
                    )
                step_results[step_id] = result
                all_results.append(result)

        return all_results

    async def _execute_step(
        self, step: ExecutionStep, params: dict, user_id: str
    ) -> StepResult:
        """Execute a single step using the appropriate agent.

        Args:
            step: The execution step
            params: Parameters for the step
            user_id: The user's ID

        Returns:
            The step result
        """
        agent = self.agents.get(step.service)
        if not agent:
            return StepResult(
                step=step.step,
                success=False,
                error=f"No agent available for service: {step.service}",
            )

        try:
            # Get step value (handle both enum and string)
            step_value = step.step.value if hasattr(step.step, 'value') else step.step

            # Handle search steps
            if step_value.startswith("search_"):
                query = params.get("search_query", "")
                filters = params.get("filters", {})
                results = await agent.search(query, user_id, filters)
                return StepResult(
                    step=step.step, success=True, data={"results": results}
                )

            # Handle recipient resolution issues before executing send/draft
            if step.step in (StepType.DRAFT_EMAIL, StepType.SEND_EMAIL):
                if params.get("clarification_needed"):
                    # Multiple email addresses found - need user clarification
                    return StepResult(
                        step=step.step,
                        success=False,
                        error="clarification_needed",
                        data={
                            "type": "clarification_needed",
                            "recipient_name": params.get("recipient_name", ""),
                            "email_options": params.get("email_options", []),
                            "message": f"I found multiple email addresses for '{params.get('recipient_name', '')}'. Which one should I use?"
                        }
                    )
                if params.get("recipient_not_found"):
                    # No matching emails found
                    return StepResult(
                        step=step.step,
                        success=False,
                        error="recipient_not_found",
                        data={
                            "type": "recipient_not_found",
                            "recipient_name": params.get("recipient_name", ""),
                            "message": f"I couldn't find an email address for '{params.get('recipient_name', '')}' in your recent emails. Could you provide their email address?"
                        }
                    )

            # Execute action steps
            return await agent.execute(step.step, params, user_id)

        except Exception as e:
            return StepResult(step=step.step, success=False, error=str(e))

    def _enrich_params(
        self, step: ExecutionStep, step_results: dict, intent: ParsedIntent, original_query: str = ""
    ) -> dict:
        """Enrich step parameters with results from dependencies.

        Args:
            step: The current step
            step_results: Results from previously executed steps
            intent: The parsed intent
            original_query: Original user query for search fallback

        Returns:
            Enriched parameters dictionary
        """
        params = step.params.copy()
        params["original_query"] = original_query

        # Get step value (handle both enum and string)
        step_value = step.step.value if hasattr(step.step, 'value') else step.step

        # Build search query from entities for search steps
        if step_value.startswith("search_"):
            entities = intent.entities
            search_parts = []

            # Special case: recipient resolution search should ONLY search for the person's name
            if entities.get("recipient_needs_resolution") and entities.get("recipient_name"):
                # For recipient resolution, search only for the person's name
                params["search_query"] = entities["recipient_name"]
                params["filters"] = {}  # No filters, just find emails from/to this person
            else:
                # Normal search: Add relevant entity values to search query
                # Exclude meta fields and message content
                excluded_keys = {
                    "action", "operation", "time_range", "time",
                    "message", "body", "content",  # Don't search for message content
                    "recipient_needs_resolution", "recipient_name",  # Meta fields
                }
                for key, value in entities.items():
                    if key not in excluded_keys:
                        if isinstance(value, str):
                            search_parts.append(value)
                        elif isinstance(value, list):
                            search_parts.extend(str(v) for v in value)

                # If no entities extracted, use the original query for semantic search
                # This ensures we always have a meaningful search query
                if not search_parts:
                    # Extract keywords from original query by removing common words
                    original_query = params.get("original_query", "")
                    if original_query:
                        stop_words = {"find", "search", "show", "get", "me", "my", "the", "a", "an", "about", "from", "in", "for", "emails", "email", "calendar", "events", "files", "documents", "send", "draft", "tell", "telling", "him", "her", "them", "that", "to"}
                        words = [w for w in original_query.lower().split() if w not in stop_words]
                        search_parts = words

                params["search_query"] = " ".join(search_parts) if search_parts else ""

                # Add filters based on entities
                filters = {}
                if entities.get("sender"):
                    filters["sender"] = entities["sender"]
                if entities.get("time_range"):
                    filters["time_range"] = entities["time_range"]
                if entities.get("time"):
                    filters["time"] = entities["time"]
                if entities.get("mime_type"):
                    filters["mime_type"] = entities["mime_type"]
                params["filters"] = filters

        # Get data from dependencies
        for dep_id in step.depends_on:
            if dep_id in step_results:
                dep_result = step_results[dep_id]
                if dep_result.success and dep_result.data:
                    # If dependency was a search, pass first result's data
                    if "results" in dep_result.data and dep_result.data["results"]:
                        first_result = dep_result.data["results"][0]

                        # For recipient resolution, we only need to extract the email address
                        # Don't pass source_data as we're composing a NEW email, not replying
                        is_recipient_resolution = intent.entities.get("recipient_needs_resolution")

                        if not is_recipient_resolution:
                            params["source_data"] = first_result

                            # Auto-fill IDs from search results
                            if "id" in first_result:
                                item_id = first_result["id"]
                                if "email_id" not in params:
                                    params["email_id"] = item_id
                                if "event_id" not in params:
                                    params["event_id"] = item_id
                                if "file_id" not in params:
                                    params["file_id"] = item_id

                            # Extract useful data for drafting emails (actual replies)
                            if step.step == StepType.DRAFT_EMAIL:
                                if "sender" in first_result:
                                    params["to"] = first_result["sender"]
                                if "subject" in first_result:
                                    params["subject"] = f"Re: {first_result['subject']}"

                    # Handle recipient resolution for send/draft email
                    if (step.step in (StepType.DRAFT_EMAIL, StepType.SEND_EMAIL)
                            and intent.entities.get("recipient_needs_resolution")):
                        resolved = self._resolve_recipient(
                            dep_result.data.get("results", []),
                            intent.entities.get("recipient_name", "")
                        )
                        if resolved.get("clarification_needed"):
                            params["clarification_needed"] = True
                            params["email_options"] = resolved["email_options"]
                            params["recipient_name"] = intent.entities.get("recipient_name", "")
                        elif resolved.get("resolved_email"):
                            params["to"] = resolved["resolved_email"]
                            # Use resolved full name for greeting (e.g., "Vinay Khobragade" from email header)
                            if resolved.get("resolved_name"):
                                # Extract first name for friendly greeting
                                full_name = resolved["resolved_name"]
                                first_name = full_name.split()[0] if full_name else ""
                                params["recipient_name"] = first_name
                        else:
                            params["recipient_not_found"] = True
                            params["recipient_name"] = intent.entities.get("recipient_name", "")

        # Map recipient to 'to' for email steps (handles follow-up email address case)
        if step.step in (StepType.DRAFT_EMAIL, StepType.SEND_EMAIL):
            if not params.get("to") and params.get("recipient"):
                params["to"] = params["recipient"]
            # NOTE: Don't map message → body here for draft_email
            # Let the Gmail agent use LLM composition to create a proper email
            # Only map for send_email (which receives the already-composed body from draft)
            if step.step == StepType.SEND_EMAIL:
                if not params.get("body") and params.get("message"):
                    params["body"] = params["message"]
            # Map topic to subject if no subject specified (only for send_email)
            # For draft_email, let LLM generate subject from message intent
            if step.step == StepType.SEND_EMAIL:
                if not params.get("subject") and params.get("topic"):
                    params["subject"] = params["topic"]
                if not params.get("subject"):
                    if params.get("message"):
                        msg = params["message"]
                        params["subject"] = msg[:50] + ("..." if len(msg) > 50 else "")
                    elif params.get("topic"):
                        params["subject"] = params["topic"]

        return params

    def _resolve_recipient(self, search_results: list[dict], recipient_name: str) -> dict:
        """Resolve a recipient name to an email address from search results.

        Args:
            search_results: List of email search results
            recipient_name: The name to resolve

        Returns:
            dict with either:
            - resolved_email: The resolved email address (if exactly one match)
            - resolved_name: The full name extracted from the email header
            - clarification_needed + email_options: If multiple matches
            - Empty dict if no matches found
        """
        if not search_results or not recipient_name:
            return {}

        # Extract unique email addresses and names from senders that match the recipient name
        recipient_lower = recipient_name.lower()
        # Map email -> full name for name extraction
        email_to_name: dict[str, str] = {}

        for result in search_results:
            sender = result.get("sender", "")
            if not sender:
                continue

            # Parse sender - could be "Name <email>" or just "email"
            sender_lower = sender.lower()
            if recipient_lower in sender_lower:
                # Extract email address and name
                if "<" in sender and ">" in sender:
                    name_part = sender[:sender.index("<")].strip()
                    email = sender[sender.index("<")+1:sender.index(">")]
                    # Clean up the name (remove quotes if present)
                    name_part = name_part.strip('"\'')
                    if email.lower() not in email_to_name and name_part:
                        email_to_name[email.lower()] = name_part
                    elif email.lower() not in email_to_name:
                        email_to_name[email.lower()] = ""
                elif "@" in sender:
                    email = sender.strip()
                    if email.lower() not in email_to_name:
                        email_to_name[email.lower()] = ""
                else:
                    continue

        # Also check recipients field if present
        for result in search_results:
            recipients = result.get("recipients", [])
            if isinstance(recipients, list):
                for recip in recipients:
                    if isinstance(recip, str) and recipient_lower in recip.lower():
                        if "<" in recip and ">" in recip:
                            name_part = recip[:recip.index("<")].strip()
                            email = recip[recip.index("<")+1:recip.index(">")]
                            name_part = name_part.strip('"\'')
                            if email.lower() not in email_to_name and name_part:
                                email_to_name[email.lower()] = name_part
                            elif email.lower() not in email_to_name:
                                email_to_name[email.lower()] = ""
                        elif "@" in recip:
                            email = recip.strip()
                            if email.lower() not in email_to_name:
                                email_to_name[email.lower()] = ""
                        else:
                            continue

        matching_emails = list(email_to_name.keys())

        if len(matching_emails) == 1:
            email = matching_emails[0]
            full_name = email_to_name[email]
            # If no name found, use the original recipient_name with proper capitalization
            if not full_name:
                full_name = recipient_name.title()
            return {"resolved_email": email, "resolved_name": full_name}
        elif len(matching_emails) > 1:
            return {
                "clarification_needed": True,
                "email_options": matching_emails[:5]  # Limit to 5 options
            }
        else:
            return {}
