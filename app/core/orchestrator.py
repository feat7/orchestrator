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
        # 1. Classify intent (LLM extracts exact params for each step)
        intent = await self.classifier.classify(query, conversation_context)

        # 2. Create execution plan
        plan = self.planner.create_plan(intent)

        # 3. Execute plan
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
            original_query: The original user query for fallback

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

            # Handle search steps - params come directly from LLM
            if step_value.startswith("search_"):
                search_query = params.get("search_query", "")
                filters = params.get("filters", {})
                results = await agent.search(search_query, user_id, filters)
                return StepResult(
                    step=step.step, success=True, data={"results": results}
                )

            # Handle recipient resolution issues before executing send/draft
            if step.step in (StepType.DRAFT_EMAIL, StepType.SEND_EMAIL):
                if params.get("clarification_needed"):
                    return StepResult(
                        step=step.step,
                        success=False,
                        error="clarification_needed",
                        data={
                            "type": "clarification_needed",
                            "recipient_name": params.get("to_name", ""),
                            "email_options": params.get("email_options", []),
                            "message": f"I found multiple email addresses for '{params.get('to_name', '')}'. Which one should I use?"
                        }
                    )
                if params.get("recipient_not_found"):
                    return StepResult(
                        step=step.step,
                        success=False,
                        error="recipient_not_found",
                        data={
                            "type": "recipient_not_found",
                            "recipient_name": params.get("to_name", ""),
                            "message": f"I couldn't find an email address for '{params.get('to_name', '')}' in your recent emails. Could you provide their email address?"
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

        Now much simpler since LLM provides exact params. We just need to:
        1. Convert date filter params to the filters dict format
        2. Pass data from dependency results to action steps

        Args:
            step: The current step
            step_results: Results from previously executed steps
            intent: The parsed intent
            original_query: Original user query for fallback

        Returns:
            Enriched parameters dictionary
        """
        params = step.params.copy()
        params["original_query"] = original_query

        step_value = step.step.value if hasattr(step.step, 'value') else step.step

        # For search steps, convert LLM params to filters dict
        if step_value.startswith("search_"):
            filters = {}

            # Drive filters
            if params.get("modified_after"):
                filters["modified_after"] = params["modified_after"]
            if params.get("modified_before"):
                filters["modified_before"] = params["modified_before"]
            if params.get("mime_type"):
                filters["mime_type"] = params["mime_type"]
            if params.get("file_name"):
                filters["name"] = params["file_name"]

            # Gmail filters
            if params.get("sender"):
                filters["sender"] = params["sender"]
            if params.get("recipient"):
                filters["recipient"] = params["recipient"]
            if params.get("subject"):
                filters["subject"] = params["subject"]
            if params.get("after_date"):
                filters["after_date"] = params["after_date"]
            if params.get("before_date"):
                filters["before_date"] = params["before_date"]
            if params.get("label"):
                filters["label"] = params["label"]

            # Calendar filters
            if params.get("start_after"):
                filters["start_after"] = params["start_after"]
            if params.get("start_before"):
                filters["start_before"] = params["start_before"]
            if params.get("attendee"):
                filters["attendee"] = params["attendee"]

            params["filters"] = filters

        # Get data from dependencies (for action steps)
        for dep_id in step.depends_on:
            if dep_id in step_results:
                dep_result = step_results[dep_id]
                if dep_result.success and dep_result.data:
                    # If dependency was a search, pass first result's data
                    if "results" in dep_result.data and dep_result.data["results"]:
                        first_result = dep_result.data["results"][0]

                        # Check if this is recipient resolution (to_name set but no to email)
                        is_recipient_resolution = params.get("to_name") and not params.get("to")

                        if not is_recipient_resolution:
                            params["source_data"] = first_result

                            # Auto-fill IDs from search results
                            # Check for falsy values (empty string, None) not just missing keys
                            if "id" in first_result:
                                item_id = first_result["id"]
                                if not params.get("email_id"):
                                    params["email_id"] = item_id
                                if not params.get("event_id"):
                                    params["event_id"] = item_id
                                if not params.get("file_id"):
                                    params["file_id"] = item_id

                        # Handle recipient resolution for draft/send email
                        if is_recipient_resolution and step.step in (StepType.DRAFT_EMAIL, StepType.SEND_EMAIL):
                            resolved = self._resolve_recipient(
                                dep_result.data.get("results", []),
                                params.get("to_name", "")
                            )
                            if resolved.get("clarification_needed"):
                                params["clarification_needed"] = True
                                params["email_options"] = resolved["email_options"]
                            elif resolved.get("resolved_email"):
                                params["to"] = resolved["resolved_email"]
                                if resolved.get("resolved_name"):
                                    full_name = resolved["resolved_name"]
                                    first_name = full_name.split()[0] if full_name else ""
                                    params["recipient_name"] = first_name
                            else:
                                params["recipient_not_found"] = True

                        # Handle attendee resolution for create_event
                        if step.step == StepType.CREATE_EVENT and params.get("attendee_names"):
                            attendee_names = params.get("attendee_names", [])
                            resolved_attendees = params.get("attendees", []) or []

                            for attendee_name in attendee_names:
                                resolved = self._resolve_recipient(
                                    dep_result.data.get("results", []),
                                    attendee_name
                                )
                                if resolved.get("resolved_email"):
                                    if resolved["resolved_email"] not in resolved_attendees:
                                        resolved_attendees.append(resolved["resolved_email"])

                            params["attendees"] = resolved_attendees

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

        recipient_lower = recipient_name.lower()
        email_to_name: dict[str, str] = {}

        for result in search_results:
            sender = result.get("sender", "")
            if not sender:
                continue

            sender_lower = sender.lower()
            if recipient_lower in sender_lower:
                if "<" in sender and ">" in sender:
                    name_part = sender[:sender.index("<")].strip()
                    email = sender[sender.index("<")+1:sender.index(">")]
                    name_part = name_part.strip('"\'')
                    if email.lower() not in email_to_name and name_part:
                        email_to_name[email.lower()] = name_part
                    elif email.lower() not in email_to_name:
                        email_to_name[email.lower()] = ""
                elif "@" in sender:
                    email = sender.strip()
                    if email.lower() not in email_to_name:
                        email_to_name[email.lower()] = ""

        # Also check recipients field
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

        matching_emails = list(email_to_name.keys())

        if len(matching_emails) == 1:
            email = matching_emails[0]
            full_name = email_to_name[email]
            if not full_name:
                full_name = recipient_name.title()
            return {"resolved_email": email, "resolved_name": full_name}
        elif len(matching_emails) > 1:
            return {
                "clarification_needed": True,
                "email_options": matching_emails[:5]
            }
        else:
            return {}
