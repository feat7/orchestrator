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
        # 1. Classify intent
        intent = await self.classifier.classify(query, conversation_context)

        # 2. Create execution plan
        plan = self.planner.create_plan(intent)

        # 3. Execute plan
        results = await self._execute_plan(plan, user_id, intent)

        return {
            "intent": intent.model_dump(),
            "plan": {
                "steps": [s.step.value if hasattr(s.step, 'value') else s.step for s in plan.steps],
                "parallel_groups": plan.parallel_groups,
            },
            "results": results,
        }

    async def _execute_plan(
        self, plan: ExecutionPlan, user_id: str, intent: ParsedIntent
    ) -> list[StepResult]:
        """Execute the plan respecting dependencies, parallelizing where possible.

        Args:
            plan: The execution plan
            user_id: The user's ID
            intent: The parsed intent

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
                params = self._enrich_params(step, step_results, intent)
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

            # Execute action steps
            return await agent.execute(step.step, params, user_id)

        except Exception as e:
            return StepResult(step=step.step, success=False, error=str(e))

    def _enrich_params(
        self, step: ExecutionStep, step_results: dict, intent: ParsedIntent
    ) -> dict:
        """Enrich step parameters with results from dependencies.

        Args:
            step: The current step
            step_results: Results from previously executed steps
            intent: The parsed intent

        Returns:
            Enriched parameters dictionary
        """
        params = step.params.copy()

        # Get step value (handle both enum and string)
        step_value = step.step.value if hasattr(step.step, 'value') else step.step

        # Build search query from entities for search steps
        if step_value.startswith("search_"):
            entities = intent.entities
            search_parts = []

            # Add relevant entity values to search query
            for key, value in entities.items():
                if key not in ("action", "operation"):
                    if isinstance(value, str):
                        search_parts.append(value)
                    elif isinstance(value, list):
                        search_parts.extend(str(v) for v in value)

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

                        # Extract useful data for drafting emails
                        if step.step == StepType.DRAFT_EMAIL:
                            if "sender" in first_result:
                                params["to"] = first_result["sender"]
                            if "subject" in first_result:
                                params["subject"] = f"Re: {first_result['subject']}"

        return params
