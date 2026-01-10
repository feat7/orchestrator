"""Query planner for creating execution DAGs from intents."""

from dataclasses import dataclass, field
from typing import Optional

from app.schemas.intent import ParsedIntent, StepType, ServiceType, ExecutionStep as IntentStep


@dataclass
class ExecutionStep:
    """A single step in the execution plan."""

    step: StepType
    service: ServiceType
    params: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)  # Step IDs this depends on
    step_id: str = ""


@dataclass
class ExecutionPlan:
    """Complete execution plan with steps and parallel groups."""

    steps: list[ExecutionStep]
    parallel_groups: list[list[str]]  # Groups of step_ids that can run in parallel


class QueryPlanner:
    """Converts parsed intents into executable plans with dependencies."""

    # Map steps to their service
    STEP_SERVICE_MAP = {
        StepType.SEARCH_GMAIL: ServiceType.GMAIL,
        StepType.GET_EMAIL: ServiceType.GMAIL,
        StepType.DRAFT_EMAIL: ServiceType.GMAIL,
        StepType.SEND_EMAIL: ServiceType.GMAIL,
        StepType.SEARCH_CALENDAR: ServiceType.GCAL,
        StepType.GET_EVENT: ServiceType.GCAL,
        StepType.CREATE_EVENT: ServiceType.GCAL,
        StepType.UPDATE_EVENT: ServiceType.GCAL,
        StepType.DELETE_EVENT: ServiceType.GCAL,
        StepType.SEARCH_DRIVE: ServiceType.GDRIVE,
        StepType.GET_FILE: ServiceType.GDRIVE,
        StepType.SHARE_FILE: ServiceType.GDRIVE,
    }

    # Steps that produce data other steps might need
    PRODUCER_STEPS = {
        StepType.SEARCH_GMAIL,
        StepType.SEARCH_CALENDAR,
        StepType.SEARCH_DRIVE,
        StepType.GET_EMAIL,
        StepType.GET_EVENT,
        StepType.GET_FILE,
    }

    # Steps that consume data from previous steps
    CONSUMER_STEPS = {
        StepType.GET_EMAIL,
        StepType.DRAFT_EMAIL,
        StepType.SEND_EMAIL,
        StepType.GET_EVENT,
        StepType.UPDATE_EVENT,
        StepType.DELETE_EVENT,
        StepType.GET_FILE,
        StepType.SHARE_FILE,
    }

    # Search steps (can run in parallel)
    SEARCH_STEPS = {
        StepType.SEARCH_GMAIL,
        StepType.SEARCH_CALENDAR,
        StepType.SEARCH_DRIVE,
    }

    def create_plan(self, intent: ParsedIntent) -> ExecutionPlan:
        """Convert a parsed intent into an executable plan.

        The intent now contains steps with their own typed params.

        Args:
            intent: The parsed intent from classification

        Returns:
            An ExecutionPlan with steps and parallel groups
        """
        execution_steps = []

        for idx, intent_step in enumerate(intent.steps):
            step_id = f"step_{idx}"

            # Get step type (handle both old and new format)
            if isinstance(intent_step, IntentStep):
                step_type = StepType(intent_step.step) if isinstance(intent_step.step, str) else intent_step.step
                params = intent_step.params.copy()
                explicit_deps = intent_step.depends_on
            else:
                # Old format: step is just a StepType
                step_type = intent_step
                params = intent.entities.copy()
                explicit_deps = None

            service = self.STEP_SERVICE_MAP.get(step_type, ServiceType.GMAIL)

            # Determine dependencies
            if explicit_deps is not None:
                # Use explicit dependencies from LLM
                depends_on = [f"step_{dep_idx}" for dep_idx in explicit_deps]
            else:
                # Compute dependencies automatically
                prev_steps = [
                    s.step if isinstance(s, IntentStep) else s
                    for s in intent.steps[:idx]
                ]
                depends_on = self._compute_dependencies(step_type, prev_steps, idx)

            exec_step = ExecutionStep(
                step=step_type,
                service=service,
                params=params,
                depends_on=depends_on,
                step_id=step_id,
            )
            execution_steps.append(exec_step)

        # Compute parallel groups
        parallel_groups = self._compute_parallel_groups(execution_steps)

        return ExecutionPlan(steps=execution_steps, parallel_groups=parallel_groups)

    def _compute_dependencies(
        self, step: StepType, previous_steps: list[StepType], current_idx: int
    ) -> list[str]:
        """Compute which previous steps this step depends on.

        Args:
            step: The current step
            previous_steps: List of steps before this one
            current_idx: Index of the current step

        Returns:
            List of step IDs that this step depends on
        """
        depends_on = []

        # Search steps have no dependencies
        if step in self.SEARCH_STEPS:
            return depends_on

        # Consumer steps depend on their producer
        if step in self.CONSUMER_STEPS:
            service = self.STEP_SERVICE_MAP.get(step)

            # Look for the most recent relevant producer
            for prev_idx in range(current_idx - 1, -1, -1):
                prev_step = previous_steps[prev_idx]
                prev_step_type = prev_step if isinstance(prev_step, StepType) else StepType(prev_step)

                # Check if this is a producer step
                if prev_step_type in self.PRODUCER_STEPS:
                    prev_service = self.STEP_SERVICE_MAP.get(prev_step_type)

                    # Depend on search from same service
                    if prev_service == service:
                        depends_on.append(f"step_{prev_idx}")
                        break

                    # Action steps can depend on any search
                    if prev_step_type in self.SEARCH_STEPS and step not in self.SEARCH_STEPS:
                        depends_on.append(f"step_{prev_idx}")
                        # Don't break - might need multiple search results

        return depends_on

    def _compute_parallel_groups(
        self, steps: list[ExecutionStep]
    ) -> list[list[str]]:
        """Compute groups of steps that can run in parallel.

        Args:
            steps: List of execution steps

        Returns:
            List of parallel groups (each group is a list of step IDs)
        """
        groups = []
        completed = set()
        remaining = {s.step_id for s in steps}

        while remaining:
            # Find all steps whose dependencies are satisfied
            ready = []
            for step in steps:
                if step.step_id in remaining:
                    if all(dep in completed for dep in step.depends_on):
                        ready.append(step.step_id)

            if not ready:
                # Circular dependency or error - break to avoid infinite loop
                # Add remaining steps as individual groups
                for step_id in remaining:
                    groups.append([step_id])
                break

            groups.append(ready)
            completed.update(ready)
            remaining -= set(ready)

        return groups

    def get_step_info(self, step: StepType) -> dict:
        """Get metadata about a step type.

        Args:
            step: The step type

        Returns:
            Dictionary with step metadata
        """
        return {
            "step": step.value,
            "service": self.STEP_SERVICE_MAP.get(step, ServiceType.GMAIL).value,
            "is_search": step in self.SEARCH_STEPS,
            "is_producer": step in self.PRODUCER_STEPS,
            "is_consumer": step in self.CONSUMER_STEPS,
        }
