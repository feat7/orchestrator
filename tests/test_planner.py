"""Tests for query planner."""

import pytest
from app.core.planner import QueryPlanner, ExecutionStep, ExecutionPlan
from app.schemas.intent import ParsedIntent, ServiceType, StepType, ExecutionStep as IntentExecutionStep


def test_planner_initialization():
    """Test QueryPlanner initialization."""
    planner = QueryPlanner()
    assert planner is not None


def test_create_plan_single_step(sample_intent):
    """Test planning with single step."""
    planner = QueryPlanner()
    plan = planner.create_plan(sample_intent)

    assert isinstance(plan, ExecutionPlan)
    assert len(plan.steps) == 1
    assert plan.steps[0].step == StepType.SEARCH_CALENDAR


def test_create_plan_multi_step(sample_multi_service_intent):
    """Test planning with multiple steps."""
    planner = QueryPlanner()
    plan = planner.create_plan(sample_multi_service_intent)

    assert len(plan.steps) == 3
    assert plan.steps[0].step == StepType.SEARCH_GMAIL
    assert plan.steps[1].step == StepType.SEARCH_CALENDAR
    assert plan.steps[2].step == StepType.DRAFT_EMAIL


def test_parallel_groups_search_steps():
    """Test that search steps can run in parallel."""
    planner = QueryPlanner()

    intent = ParsedIntent(
        services=[ServiceType.GMAIL, ServiceType.GCAL, ServiceType.GDRIVE],
        operation="search",
        entities={},
        steps=[
            IntentExecutionStep(step=StepType.SEARCH_GMAIL, params={"search_query": "test"}),
            IntentExecutionStep(step=StepType.SEARCH_CALENDAR, params={"search_query": "test"}),
            IntentExecutionStep(step=StepType.SEARCH_DRIVE, params={"search_query": "test"}),
        ],
    )

    plan = planner.create_plan(intent)

    # All search steps should be in the first parallel group
    assert len(plan.parallel_groups) == 1
    assert len(plan.parallel_groups[0]) == 3


def test_dependencies_action_after_search():
    """Test that action steps depend on search steps."""
    planner = QueryPlanner()

    intent = ParsedIntent(
        services=[ServiceType.GMAIL],
        operation="update",
        entities={},
        steps=[
            IntentExecutionStep(step=StepType.SEARCH_GMAIL, params={"search_query": "test"}),
            IntentExecutionStep(step=StepType.DRAFT_EMAIL, params={"message": "test"}, depends_on=[0]),
        ],
    )

    plan = planner.create_plan(intent)

    # Draft email should depend on search
    draft_step = plan.steps[1]
    assert len(draft_step.depends_on) > 0
    assert "step_0" in draft_step.depends_on


def test_parallel_groups_with_dependencies():
    """Test parallel groups respect dependencies."""
    planner = QueryPlanner()

    intent = ParsedIntent(
        services=[ServiceType.GMAIL, ServiceType.GCAL],
        operation="update",
        entities={},
        steps=[
            IntentExecutionStep(step=StepType.SEARCH_GMAIL, params={"search_query": "test"}),
            IntentExecutionStep(step=StepType.SEARCH_CALENDAR, params={"search_query": "test"}),
            IntentExecutionStep(step=StepType.DRAFT_EMAIL, params={"message": "test"}, depends_on=[0, 1]),
        ],
    )

    plan = planner.create_plan(intent)

    # Should have 2 groups: searches first, then draft
    assert len(plan.parallel_groups) == 2

    # First group should have search steps
    first_group = plan.parallel_groups[0]
    assert "step_0" in first_group
    assert "step_1" in first_group

    # Second group should have draft
    second_group = plan.parallel_groups[1]
    assert "step_2" in second_group


def test_step_service_mapping():
    """Test step to service mapping."""
    planner = QueryPlanner()

    assert planner.STEP_SERVICE_MAP[StepType.SEARCH_GMAIL] == ServiceType.GMAIL
    assert planner.STEP_SERVICE_MAP[StepType.SEARCH_CALENDAR] == ServiceType.GCAL
    assert planner.STEP_SERVICE_MAP[StepType.SEARCH_DRIVE] == ServiceType.GDRIVE


def test_get_step_info():
    """Test step info retrieval."""
    planner = QueryPlanner()

    info = planner.get_step_info(StepType.SEARCH_GMAIL)

    assert info["step"] == "search_gmail"
    assert info["service"] == "gmail"
    assert info["is_search"] is True
    assert info["is_producer"] is True


def test_execution_step_dataclass():
    """Test ExecutionStep dataclass."""
    step = ExecutionStep(
        step=StepType.SEARCH_GMAIL,
        service=ServiceType.GMAIL,
        params={"query": "test"},
        depends_on=[],
        step_id="step_0",
    )

    assert step.step == StepType.SEARCH_GMAIL
    assert step.service == ServiceType.GMAIL
    assert step.params == {"query": "test"}
    assert step.step_id == "step_0"
