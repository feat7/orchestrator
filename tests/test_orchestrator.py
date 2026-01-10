"""Tests for the Orchestrator component."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.core.orchestrator import Orchestrator
from app.core.planner import QueryPlanner, ExecutionPlan, ExecutionStep
from app.core.intent import IntentClassifier
from app.schemas.intent import ParsedIntent, ServiceType, StepType, StepResult, ExecutionStep as IntentExecutionStep


class MockIntentClassifier:
    """Mock intent classifier for testing."""

    def __init__(self, intent: ParsedIntent):
        self.intent = intent

    async def classify(self, query: str, context=None) -> ParsedIntent:
        return self.intent


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, search_results=None, execute_result=None):
        self.search_results = search_results or []
        self.execute_result = execute_result or StepResult(
            step=StepType.SEARCH_GMAIL, success=True, data={"results": []}
        )
        self.search_calls = []
        self.execute_calls = []

    async def search(self, query: str, user_id: str, filters=None):
        self.search_calls.append({"query": query, "user_id": user_id, "filters": filters})
        return self.search_results

    async def execute(self, step: StepType, params: dict, user_id: str):
        self.execute_calls.append({"step": step, "params": params, "user_id": user_id})
        return self.execute_result


@pytest.fixture
def simple_intent():
    """Create a simple single-service intent."""
    return ParsedIntent(
        services=[ServiceType.GMAIL],
        operation="search",
        entities={"topic": "meeting notes"},
        steps=[IntentExecutionStep(step=StepType.SEARCH_GMAIL, params={"search_query": "meeting notes"})],
        confidence=0.95,
    )


@pytest.fixture
def multi_service_intent():
    """Create a multi-service intent."""
    return ParsedIntent(
        services=[ServiceType.GMAIL, ServiceType.GCAL],
        operation="search",
        entities={"topic": "project meeting"},
        steps=[
            IntentExecutionStep(step=StepType.SEARCH_GMAIL, params={"search_query": "project meeting"}),
            IntentExecutionStep(step=StepType.SEARCH_CALENDAR, params={"search_query": "project meeting"}),
        ],
        confidence=0.9,
    )


@pytest.fixture
def action_intent():
    """Create an intent with action step."""
    return ParsedIntent(
        services=[ServiceType.GMAIL],
        operation="draft",
        entities={"action": "draft", "to": "test@example.com"},
        steps=[
            IntentExecutionStep(step=StepType.SEARCH_GMAIL, params={"search_query": "email"}),
            IntentExecutionStep(step=StepType.DRAFT_EMAIL, params={"message": "reply"}, depends_on=[0]),
        ],
        confidence=0.85,
    )


def test_orchestrator_initialization(simple_intent):
    """Test orchestrator can be initialized."""
    classifier = MockIntentClassifier(simple_intent)
    planner = QueryPlanner()
    agents = {ServiceType.GMAIL: MockAgent()}

    orchestrator = Orchestrator(classifier, planner, agents)

    assert orchestrator.classifier == classifier
    assert orchestrator.planner == planner
    assert orchestrator.agents == agents


@pytest.mark.asyncio
async def test_execute_query_single_service(simple_intent):
    """Test executing a simple single-service query."""
    search_results = [
        {"id": "msg1", "subject": "Meeting Notes", "sender": "test@example.com"}
    ]
    mock_agent = MockAgent(search_results=search_results)

    classifier = MockIntentClassifier(simple_intent)
    planner = QueryPlanner()
    agents = {ServiceType.GMAIL: mock_agent}

    orchestrator = Orchestrator(classifier, planner, agents)
    user_id = str(uuid4())

    result = await orchestrator.execute_query(
        query="Find my meeting notes",
        user_id=user_id,
    )

    assert "intent" in result
    assert "plan" in result
    assert "results" in result
    assert result["intent"]["operation"] == "search"
    assert len(mock_agent.search_calls) == 1


@pytest.mark.asyncio
async def test_execute_query_multi_service(multi_service_intent):
    """Test executing a multi-service query."""
    gmail_results = [{"id": "msg1", "subject": "Project Update"}]
    gcal_results = [{"id": "evt1", "title": "Project Meeting"}]

    gmail_agent = MockAgent(search_results=gmail_results)
    gcal_agent = MockAgent(search_results=gcal_results)

    classifier = MockIntentClassifier(multi_service_intent)
    planner = QueryPlanner()
    agents = {
        ServiceType.GMAIL: gmail_agent,
        ServiceType.GCAL: gcal_agent,
    }

    orchestrator = Orchestrator(classifier, planner, agents)
    user_id = str(uuid4())

    result = await orchestrator.execute_query(
        query="Find project meeting info",
        user_id=user_id,
    )

    assert len(result["results"]) == 2
    # Both agents should have been called
    assert len(gmail_agent.search_calls) == 1
    assert len(gcal_agent.search_calls) == 1


@pytest.mark.asyncio
async def test_execute_query_with_dependencies(action_intent):
    """Test executing a query with step dependencies."""
    search_results = [
        {"id": "msg1", "subject": "Original Email", "sender": "sender@example.com"}
    ]
    draft_result = StepResult(
        step=StepType.DRAFT_EMAIL,
        success=True,
        data={"draft_id": "draft123"},
    )

    mock_agent = MockAgent(search_results=search_results, execute_result=draft_result)

    classifier = MockIntentClassifier(action_intent)
    planner = QueryPlanner()
    agents = {ServiceType.GMAIL: mock_agent}

    orchestrator = Orchestrator(classifier, planner, agents)
    user_id = str(uuid4())

    result = await orchestrator.execute_query(
        query="Reply to the latest email",
        user_id=user_id,
    )

    # Should have search and draft results
    assert len(result["results"]) == 2
    # Both search and execute should have been called
    assert len(mock_agent.search_calls) == 1
    assert len(mock_agent.execute_calls) == 1


@pytest.mark.asyncio
async def test_execute_query_with_conversation_context(simple_intent):
    """Test executing a query with conversation context."""
    mock_agent = MockAgent(search_results=[])
    classifier = MockIntentClassifier(simple_intent)
    planner = QueryPlanner()
    agents = {ServiceType.GMAIL: mock_agent}

    orchestrator = Orchestrator(classifier, planner, agents)
    user_id = str(uuid4())

    context = [
        {"query": "Find emails from John", "intent": {"operation": "search"}}
    ]

    result = await orchestrator.execute_query(
        query="Show me more details",
        user_id=user_id,
        conversation_context=context,
    )

    assert result is not None
    assert "results" in result


@pytest.mark.asyncio
async def test_execute_query_missing_agent(simple_intent):
    """Test executing a query when agent is missing."""
    classifier = MockIntentClassifier(simple_intent)
    planner = QueryPlanner()
    agents = {}  # No agents

    orchestrator = Orchestrator(classifier, planner, agents)
    user_id = str(uuid4())

    result = await orchestrator.execute_query(
        query="Find my emails",
        user_id=user_id,
    )

    # Should have a failed result
    assert len(result["results"]) == 1
    assert result["results"][0].success is False
    assert "No agent available" in result["results"][0].error


@pytest.mark.asyncio
async def test_execute_query_agent_error(simple_intent):
    """Test handling when agent raises an exception."""
    class ErrorAgent:
        async def search(self, query, user_id, filters=None):
            raise RuntimeError("API Error")

        async def execute(self, step, params, user_id):
            raise RuntimeError("API Error")

    classifier = MockIntentClassifier(simple_intent)
    planner = QueryPlanner()
    agents = {ServiceType.GMAIL: ErrorAgent()}

    orchestrator = Orchestrator(classifier, planner, agents)
    user_id = str(uuid4())

    result = await orchestrator.execute_query(
        query="Find my emails",
        user_id=user_id,
    )

    # Should have a failed result with error message
    assert len(result["results"]) == 1
    assert result["results"][0].success is False
    assert "API Error" in result["results"][0].error


def test_enrich_params_for_search(simple_intent):
    """Test parameter enrichment for search steps."""
    classifier = MockIntentClassifier(simple_intent)
    planner = QueryPlanner()
    agents = {}

    orchestrator = Orchestrator(classifier, planner, agents)

    # Now params come from the step itself (LLM output)
    step = ExecutionStep(
        step_id=0,
        step=StepType.SEARCH_GMAIL,
        service=ServiceType.GMAIL,
        params={"search_query": "meeting notes"},
        depends_on=[],
    )

    params = orchestrator._enrich_params(step, {}, simple_intent)

    # search_query comes from step params, enrichment adds filters
    assert "search_query" in params
    assert "meeting notes" in params["search_query"]
    assert "filters" in params


def test_enrich_params_with_dependency_results():
    """Test parameter enrichment using results from dependencies."""
    intent = ParsedIntent(
        services=[ServiceType.GMAIL],
        operation="draft",
        entities={"action": "reply"},
        steps=[
            IntentExecutionStep(step=StepType.SEARCH_GMAIL, params={"search_query": "hello"}),
            IntentExecutionStep(step=StepType.DRAFT_EMAIL, params={"message": "reply"}, depends_on=[0]),
        ],
        confidence=0.9,
    )

    classifier = MockIntentClassifier(intent)
    planner = QueryPlanner()
    agents = {}

    orchestrator = Orchestrator(classifier, planner, agents)

    # Step with dependency on search results
    step = ExecutionStep(
        step_id=1,
        step=StepType.DRAFT_EMAIL,
        service=ServiceType.GMAIL,
        params={"message": "reply"},
        depends_on=[0],
    )

    # Simulate search results from dependency
    search_result = StepResult(
        step=StepType.SEARCH_GMAIL,
        success=True,
        data={
            "results": [
                {"id": "msg1", "subject": "Hello", "sender": "test@example.com"}
            ]
        },
    )

    params = orchestrator._enrich_params(step, {0: search_result}, intent)

    # Should auto-fill email_id from search results
    assert params.get("email_id") == "msg1"
    # Source data should be passed
    assert params.get("source_data") is not None
