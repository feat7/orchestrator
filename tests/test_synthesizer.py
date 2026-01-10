"""Tests for the ResponseSynthesizer component."""

import pytest
from unittest.mock import AsyncMock

from app.core.synthesizer import ResponseSynthesizer
from app.core.llm import LLMProvider
from app.schemas.intent import ParsedIntent, ServiceType, StepType, StepResult, ExecutionStep


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    @property
    def name(self) -> str:
        return "mock"

    def __init__(self, response: str = "Mock response"):
        self.response = response
        self.complete_calls = []

    async def complete(self, prompt: str, system: str = None, json_mode: bool = False) -> str:
        self.complete_calls.append({"prompt": prompt, "system": system, "json_mode": json_mode})
        return self.response

    async def embed(self, text: str) -> list[float]:
        return [0.0] * 1536


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    return MockLLMProvider("Here is a summary of your emails.")


@pytest.fixture
def simple_intent():
    """Create a simple intent."""
    return ParsedIntent(
        services=[ServiceType.GMAIL],
        operation="search",
        entities={"topic": "meeting"},
        steps=[ExecutionStep(step=StepType.SEARCH_GMAIL, params={"search_query": "meeting"})],
        confidence=0.95,
    )


@pytest.fixture
def multi_service_intent():
    """Create a multi-service intent."""
    return ParsedIntent(
        services=[ServiceType.GMAIL, ServiceType.GCAL],
        operation="search",
        entities={"topic": "project"},
        steps=[
            ExecutionStep(step=StepType.SEARCH_GMAIL, params={"search_query": "project"}),
            ExecutionStep(step=StepType.SEARCH_CALENDAR, params={"search_query": "project"})
        ],
        confidence=0.9,
    )


def test_synthesizer_initialization(mock_llm):
    """Test synthesizer can be initialized."""
    synthesizer = ResponseSynthesizer(mock_llm)
    assert synthesizer.llm == mock_llm


@pytest.mark.asyncio
async def test_synthesize_success_response(mock_llm, simple_intent):
    """Test synthesizing a successful response."""
    synthesizer = ResponseSynthesizer(mock_llm)

    results = [
        StepResult(
            step=StepType.SEARCH_GMAIL,
            success=True,
            data={"results": [{"id": "1", "subject": "Meeting Notes"}]},
        )
    ]

    response = await synthesizer.synthesize(
        query="Find my meeting notes",
        intent=simple_intent,
        results=results,
    )

    assert response == "Here is a summary of your emails."
    assert len(mock_llm.complete_calls) == 1
    # Check that the prompt includes query and results
    call = mock_llm.complete_calls[0]
    assert "meeting notes" in call["prompt"].lower()
    assert call["system"] is not None


@pytest.mark.asyncio
async def test_synthesize_with_failed_step(mock_llm, simple_intent):
    """Test synthesizing response with a failed step."""
    mock_llm.response = "I couldn't find the emails due to an error."
    synthesizer = ResponseSynthesizer(mock_llm)

    results = [
        StepResult(
            step=StepType.SEARCH_GMAIL,
            success=False,
            error="API rate limit exceeded",
        )
    ]

    response = await synthesizer.synthesize(
        query="Find my emails",
        intent=simple_intent,
        results=results,
    )

    assert response == "I couldn't find the emails due to an error."
    # Check the prompt includes the error
    call = mock_llm.complete_calls[0]
    assert "FAILED" in call["prompt"]


@pytest.mark.asyncio
async def test_synthesize_multi_service_response(mock_llm, multi_service_intent):
    """Test synthesizing response from multiple services."""
    mock_llm.response = "Found 2 emails and 3 events related to your project."
    synthesizer = ResponseSynthesizer(mock_llm)

    results = [
        StepResult(
            step=StepType.SEARCH_GMAIL,
            success=True,
            data={"results": [{"id": "1", "subject": "Project Update"}]},
        ),
        StepResult(
            step=StepType.SEARCH_CALENDAR,
            success=True,
            data={"results": [{"id": "e1", "title": "Project Meeting"}]},
        ),
    ]

    response = await synthesizer.synthesize(
        query="Find project info",
        intent=multi_service_intent,
        results=results,
    )

    assert "Found 2 emails and 3 events" in response


@pytest.mark.asyncio
async def test_synthesize_with_empty_results(mock_llm, simple_intent):
    """Test synthesizing response when no results found."""
    mock_llm.response = "I couldn't find any matching emails."
    synthesizer = ResponseSynthesizer(mock_llm)

    results = [
        StepResult(
            step=StepType.SEARCH_GMAIL,
            success=True,
            data={"results": []},
        )
    ]

    response = await synthesizer.synthesize(
        query="Find emails from nobody",
        intent=simple_intent,
        results=results,
    )

    assert response == "I couldn't find any matching emails."
    # Check prompt mentions no results
    call = mock_llm.complete_calls[0]
    assert "No results found" in call["prompt"]


@pytest.mark.asyncio
async def test_synthesize_with_draft_created(mock_llm):
    """Test synthesizing response when draft is created."""
    synthesizer = ResponseSynthesizer(mock_llm)

    intent = ParsedIntent(
        services=[ServiceType.GMAIL],
        operation="draft",
        entities={"to": "test@example.com"},
        steps=[ExecutionStep(step=StepType.DRAFT_EMAIL, params={"to": "test@example.com", "message": "Hello"})],
        confidence=0.9,
    )

    results = [
        StepResult(
            step=StepType.DRAFT_EMAIL,
            success=True,
            data={"draft_id": "draft123", "to": "test@example.com", "subject": "Hello", "body": "Test body"},
        )
    ]

    response = await synthesizer.synthesize(
        query="Draft an email to test@example.com",
        intent=intent,
        results=results,
    )

    # Synthesizer now returns formatted draft response directly
    assert "I've drafted an email for you" in response
    assert "draft123" in response
    assert "test@example.com" in response
    assert "Would you like me to send it?" in response


@pytest.mark.asyncio
async def test_synthesize_error_response(mock_llm):
    """Test synthesizing an error response."""
    mock_llm.response = "I'm sorry, something went wrong. Please try again."
    synthesizer = ResponseSynthesizer(mock_llm)

    response = await synthesizer.synthesize_error(
        query="Find my emails",
        error="Database connection failed",
    )

    assert "sorry" in response.lower() or "went wrong" in response.lower()
    call = mock_llm.complete_calls[0]
    assert "Database connection failed" in call["prompt"]


def test_format_results_email_items(mock_llm):
    """Test formatting email items in results."""
    synthesizer = ResponseSynthesizer(mock_llm)

    results = [
        StepResult(
            step=StepType.SEARCH_GMAIL,
            success=True,
            data={
                "results": [
                    {
                        "id": "1",
                        "subject": "Weekly Update",
                        "sender": "boss@company.com",
                        "received_at": "2024-01-15",
                    }
                ]
            },
        )
    ]

    formatted = synthesizer._format_results(results)

    assert "SUCCESS" in formatted
    assert "Weekly Update" in formatted
    assert "boss@company.com" in formatted


def test_format_results_event_items(mock_llm):
    """Test formatting calendar event items."""
    synthesizer = ResponseSynthesizer(mock_llm)

    results = [
        StepResult(
            step=StepType.SEARCH_CALENDAR,
            success=True,
            data={
                "results": [
                    {
                        "id": "e1",
                        "title": "Team Standup",
                        "start_time": "2024-01-15 09:00",
                        "location": "Conference Room A",
                    }
                ]
            },
        )
    ]

    formatted = synthesizer._format_results(results)

    assert "SUCCESS" in formatted
    assert "Team Standup" in formatted
    assert "Conference Room A" in formatted


def test_format_results_file_items(mock_llm):
    """Test formatting Drive file items."""
    synthesizer = ResponseSynthesizer(mock_llm)

    results = [
        StepResult(
            step=StepType.SEARCH_DRIVE,
            success=True,
            data={
                "results": [
                    {
                        "id": "f1",
                        "name": "Project Proposal.pdf",
                        "mime_type": "application/pdf",
                        "web_link": "https://drive.google.com/file/d/f1/view",
                    }
                ]
            },
        )
    ]

    formatted = synthesizer._format_results(results)

    assert "SUCCESS" in formatted
    assert "Project Proposal.pdf" in formatted
    assert "application/pdf" in formatted
    # Should include web link for clicking
    assert "https://drive.google.com/file/d/f1/view" in formatted


def test_format_results_action_results(mock_llm):
    """Test formatting action results (draft, send, delete)."""
    synthesizer = ResponseSynthesizer(mock_llm)

    results = [
        StepResult(
            step=StepType.DRAFT_EMAIL,
            success=True,
            data={"draft_id": "draft123"},
        ),
        StepResult(
            step=StepType.SEND_EMAIL,
            success=True,
            data={"message_id": "msg456"},
        ),
        StepResult(
            step=StepType.DELETE_EVENT,
            success=True,
            data={"deleted": True},
        ),
    ]

    formatted = synthesizer._format_results(results)

    assert "draft123" in formatted
    assert "msg456" in formatted
    assert "deleted successfully" in formatted


def test_format_results_file_content(mock_llm):
    """Test formatting get_file results with content_preview."""
    synthesizer = ResponseSynthesizer(mock_llm)

    # This simulates get_file returning file data with content
    results = [
        StepResult(
            step=StepType.GET_FILE,
            success=True,
            data={
                "id": "file123",
                "name": "Orchestator",
                "mime_type": "application/vnd.google-apps.document",
                "content_preview": "This is a simple drive file to test orchestrator.",
                "modified_at": "2026-01-09T10:00:00",
                "web_link": "https://docs.google.com/document/d/file123",
            },
        )
    ]

    formatted = synthesizer._format_results(results)

    # Should show file name
    assert "Orchestator" in formatted
    # Should show CONTENT section
    assert "CONTENT" in formatted
    # Should show the actual file content
    assert "simple drive file to test orchestrator" in formatted
    # Should show the web link for opening in browser
    assert "https://docs.google.com/document/d/file123" in formatted


def test_format_item_generic(mock_llm):
    """Test formatting generic items."""
    synthesizer = ResponseSynthesizer(mock_llm)

    item = {"some_key": "some_value", "another": 123}
    formatted = synthesizer._format_item(item)

    # Should produce a string representation
    assert "some_key" in formatted or str(item) in formatted
