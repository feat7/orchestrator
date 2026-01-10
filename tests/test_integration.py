"""Integration tests for the full query flow."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.core.orchestrator import Orchestrator
from app.core.planner import QueryPlanner
from app.core.synthesizer import ResponseSynthesizer
from app.core.intent import IntentClassifier
from app.core.llm import LLMProvider
from app.agents.gmail import GmailAgent
from app.agents.gcal import GcalAgent
from app.agents.gdrive import GdriveAgent
from app.schemas.intent import ParsedIntent, ServiceType, StepType, StepResult


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for integration tests."""

    @property
    def name(self) -> str:
        return "mock"

    def __init__(self, intent_response: str = None, synthesis_response: str = None):
        self.intent_response = intent_response or '''{
            "services": ["gmail"],
            "operation": "search",
            "entities": {"topic": "meeting"},
            "steps": ["search_gmail"],
            "confidence": 0.95
        }'''
        self.synthesis_response = synthesis_response or "I found 2 emails about meetings."
        self.complete_calls = []

    async def complete(self, prompt: str, system: str = None, json_mode: bool = False) -> str:
        self.complete_calls.append({"prompt": prompt, "system": system, "json_mode": json_mode})
        if json_mode or "classify" in prompt.lower() or "intent" in system.lower() if system else False:
            return self.intent_response
        return self.synthesis_response

    async def embed(self, text: str) -> list[float]:
        return [0.1] * 1536


class MockEmbeddingService:
    """Mock embedding service for integration tests."""

    async def embed(self, text: str) -> list[float]:
        return [0.1] * 1536


class MockGmailService:
    """Mock Gmail service for integration tests."""

    async def search_emails(self, user_id, embedding, filters=None, limit=10, similarity_threshold=0.25):
        return [
            {"id": "msg1", "subject": "Team Meeting Notes", "sender": "colleague@example.com"},
            {"id": "msg2", "subject": "Project Meeting Summary", "sender": "boss@example.com"},
        ]

    async def search_emails_bm25(self, user_id, query, filters=None, limit=20):
        """BM25/full-text search for hybrid search."""
        return [
            {"id": "msg1", "subject": "Team Meeting Notes", "sender": "colleague@example.com"},
            {"id": "msg2", "subject": "Project Meeting Summary", "sender": "boss@example.com"},
        ]

    async def get_email(self, user_id, email_id):
        return {"id": email_id, "subject": "Test Email", "body": "Email content"}

    async def create_draft(self, user_id, to, subject, body):
        return {"id": f"draft_{uuid4().hex[:8]}"}

    async def send_email(self, user_id, to, subject, body):
        return {"message_id": f"msg_{uuid4().hex[:8]}", "status": "sent"}

    async def send_draft(self, user_id, draft_id):
        return {"id": f"sent_{uuid4().hex[:8]}", "status": "sent", "draft_id": draft_id}


class MockCalendarService:
    """Mock Calendar service for integration tests."""

    async def search_events(self, user_id, embedding, filters=None, limit=10, similarity_threshold=0.25):
        return [
            {"id": "evt1", "title": "Team Standup", "start_time": "2024-01-15 09:00"},
            {"id": "evt2", "title": "Project Review", "start_time": "2024-01-15 14:00"},
        ]

    async def search_events_bm25(self, user_id, query, filters=None, limit=20):
        """BM25/full-text search for hybrid search."""
        return [
            {"id": "evt1", "title": "Team Standup", "start_time": "2024-01-15 09:00"},
            {"id": "evt2", "title": "Project Review", "start_time": "2024-01-15 14:00"},
        ]

    async def get_event(self, user_id, event_id):
        return {"id": event_id, "title": "Test Event"}

    async def create_event(self, user_id, title, start_time, end_time, **kwargs):
        return {"event_id": f"evt_{uuid4().hex[:8]}"}

    async def update_event(self, user_id, event_id, updates):
        return {"event_id": event_id, "updated": True}

    async def delete_event(self, user_id, event_id):
        return None


class MockDriveService:
    """Mock Drive service for integration tests."""

    async def search_files(self, user_id, embedding, filters=None, limit=10, similarity_threshold=0.25):
        return [
            {"id": "file1", "name": "Meeting Notes.docx", "mime_type": "application/vnd.google-apps.document"},
        ]

    async def search_files_bm25(self, user_id, query, filters=None, limit=20):
        """BM25/full-text search for hybrid search."""
        return [
            {"id": "file1", "name": "Meeting Notes.docx", "mime_type": "application/vnd.google-apps.document"},
        ]

    async def get_file(self, user_id, file_id):
        return {"id": file_id, "name": "Test File"}

    async def share_file(self, user_id, file_id, email, role="reader"):
        return {"shared": True}


@pytest.fixture
def mock_llm():
    """Create mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def mock_embedding_service():
    """Create mock embedding service."""
    return MockEmbeddingService()


@pytest.fixture
def mock_gmail_service():
    """Create mock Gmail service."""
    return MockGmailService()


@pytest.fixture
def mock_calendar_service():
    """Create mock Calendar service."""
    return MockCalendarService()


@pytest.fixture
def mock_drive_service():
    """Create mock Drive service."""
    return MockDriveService()


@pytest.fixture
def gmail_agent(mock_gmail_service, mock_embedding_service):
    """Create Gmail agent with mocks."""
    return GmailAgent(mock_gmail_service, mock_embedding_service)


@pytest.fixture
def gcal_agent(mock_calendar_service, mock_embedding_service):
    """Create Calendar agent with mocks."""
    return GcalAgent(mock_calendar_service, mock_embedding_service)


@pytest.fixture
def gdrive_agent(mock_drive_service, mock_embedding_service):
    """Create Drive agent with mocks."""
    return GdriveAgent(mock_drive_service, mock_embedding_service)


class TestFullQueryFlow:
    """Test the complete query flow from intent to response."""

    @pytest.mark.asyncio
    async def test_simple_gmail_search_flow(self, mock_llm, gmail_agent):
        """Test simple Gmail search query end-to-end."""
        # Setup intent classifier
        classifier = IntentClassifier(mock_llm)
        planner = QueryPlanner()
        agents = {ServiceType.GMAIL: gmail_agent}

        # Create orchestrator
        orchestrator = Orchestrator(classifier, planner, agents)

        # Execute query
        user_id = str(uuid4())
        result = await orchestrator.execute_query(
            query="Find my meeting emails",
            user_id=user_id,
        )

        # Verify the full flow
        assert "intent" in result
        assert "plan" in result
        assert "results" in result

        # Check intent was classified
        assert result["intent"]["operation"] == "search"
        assert ServiceType.GMAIL.value in [s for s in result["intent"]["services"]]

        # Check plan was created
        assert len(result["plan"]["steps"]) > 0

        # Check results were produced
        assert len(result["results"]) > 0
        assert result["results"][0].success is True

    @pytest.mark.asyncio
    async def test_multi_service_search_flow(self, mock_llm, gmail_agent, gcal_agent):
        """Test multi-service search query end-to-end."""
        # Configure LLM to return multi-service intent with ExecutionStep format
        mock_llm.intent_response = '''{
            "services": ["gmail", "gcal"],
            "operation": "search",
            "steps": [
                {"step": "search_gmail", "params": {"search_query": "project meeting"}},
                {"step": "search_calendar", "params": {"search_query": "project meeting"}}
            ],
            "confidence": 0.9
        }'''

        classifier = IntentClassifier(mock_llm)
        planner = QueryPlanner()
        agents = {
            ServiceType.GMAIL: gmail_agent,
            ServiceType.GCAL: gcal_agent,
        }

        orchestrator = Orchestrator(classifier, planner, agents)

        user_id = str(uuid4())
        result = await orchestrator.execute_query(
            query="Find all project meeting info in emails and calendar",
            user_id=user_id,
        )

        # Should have results from both services
        assert len(result["results"]) == 2
        assert all(r.success for r in result["results"])

    @pytest.mark.asyncio
    async def test_action_flow_with_dependency(self, mock_llm, gmail_agent):
        """Test action flow that depends on search results."""
        # Configure LLM to return action intent with ExecutionStep format
        mock_llm.intent_response = '''{
            "services": ["gmail"],
            "operation": "draft",
            "steps": [
                {"step": "search_gmail", "params": {"search_query": "meeting"}},
                {"step": "draft_email", "params": {"message": "reply to meeting", "to_name": "test"}, "depends_on": [0]}
            ],
            "confidence": 0.85
        }'''

        classifier = IntentClassifier(mock_llm)
        planner = QueryPlanner()
        agents = {ServiceType.GMAIL: gmail_agent}

        orchestrator = Orchestrator(classifier, planner, agents)

        user_id = str(uuid4())
        result = await orchestrator.execute_query(
            query="Reply to the latest meeting email",
            user_id=user_id,
        )

        # Should have search and draft results
        assert len(result["results"]) == 2

        # First step (search) should succeed
        assert result["results"][0].success is True
        assert result["results"][0].data is not None

    @pytest.mark.asyncio
    async def test_full_flow_with_synthesizer(self, mock_llm, gmail_agent):
        """Test complete flow including response synthesis."""
        classifier = IntentClassifier(mock_llm)
        planner = QueryPlanner()
        agents = {ServiceType.GMAIL: gmail_agent}
        synthesizer = ResponseSynthesizer(mock_llm)

        orchestrator = Orchestrator(classifier, planner, agents)

        user_id = str(uuid4())
        result = await orchestrator.execute_query(
            query="Find my meeting emails",
            user_id=user_id,
        )

        # Synthesize response
        intent = ParsedIntent(**result["intent"])
        response = await synthesizer.synthesize(
            query="Find my meeting emails",
            intent=intent,
            results=result["results"],
        )

        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_conversation_context_flow(self, mock_llm, gmail_agent):
        """Test flow with conversation context."""
        classifier = IntentClassifier(mock_llm)
        planner = QueryPlanner()
        agents = {ServiceType.GMAIL: gmail_agent}

        orchestrator = Orchestrator(classifier, planner, agents)

        user_id = str(uuid4())

        # First query
        result1 = await orchestrator.execute_query(
            query="Find meeting emails from John",
            user_id=user_id,
        )

        # Second query with context
        context = [
            {"query": "Find meeting emails from John", "intent": result1["intent"]}
        ]

        result2 = await orchestrator.execute_query(
            query="Show me more details",
            user_id=user_id,
            conversation_context=context,
        )

        assert result2 is not None
        assert "results" in result2


class TestErrorHandling:
    """Test error handling in the integration flow."""

    @pytest.mark.asyncio
    async def test_missing_agent_graceful_failure(self, mock_llm):
        """Test graceful failure when agent is missing."""
        classifier = IntentClassifier(mock_llm)
        planner = QueryPlanner()
        agents = {}  # No agents configured

        orchestrator = Orchestrator(classifier, planner, agents)

        user_id = str(uuid4())
        result = await orchestrator.execute_query(
            query="Find my emails",
            user_id=user_id,
        )

        # Should have failed result
        assert len(result["results"]) > 0
        assert result["results"][0].success is False
        assert "No agent" in result["results"][0].error

    @pytest.mark.asyncio
    async def test_service_error_graceful_handling(self, mock_llm, mock_embedding_service):
        """Test graceful handling of service errors."""

        class FailingGmailService:
            async def search_emails(self, *args, **kwargs):
                raise RuntimeError("Gmail API temporarily unavailable")

            async def search_emails_bm25(self, *args, **kwargs):
                raise RuntimeError("Gmail API temporarily unavailable")

        failing_agent = GmailAgent(FailingGmailService(), mock_embedding_service)

        classifier = IntentClassifier(mock_llm)
        planner = QueryPlanner()
        agents = {ServiceType.GMAIL: failing_agent}

        orchestrator = Orchestrator(classifier, planner, agents)

        user_id = str(uuid4())
        result = await orchestrator.execute_query(
            query="Find my emails",
            user_id=user_id,
        )

        # Should have error result but not crash
        assert len(result["results"]) > 0
        assert result["results"][0].success is False
        assert "unavailable" in result["results"][0].error.lower()


class TestAPIIntegration:
    """Test API endpoint integration."""

    @pytest.mark.asyncio
    async def test_query_schema_validation(self):
        """Test query request/response schema validation."""
        from app.schemas.query import QueryRequest, QueryResponse, ActionTaken

        # Valid request
        request = QueryRequest(query="Find my emails")
        assert request.query == "Find my emails"
        assert request.conversation_id is None

        # Valid response
        conv_id = uuid4()
        response = QueryResponse(
            response="Found 2 emails",
            actions_taken=[ActionTaken(step="search_gmail", success=True)],
            conversation_id=conv_id,
        )
        assert response.response == "Found 2 emails"
        assert len(response.actions_taken) == 1

    @pytest.mark.asyncio
    async def test_sync_trigger_schema_validation(self):
        """Test sync trigger request schema validation."""
        from app.schemas.query import SyncTriggerRequest

        # Default (all services)
        request = SyncTriggerRequest()
        assert request.service == "all"

        # Specific service
        request = SyncTriggerRequest(service="gmail")
        assert request.service == "gmail"

    @pytest.mark.asyncio
    async def test_sync_status_schema_validation(self):
        """Test sync status response schema validation."""
        from app.schemas.query import SyncStatusResponse
        from datetime import datetime

        response = SyncStatusResponse(
            gmail_last_sync=datetime.utcnow(),
            gmail_status="completed",
            gcal_last_sync=None,
            gcal_status="never_synced",
            gdrive_last_sync=None,
            gdrive_status="never_synced",
        )

        assert response.gmail_status == "completed"
        assert response.gcal_status == "never_synced"

    @pytest.mark.asyncio
    async def test_health_response_schema_validation(self):
        """Test health response schema validation."""
        from app.schemas.query import HealthResponse

        response = HealthResponse(
            status="healthy",
            database="connected",
            redis="connected",
        )

        assert response.status == "healthy"
        assert response.database == "connected"


class TestEndToEndScenarios:
    """Test realistic end-to-end scenarios."""

    @pytest.mark.asyncio
    async def test_find_flight_booking_scenario(
        self, mock_llm, gmail_agent, gcal_agent
    ):
        """Test finding flight booking info across services."""
        mock_llm.intent_response = '''{
            "services": ["gmail", "gcal"],
            "operation": "search",
            "steps": [
                {"step": "search_gmail", "params": {"search_query": "Turkish Airlines flight booking"}},
                {"step": "search_calendar", "params": {"search_query": "flight"}}
            ],
            "confidence": 0.9
        }'''

        classifier = IntentClassifier(mock_llm)
        planner = QueryPlanner()
        agents = {
            ServiceType.GMAIL: gmail_agent,
            ServiceType.GCAL: gcal_agent,
        }

        orchestrator = Orchestrator(classifier, planner, agents)

        result = await orchestrator.execute_query(
            query="Find my Turkish Airlines flight booking",
            user_id=str(uuid4()),
        )

        assert len(result["results"]) == 2
        assert all(r.success for r in result["results"])

    @pytest.mark.asyncio
    async def test_schedule_meeting_scenario(self, mock_llm, gcal_agent):
        """Test scheduling a meeting."""
        mock_llm.intent_response = '''{
            "services": ["gcal"],
            "operation": "create",
            "entities": {"event": "meeting", "time": "tomorrow 2pm"},
            "steps": ["create_event"],
            "confidence": 0.88
        }'''

        classifier = IntentClassifier(mock_llm)
        planner = QueryPlanner()
        agents = {ServiceType.GCAL: gcal_agent}

        orchestrator = Orchestrator(classifier, planner, agents)

        result = await orchestrator.execute_query(
            query="Schedule a team meeting for tomorrow at 2pm",
            user_id=str(uuid4()),
        )

        assert len(result["results"]) > 0

    @pytest.mark.asyncio
    async def test_find_and_share_document_scenario(
        self, mock_llm, gdrive_agent
    ):
        """Test finding and sharing a document."""
        mock_llm.intent_response = '''{
            "services": ["gdrive"],
            "operation": "search",
            "entities": {"file": "project proposal"},
            "steps": ["search_drive"],
            "confidence": 0.92
        }'''

        classifier = IntentClassifier(mock_llm)
        planner = QueryPlanner()
        agents = {ServiceType.GDRIVE: gdrive_agent}

        orchestrator = Orchestrator(classifier, planner, agents)

        result = await orchestrator.execute_query(
            query="Find the project proposal document",
            user_id=str(uuid4()),
        )

        assert len(result["results"]) > 0
        assert result["results"][0].success is True
