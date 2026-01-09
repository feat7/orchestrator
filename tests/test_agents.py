"""Tests for the Agent components (Gmail, Calendar, Drive)."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.agents.gmail import GmailAgent
from app.agents.gcal import GcalAgent
from app.agents.gdrive import GdriveAgent
from app.schemas.intent import StepType, StepResult


class MockEmbeddingService:
    """Mock embedding service for testing."""

    async def embed(self, text: str) -> list[float]:
        return [0.0] * 1536


class MockGmailService:
    """Mock Gmail service for testing."""

    def __init__(self):
        self.search_results = []
        self.get_result = None
        self.draft_result = {"draft_id": "draft123"}
        self.send_result = {"message_id": "msg456"}
        self.search_calls = []
        self.get_calls = []

    async def search_emails(self, user_id, embedding, filters=None, limit=10):
        self.search_calls.append({"user_id": user_id, "filters": filters, "limit": limit})
        return self.search_results

    async def get_email(self, user_id, email_id):
        self.get_calls.append({"user_id": user_id, "email_id": email_id})
        return self.get_result

    async def create_draft(self, user_id, to, subject, body):
        return self.draft_result

    async def send_email(self, user_id, to, subject, body):
        return self.send_result


class MockCalendarService:
    """Mock Calendar service for testing."""

    def __init__(self):
        self.search_results = []
        self.get_result = None
        self.create_result = {"event_id": "evt123"}
        self.update_result = {"event_id": "evt123", "updated": True}
        self.search_calls = []

    async def search_events(self, user_id, embedding, filters=None, limit=10):
        self.search_calls.append({"user_id": user_id, "filters": filters})
        return self.search_results

    async def get_event(self, user_id, event_id):
        return self.get_result

    async def create_event(self, user_id, title, start_time, end_time, attendees=None, description="", location=""):
        return self.create_result

    async def update_event(self, user_id, event_id, updates):
        return self.update_result

    async def delete_event(self, user_id, event_id):
        return None


class MockDriveService:
    """Mock Drive service for testing."""

    def __init__(self):
        self.search_results = []
        self.get_result = None
        self.share_result = {"shared": True}
        self.search_calls = []

    async def search_files(self, user_id, embedding, filters=None, limit=10):
        self.search_calls.append({"user_id": user_id, "filters": filters})
        return self.search_results

    async def get_file(self, user_id, file_id):
        return self.get_result

    async def share_file(self, user_id, file_id, email, role="reader"):
        return self.share_result


# =============================================================================
# Gmail Agent Tests
# =============================================================================


@pytest.fixture
def gmail_agent():
    """Create Gmail agent with mocks."""
    gmail_service = MockGmailService()
    embedding_service = MockEmbeddingService()
    return GmailAgent(gmail_service, embedding_service)


def test_gmail_agent_initialization(gmail_agent):
    """Test Gmail agent can be initialized."""
    assert gmail_agent.gmail is not None
    assert gmail_agent.embeddings is not None


@pytest.mark.asyncio
async def test_gmail_agent_search(gmail_agent):
    """Test Gmail agent search."""
    gmail_agent.gmail.search_results = [
        {"id": "msg1", "subject": "Test Email", "sender": "test@example.com"}
    ]
    user_id = str(uuid4())

    results = await gmail_agent.search("test query", user_id, {"sender": "test@example.com"})

    assert len(results) == 1
    assert results[0]["subject"] == "Test Email"
    assert len(gmail_agent.gmail.search_calls) == 1


@pytest.mark.asyncio
async def test_gmail_agent_execute_get_email(gmail_agent):
    """Test Gmail agent execute GET_EMAIL step."""
    gmail_agent.gmail.get_result = {
        "id": "msg1",
        "subject": "Test Email",
        "body": "Hello World",
    }
    user_id = str(uuid4())

    result = await gmail_agent.execute(
        StepType.GET_EMAIL,
        {"email_id": "msg1"},
        user_id,
    )

    assert result.success is True
    assert result.data["subject"] == "Test Email"


@pytest.mark.asyncio
async def test_gmail_agent_execute_get_email_not_found(gmail_agent):
    """Test Gmail agent execute GET_EMAIL when email not found."""
    gmail_agent.gmail.get_result = None
    user_id = str(uuid4())

    result = await gmail_agent.execute(
        StepType.GET_EMAIL,
        {"email_id": "nonexistent"},
        user_id,
    )

    assert result.success is False
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_gmail_agent_execute_draft_email(gmail_agent):
    """Test Gmail agent execute DRAFT_EMAIL step."""
    user_id = str(uuid4())

    result = await gmail_agent.execute(
        StepType.DRAFT_EMAIL,
        {"to": "recipient@example.com", "subject": "Hello", "body": "Test body"},
        user_id,
    )

    assert result.success is True
    assert result.data["draft_id"] == "draft123"


@pytest.mark.asyncio
async def test_gmail_agent_execute_draft_email_with_source_data(gmail_agent):
    """Test Gmail agent draft email using source data from search."""
    user_id = str(uuid4())

    result = await gmail_agent.execute(
        StepType.DRAFT_EMAIL,
        {
            "source_data": {
                "sender": "original@example.com",
                "subject": "Original Subject",
            },
            "body": "Reply body",
        },
        user_id,
    )

    assert result.success is True


@pytest.mark.asyncio
async def test_gmail_agent_execute_send_email(gmail_agent):
    """Test Gmail agent execute SEND_EMAIL step."""
    user_id = str(uuid4())

    result = await gmail_agent.execute(
        StepType.SEND_EMAIL,
        {"to": "recipient@example.com", "subject": "Hello", "body": "Test body"},
        user_id,
    )

    assert result.success is True
    assert result.data["message_id"] == "msg456"


@pytest.mark.asyncio
async def test_gmail_agent_execute_unsupported_step(gmail_agent):
    """Test Gmail agent with unsupported step type."""
    user_id = str(uuid4())

    result = await gmail_agent.execute(
        StepType.CREATE_EVENT,  # Not a Gmail step
        {},
        user_id,
    )

    assert result.success is False
    assert "Unsupported" in result.error


@pytest.mark.asyncio
async def test_gmail_agent_get_context(gmail_agent):
    """Test Gmail agent get_context method."""
    gmail_agent.gmail.get_result = {"id": "msg1", "subject": "Test", "body": "Full body"}
    user_id = str(uuid4())

    context = await gmail_agent.get_context("msg1", user_id)

    assert context is not None
    assert context["subject"] == "Test"


# =============================================================================
# Calendar Agent Tests
# =============================================================================


@pytest.fixture
def gcal_agent():
    """Create Calendar agent with mocks."""
    calendar_service = MockCalendarService()
    embedding_service = MockEmbeddingService()
    return GcalAgent(calendar_service, embedding_service)


def test_gcal_agent_initialization(gcal_agent):
    """Test Calendar agent can be initialized."""
    assert gcal_agent.calendar is not None
    assert gcal_agent.embeddings is not None


@pytest.mark.asyncio
async def test_gcal_agent_search(gcal_agent):
    """Test Calendar agent search."""
    gcal_agent.calendar.search_results = [
        {"id": "evt1", "title": "Team Meeting", "start_time": "2024-01-15 09:00"}
    ]
    user_id = str(uuid4())

    results = await gcal_agent.search("meeting", user_id, {"time_range": "next_week"})

    assert len(results) == 1
    assert results[0]["title"] == "Team Meeting"


@pytest.mark.asyncio
async def test_gcal_agent_execute_get_event(gcal_agent):
    """Test Calendar agent execute GET_EVENT step."""
    gcal_agent.calendar.get_result = {
        "id": "evt1",
        "title": "Team Meeting",
        "start_time": "2024-01-15 09:00",
    }
    user_id = str(uuid4())

    result = await gcal_agent.execute(
        StepType.GET_EVENT,
        {"event_id": "evt1"},
        user_id,
    )

    assert result.success is True
    assert result.data["title"] == "Team Meeting"


@pytest.mark.asyncio
async def test_gcal_agent_execute_get_event_not_found(gcal_agent):
    """Test Calendar agent execute GET_EVENT when not found."""
    gcal_agent.calendar.get_result = None
    user_id = str(uuid4())

    result = await gcal_agent.execute(
        StepType.GET_EVENT,
        {"event_id": "nonexistent"},
        user_id,
    )

    assert result.success is False
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_gcal_agent_execute_create_event(gcal_agent):
    """Test Calendar agent execute CREATE_EVENT step."""
    user_id = str(uuid4())

    result = await gcal_agent.execute(
        StepType.CREATE_EVENT,
        {
            "title": "New Meeting",
            "start_time": "2024-01-20 10:00",
            "end_time": "2024-01-20 11:00",
            "attendees": ["test@example.com"],
        },
        user_id,
    )

    assert result.success is True
    assert result.data["event_id"] == "evt123"


@pytest.mark.asyncio
async def test_gcal_agent_execute_update_event(gcal_agent):
    """Test Calendar agent execute UPDATE_EVENT step."""
    user_id = str(uuid4())

    result = await gcal_agent.execute(
        StepType.UPDATE_EVENT,
        {"event_id": "evt1", "updates": {"title": "Updated Meeting"}},
        user_id,
    )

    assert result.success is True


@pytest.mark.asyncio
async def test_gcal_agent_execute_delete_event(gcal_agent):
    """Test Calendar agent execute DELETE_EVENT step."""
    user_id = str(uuid4())

    result = await gcal_agent.execute(
        StepType.DELETE_EVENT,
        {"event_id": "evt1"},
        user_id,
    )

    assert result.success is True
    assert result.data["deleted"] is True


@pytest.mark.asyncio
async def test_gcal_agent_get_context(gcal_agent):
    """Test Calendar agent get_context method."""
    gcal_agent.calendar.get_result = {
        "id": "evt1",
        "title": "Test Event",
        "description": "Full description",
    }
    user_id = str(uuid4())

    context = await gcal_agent.get_context("evt1", user_id)

    assert context is not None
    assert context["title"] == "Test Event"


# =============================================================================
# Drive Agent Tests
# =============================================================================


@pytest.fixture
def gdrive_agent():
    """Create Drive agent with mocks."""
    drive_service = MockDriveService()
    embedding_service = MockEmbeddingService()
    return GdriveAgent(drive_service, embedding_service)


def test_gdrive_agent_initialization(gdrive_agent):
    """Test Drive agent can be initialized."""
    assert gdrive_agent.drive is not None
    assert gdrive_agent.embeddings is not None


@pytest.mark.asyncio
async def test_gdrive_agent_search(gdrive_agent):
    """Test Drive agent search."""
    gdrive_agent.drive.search_results = [
        {"id": "file1", "name": "Project.pdf", "mime_type": "application/pdf"}
    ]
    user_id = str(uuid4())

    results = await gdrive_agent.search("project", user_id, {"mime_type": "application/pdf"})

    assert len(results) == 1
    assert results[0]["name"] == "Project.pdf"


@pytest.mark.asyncio
async def test_gdrive_agent_execute_get_file(gdrive_agent):
    """Test Drive agent execute GET_FILE step."""
    gdrive_agent.drive.get_result = {
        "id": "file1",
        "name": "Document.docx",
        "content": "File content",
    }
    user_id = str(uuid4())

    result = await gdrive_agent.execute(
        StepType.GET_FILE,
        {"file_id": "file1"},
        user_id,
    )

    assert result.success is True
    assert result.data["name"] == "Document.docx"


@pytest.mark.asyncio
async def test_gdrive_agent_execute_get_file_not_found(gdrive_agent):
    """Test Drive agent execute GET_FILE when not found."""
    gdrive_agent.drive.get_result = None
    user_id = str(uuid4())

    result = await gdrive_agent.execute(
        StepType.GET_FILE,
        {"file_id": "nonexistent"},
        user_id,
    )

    assert result.success is False
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_gdrive_agent_execute_share_file(gdrive_agent):
    """Test Drive agent execute SHARE_FILE step."""
    user_id = str(uuid4())

    result = await gdrive_agent.execute(
        StepType.SHARE_FILE,
        {"file_id": "file1", "email": "share@example.com", "role": "editor"},
        user_id,
    )

    assert result.success is True
    assert result.data["shared"] is True


@pytest.mark.asyncio
async def test_gdrive_agent_execute_unsupported_step(gdrive_agent):
    """Test Drive agent with unsupported step type."""
    user_id = str(uuid4())

    result = await gdrive_agent.execute(
        StepType.SEND_EMAIL,  # Not a Drive step
        {},
        user_id,
    )

    assert result.success is False
    assert "Unsupported" in result.error


@pytest.mark.asyncio
async def test_gdrive_agent_get_context(gdrive_agent):
    """Test Drive agent get_context method."""
    gdrive_agent.drive.get_result = {
        "id": "file1",
        "name": "Test File",
        "content_preview": "Preview content",
    }
    user_id = str(uuid4())

    context = await gdrive_agent.get_context("file1", user_id)

    assert context is not None
    assert context["name"] == "Test File"


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_gmail_agent_handles_service_error():
    """Test Gmail agent handles service errors gracefully."""

    class ErrorGmailService:
        async def get_email(self, user_id, email_id):
            raise RuntimeError("API Error")

        async def search_emails(self, *args, **kwargs):
            raise RuntimeError("API Error")

    agent = GmailAgent(ErrorGmailService(), MockEmbeddingService())
    user_id = str(uuid4())

    result = await agent.execute(StepType.GET_EMAIL, {"email_id": "123"}, user_id)

    assert result.success is False
    assert "API Error" in result.error


@pytest.mark.asyncio
async def test_gcal_agent_handles_service_error():
    """Test Calendar agent handles service errors gracefully."""

    class ErrorCalendarService:
        async def get_event(self, user_id, event_id):
            raise RuntimeError("Calendar API Error")

        async def search_events(self, *args, **kwargs):
            raise RuntimeError("Calendar API Error")

    agent = GcalAgent(ErrorCalendarService(), MockEmbeddingService())
    user_id = str(uuid4())

    result = await agent.execute(StepType.GET_EVENT, {"event_id": "123"}, user_id)

    assert result.success is False
    assert "Calendar API Error" in result.error


@pytest.mark.asyncio
async def test_gdrive_agent_handles_service_error():
    """Test Drive agent handles service errors gracefully."""

    class ErrorDriveService:
        async def get_file(self, user_id, file_id):
            raise RuntimeError("Drive API Error")

        async def search_files(self, *args, **kwargs):
            raise RuntimeError("Drive API Error")

    agent = GdriveAgent(ErrorDriveService(), MockEmbeddingService())
    user_id = str(uuid4())

    result = await agent.execute(StepType.GET_FILE, {"file_id": "123"}, user_id)

    assert result.success is False
    assert "Drive API Error" in result.error
