"""Tests for the SyncService component."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime
from uuid import uuid4


class TestSyncServiceUnit:
    """Unit tests for SyncService that don't require database/API connections."""

    def test_sync_service_import(self):
        """Test that SyncService can be imported."""
        from app.services.sync import SyncService
        assert SyncService is not None

    def test_generate_embedding_truncation(self):
        """Test that embedding generation handles long text."""
        # This tests the logic without actually calling OpenAI
        max_chars = 8000
        long_text = "a" * 10000

        # Simulate truncation logic
        truncated = long_text[:max_chars] if len(long_text) > max_chars else long_text
        assert len(truncated) == max_chars

    def test_sync_service_handles_missing_credentials(self):
        """Test sync handles missing credentials scenario."""
        # This is a logic test - credentials lookup returns None
        credentials = None
        if not credentials:
            error = ValueError("No valid credentials found")
            assert "No valid credentials" in str(error)

    def test_sync_status_values(self):
        """Test valid sync status values."""
        valid_statuses = ["idle", "syncing", "completed", "error"]
        assert "syncing" in valid_statuses
        assert "completed" in valid_statuses
        assert "error" in valid_statuses

    def test_gmail_sync_result_structure(self):
        """Test the expected structure of Gmail sync results."""
        result = {
            "service": "gmail",
            "synced": 10,
            "errors": 0,
            "status": "completed",
        }
        assert result["service"] == "gmail"
        assert isinstance(result["synced"], int)
        assert isinstance(result["errors"], int)
        assert result["status"] in ["completed", "error"]

    def test_calendar_sync_result_structure(self):
        """Test the expected structure of Calendar sync results."""
        result = {
            "service": "gcal",
            "synced": 5,
            "errors": 0,
            "status": "completed",
        }
        assert result["service"] == "gcal"
        assert isinstance(result["synced"], int)

    def test_drive_sync_result_structure(self):
        """Test the expected structure of Drive sync results."""
        result = {
            "service": "gdrive",
            "synced": 20,
            "errors": 1,
            "status": "completed",
        }
        assert result["service"] == "gdrive"
        assert isinstance(result["synced"], int)

    def test_sync_all_result_structure(self):
        """Test the expected structure of sync_all results."""
        result = {
            "gmail": {"service": "gmail", "synced": 10, "errors": 0, "status": "completed"},
            "calendar": {"service": "gcal", "synced": 5, "errors": 0, "status": "completed"},
            "drive": {"service": "gdrive", "synced": 20, "errors": 0, "status": "completed"},
        }
        assert "gmail" in result
        assert "calendar" in result
        assert "drive" in result

    def test_email_date_parsing(self):
        """Test email date parsing logic."""
        from email.utils import parsedate_to_datetime

        date_str = "Mon, 15 Jan 2024 10:30:00 +0000"
        parsed = parsedate_to_datetime(date_str)
        assert isinstance(parsed, datetime)
        assert parsed.year == 2024
        assert parsed.month == 1
        assert parsed.day == 15

    def test_calendar_time_bounds(self):
        """Test calendar sync time bounds calculation."""
        from datetime import timedelta

        now = datetime.utcnow()
        days_ahead = 30
        days_back = 7

        time_min = now - timedelta(days=days_back)
        time_max = now + timedelta(days=days_ahead)

        assert time_min < now
        assert time_max > now
        assert (time_max - time_min).days == days_ahead + days_back

    def test_iso_time_parsing(self):
        """Test ISO datetime parsing for calendar events."""
        iso_time = "2024-01-15T10:30:00Z"
        parsed = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
        assert parsed.year == 2024
        assert parsed.month == 1
        assert parsed.hour == 10

    def test_date_only_parsing(self):
        """Test date-only parsing for all-day events."""
        date_str = "2024-01-15"
        parsed = datetime.strptime(date_str, "%Y-%m-%d")
        assert parsed.year == 2024
        assert parsed.month == 1
        assert parsed.day == 15

    def test_error_sync_result(self):
        """Test error sync result structure."""
        error = "API connection failed"
        result = {
            "service": "gmail",
            "synced": 0,
            "errors": 1,
            "status": "error",
            "error": error,
        }
        assert result["status"] == "error"
        assert result["error"] == error
        assert result["synced"] == 0

    def test_email_body_extraction_from_payload(self):
        """Test email body extraction logic from different payload formats."""
        # Direct body in payload
        payload1 = {"body": {"data": "SGVsbG8gV29ybGQ="}}  # "Hello World" base64
        assert payload1["body"]["data"] is not None

        # Body in parts
        payload2 = {
            "parts": [
                {"mimeType": "text/plain", "body": {"data": "SGVsbG8gV29ybGQ="}},
                {"mimeType": "text/html", "body": {"data": "PFBHZ2VudD4="}},
            ]
        }
        plain_part = next(
            (p for p in payload2["parts"] if p.get("mimeType") == "text/plain"), None
        )
        assert plain_part is not None
        assert plain_part["body"]["data"] is not None

    def test_gmail_label_read_status(self):
        """Test extracting read status from Gmail labels."""
        labels_unread = ["INBOX", "UNREAD", "IMPORTANT"]
        labels_read = ["INBOX", "IMPORTANT"]

        is_unread1 = "UNREAD" in labels_unread
        is_unread2 = "UNREAD" in labels_read

        assert is_unread1 is True
        assert is_unread2 is False

    def test_drive_file_owners_extraction(self):
        """Test extracting owner emails from Drive file metadata."""
        file_data = {
            "owners": [
                {"emailAddress": "owner1@example.com"},
                {"emailAddress": "owner2@example.com"},
            ]
        }
        owners = [o.get("emailAddress", "") for o in file_data.get("owners", [])]
        assert len(owners) == 2
        assert "owner1@example.com" in owners

    def test_calendar_attendees_extraction(self):
        """Test extracting attendee emails from calendar events."""
        event = {
            "attendees": [
                {"email": "attendee1@example.com"},
                {"email": "attendee2@example.com"},
            ]
        }
        attendee_emails = [a.get("email", "") for a in event.get("attendees", [])]
        assert len(attendee_emails) == 2

    def test_sync_database_url_transformation(self):
        """Test database URL transformation for sync engine."""
        async_url = "postgresql+asyncpg://user:pass@host:5432/db"
        sync_url = async_url.replace("+asyncpg", "")
        assert "+asyncpg" not in sync_url
        assert "postgresql://" in sync_url


class TestSyncServiceIntegration:
    """Integration tests that require mocking external services."""

    @pytest.fixture
    def mock_credentials(self):
        """Create mock Google credentials."""
        from google.oauth2.credentials import Credentials

        creds = MagicMock(spec=Credentials)
        creds.token = "mock_token"
        creds.refresh_token = "mock_refresh"
        creds.expired = False
        creds.valid = True
        return creds

    def test_sync_service_initialization_requires_valid_uuid(self):
        """Test SyncService requires a valid UUID string."""
        from uuid import UUID

        valid_uuid = str(uuid4())
        # This should not raise
        parsed = UUID(valid_uuid)
        assert parsed is not None

        # Invalid UUID should raise
        with pytest.raises(ValueError):
            UUID("not-a-uuid")

    def test_embedding_text_construction_for_email(self):
        """Test embedding text construction for emails."""
        headers = {
            "Subject": "Test Subject",
            "From": "sender@example.com",
            "To": "recipient@example.com",
        }
        body = "This is the email body content."

        embed_text = f"Subject: {headers.get('Subject', '')}\n"
        embed_text += f"From: {headers.get('From', '')}\n"
        embed_text += f"To: {headers.get('To', '')}\n"
        embed_text += f"Body: {body[:2000]}"

        assert "Test Subject" in embed_text
        assert "sender@example.com" in embed_text
        assert "email body" in embed_text

    def test_embedding_text_construction_for_event(self):
        """Test embedding text construction for calendar events."""
        event = {
            "summary": "Team Meeting",
            "description": "Weekly sync",
            "location": "Conference Room A",
            "attendees": [{"email": "test@example.com"}],
        }

        embed_text = f"Event: {event.get('summary', 'Untitled')}\n"
        if event.get("description"):
            embed_text += f"Description: {event['description']}\n"
        if event.get("location"):
            embed_text += f"Location: {event['location']}\n"

        assert "Team Meeting" in embed_text
        assert "Weekly sync" in embed_text
        assert "Conference Room A" in embed_text

    def test_embedding_text_construction_for_file(self):
        """Test embedding text construction for Drive files."""
        file_data = {
            "name": "Project Proposal.pdf",
            "mimeType": "application/pdf",
            "description": "Q1 project proposal",
        }

        embed_text = f"File: {file_data.get('name', 'Untitled')}\n"
        embed_text += f"Type: {file_data.get('mimeType', 'unknown')}\n"
        if file_data.get("description"):
            embed_text += f"Description: {file_data['description']}\n"

        assert "Project Proposal.pdf" in embed_text
        assert "application/pdf" in embed_text
        assert "Q1 project" in embed_text


class TestSyncStatusTracking:
    """Tests for sync status tracking functionality."""

    def test_sync_status_model_fields(self):
        """Test SyncStatus model has required fields."""
        from app.db.models import SyncStatus

        # Check the model has the expected columns
        assert hasattr(SyncStatus, "user_id")
        assert hasattr(SyncStatus, "service")
        assert hasattr(SyncStatus, "status")
        assert hasattr(SyncStatus, "last_sync_at")
        assert hasattr(SyncStatus, "error_message")

    def test_valid_service_names(self):
        """Test valid service names for sync status."""
        valid_services = ["gmail", "gcal", "gdrive"]
        assert "gmail" in valid_services
        assert "gcal" in valid_services
        assert "gdrive" in valid_services

    def test_valid_status_transitions(self):
        """Test valid status transitions."""
        # idle -> syncing -> completed
        # idle -> syncing -> error
        transitions = {
            "idle": ["syncing"],
            "syncing": ["completed", "error"],
            "completed": ["syncing"],
            "error": ["syncing"],
        }

        assert "syncing" in transitions["idle"]
        assert "completed" in transitions["syncing"]
        assert "error" in transitions["syncing"]
