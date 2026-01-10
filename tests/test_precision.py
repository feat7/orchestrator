"""Tests for Precision@5 benchmark evaluation."""

import pytest
from app.evaluation.benchmark import (
    BenchmarkQuery,
    calculate_precision_at_k,
    _check_keywords_in_fields,
    _check_sender_pattern,
    _check_mime_type,
    SEARCH_BENCHMARK,
)


class TestRelevanceCheckers:
    """Test relevance checker functions."""

    def test_check_keywords_in_fields_match(self):
        """Test keyword matching in fields."""
        result = {
            "subject": "Turkish Airlines Flight Confirmation",
            "body_preview": "Your booking is confirmed",
        }
        assert _check_keywords_in_fields(
            result, ["turkish", "flight"], ["subject", "body_preview"]
        ) is True

    def test_check_keywords_in_fields_no_match(self):
        """Test keyword no match."""
        result = {
            "subject": "Meeting Notes",
            "body_preview": "Here are the meeting notes",
        }
        assert _check_keywords_in_fields(
            result, ["turkish", "flight"], ["subject", "body_preview"]
        ) is False

    def test_check_keywords_case_insensitive(self):
        """Test case insensitive matching."""
        result = {"subject": "TURKISH AIRLINES"}
        assert _check_keywords_in_fields(
            result, ["turkish"], ["subject"]
        ) is True

    def test_check_sender_pattern(self):
        """Test sender pattern matching."""
        result = {"sender": "support@turkishairlines.com"}
        assert _check_sender_pattern(result, "turkish") is True

    def test_check_mime_type(self):
        """Test MIME type matching."""
        result = {"mime_type": "application/pdf"}
        assert _check_mime_type(result, ["pdf"]) is True
        assert _check_mime_type(result, ["spreadsheet"]) is False


class TestPrecisionCalculation:
    """Test Precision@K calculation."""

    def test_precision_all_relevant(self):
        """Test P@5 when all results are relevant."""
        results = [
            {"subject": "Turkish flight"},
            {"subject": "Turkish Airlines booking"},
            {"subject": "Your Turkish flight"},
            {"subject": "Flight confirmation Turkish"},
            {"subject": "Turkish Airlines receipt"},
        ]
        relevance_check = lambda r: "turkish" in r.get("subject", "").lower()

        precision, relevant, total = calculate_precision_at_k(
            results, relevance_check, k=5
        )

        assert precision == 1.0
        assert relevant == 5
        assert total == 5

    def test_precision_some_relevant(self):
        """Test P@5 when some results are relevant."""
        results = [
            {"subject": "Turkish flight"},
            {"subject": "Meeting notes"},
            {"subject": "Turkish Airlines booking"},
            {"subject": "Random email"},
            {"subject": "Another random"},
        ]
        relevance_check = lambda r: "turkish" in r.get("subject", "").lower()

        precision, relevant, total = calculate_precision_at_k(
            results, relevance_check, k=5
        )

        assert precision == 0.4  # 2/5
        assert relevant == 2
        assert total == 5

    def test_precision_none_relevant(self):
        """Test P@5 when no results are relevant."""
        results = [
            {"subject": "Meeting notes"},
            {"subject": "Project update"},
            {"subject": "Lunch plans"},
            {"subject": "Weekly report"},
            {"subject": "Coffee chat"},
        ]
        relevance_check = lambda r: "turkish" in r.get("subject", "").lower()

        precision, relevant, total = calculate_precision_at_k(
            results, relevance_check, k=5
        )

        assert precision == 0.0
        assert relevant == 0
        assert total == 5

    def test_precision_fewer_than_k_results(self):
        """Test P@K with fewer than K results."""
        results = [
            {"subject": "Turkish flight"},
            {"subject": "Turkish Airlines"},
        ]
        relevance_check = lambda r: "turkish" in r.get("subject", "").lower()

        precision, relevant, total = calculate_precision_at_k(
            results, relevance_check, k=5
        )

        # Still divides by K=5
        assert precision == 0.4  # 2/5
        assert relevant == 2
        assert total == 2

    def test_precision_empty_results(self):
        """Test P@K with no results."""
        results = []
        relevance_check = lambda r: True

        precision, relevant, total = calculate_precision_at_k(
            results, relevance_check, k=5
        )

        assert precision == 0.0
        assert relevant == 0
        assert total == 0


class TestBenchmarkQueries:
    """Test benchmark query definitions."""

    def test_benchmark_queries_exist(self):
        """Test that benchmark queries are defined."""
        assert len(SEARCH_BENCHMARK) > 0

    def test_benchmark_covers_all_services(self):
        """Test that benchmark covers all services."""
        services = {q.service for q in SEARCH_BENCHMARK}
        assert "gmail" in services
        assert "gcal" in services
        assert "gdrive" in services

    def test_gmail_benchmark_relevance(self):
        """Test Gmail benchmark relevance checks work."""
        # Find a Gmail benchmark query (ComfyUI workflow)
        gmail_query = next(
            q for q in SEARCH_BENCHMARK
            if q.service == "gmail" and "ComfyUI" in q.query
        )

        # Test relevant result
        relevant_result = {
            "subject": "ComfyUI Workflow Update - New Features",
            "body_preview": "Check out the latest workflow improvements",
            "sender": "updates@example.com",
        }
        assert gmail_query.relevance_check(relevant_result) is True

        # Test irrelevant result
        irrelevant_result = {
            "subject": "Meeting notes",
            "body_preview": "Here are the notes from today",
            "sender": "colleague@company.com",
        }
        assert gmail_query.relevance_check(irrelevant_result) is False

    def test_calendar_benchmark_relevance(self):
        """Test Calendar benchmark relevance checks work."""
        # Find a calendar benchmark query
        cal_query = next(
            q for q in SEARCH_BENCHMARK
            if q.service == "gcal" and "standup" in q.query.lower()
        )

        # Test relevant result
        relevant_result = {
            "title": "Daily Team Standup",
            "description": "Daily sync meeting",
        }
        assert cal_query.relevance_check(relevant_result) is True

        # Test irrelevant result
        irrelevant_result = {
            "title": "Lunch with Sarah",
            "description": "Catch up over lunch",
        }
        assert cal_query.relevance_check(irrelevant_result) is False

    def test_drive_benchmark_relevance(self):
        """Test Drive benchmark relevance checks work."""
        # Find a drive benchmark query (project document)
        drive_query = next(
            q for q in SEARCH_BENCHMARK
            if q.service == "gdrive" and "project" in q.query.lower()
        )

        # Test relevant result
        relevant_result = {
            "name": "Project Document 2024.docx",
            "content_preview": "This project outlines...",
            "mime_type": "application/vnd.google-apps.document",
        }
        assert drive_query.relevance_check(relevant_result) is True

        # Test irrelevant result
        irrelevant_result = {
            "name": "vacation_photos.jpg",
            "content_preview": "",
            "mime_type": "image/jpeg",
        }
        assert drive_query.relevance_check(irrelevant_result) is False


@pytest.mark.asyncio
async def test_run_search_benchmark_with_mock_agents():
    """Test running benchmark with mock agents."""
    from app.evaluation.benchmark import run_search_benchmark

    class MockAgent:
        def __init__(self, results):
            self.results = results

        async def search(self, query, user_id, filters):
            return self.results

    # Create mock agents with relevant results
    gmail_agent = MockAgent([
        {"id": "1", "subject": "Turkish Airlines Booking", "body_preview": "Flight TK123"},
        {"id": "2", "subject": "Flight Confirmation", "body_preview": "Turkish Airlines"},
        {"id": "3", "subject": "Turkish flight receipt", "body_preview": ""},
        {"id": "4", "subject": "Meeting notes", "body_preview": ""},
        {"id": "5", "subject": "Random email", "body_preview": ""},
    ])

    gcal_agent = MockAgent([
        {"id": "1", "title": "Team Standup", "description": "Daily sync"},
        {"id": "2", "title": "Daily Standup", "description": "Team meeting"},
        {"id": "3", "title": "Standup", "description": ""},
        {"id": "4", "title": "Lunch", "description": ""},
        {"id": "5", "title": "Coffee", "description": ""},
    ])

    gdrive_agent = MockAgent([
        {"id": "1", "name": "Project Proposal.docx", "content_preview": ""},
        {"id": "2", "name": "Proposal Draft", "content_preview": "project"},
        {"id": "3", "name": "proposal_v2.pdf", "content_preview": ""},
        {"id": "4", "name": "random.txt", "content_preview": ""},
        {"id": "5", "name": "photo.jpg", "content_preview": ""},
    ])

    agents = {
        "gmail": gmail_agent,
        "gcal": gcal_agent,
        "gdrive": gdrive_agent,
    }

    result = await run_search_benchmark(agents, "test-user-id", None)

    assert "overall_precision_at_5" in result
    assert "per_service" in result
    assert "details" in result
    assert result["target_precision"] == 0.8

    # At least some queries should have results
    assert result["queries_evaluated"] > 0
