"""Pytest configuration and fixtures."""

import pytest
import pytest_asyncio

# Configure pytest-asyncio to use auto mode
pytest_plugins = ('pytest_asyncio',)
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock

from app.main import app
from app.core.llm import LLMProvider


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    async def complete(self, prompt: str, system: str = None, json_mode: bool = False) -> str:
        """Return mock intent classification."""
        if json_mode or "classify" in prompt.lower():
            return '''{
                "services": ["gmail"],
                "operation": "search",
                "entities": {"topic": "test"},
                "steps": ["search_gmail"],
                "confidence": 0.95
            }'''
        return "This is a mock response."

    async def embed(self, text: str) -> list[float]:
        """Return mock embedding."""
        # Return a deterministic mock embedding based on text length
        import hashlib
        hash_bytes = hashlib.md5(text.encode()).digest()
        # Convert to list of floats
        return [float(b) / 255.0 for b in hash_bytes] * 96  # 1536 dimensions


@pytest.fixture
def mock_llm():
    """Provide mock LLM provider."""
    return MockLLMProvider()


@pytest_asyncio.fixture
async def client():
    """Provide async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_query():
    """Provide sample query for testing."""
    return "What's on my calendar next week?"


@pytest.fixture
def sample_intent():
    """Provide sample parsed intent."""
    from app.schemas.intent import ParsedIntent, ServiceType, StepType

    return ParsedIntent(
        services=[ServiceType.GCAL],
        operation="search",
        entities={"time_range": "next_week"},
        steps=[StepType.SEARCH_CALENDAR],
        confidence=0.95,
    )


@pytest.fixture
def sample_multi_service_intent():
    """Provide sample multi-service intent."""
    from app.schemas.intent import ParsedIntent, ServiceType, StepType

    return ParsedIntent(
        services=[ServiceType.GMAIL, ServiceType.GCAL],
        operation="update",
        entities={"airline": "Turkish Airlines", "action": "cancel"},
        steps=[StepType.SEARCH_GMAIL, StepType.SEARCH_CALENDAR, StepType.DRAFT_EMAIL],
        confidence=0.9,
    )
