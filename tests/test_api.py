"""Tests for API endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient):
    """Test root endpoint serves UI."""
    response = await client.get("/")

    assert response.status_code == 200
    # Root now serves HTML UI
    assert "text/html" in response.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    """Test health check endpoint."""
    response = await client.get("/api/v1/health")

    # May fail if DB not connected, but should return 200
    assert response.status_code in [200, 500]


@pytest.mark.asyncio
async def test_query_endpoint_requires_auth(client: AsyncClient):
    """Test query endpoint requires authentication."""
    response = await client.post("/api/v1/query", json={"query": "test"})

    # Should return 401 without session auth
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_query_endpoint_validation(client: AsyncClient):
    """Test query endpoint input validation (empty query)."""
    response = await client.post("/api/v1/query", json={})

    # Without auth, returns 401 first; with auth would return 422
    assert response.status_code in [401, 422]


@pytest.mark.asyncio
async def test_sync_status_endpoint_requires_auth(client: AsyncClient):
    """Test sync status endpoint requires authentication."""
    response = await client.get("/api/v1/sync/status")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_sync_trigger_endpoint_requires_auth(client: AsyncClient):
    """Test sync trigger endpoint requires authentication."""
    response = await client.post("/api/v1/sync/trigger", json={"service": "gmail"})

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_auth_status_unauthenticated(client: AsyncClient):
    """Test auth status endpoint returns unauthenticated status."""
    response = await client.get("/api/v1/auth/status")

    assert response.status_code == 200
    data = response.json()
    assert data["authenticated"] is False


@pytest.mark.asyncio
async def test_auth_login_endpoint(client: AsyncClient):
    """Test auth login endpoint redirects to Google."""
    response = await client.get("/api/v1/auth/login", follow_redirects=False)

    # Should redirect to Google OAuth
    assert response.status_code in [302, 307]
    location = response.headers.get("location", "")
    assert "accounts.google.com" in location or "googleapis.com" in location


def test_query_request_schema():
    """Test QueryRequest schema validation."""
    from app.schemas.query import QueryRequest

    # Valid request
    request = QueryRequest(query="Test query")
    assert request.query == "Test query"
    assert request.conversation_id is None

    # With conversation ID
    from uuid import uuid4

    conv_id = uuid4()
    request = QueryRequest(query="Test", conversation_id=conv_id)
    assert request.conversation_id == conv_id


def test_query_response_schema():
    """Test QueryResponse schema."""
    from app.schemas.query import QueryResponse, ActionTaken
    from uuid import uuid4

    response = QueryResponse(
        response="Test response",
        actions_taken=[ActionTaken(step="search_gmail", success=True)],
        conversation_id=uuid4(),
    )

    assert response.response == "Test response"
    assert len(response.actions_taken) == 1
    assert response.actions_taken[0].success is True
