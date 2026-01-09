"""Schemas for API requests and responses."""

from pydantic import BaseModel, Field
from typing import Optional, Any
from uuid import UUID
from datetime import datetime


class QueryRequest(BaseModel):
    """Request payload for processing a natural language query."""

    query: str = Field(description="The natural language query to process")
    conversation_id: Optional[UUID] = Field(
        default=None, description="Optional conversation ID for multi-turn context"
    )


class ActionTaken(BaseModel):
    """Represents a single action taken during query processing."""

    step: str = Field(description="The step that was executed")
    success: bool = Field(description="Whether the step succeeded")
    data: Optional[dict[str, Any]] = Field(default=None)
    error: Optional[str] = Field(default=None)


class QueryResponse(BaseModel):
    """Response payload after processing a query."""

    response: str = Field(description="Natural language response to the user")
    actions_taken: list[ActionTaken] = Field(
        description="List of actions taken during processing"
    )
    conversation_id: UUID = Field(description="Conversation ID for follow-up queries")
    intent: Optional[dict[str, Any]] = Field(
        default=None, description="The parsed intent (for debugging)"
    )
    needs_clarification: bool = Field(
        default=False,
        description="True if the AI needs more info before proceeding"
    )
    options: Optional[list[str]] = Field(
        default=None,
        description="Suggested options for clarification"
    )
    latency_ms: Optional[int] = Field(
        default=None,
        description="Server-side processing time in milliseconds"
    )


class SyncTriggerRequest(BaseModel):
    """Request to trigger a sync operation."""

    service: Optional[str] = Field(
        default="all", description="Service to sync (gmail, gcal, gdrive, or all)"
    )


class SyncStatusResponse(BaseModel):
    """Response showing sync status for all services."""

    gmail_last_sync: Optional[datetime] = Field(default=None)
    gmail_status: str = Field(default="unknown")
    gcal_last_sync: Optional[datetime] = Field(default=None)
    gcal_status: str = Field(default="unknown")
    gdrive_last_sync: Optional[datetime] = Field(default=None)
    gdrive_status: str = Field(default="unknown")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy")
    database: str = Field(default="connected")
    redis: str = Field(default="connected")


class MetricsResponse(BaseModel):
    """Performance metrics response."""

    total_queries: int = Field(description="Total queries processed")
    avg_latency_ms: float = Field(description="Average latency in milliseconds")
    p50_latency_ms: float = Field(description="Median latency")
    p95_latency_ms: float = Field(description="95th percentile latency")
    p99_latency_ms: float = Field(description="99th percentile latency")
    cache_hit_rate: float = Field(description="Cache hit rate (0-1)")
    embedding_latency_ms: float = Field(description="Average embedding query latency")
    search_precision_at_5: Optional[float] = Field(
        default=None,
        description="Precision@5 for search results"
    )
