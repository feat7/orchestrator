"""Schemas for intent classification and step execution."""

from pydantic import BaseModel, Field
from typing import Literal, Optional, Any
from enum import Enum


class ServiceType(str, Enum):
    """Available Google Workspace services."""

    GMAIL = "gmail"
    GCAL = "gcal"
    GDRIVE = "gdrive"


class StepType(str, Enum):
    """Available execution steps for the orchestrator."""

    # Gmail steps
    SEARCH_GMAIL = "search_gmail"
    GET_EMAIL = "get_email"
    DRAFT_EMAIL = "draft_email"
    SEND_EMAIL = "send_email"

    # Calendar steps
    SEARCH_CALENDAR = "search_calendar"
    GET_EVENT = "get_event"
    CREATE_EVENT = "create_event"
    UPDATE_EVENT = "update_event"
    DELETE_EVENT = "delete_event"

    # Drive steps
    SEARCH_DRIVE = "search_drive"
    GET_FILE = "get_file"
    SHARE_FILE = "share_file"


class ParsedIntent(BaseModel):
    """Structured representation of a user's intent."""

    services: list[ServiceType] = Field(
        description="Services needed to fulfill the request"
    )
    operation: Literal["search", "create", "update", "delete", "send", "draft"] = Field(
        description="Primary operation type"
    )
    entities: dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted entities (names, dates, companies, etc.)",
    )
    steps: list[StepType] = Field(
        description="Ordered list of steps to execute"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score of the classification"
    )

    class Config:
        use_enum_values = True


class StepResult(BaseModel):
    """Result of executing a single step."""

    step: StepType = Field(description="The step that was executed")
    success: bool = Field(description="Whether the step succeeded")
    data: Optional[dict[str, Any]] = Field(
        default=None, description="Data returned by the step"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if the step failed"
    )

    class Config:
        use_enum_values = True
