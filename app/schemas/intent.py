"""Schemas for intent classification and step execution."""

from pydantic import BaseModel, Field
from typing import Literal, Optional, Any, Union
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


# =============================================================================
# Step Parameter Schemas - LLM outputs these directly
# =============================================================================

class SearchDriveParams(BaseModel):
    """Parameters for search_drive step."""
    search_query: str = Field(default="", description="Semantic search query for file content/name")
    modified_after: Optional[str] = Field(default=None, description="ISO date - only files modified after this date")
    modified_before: Optional[str] = Field(default=None, description="ISO date - only files modified before this date")
    mime_type: Optional[str] = Field(default=None, description="Filter by MIME type (e.g., 'application/pdf', 'document', 'spreadsheet')")
    file_name: Optional[str] = Field(default=None, description="Filter by file name (partial match)")


class SearchGmailParams(BaseModel):
    """Parameters for search_gmail step."""
    search_query: str = Field(default="", description="Semantic search query for email content")
    sender: Optional[str] = Field(default=None, description="Filter by sender email or name")
    recipient: Optional[str] = Field(default=None, description="Filter by recipient email or name")
    subject: Optional[str] = Field(default=None, description="Filter by subject line")
    after_date: Optional[str] = Field(default=None, description="ISO date - only emails after this date")
    before_date: Optional[str] = Field(default=None, description="ISO date - only emails before this date")
    has_attachment: Optional[bool] = Field(default=None, description="Filter for emails with attachments")


class SearchCalendarParams(BaseModel):
    """Parameters for search_calendar step."""
    search_query: str = Field(default="", description="Semantic search query for event content")
    start_after: Optional[str] = Field(default=None, description="ISO datetime - only events starting after this time")
    start_before: Optional[str] = Field(default=None, description="ISO datetime - only events starting before this time")
    attendee: Optional[str] = Field(default=None, description="Filter by attendee email or name")


class GetFileParams(BaseModel):
    """Parameters for get_file step."""
    file_id: Optional[str] = Field(default=None, description="File ID to retrieve")
    file_name: Optional[str] = Field(default=None, description="File name to search for (if file_id not known)")


class GetEmailParams(BaseModel):
    """Parameters for get_email step."""
    email_id: Optional[str] = Field(default=None, description="Email ID to retrieve")


class GetEventParams(BaseModel):
    """Parameters for get_event step."""
    event_id: Optional[str] = Field(default=None, description="Event ID to retrieve")


class DraftEmailParams(BaseModel):
    """Parameters for draft_email step."""
    to: Optional[str] = Field(default=None, description="Recipient email address")
    to_name: Optional[str] = Field(default=None, description="Recipient name (for resolution if email not known)")
    subject: Optional[str] = Field(default=None, description="Email subject")
    message: str = Field(description="Message intent/content to compose")


class SendEmailParams(BaseModel):
    """Parameters for send_email step - only used after draft confirmation."""
    draft_id: Optional[str] = Field(default=None, description="Draft ID to send")
    to: Optional[str] = Field(default=None, description="Recipient email")
    subject: Optional[str] = Field(default=None, description="Email subject")
    body: Optional[str] = Field(default=None, description="Email body")


class CreateEventParams(BaseModel):
    """Parameters for create_event step."""
    title: str = Field(description="Event title")
    start_time: str = Field(description="ISO datetime for event start")
    end_time: str = Field(description="ISO datetime for event end")
    attendees: Optional[list[str]] = Field(default=None, description="List of attendee emails")
    description: Optional[str] = Field(default=None, description="Event description")
    location: Optional[str] = Field(default=None, description="Event location")


class UpdateEventParams(BaseModel):
    """Parameters for update_event step."""
    event_id: Optional[str] = Field(default=None, description="Event ID to update")
    title: Optional[str] = Field(default=None, description="New title")
    start_time: Optional[str] = Field(default=None, description="New start time")
    end_time: Optional[str] = Field(default=None, description="New end time")
    attendees: Optional[list[str]] = Field(default=None, description="Updated attendees")
    description: Optional[str] = Field(default=None, description="Updated description")


class DeleteEventParams(BaseModel):
    """Parameters for delete_event step."""
    event_id: Optional[str] = Field(default=None, description="Event ID to delete")


class ShareFileParams(BaseModel):
    """Parameters for share_file step."""
    file_id: Optional[str] = Field(default=None, description="File ID to share")
    file_name: Optional[str] = Field(default=None, description="File name (for resolution)")
    email: str = Field(description="Email address to share with")
    role: str = Field(default="reader", description="Permission role: reader, writer, commenter")


# Union type for all step params
StepParams = Union[
    SearchDriveParams, SearchGmailParams, SearchCalendarParams,
    GetFileParams, GetEmailParams, GetEventParams,
    DraftEmailParams, SendEmailParams,
    CreateEventParams, UpdateEventParams, DeleteEventParams,
    ShareFileParams
]


class ExecutionStep(BaseModel):
    """A single step to execute with its parameters."""
    step: StepType = Field(description="The step type to execute")
    params: dict[str, Any] = Field(default_factory=dict, description="Parameters for this step")
    depends_on: Optional[list[int]] = Field(default=None, description="Indices of steps this depends on (for chaining)")


class ParsedIntent(BaseModel):
    """Structured representation of a user's intent with typed step parameters."""

    services: list[ServiceType] = Field(
        default_factory=list,
        description="Services needed to fulfill the request"
    )
    operation: Literal["search", "create", "update", "delete", "send", "draft", "chat"] = Field(
        description="Primary operation type"
    )
    steps: list[ExecutionStep] = Field(
        default_factory=list,
        description="Ordered list of steps to execute with their parameters"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score of the classification"
    )

    # Response for chat operations (no action needed)
    response: Optional[str] = Field(
        default=None,
        description="Direct response for chat operations"
    )

    # Keep entities for backward compatibility during transition
    entities: dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted entities (deprecated, use step params instead)",
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
