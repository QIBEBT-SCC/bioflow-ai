import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AgentName(StrEnum):
    WORKFLOW_BUILDER = 'workflow-builder'
    WORKFLOW_ASSISTANT = 'workflow-assistant'
    WORKFLOW_FIXER = 'workflow-fixer'
    SAMPLE_MANAGER = 'sample-manager'
    TOOL_GENERATOR = 'tool-generator'


class UserLanguage(StrEnum):
    ZH = 'zh'
    EN = 'en'


class AgentRunStatus(StrEnum):
    QUEUED = 'queued'
    RUNNING = 'running'
    WAITING_INPUT = 'waiting_input'
    CANCEL_REQUESTED = 'cancel_requested'
    CANCELLED = 'cancelled'
    COMPLETED = 'completed'
    FAILED = 'failed'


ACTIVE_AGENT_RUN_STATUSES = frozenset(
    {
        AgentRunStatus.QUEUED,
        AgentRunStatus.RUNNING,
        AgentRunStatus.WAITING_INPUT,
        AgentRunStatus.CANCEL_REQUESTED,
    }
)
TERMINAL_AGENT_RUN_STATUSES = frozenset({AgentRunStatus.CANCELLED, AgentRunStatus.COMPLETED, AgentRunStatus.FAILED})


class AgentEventType(StrEnum):
    QUEUED = 'run.queued'
    STARTED = 'run.started'
    PROGRESS = 'run.progress'
    INTERRUPTED = 'run.interrupted'
    RESOURCE_CHANGED = 'resource.changed'
    CANCEL_REQUESTED = 'run.cancel_requested'
    CANCELLED = 'run.cancelled'
    COMPLETED = 'run.completed'
    FAILED = 'run.failed'


class TextMessagePart(BaseModel):
    type: Literal['text'] = 'text'
    text: str


class AgentMessagePublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    session_uid: uuid.UUID
    run_uid: uuid.UUID
    role: Literal['user', 'assistant']
    parts: list[TextMessagePart]
    create_time: datetime


class AgentRunPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    session_uid: uuid.UUID
    project_id: int | None
    agent_name: AgentName
    status: AgentRunStatus
    interrupt_payload: dict[str, Any] | None
    result_payload: dict[str, Any] | None
    error_code: str | None
    error_message: str | None
    cancel_requested: bool
    heartbeat_at: datetime | None
    retry_of_uid: uuid.UUID | None
    create_time: datetime
    start_time: datetime | None
    end_time: datetime | None
    update_time: datetime


class AgentSessionPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    owner_id: int
    project_id: int | None
    title: str
    create_time: datetime
    update_time: datetime
    latest_run: AgentRunPublic | None = None


class AgentSessionPage(BaseModel):
    limit: int
    has_more: bool
    next_cursor: datetime | None
    data: list[AgentSessionPublic]


class AgentSessionCreate(BaseModel):
    project_id: int | None = None
    title: str = Field(default='', max_length=200)


class AgentSessionUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class AgentLanguagePreference(BaseModel):
    language: UserLanguage | None = None

    @field_validator('language', mode='before')
    @classmethod
    def normalize_empty_language(cls, value: Any) -> Any:
        """Treat an omitted or blank language as no language preference."""
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value


class AgentRunCreate(AgentLanguagePreference):
    agent_name: AgentName
    text: str = Field(min_length=1, max_length=100_000)
    source_run_uid: uuid.UUID | None = None

    @model_validator(mode='after')
    def validate_fix_request(self) -> 'AgentRunCreate':
        """Require a concrete message when starting workflow repair."""
        if self.agent_name == AgentName.WORKFLOW_FIXER and not self.text.strip():
            message = 'workflow-fixer requires a non-empty request'
            raise ValueError(message)
        return self


class AgentRunResume(AgentLanguagePreference):
    approved: bool = False
    feedback: str | None = Field(default=None, max_length=100_000)

    @model_validator(mode='after')
    def validate_resume_input(self) -> 'AgentRunResume':
        """Require either explicit approval or non-empty feedback."""
        if not self.approved and not (self.feedback and self.feedback.strip()):
            message = 'feedback is required when approved is false'
            raise ValueError(message)
        return self


class AgentRunRetry(AgentLanguagePreference):
    pass


class AgentEventPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run_uid: uuid.UUID
    event_type: AgentEventType
    payload: dict[str, Any]
    create_time: datetime


class AgentRunDispatchResponse(BaseModel):
    run: AgentRunPublic
