"""Schemas and states owned by the V2 workflow runtime."""

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.react_flow import Edge, WorkflowDefinition
from app.schemas.workflow_config import ToolOutput


class RunStatus(StrEnum):
    """User-visible outcome of a V2 workflow run."""

    PENDING = 'pending'
    RUNNING = 'running'
    SUCCEEDED = 'succeeded'
    FAILED = 'failed'


class NodeRunStatus(StrEnum):
    """Persistent lifecycle of one executable graph node."""

    PENDING = 'pending'
    READY = 'ready'
    QUEUED = 'queued'
    RUNNING = 'running'
    SUCCEEDED = 'succeeded'
    FAILED = 'failed'
    BLOCKED = 'blocked'


class OutboxStatus(StrEnum):
    """Publication lifecycle for a ready node."""

    PENDING = 'pending'
    PUBLISHED = 'published'


TERMINAL_NODE_STATUSES = {
    NodeRunStatus.SUCCEEDED,
    NodeRunStatus.FAILED,
    NodeRunStatus.BLOCKED,
}

ACTIVE_NODE_STATUSES = {
    NodeRunStatus.PENDING,
    NodeRunStatus.READY,
    NodeRunStatus.QUEUED,
    NodeRunStatus.RUNNING,
}


class NodeRunStatistics(BaseModel):
    """Status counts for the current run generation."""

    total: int = 0
    pending: int = 0
    ready: int = 0
    queued: int = 0
    running: int = 0
    succeeded: int = 0
    failed: int = 0
    blocked: int = 0


class NodeRunData(BaseModel):
    """Runtime information injected into a workflow node."""

    uid: uuid.UUID
    status: NodeRunStatus
    create_time: datetime | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None


class WorkflowRunRecord(BaseModel):
    """Compact V2 run response."""

    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    generation: int
    name: str
    owner_id: int
    project_id: int | None
    workflow_uid: uuid.UUID | None
    sample_uid: uuid.UUID | None
    base_dir: str
    status: RunStatus
    settled: bool
    create_time: datetime | None
    start_time: datetime | None
    end_time: datetime | None


class WorkflowRunPublic(WorkflowRunRecord):
    """Run response with graph progress for the current generation."""

    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    node_statistics: NodeRunStatistics = Field(default_factory=NodeRunStatistics)


class NodeRunPublic(BaseModel):
    """Public state for a V2 node execution."""

    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    run_uid: uuid.UUID
    generation: int
    definition_node_id: str
    node_type: str
    name: str
    track_progress: bool
    status: NodeRunStatus
    base_dir: str | None
    commands: str | None
    error_message: str | None
    tool_output: ToolOutput | None
    system: str | None
    hostname: str | None
    worker_id: str | None
    create_time: datetime | None
    queued_at: datetime | None
    start_time: datetime | None
    heartbeat_at: datetime | None
    end_time: datetime | None


class PaginatedWorkflowRuns(BaseModel):
    """Paginated V2 runs."""

    total: int
    offset: int
    limit: int
    has_more: bool
    data: list[WorkflowRunPublic]


class WorkflowRunStatistics(BaseModel):
    """Aggregate counts for V2 workflow runs."""

    total: int = 0
    pending: int = 0
    running: int = 0
    succeeded: int = 0
    failed: int = 0


class WorkflowRunRequest(BaseModel):
    """Ad-hoc editor workflow execution request."""

    workflow: WorkflowDefinition
    template_name: str = 'unname'


class ProjectWorkflowRunRequest(BaseModel):
    """V2 project execution request; retries and auto-summary are intentionally absent."""

    model_config = ConfigDict(extra='forbid')

    sample_uids: list[uuid.UUID] | None = None
    run_name_prefix: str | None = None


class FileContentRequest(BaseModel):
    """Relative run output path."""

    path: str
