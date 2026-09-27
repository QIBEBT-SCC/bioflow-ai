"""Schemas and states owned by the durable workflow runtime."""

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.react_flow import Edge, WorkflowInterface
from app.schemas.workflow_config import ToolOutput


class RunStatus(StrEnum):
    """User-visible outcome of a workflow run."""

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
    run_uid: uuid.UUID | None = None
    status: NodeRunStatus
    create_time: datetime | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    item_statistics: dict[str, int] | None = None


class WorkflowRunRecord(BaseModel):
    """Compact workflow-run response."""

    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    generation: int
    name: str
    owner_id: int
    project_id: int | None
    workflow_uid: uuid.UUID | None
    sample_uid: uuid.UUID | None
    parent_node_uid: uuid.UUID | None = None
    parent_run_uid: uuid.UUID | None = None
    parent_run_name: str | None = None
    parent_node_name: str | None = None
    item_key: str | None = None
    item_name: str | None = None
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
    interface: WorkflowInterface | None = None
    node_statistics: NodeRunStatistics = Field(default_factory=NodeRunStatistics)


class NodeRunRecord(BaseModel):
    """Task-center projection for one current-generation node execution."""

    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    run_uid: uuid.UUID
    run_name: str
    generation: int
    definition_node_id: str
    node_type: str
    name: str
    status: NodeRunStatus
    owner_id: int
    owner_username: str
    project_id: int | None
    workflow_uid: uuid.UUID | None
    sample_uid: uuid.UUID | None
    parent_node_uid: uuid.UUID | None = None
    item_name: str | None = None
    create_time: datetime | None
    start_time: datetime | None
    end_time: datetime | None


class NodeRunPublic(NodeRunRecord):
    """Detailed public state for one current-generation node execution."""

    track_progress: bool
    tool_uid: uuid.UUID | None
    tool_name: str | None
    tool_description: str | None
    base_dir: str | None
    commands: str | None
    error_message: str | None
    tool_output: ToolOutput | None
    system: str | None
    hostname: str | None
    worker_id: str | None
    queued_at: datetime | None
    heartbeat_at: datetime | None


class PaginatedNodeRuns(BaseModel):
    """Paginated current-generation node executions."""

    total: int
    offset: int
    limit: int
    has_more: bool
    data: list[NodeRunRecord]


class PaginatedWorkflowRuns(BaseModel):
    """Paginated workflow runs."""

    total: int
    offset: int
    limit: int
    has_more: bool
    data: list[WorkflowRunPublic]


class WorkflowRunStatistics(BaseModel):
    """Aggregate counts for workflow runs."""

    total: int = 0
    pending: int = 0
    running: int = 0
    succeeded: int = 0
    failed: int = 0


class ProjectWorkflowRunRequest(BaseModel):
    """Project execution request; retries and auto-summary are intentionally absent."""

    model_config = ConfigDict(extra='forbid')

    sample_uids: list[uuid.UUID] | None = None
    run_name_prefix: str | None = None


class FileContentRequest(BaseModel):
    """Relative run output path."""

    path: str


class FilePreviewMetaRequest(BaseModel):
    """Metadata lookup for one file in the active run generation."""

    model_config = ConfigDict(extra='forbid')

    path: str = Field(min_length=1, max_length=4096)
    generation: int = Field(ge=1)
    query: str = Field(default='', max_length=100)


class FilePreviewRegionRequest(BaseModel):
    """A 0-based, half-open genomic viewport."""

    model_config = ConfigDict(extra='forbid')

    path: str = Field(min_length=1, max_length=4096)
    generation: int = Field(ge=1)
    chrom: str = Field(min_length=1, max_length=255)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    bins: int = Field(default=512, ge=1, le=512)
    chrom2: str | None = Field(default=None, min_length=1, max_length=255)
    start2: int | None = Field(default=None, ge=0)
    end2: int | None = Field(default=None, gt=0)
    resolution: int | None = Field(default=None, gt=0)


class FileTablePreviewMetaRequest(BaseModel):
    """Metadata lookup for a CSV or TSV in the active run generation."""

    model_config = ConfigDict(extra='forbid')

    path: str = Field(min_length=1, max_length=4096)
    generation: int = Field(ge=1)


class TablePreviewSort(BaseModel):
    """One server-side table sort key."""

    model_config = ConfigDict(extra='forbid')

    column: str = Field(min_length=1, max_length=1024)
    direction: Literal['asc', 'desc']


class TablePreviewFilter(BaseModel):
    """One bounded, type-aware table filter."""

    model_config = ConfigDict(extra='forbid')

    column: str = Field(min_length=1, max_length=1024)
    operator: Literal['contains', 'equals', 'not_equals', 'gt', 'gte', 'lt', 'lte', 'is_null', 'is_not_null']
    value: str | None = Field(default=None, max_length=1024)


class FileTablePreviewQueryRequest(FileTablePreviewMetaRequest):
    """A projected, paginated table-preview query."""

    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1)
    columns: list[str] = Field(min_length=1)
    sort: TablePreviewSort | None = None
    filters: list[TablePreviewFilter] = Field(default_factory=list)


class FileTablePreviewProfileRequest(FileTablePreviewMetaRequest):
    """On-demand summary for one table column."""

    column: str = Field(min_length=1, max_length=1024)
    filters: list[TablePreviewFilter] = Field(default_factory=list)


class FileVcfPreviewMetaRequest(BaseModel):
    """Metadata lookup for a VCF in the active run generation."""

    model_config = ConfigDict(extra='forbid')

    path: str = Field(min_length=1, max_length=4096)
    generation: int = Field(ge=1)


class FileVcfPreviewQueryRequest(FileVcfPreviewMetaRequest):
    """A filtered page of VCF records in file order."""

    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1)
    contig: str | None = Field(default=None, min_length=1, max_length=1024)
    variant_type: Literal['all', 'snp', 'mnv', 'indel', 'sv', 'mixed', 'other'] = 'all'
    filter_status: Literal['all', 'pass', 'filtered', 'unfiltered'] = 'all'
    search: str = Field(default='', max_length=1024)
    min_qual: float | None = None


class FileVcfPreviewDetailRequest(FileVcfPreviewMetaRequest):
    """One VCF record selected by its zero-based body index."""

    record_index: int = Field(ge=0)


class OutputFileItem(BaseModel):
    """One output file in a workflow-run file tree."""

    type: Literal['file'] = 'file'
    path: str
    name: str


class OutputFolderItem(BaseModel):
    """One output folder in a workflow-run file tree."""

    type: Literal['folder'] = 'folder'
    path: str
    name: str
    children: list['OutputFileNode'] = Field(default_factory=list)


OutputFileNode = Annotated[OutputFileItem | OutputFolderItem, Field(discriminator='type')]
OutputFolderItem.model_rebuild()


class NodeRunUpdate(BaseModel):
    """Executor-owned metadata that may be persisted while a node is running."""

    commands: str | None = None
    system: str | None = None
    hostname: str | None = None


class MonitorPublic(BaseModel):
    """Transient container resource sample exposed by the node-run API."""

    model_config = ConfigDict(from_attributes=True)

    cpu_usage: float
    mem_usage: float
    mem_used: float
    io_in: float
    io_out: float
    time: datetime
