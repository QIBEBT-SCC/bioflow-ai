import uuid
from datetime import datetime

from pydantic import BaseModel

from app.schemas.workflow_config import ExecutionScope


class ProjectWorkflowAdd(BaseModel):
    workflow_uid: uuid.UUID


class ProjectWorkflowPublic(BaseModel):
    workflow_uid: uuid.UUID
    workflow_name: str
    execution_scope: ExecutionScope
    import_time: datetime
    enabled: bool
    auto_summary: bool


class WorkflowRunRequest(BaseModel):
    sample_uids: list[uuid.UUID] | None = None
    run_name_prefix: str | None = None
    auto_summary: bool | None = None  # None = 使用 workflow 默认值


class WorkflowRunResult(BaseModel):
    run_uids: list[uuid.UUID]
    count: int
