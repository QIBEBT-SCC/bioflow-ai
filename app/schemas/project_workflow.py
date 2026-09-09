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


class WorkflowRunResult(BaseModel):
    run_uids: list[uuid.UUID]
    count: int
