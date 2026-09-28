import uuid

from pydantic import BaseModel, ConfigDict

from app.schemas.react_flow import WorkflowDefinition
from app.schemas.workflow_config import ExecutionScope, WorkflowType


class WorkflowCreateProp(BaseModel):
    name: str
    description: str
    wf_type: WorkflowType
    execution_scope: ExecutionScope = ExecutionScope.SAMPLE_LEVEL
    workflow: WorkflowDefinition
    public: bool


class WorkflowPortSimple(BaseModel):
    id: str
    name: str
    description: str = ''


class WorkflowSimple(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    uid: str | uuid.UUID
    name: str
    description: str
    execution_scope: ExecutionScope
    wf_type: WorkflowType
    inputs: list[WorkflowPortSimple]
    outputs: list[WorkflowPortSimple]


class WorkflowSearchResult(WorkflowSimple):
    relevance_score: float


class WorkflowPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    name: str
    owner_id: int
    description: str
    wf_type: WorkflowType
    execution_scope: ExecutionScope
    workflow: WorkflowDefinition
    public: bool


class WorkflowUpdate(BaseModel):
    description: str | None = None
    public: bool | None = None
    name: str | None = None
    execution_scope: ExecutionScope | None = None
    workflow: WorkflowDefinition | None = None
