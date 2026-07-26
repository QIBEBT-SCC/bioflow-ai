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
    auto_summary: bool = False
    summary_prompt: str | None = None


class WorkflowSimple(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    uid: str | uuid.UUID
    name: str
    description: str
    execution_scope: ExecutionScope


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
    auto_summary: bool
    summary_prompt: str | None


class WorkflowUpdate(BaseModel):
    name: str | None = None
    execution_scope: ExecutionScope | None = None
    workflow: WorkflowDefinition | None = None
    auto_summary: bool | None = None
    summary_prompt: str | None = None
