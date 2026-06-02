import uuid

from pydantic import BaseModel, ConfigDict

from app.models.workflow import ExecutionScope, WorkflowType
from app.schemas.react_flow import WorkflowDefinition


class WorkflowCreateProp(BaseModel):
    name: str
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


class WorkflowUpdate(BaseModel):
    name: str | None = None
    execution_scope: ExecutionScope | None = None
    workflow: WorkflowDefinition | None = None
    auto_summary: bool | None = None
    summary_prompt: str | None = None
