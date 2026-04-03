from typing import Optional

from pydantic import BaseModel

from app.models.workflow import ExecutionScope, WorkflowType
from app.schemas.react_flow import WorkflowDefinition


class WorkflowCreateProp(BaseModel):
    name: str
    wf_type: WorkflowType
    execution_scope: ExecutionScope = ExecutionScope.SAMPLE_LEVEL
    workflow: WorkflowDefinition
    public: bool


class WorkflowUpdate(BaseModel):
    name: Optional[str] = None
    execution_scope: Optional[ExecutionScope] = None
    workflow: Optional[WorkflowDefinition] = None
