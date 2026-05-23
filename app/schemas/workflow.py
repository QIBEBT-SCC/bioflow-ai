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
    auto_summary: bool = False
    summary_prompt: Optional[str] = None


class WorkflowUpdate(BaseModel):
    name: Optional[str] = None
    execution_scope: Optional[ExecutionScope] = None
    workflow: Optional[WorkflowDefinition] = None
    auto_summary: Optional[bool] = None
    summary_prompt: Optional[str] = None
