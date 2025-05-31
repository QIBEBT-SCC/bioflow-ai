from pydantic import BaseModel

from app.models.workflow import WorkflowType
from app.schemas.react_flow import WorkflowDefinition


class WorkflowCreateProp(BaseModel):
    name: str
    wf_type: WorkflowType
    workflow: WorkflowDefinition
    public: bool


class WorkflowUpdate(BaseModel):
    workflow: WorkflowDefinition
