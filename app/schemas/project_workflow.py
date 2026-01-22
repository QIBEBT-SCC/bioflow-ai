import uuid
from datetime import datetime

from pydantic import BaseModel


class ProjectWorkflowAdd(BaseModel):
    """添加工作流到项目的请求"""
    workflow_uid: uuid.UUID


class ProjectWorkflowPublic(BaseModel):
    """项目中的工作流信息"""
    workflow_uid: uuid.UUID
    workflow_name: str
    import_time: datetime
    enabled: bool


class WorkflowRunRequest(BaseModel):
    """运行工作流的请求"""
    sample_uids: list[uuid.UUID]
    run_name_prefix: str | None = None


class WorkflowRunResult(BaseModel):
    """批量运行工作流的结果"""
    run_uids: list[uuid.UUID]
    count: int
