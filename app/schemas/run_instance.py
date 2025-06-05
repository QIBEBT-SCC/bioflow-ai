from datetime import datetime
from typing import Optional

from pydantic import BaseModel, UUID7

from app.models.workflow import Status
from app.schemas.react_flow import WorkflowDefinition
from app.schemas.user import UserPublic


class RunUpdate(BaseModel):
    workflow: Optional[WorkflowDefinition] = None
    status: Optional[Status] = None

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class TaskStat(BaseModel):
    total: int
    waiting: int
    running: int
    success: int
    error: int


class RunPublic(BaseModel):
    uid: UUID7
    name: str
    owner: UserPublic

    status: Status
    task_stats: TaskStat

    create_time: datetime
    start_time: datetime
    end_time: datetime


class RunInfo4Task(BaseModel):
    uid: UUID7
    name: str
