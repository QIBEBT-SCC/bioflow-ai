import uuid
from datetime import datetime
from typing import Optional

from pydantic import UUID7, BaseModel, ConfigDict

from app.models.workflow import Status, ToolOutput
from app.schemas.user import UserPublic


class RunInfo4Task(BaseModel):
    uid: UUID7
    name: str


class ToolInfo4Task(BaseModel):
    uid: uuid.UUID
    name: str
    description: str


class SimpleTaskPublic(BaseModel):
    uid: UUID7
    name: str
    owner: UserPublic
    run_instance: RunInfo4Task

    status: Status

    create_time: Optional[datetime]
    start_time: Optional[datetime]
    end_time: Optional[datetime]


class TaskPublic(BaseModel):
    uid: UUID7
    name: str
    owner: UserPublic
    run_instance: RunInfo4Task
    tool: ToolInfo4Task
    commands: Optional[str]
    tool_output: Optional[ToolOutput]
    status: Status

    system: Optional[str]
    hostname: Optional[str]

    create_time: Optional[datetime]
    start_time: Optional[datetime]
    end_time: Optional[datetime]


class TaskUpdate(BaseModel):
    commands: Optional[str] = None
    tool_output: Optional[ToolOutput] = None
    status: Optional[Status] = None

    system: Optional[str] = None
    hostname: Optional[str] = None

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class MonitorPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    cpu_usage: float
    mem_usage: float
    mem_used: float
    io_in: float
    io_out: float
    time: datetime
