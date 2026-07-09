import uuid
from datetime import datetime

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

    create_time: datetime | None
    start_time: datetime | None
    end_time: datetime | None


class TaskPublic(BaseModel):
    uid: UUID7
    name: str
    owner: UserPublic
    run_instance: RunInfo4Task
    tool: ToolInfo4Task | None = None
    commands: str | None
    tool_output: ToolOutput | None
    status: Status

    system: str | None
    hostname: str | None

    create_time: datetime | None
    start_time: datetime | None
    end_time: datetime | None


class TaskUpdate(BaseModel):
    commands: str | None = None
    tool_output: ToolOutput | None = None
    status: Status | None = None

    system: str | None = None
    hostname: str | None = None

    start_time: datetime | None = None
    end_time: datetime | None = None


class MonitorPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    cpu_usage: float
    mem_usage: float
    mem_used: float
    io_in: float
    io_out: float
    time: datetime
