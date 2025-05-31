from datetime import datetime
from typing import Optional

from pydantic import BaseModel, UUID7
from sqlmodel import SQLModel

from app.models.workflow import Status, ToolOutput
from app.schemas.user import UserPublic


class SimpleTaskPublic(BaseModel):
    uid: UUID7
    name: str
    owner: UserPublic
    status: Status

    create_time: Optional[datetime]
    start_time: Optional[datetime]
    end_time: Optional[datetime]


class TaskPublic(BaseModel):
    uid: UUID7
    instance_uid: UUID7
    name: str
    owner: UserPublic
    commands: Optional[str]
    tool_output: Optional[ToolOutput]
    status: Status

    system: Optional[str]
    hostname: Optional[str]

    create_time: Optional[datetime]
    start_time: Optional[datetime]
    end_time: Optional[datetime]


class TaskUpdate(SQLModel):
    commands: Optional[str] = None
    tool_output: Optional[ToolOutput] = None
    status: Optional[Status] = None

    system: Optional[str] = None
    hostname: Optional[str] = None

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class MonitorPublic(BaseModel):
    cpu_usage: float
    mem_usage: float
    mem_used: float
    io_in: float
    io_out: float
    time: datetime
