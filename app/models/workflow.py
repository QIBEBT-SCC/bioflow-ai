import uuid
from datetime import datetime, UTC
from enum import IntEnum
from pathlib import Path
from typing import Optional

import uuid_extensions
from pydantic import BaseModel
from sqlalchemy import Column, UUID, ForeignKey
from sqlmodel import SQLModel, Field, Relationship

from app.models.decorator.special_field import PydanticJson
from app.schemas.react_flow import WorkflowDefinition
from app.models.tools import ToolDefine
from app.models.user import User


class Status(IntEnum):
    WAITING = 0
    RUNNING = 1
    ERROR = 2
    SUCCESS = 3


class WorkflowType(IntEnum):
    SUBMODULE = 0
    TEMPLATE = 1


class Workflow(SQLModel, table=True):
    __tablename__ = 'workflows'

    uid: uuid.UUID = Field(default_factory=uuid_extensions.uuid7, primary_key=True)
    name: str
    owner_id: int = Field(foreign_key='users.id')
    wf_type: WorkflowType
    workflow: WorkflowDefinition = Field(sa_column=Column(PydanticJson(WorkflowDefinition)))
    public: bool = False

    owner: Optional['User'] = Relationship()


class RunInstance(SQLModel, table=True):
    __tablename__ = 'run_instances'

    uid: uuid.UUID = Field(default_factory=uuid_extensions.uuid7, primary_key=True)
    name: str
    project_id: Optional[int] = Field(default=None, foreign_key='projects.id')
    owner_id: int = Field(foreign_key='users.id')
    workflow: WorkflowDefinition = Field(sa_column=Column(PydanticJson(WorkflowDefinition)))

    status: Status

    create_time: Optional[datetime] = Field(default_factory=lambda: datetime.now(UTC))
    start_time: Optional[datetime] = Field(default=None)
    end_time: Optional[datetime] = Field(default=None)

    owner: Optional['User'] = Relationship()
    tasks: list['TaskInstance'] = Relationship(back_populates='run_instance')


class ToolOutput(BaseModel):
    result: dict[str, Path] = Field(default_factory=dict)
    log: Optional[Path] = None
    reports: dict[str, Path] = Field(default_factory=dict)

    def all_files(self) -> dict[str, Path]:
        files = (self.result | self.reports).copy()
        if self.log:
            files['log'] = self.log
        return files

    def all_exist(self):
        return all(file.exists() for file in self.all_files().values())


class TaskInstance(SQLModel, table=True):
    __tablename__ = 'task_instances'

    uid: uuid.UUID = Field(default_factory=uuid_extensions.uuid7, primary_key=True)
    instance_uid: uuid.UUID = Field(sa_column=Column(UUID(as_uuid=True), ForeignKey('run_instances.uid')))
    tool_uid: uuid.UUID = Field(foreign_key='tools.uid')
    owner_id: int = Field(foreign_key='users.id')
    name: str
    commands: Optional[str] = Field(default=None)
    tool_output: Optional[ToolOutput] = Field(sa_column=Column(PydanticJson(ToolOutput)))
    status: Status

    system: Optional[str]
    hostname: Optional[str]

    create_time: Optional[datetime] = Field(default_factory=lambda: datetime.now(UTC))
    start_time: Optional[datetime] = Field(default=None)
    end_time: Optional[datetime] = Field(default=None)

    tool: Optional['ToolDefine'] = Relationship()
    owner: Optional['User'] = Relationship()
    run_instance: Optional['RunInstance'] = Relationship(back_populates='tasks')


class TaskMonitor(SQLModel, table=True):
    __tablename__ = 'task_monitor_logs'

    id: Optional[int] = Field(default=None, primary_key=True)
    task_uid: uuid.UUID = Field(sa_column=Column(UUID(as_uuid=True), ForeignKey('task_instances.uid')))
    cpu_usage: float
    mem_usage: float
    mem_used: float
    io_in: float
    io_out: float
    time: datetime
