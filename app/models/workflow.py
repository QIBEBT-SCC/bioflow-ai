from datetime import datetime, UTC
from enum import IntEnum
from pathlib import Path
from typing import Optional

from pydantic import UUID7, BaseModel
from sqlalchemy import Column, UUID, ForeignKey
from sqlmodel import SQLModel, Field, Relationship

from app.models.decorator.special_field import PydanticJson
from app.models.react_flow import WorkflowDefinition
from app.models.tools import ToolData
from app.models.user import User, UserPublic


# TABLE project_raw

# TABLE run_instance
# uid, owner, project_uid, templet(str), status, time..

# TABLE task
# uid, instance_uid, owner, task_type(name), web_node_id,other_args, status, local_path, result, time...


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

    uid: UUID7 = Field(sa_column=Column(UUID(as_uuid=True), primary_key=True))
    name: str
    owner_id: int = Field(foreign_key='users.id')
    wf_type: WorkflowType
    workflow: WorkflowDefinition = Field(sa_column=Column(PydanticJson(WorkflowDefinition)))
    public: bool = False

    owner: Optional['User'] = Relationship()


class WorkflowCreateProp(BaseModel):
    name: str
    wf_type: WorkflowType
    workflow: WorkflowDefinition
    public: bool


class WorkflowUpdate(BaseModel):
    workflow: WorkflowDefinition


class RunInstance(SQLModel, table=True):
    __tablename__ = 'run_instances'

    uid: UUID7 = Field(sa_column=Column(UUID(as_uuid=True), primary_key=True))
    project_id: Optional[int] = Field(default=None, foreign_key='projects.id')
    owner_id: int = Field(foreign_key='users.id')
    workflow: WorkflowDefinition = Field(sa_column=Column(PydanticJson(WorkflowDefinition)))

    status: Status

    create_time: Optional[datetime] = Field(default_factory=lambda: datetime.now(UTC))
    start_time: Optional[datetime] = Field(default_factory=lambda: datetime.now(UTC))
    end_time: Optional[datetime] = Field(default_factory=lambda: datetime.now(UTC))

    owner: Optional['User'] = Relationship()


class RunUpdate(SQLModel):
    workflow: Optional[WorkflowDefinition] = None
    status: Optional[Status] = None

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


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

    uid: UUID7 = Field(sa_column=Column(UUID(as_uuid=True), primary_key=True))
    instance_uid: UUID7 = Field(
        sa_column=Column(UUID(as_uuid=True), ForeignKey('run_instances.uid'))
    )
    tool_uid: UUID7 = Field(sa_column=Column(UUID(as_uuid=True), ForeignKey('tools.uid')))
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

    tool: Optional['ToolData'] = Relationship()
    owner: Optional['User'] = Relationship()


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


class TaskMonitor(SQLModel, table=True):
    __tablename__ = 'task_monitor_logs'

    id: Optional[int] = Field(default=None, primary_key=True)
    task_uid: UUID7 = Field(sa_column=Column(UUID(as_uuid=True), ForeignKey('task_instances.uid')))
    cpu_usage: float
    mem_usage: float
    mem_used: float
    io_in: float
    io_out: float
    time: datetime


class MonitorPublic(BaseModel):
    cpu_usage: float
    mem_usage: float
    mem_used: float
    io_in: float
    io_out: float
    time: datetime
