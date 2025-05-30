from datetime import datetime, UTC
from enum import IntEnum
from typing import Optional

from pydantic import UUID7, BaseModel
from sqlalchemy import Column, UUID, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import SQLModel, Field, Relationship

from app.models.decorator.special_field import PydanticJson
from app.models.react_flow import WorkflowDefinition
from app.models.user import User


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


class TaskInstance(SQLModel, table=True):
    __tablename__ = 'task_instances'

    uid: UUID7 = Field(sa_column=Column(UUID(as_uuid=True), primary_key=True))
    instance_uid: UUID7 = Field(
        sa_column=Column(UUID(as_uuid=True), ForeignKey('run_instances.uid'))
    )
    owner_id: int = Field(foreign_key='users.id')
    name: str
    commands: Optional[str] = Field(default=None)
    result: Optional[dict] = Field(sa_type=JSONB)
    status: Status

    system: Optional[str]
    hostname: Optional[str]
    owner: Optional['User'] = Relationship()

    create_time: Optional[datetime] = Field(default_factory=lambda: datetime.now(UTC))
    start_time: Optional[datetime] = Field(default=None)
    end_time: Optional[datetime] = Field(default=None)


class TaskUpdate(SQLModel):
    commands: Optional[str] = None
    result: Optional[dict] = None
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
