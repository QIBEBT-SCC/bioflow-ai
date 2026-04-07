import uuid
from datetime import UTC, datetime
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import uuid_extensions
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column
from sqlmodel import Field, Relationship, SQLModel

from app.models.decorator.special_field import PydanticJson
from app.schemas.react_flow import WorkflowDefinition

if TYPE_CHECKING:
    from app.models.resource import Sample
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


class ExecutionScope(IntEnum):
    SAMPLE_LEVEL = 0
    PROJECT_LEVEL = 1


class Workflow(SQLModel, table=True):
    __tablename__ = 'workflows'

    uid: uuid.UUID = Field(default_factory=uuid_extensions.uuid7, primary_key=True)
    name: str
    owner_id: int = Field(foreign_key='users.id')
    wf_type: WorkflowType
    execution_scope: ExecutionScope = Field(default=ExecutionScope.SAMPLE_LEVEL)
    workflow: WorkflowDefinition = Field(sa_column=Column(PydanticJson(WorkflowDefinition)))
    public: bool = False

    owner: Optional['User'] = Relationship()


class RunInstance(SQLModel, table=True):
    __tablename__ = 'run_instances'

    uid: uuid.UUID = Field(default_factory=uuid_extensions.uuid7, primary_key=True)
    name: str
    owner_id: int = Field(foreign_key='users.id')

    # 项目关联 (可选, 调试模式下为 None)
    project_id: Optional[int] = Field(default=None, foreign_key='projects.id')
    workflow_uid: Optional[uuid.UUID] = Field(default=None)
    sample_uid: Optional[uuid.UUID] = Field(default=None, foreign_key='samples.uid')

    base_dir: Optional[str] = Field(default=None)
    workflow: WorkflowDefinition = Field(sa_column=Column(PydanticJson(WorkflowDefinition)))

    status: Status
    create_time: Optional[datetime] = Field(default_factory=lambda: datetime.now(UTC))
    start_time: Optional[datetime] = Field(default=None)
    end_time: Optional[datetime] = Field(default=None)

    owner: Optional['User'] = Relationship()
    sample: Optional['Sample'] = Relationship()
    tasks: list['TaskInstance'] = Relationship(back_populates='run_instance')


class ToolOutput(BaseModel):
    model_config = ConfigDict(from_attributes=True)

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
    name: str
    instance_uid: uuid.UUID = Field(foreign_key='run_instances.uid')
    owner_id: int = Field(foreign_key='users.id')

    # 工具任务才有
    tool_uid: Optional[uuid.UUID] = Field(default=None, foreign_key='tools.uid')
    commands: Optional[str] = Field(default=None)

    base_dir: Optional[str] = Field(default=None)
    tool_output: Optional[ToolOutput] = Field(default=None, sa_column=Column(PydanticJson(ToolOutput)))

    system: Optional[str] = Field(default=None)
    hostname: Optional[str] = Field(default=None)

    status: Status
    create_time: Optional[datetime] = Field(default_factory=lambda: datetime.now(UTC))
    start_time: Optional[datetime] = Field(default=None)
    end_time: Optional[datetime] = Field(default=None)

    tool: Optional['ToolDefine'] = Relationship()
    owner: Optional['User'] = Relationship()
    run_instance: Optional['RunInstance'] = Relationship(back_populates='tasks')


class TaskMonitor(SQLModel, table=True):
    __tablename__ = 'task_monitor_logs'

    id: Optional[int] = Field(default=None, primary_key=True)
    task_uid: uuid.UUID = Field(default_factory=uuid_extensions.uuid7, foreign_key='task_instances.uid')
    cpu_usage: float
    mem_usage: float
    mem_used: float
    io_in: float
    io_out: float
    time: datetime
