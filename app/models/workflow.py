import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import ForeignKey, true
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base
from app.models.decorator.special_field import IntEnumType, PydanticJson
from app.models.project import Project
from app.models.resource import Sample
from app.models.tools import ToolDefine
from app.models.user import User
from app.schemas.react_flow import WorkflowDefinition
from app.schemas.workflow_config import ExecutionScope, Status, ToolOutput, WorkflowType
from app.utils.system import random_uid, utc_now


class Workflow(Base):
    __tablename__ = 'workflows'

    uid: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=random_uid)
    name: Mapped[str]
    owner_id: Mapped[int] = mapped_column(ForeignKey(User.id))
    description: Mapped[str]
    wf_type: Mapped[WorkflowType] = mapped_column(IntEnumType(WorkflowType), nullable=False)
    execution_scope: Mapped[ExecutionScope] = mapped_column(
        IntEnumType(ExecutionScope),
        default=ExecutionScope.SAMPLE_LEVEL,
        nullable=False,
    )
    workflow: Mapped[WorkflowDefinition] = mapped_column(PydanticJson(WorkflowDefinition), nullable=True)
    public: Mapped[bool] = mapped_column(default=False)
    auto_summary: Mapped[bool] = mapped_column(default=False)
    summary_prompt: Mapped[str | None] = mapped_column(default=None)

    owner: Mapped[Optional[User]] = relationship(User)


class ProjectWorkflow(Base):
    """Project-to-workflow template association."""

    __tablename__ = 'project_workflows'

    project_id: Mapped[int] = mapped_column(
        ForeignKey(Project.id, ondelete='CASCADE'),
        primary_key=True,
    )
    workflow_uid: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(Workflow.uid, ondelete='CASCADE'),
        primary_key=True,
    )
    import_time: Mapped[datetime] = mapped_column(default=utc_now)
    enabled: Mapped[bool] = mapped_column(default=True)

    project: Mapped[Optional[Project]] = relationship(Project)
    workflow: Mapped[Optional[Workflow]] = relationship(Workflow)


class RunInstance(Base):
    __tablename__ = 'run_instances'

    uid: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=random_uid)
    name: Mapped[str]
    owner_id: Mapped[int] = mapped_column(ForeignKey(User.id))

    # 项目关联 (可选, 调试模式下为 None)
    project_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(Project.id, ondelete='CASCADE'),
        default=None,
    )
    workflow_uid: Mapped[Optional[uuid.UUID]] = mapped_column(default=None)
    sample_uid: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey(Sample.uid, ondelete='SET NULL'),
        default=None,
    )

    base_dir: Mapped[Optional[str]] = mapped_column(default=None)
    workflow: Mapped[WorkflowDefinition] = mapped_column(PydanticJson(WorkflowDefinition), nullable=True)

    status: Mapped[Status] = mapped_column(IntEnumType(Status), nullable=False)
    create_time: Mapped[Optional[datetime]] = mapped_column(default=utc_now)
    start_time: Mapped[Optional[datetime]] = mapped_column(default=None)
    end_time: Mapped[Optional[datetime]] = mapped_column(default=None)

    owner: Mapped[Optional[User]] = relationship(User)
    sample: Mapped[Optional[Sample]] = relationship(Sample)
    tasks: Mapped[list['TaskInstance']] = relationship(
        back_populates='run_instance',
        cascade='all, delete-orphan',
        passive_deletes=True,
    )


class TaskInstance(Base):
    __tablename__ = 'task_instances'

    uid: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=random_uid)
    name: Mapped[str]
    instance_uid: Mapped[uuid.UUID] = mapped_column(ForeignKey(RunInstance.uid, ondelete='CASCADE'))
    owner_id: Mapped[int] = mapped_column(ForeignKey(User.id))
    track_progress: Mapped[bool] = mapped_column(default=True, server_default=true(), nullable=False)

    # 工具任务才有
    tool_uid: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey(ToolDefine.uid),
        default=None,
    )
    commands: Mapped[Optional[str]] = mapped_column(default=None)

    base_dir: Mapped[Optional[str]] = mapped_column(default=None)
    tool_output: Mapped[Optional[ToolOutput]] = mapped_column(PydanticJson(ToolOutput), default=None, nullable=True)

    system: Mapped[Optional[str]] = mapped_column(default=None)
    hostname: Mapped[Optional[str]] = mapped_column(default=None)

    status: Mapped[Status] = mapped_column(IntEnumType(Status), nullable=False)
    create_time: Mapped[Optional[datetime]] = mapped_column(default=utc_now)
    start_time: Mapped[Optional[datetime]] = mapped_column(default=None)
    end_time: Mapped[Optional[datetime]] = mapped_column(default=None)

    tool: Mapped[Optional[ToolDefine]] = relationship(ToolDefine)
    owner: Mapped[Optional[User]] = relationship(User)
    run_instance: Mapped[Optional[RunInstance]] = relationship(back_populates='tasks')


class TaskMonitor(Base):
    __tablename__ = 'task_monitor_logs'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    task_uid: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(TaskInstance.uid, ondelete='CASCADE'),
        default=random_uid,
    )
    cpu_usage: Mapped[float]
    mem_usage: Mapped[float]
    mem_used: Mapped[float]
    io_in: Mapped[float]
    io_out: Mapped[float]
    time: Mapped[datetime]
