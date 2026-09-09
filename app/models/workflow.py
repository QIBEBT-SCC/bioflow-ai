import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, Index, UniqueConstraint, false, text, true
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base
from app.models.decorator.special_field import IntEnumType, PydanticJson
from app.models.project import Project
from app.models.resource import Sample
from app.models.tools import ToolDefine
from app.models.user import User
from app.schemas.react_flow import WorkflowDefinition
from app.schemas.workflow_config import ExecutionScope, ToolOutput, WorkflowType
from app.schemas.workflow_run import NodeRunStatus, OutboxStatus, RunStatus
from app.utils.system import random_uid, utc_now


class Workflow(Base):
    """Reusable workflow definition edited by users and agents."""

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
    workflow: Mapped[WorkflowDefinition] = mapped_column(PydanticJson(WorkflowDefinition), nullable=False)
    public: Mapped[bool] = mapped_column(default=False)

    owner: Mapped[User] = relationship(User)


class ProjectWorkflow(Base):
    """Project-to-workflow template association."""

    __tablename__ = 'project_workflows'

    project_id: Mapped[int] = mapped_column(ForeignKey(Project.id, ondelete='CASCADE'), primary_key=True)
    workflow_uid: Mapped[uuid.UUID] = mapped_column(ForeignKey(Workflow.uid, ondelete='CASCADE'), primary_key=True)
    import_time: Mapped[datetime] = mapped_column(default=utc_now)
    enabled: Mapped[bool] = mapped_column(default=True)

    project: Mapped[Project] = relationship(Project)
    workflow: Mapped[Workflow] = relationship(Workflow)


class WorkflowRun(Base):
    """One reusable project/sample run identity and its active generation."""

    __tablename__ = 'workflow_runs'
    __table_args__ = (
        Index('ix_workflow_runs_owner_created', 'owner_id', 'create_time'),
        Index(
            'uq_workflow_runs_sample',
            'project_id',
            'workflow_uid',
            'sample_uid',
            unique=True,
            postgresql_where=text('sample_uid IS NOT NULL'),
        ),
        Index(
            'uq_workflow_runs_project',
            'project_id',
            'workflow_uid',
            unique=True,
            postgresql_where=text('sample_uid IS NULL AND project_id IS NOT NULL AND workflow_uid IS NOT NULL'),
        ),
    )

    uid: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=random_uid)
    generation: Mapped[int] = mapped_column(default=1, nullable=False)
    name: Mapped[str]
    owner_id: Mapped[int] = mapped_column(ForeignKey(User.id))
    project_id: Mapped[int | None] = mapped_column(ForeignKey(Project.id, ondelete='CASCADE'), default=None)
    workflow_uid: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(Workflow.uid, ondelete='SET NULL'),
        default=None,
    )
    sample_uid: Mapped[uuid.UUID | None] = mapped_column(ForeignKey(Sample.uid, ondelete='SET NULL'), default=None)
    base_dir: Mapped[str]
    workflow: Mapped[WorkflowDefinition] = mapped_column(PydanticJson(WorkflowDefinition), nullable=False)
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus, native_enum=False, length=16),
        default=RunStatus.PENDING,
        nullable=False,
    )
    settled: Mapped[bool] = mapped_column(default=False, server_default=false(), nullable=False)
    create_time: Mapped[datetime] = mapped_column(default=utc_now)
    start_time: Mapped[datetime | None] = mapped_column(default=None)
    end_time: Mapped[datetime | None] = mapped_column(default=None)

    node_runs: Mapped[list['NodeRun']] = relationship(
        back_populates='run',
        cascade='all, delete-orphan',
        passive_deletes=True,
    )


class NodeRun(Base):
    """Execution state for one executable node in one run generation."""

    __tablename__ = 'workflow_node_runs'
    __table_args__ = (
        UniqueConstraint('run_uid', 'generation', 'definition_node_id', name='uq_node_run_definition'),
        Index('ix_node_runs_run_generation_status', 'run_uid', 'generation', 'status'),
        Index('ix_node_runs_heartbeat', 'status', 'heartbeat_at'),
    )

    uid: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=random_uid)
    run_uid: Mapped[uuid.UUID] = mapped_column(ForeignKey(WorkflowRun.uid, ondelete='CASCADE'), nullable=False)
    generation: Mapped[int]
    definition_node_id: Mapped[str]
    node_type: Mapped[str]
    name: Mapped[str]
    position_index: Mapped[int]
    track_progress: Mapped[bool] = mapped_column(default=True, server_default=true(), nullable=False)
    prepared_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    status: Mapped[NodeRunStatus] = mapped_column(
        Enum(NodeRunStatus, native_enum=False, length=16),
        default=NodeRunStatus.PENDING,
        nullable=False,
    )
    tool_uid: Mapped[uuid.UUID | None] = mapped_column(ForeignKey(ToolDefine.uid), default=None)
    base_dir: Mapped[str | None] = mapped_column(default=None)
    commands: Mapped[str | None] = mapped_column(default=None)
    error_message: Mapped[str | None] = mapped_column(default=None)
    tool_output: Mapped[ToolOutput | None] = mapped_column(PydanticJson(ToolOutput), default=None, nullable=True)
    system: Mapped[str | None] = mapped_column(default=None)
    hostname: Mapped[str | None] = mapped_column(default=None)
    worker_id: Mapped[str | None] = mapped_column(default=None)
    create_time: Mapped[datetime] = mapped_column(default=utc_now)
    queued_at: Mapped[datetime | None] = mapped_column(default=None)
    start_time: Mapped[datetime | None] = mapped_column(default=None)
    heartbeat_at: Mapped[datetime | None] = mapped_column(default=None)
    end_time: Mapped[datetime | None] = mapped_column(default=None)

    run: Mapped[WorkflowRun] = relationship(back_populates='node_runs')


class NodeDependency(Base):
    """One data-binding edge between two node runs."""

    __tablename__ = 'workflow_node_dependencies'
    __table_args__ = (
        UniqueConstraint(
            'run_uid',
            'generation',
            'source_node_uid',
            'source_handle',
            'target_node_uid',
            'target_handle',
            name='uq_node_dependency_binding',
        ),
        Index('ix_node_dependencies_target', 'target_node_uid'),
        Index('ix_node_dependencies_source', 'source_node_uid'),
    )

    uid: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=random_uid)
    run_uid: Mapped[uuid.UUID] = mapped_column(ForeignKey(WorkflowRun.uid, ondelete='CASCADE'), nullable=False)
    generation: Mapped[int]
    source_node_uid: Mapped[uuid.UUID] = mapped_column(ForeignKey(NodeRun.uid, ondelete='CASCADE'), nullable=False)
    target_node_uid: Mapped[uuid.UUID] = mapped_column(ForeignKey(NodeRun.uid, ondelete='CASCADE'), nullable=False)
    source_handle: Mapped[str]
    target_handle: Mapped[str]


class WorkflowOutbox(Base):
    """Durable intent to publish one ready node run to Celery."""

    __tablename__ = 'workflow_outbox'
    __table_args__ = (
        UniqueConstraint('node_run_uid', name='uq_workflow_outbox_node_run'),
        Index('ix_workflow_outbox_status_created', 'status', 'create_time'),
    )

    uid: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=random_uid)
    run_uid: Mapped[uuid.UUID] = mapped_column(ForeignKey(WorkflowRun.uid, ondelete='CASCADE'), nullable=False)
    node_run_uid: Mapped[uuid.UUID] = mapped_column(ForeignKey(NodeRun.uid, ondelete='CASCADE'), nullable=False)
    generation: Mapped[int]
    status: Mapped[OutboxStatus] = mapped_column(
        Enum(OutboxStatus, native_enum=False, length=16),
        default=OutboxStatus.PENDING,
        nullable=False,
    )
    publish_attempts: Mapped[int] = mapped_column(default=0, nullable=False)
    claim_token: Mapped[uuid.UUID | None] = mapped_column(default=None)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    last_error: Mapped[str | None] = mapped_column(default=None)
    create_time: Mapped[datetime] = mapped_column(default=utc_now)
    published_at: Mapped[datetime | None] = mapped_column(default=None)
