"""Database models for the durable V2 DAG runtime."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, Index, UniqueConstraint, false, text, true
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base
from app.models.decorator.special_field import PydanticJson
from app.schemas.react_flow import WorkflowDefinition
from app.schemas.workflow_config import ToolOutput
from app.utils.system import random_uid, utc_now
from app.v2.schemas.workflow import NodeRunStatus, OutboxStatus, RunStatus


class WorkflowRunV2(Base):
    """One reusable project/sample run identity and its active generation."""

    __tablename__ = 'workflow_runs_v2'
    __table_args__ = (
        Index('ix_workflow_runs_v2_owner_created', 'owner_id', 'create_time'),
        Index(
            'uq_workflow_runs_v2_sample',
            'project_id',
            'workflow_uid',
            'sample_uid',
            unique=True,
            postgresql_where=text('sample_uid IS NOT NULL'),
        ),
        Index(
            'uq_workflow_runs_v2_project',
            'project_id',
            'workflow_uid',
            unique=True,
            postgresql_where=text('sample_uid IS NULL AND project_id IS NOT NULL AND workflow_uid IS NOT NULL'),
        ),
    )

    uid: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=random_uid)
    generation: Mapped[int] = mapped_column(default=1, nullable=False)
    name: Mapped[str]
    owner_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    project_id: Mapped[int | None] = mapped_column(ForeignKey('projects.id', ondelete='CASCADE'), default=None)
    workflow_uid: Mapped[uuid.UUID | None] = mapped_column(default=None)
    sample_uid: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('samples.uid', ondelete='SET NULL'),
        default=None,
    )
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

    node_runs: Mapped[list['NodeRunV2']] = relationship(
        back_populates='run',
        cascade='all, delete-orphan',
        passive_deletes=True,
    )


class NodeRunV2(Base):
    """Execution state for one executable node in one run generation."""

    __tablename__ = 'workflow_node_runs_v2'
    __table_args__ = (
        UniqueConstraint('run_uid', 'generation', 'definition_node_id', name='uq_node_run_v2_definition'),
        Index('ix_node_runs_v2_run_generation_status', 'run_uid', 'generation', 'status'),
        Index('ix_node_runs_v2_heartbeat', 'status', 'heartbeat_at'),
    )

    uid: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=random_uid)
    run_uid: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('workflow_runs_v2.uid', ondelete='CASCADE'),
        nullable=False,
    )
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
    tool_uid: Mapped[uuid.UUID | None] = mapped_column(ForeignKey('tools.uid'), default=None)
    base_dir: Mapped[str | None] = mapped_column(default=None)
    commands: Mapped[str | None] = mapped_column(default=None)
    error_message: Mapped[str | None] = mapped_column(default=None)
    tool_output: Mapped[ToolOutput | None] = mapped_column(
        PydanticJson(ToolOutput),
        default=None,
        nullable=True,
    )
    system: Mapped[str | None] = mapped_column(default=None)
    hostname: Mapped[str | None] = mapped_column(default=None)
    worker_id: Mapped[str | None] = mapped_column(default=None)
    create_time: Mapped[datetime] = mapped_column(default=utc_now)
    queued_at: Mapped[datetime | None] = mapped_column(default=None)
    start_time: Mapped[datetime | None] = mapped_column(default=None)
    heartbeat_at: Mapped[datetime | None] = mapped_column(default=None)
    end_time: Mapped[datetime | None] = mapped_column(default=None)

    run: Mapped[WorkflowRunV2] = relationship(back_populates='node_runs')


class NodeDependencyV2(Base):
    """One data binding edge between two V2 node runs."""

    __tablename__ = 'workflow_node_dependencies_v2'
    __table_args__ = (
        UniqueConstraint(
            'run_uid',
            'generation',
            'source_node_uid',
            'source_handle',
            'target_node_uid',
            'target_handle',
            name='uq_node_dependency_v2_binding',
        ),
        Index('ix_node_dependencies_v2_target', 'target_node_uid'),
        Index('ix_node_dependencies_v2_source', 'source_node_uid'),
    )

    uid: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=random_uid)
    run_uid: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('workflow_runs_v2.uid', ondelete='CASCADE'),
        nullable=False,
    )
    generation: Mapped[int]
    source_node_uid: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('workflow_node_runs_v2.uid', ondelete='CASCADE'),
        nullable=False,
    )
    target_node_uid: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('workflow_node_runs_v2.uid', ondelete='CASCADE'),
        nullable=False,
    )
    source_handle: Mapped[str]
    target_handle: Mapped[str]


class WorkflowOutboxV2(Base):
    """Durable intent to publish one ready NodeRun to Celery."""

    __tablename__ = 'workflow_outbox_v2'
    __table_args__ = (
        UniqueConstraint('node_run_uid', name='uq_workflow_outbox_v2_node_run'),
        Index('ix_workflow_outbox_v2_status_created', 'status', 'create_time'),
    )

    uid: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=random_uid)
    run_uid: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('workflow_runs_v2.uid', ondelete='CASCADE'),
        nullable=False,
    )
    node_run_uid: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('workflow_node_runs_v2.uid', ondelete='CASCADE'),
        nullable=False,
    )
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
