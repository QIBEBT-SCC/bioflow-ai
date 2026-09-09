import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, BigInteger, DateTime, ForeignKey, Index, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base
from app.models.project import Project
from app.models.user import User
from app.utils.system import random_uid, utc_now


class AgentSession(Base):
    """A durable, user-owned conversation scoped to one project or globally."""

    __tablename__ = 'agent_sessions'

    uid: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=random_uid)
    owner_id: Mapped[int] = mapped_column(ForeignKey(User.id, ondelete='CASCADE'), index=True)
    project_id: Mapped[int | None] = mapped_column(ForeignKey(Project.id, ondelete='CASCADE'), index=True)
    title: Mapped[str] = mapped_column(String(200), default='')
    create_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    update_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)

    owner: Mapped[User] = relationship()
    project: Mapped[Project | None] = relationship()
    runs: Mapped[list['AgentRun']] = relationship(
        back_populates='agent_session',
        cascade='all, delete-orphan',
        passive_deletes=True,
    )
    messages: Mapped[list['AgentMessage']] = relationship(
        back_populates='agent_session',
        cascade='all, delete-orphan',
        passive_deletes=True,
    )


class AgentRun(Base):
    """One independently scheduled execution of a routable agent."""

    __tablename__ = 'agent_runs'
    __table_args__ = (
        Index(
            'uq_agent_runs_one_active_per_session',
            'session_uid',
            unique=True,
            postgresql_where=text("status IN ('queued', 'running', 'waiting_input', 'cancel_requested')"),
        ),
    )

    uid: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=random_uid)
    session_uid: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(AgentSession.uid, ondelete='CASCADE'),
        index=True,
    )
    project_id: Mapped[int | None] = mapped_column(ForeignKey(Project.id, ondelete='SET NULL'), index=True)
    agent_name: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default='queued', index=True)
    checkpoint_thread_id: Mapped[str] = mapped_column(String(64), unique=True)
    celery_task_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    interrupt_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    result_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(default=False)
    retry_of_uid: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('agent_runs.uid', ondelete='SET NULL'),
        nullable=True,
    )
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    create_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    start_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    update_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)

    agent_session: Mapped[AgentSession] = relationship(back_populates='runs')
    messages: Mapped[list['AgentMessage']] = relationship(
        back_populates='run',
        cascade='all, delete-orphan',
        passive_deletes=True,
    )
    events: Mapped[list['AgentEvent']] = relationship(
        back_populates='run',
        cascade='all, delete-orphan',
        passive_deletes=True,
    )


class AgentRunTarget(Base):
    """The workflow execution inspected or fixed by a targeted agent run."""

    __tablename__ = 'agent_run_targets'

    agent_run_uid: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(AgentRun.uid, ondelete='CASCADE'),
        primary_key=True,
    )
    source_run_uid: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('workflow_runs.uid', ondelete='CASCADE'),
        index=True,
    )


class AgentMessage(Base):
    """A durable domain message independent of any frontend SDK."""

    __tablename__ = 'agent_messages'

    uid: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=random_uid)
    session_uid: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(AgentSession.uid, ondelete='CASCADE'),
        index=True,
    )
    run_uid: Mapped[uuid.UUID] = mapped_column(ForeignKey(AgentRun.uid, ondelete='CASCADE'), index=True)
    role: Mapped[str] = mapped_column(String(16))
    parts: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    create_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)

    agent_session: Mapped[AgentSession] = relationship(back_populates='messages')
    run: Mapped[AgentRun] = relationship(back_populates='messages')


class AgentEvent(Base):
    """An append-only, replayable event emitted by an agent run."""

    __tablename__ = 'agent_events'

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_uid: Mapped[uuid.UUID] = mapped_column(ForeignKey(AgentRun.uid, ondelete='CASCADE'), index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    create_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)

    run: Mapped[AgentRun] = relationship(back_populates='events')
