# ruff: noqa: D103

import uuid
from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.agent import AgentEvent, AgentMessage, AgentRun, AgentRunTarget, AgentSession
from app.schemas.agent import ACTIVE_AGENT_RUN_STATUSES


def get_session(session: Session, session_uid: uuid.UUID) -> AgentSession | None:
    return session.get(AgentSession, session_uid)


def get_sessions_page(
    session: Session,
    owner_id: int,
    project_id: int | None,
    cursor: datetime | None,
    limit: int,
) -> Sequence[AgentSession]:
    statement = select(AgentSession).where(
        AgentSession.owner_id == owner_id,
        AgentSession.project_id == project_id,
    )
    if cursor is not None:
        statement = statement.where(AgentSession.update_time < cursor)
    return session.scalars(statement.order_by(AgentSession.update_time.desc()).limit(limit + 1)).all()


def get_run(session: Session, run_uid: uuid.UUID) -> AgentRun | None:
    return session.get(AgentRun, run_uid)


def get_run_target(session: Session, run_uid: uuid.UUID) -> AgentRunTarget | None:
    return session.get(AgentRunTarget, run_uid)


def get_latest_completed_diagnosis(
    session: Session,
    session_uid: uuid.UUID,
    source_run_uid: uuid.UUID,
) -> AgentRun | None:
    """Return the newest completed diagnosis for this conversation and source run."""
    statement = (
        select(AgentRun)
        .join(AgentRunTarget, AgentRunTarget.agent_run_uid == AgentRun.uid)
        .where(
            AgentRun.session_uid == session_uid,
            AgentRun.agent_name == 'workflow-diagnoser',
            AgentRun.status == 'completed',
            AgentRunTarget.source_run_uid == source_run_uid,
        )
        .order_by(AgentRun.end_time.desc().nullslast(), AgentRun.create_time.desc(), AgentRun.uid.desc())
        .limit(1)
    )
    return session.scalar(statement)


def get_latest_run(session: Session, session_uid: uuid.UUID) -> AgentRun | None:
    statement = (
        select(AgentRun).where(AgentRun.session_uid == session_uid).order_by(AgentRun.create_time.desc()).limit(1)
    )
    return session.scalar(statement)


def list_runs(session: Session, session_uid: uuid.UUID) -> Sequence[AgentRun]:
    statement = (
        select(AgentRun)
        .where(AgentRun.session_uid == session_uid)
        .order_by(AgentRun.create_time.asc(), AgentRun.uid.asc())
    )
    return session.scalars(statement).all()


def get_active_run(session: Session, session_uid: uuid.UUID) -> AgentRun | None:
    statement = (
        select(AgentRun)
        .where(
            AgentRun.session_uid == session_uid,
            AgentRun.status.in_([status.value for status in ACTIVE_AGENT_RUN_STATUSES]),
        )
        .order_by(AgentRun.create_time.desc())
        .limit(1)
    )
    return session.scalar(statement)


def list_messages(session: Session, session_uid: uuid.UUID) -> Sequence[AgentMessage]:
    statement = (
        select(AgentMessage)
        .where(AgentMessage.session_uid == session_uid)
        .order_by(AgentMessage.create_time.asc(), AgentMessage.uid.asc())
    )
    return session.scalars(statement).all()


def list_events(session: Session, run_uid: uuid.UUID, after_id: int = 0) -> Sequence[AgentEvent]:
    statement = (
        select(AgentEvent).where(AgentEvent.run_uid == run_uid, AgentEvent.id > after_id).order_by(AgentEvent.id.asc())
    )
    return session.scalars(statement).all()
