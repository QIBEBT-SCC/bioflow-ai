# ruff: noqa: D103

import uuid
from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from bioflow_ai.models.agent import AgentEvent, AgentMessage, AgentRun, AgentRunTarget, AgentSession
from bioflow_ai.schemas.agent import ACTIVE_AGENT_RUN_STATUSES, AgentEventType


def get_session(session: Session, session_uid: uuid.UUID) -> AgentSession | None:
    return session.get(AgentSession, session_uid)


def get_sessions_page(
    session: Session,
    owner_id: int | None,
    project_id: int | None,
    cursor: datetime | None,
    limit: int,
) -> Sequence[AgentSession]:
    statement = select(AgentSession).where(AgentSession.project_id == project_id)
    if owner_id is not None:
        statement = statement.where(AgentSession.owner_id == owner_id)
    if cursor is not None:
        statement = statement.where(AgentSession.update_time < cursor)
    return session.scalars(statement.order_by(AgentSession.update_time.desc()).limit(limit + 1)).all()


def get_run(session: Session, run_uid: uuid.UUID) -> AgentRun | None:
    return session.get(AgentRun, run_uid)


def get_run_target(session: Session, run_uid: uuid.UUID) -> AgentRunTarget | None:
    return session.get(AgentRunTarget, run_uid)


def list_completed_diagnoses(
    session: Session,
    session_uid: uuid.UUID,
    source_run_uid: uuid.UUID,
) -> Sequence[AgentRun]:
    """List completed assistant runs for this conversation and target, newest first."""
    statement = (
        select(AgentRun)
        .join(AgentRunTarget, AgentRunTarget.agent_run_uid == AgentRun.uid)
        .where(
            AgentRun.session_uid == session_uid,
            AgentRun.agent_name == 'workflow-assistant',
            AgentRun.status == 'completed',
            AgentRunTarget.source_run_uid == source_run_uid,
        )
        .order_by(AgentRun.end_time.desc().nullslast(), AgentRun.create_time.desc(), AgentRun.uid.desc())
    )
    return session.scalars(statement).all()


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


def has_active_project_run(session: Session, project_id: int) -> bool:
    statement = (
        select(AgentRun.uid)
        .where(
            AgentRun.project_id == project_id,
            AgentRun.status.in_([status.value for status in ACTIVE_AGENT_RUN_STATUSES]),
        )
        .limit(1)
    )
    return session.scalar(statement) is not None


def list_project_artifact_runs(session: Session, project_id: int, owner_id: int | None) -> Sequence[AgentRun]:
    statement = (
        select(AgentRun)
        .join(AgentSession, AgentSession.uid == AgentRun.session_uid)
        .where(
            AgentRun.project_id == project_id,
            AgentRun.status == 'completed',
            AgentRun.agent_name.in_(['workflow-assistant', 'workflow-fixer']),
        )
        .order_by(AgentRun.end_time.desc().nullslast(), AgentRun.create_time.desc(), AgentRun.uid.desc())
    )
    if owner_id is not None:
        statement = statement.where(AgentSession.owner_id == owner_id)
    return session.scalars(statement).all()


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


INPUT_EVENT_TYPES = (AgentEventType.INPUT_QUEUED, AgentEventType.INPUT_APPLIED, AgentEventType.INPUT_CLOSED)


def list_input_events(
    session: Session,
    run_uid: uuid.UUID,
    after_id: int = 0,
    event_types: Sequence[AgentEventType] = INPUT_EVENT_TYPES,
) -> Sequence[AgentEvent]:
    """Read the durable input journal independently of progress events."""
    return session.scalars(
        select(AgentEvent)
        .where(
            AgentEvent.run_uid == run_uid,
            AgentEvent.id > after_id,
            AgentEvent.event_type.in_([event_type.value for event_type in event_types]),
        )
        .order_by(AgentEvent.id)
    ).all()


def list_session_input_events(session: Session, session_uid: uuid.UUID) -> Sequence[AgentEvent]:
    """Identify corrections that should be included in later conversation history."""
    return session.scalars(
        select(AgentEvent)
        .join(AgentRun, AgentRun.uid == AgentEvent.run_uid)
        .where(
            AgentRun.session_uid == session_uid,
            AgentEvent.event_type.in_([AgentEventType.INPUT_QUEUED.value, AgentEventType.INPUT_APPLIED.value]),
        )
    ).all()


def insert_run_records(
    session: Session,
    run: AgentRun,
    message: AgentMessage,
    event: AgentEvent,
    target: AgentRunTarget | None,
) -> None:
    """Flush the parent run before staging records that reference it."""
    session.add(run)
    session.flush()
    session.add_all([message, event, *([target] if target is not None else [])])


def insert_session(session: Session, agent_session: AgentSession) -> None:
    """Stage a conversation without flushing or committing."""
    session.add(agent_session)


def delete_session(session: Session, agent_session: AgentSession) -> None:
    """Stage conversation deletion in the caller's transaction."""
    session.delete(agent_session)


def insert_event(session: Session, event: AgentEvent) -> None:
    """Stage an event alongside its run state change."""
    session.add(event)


def insert_message(session: Session, message: AgentMessage) -> None:
    """Stage a message in the caller's transaction."""
    session.add(message)


def refresh_run_for_update(session: Session, run: AgentRun) -> None:
    """Refresh and lock a run before the service validates its transition."""
    session.refresh(run, with_for_update=True)
