# ruff: noqa: D103

import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import app.crud.agent as agent_crud
import app.crud.project_workflow as project_workflow_crud
from app.models.agent import AgentEvent, AgentMessage, AgentRun, AgentSession
from app.models.project import Project
from app.schemas.agent import (
    ACTIVE_AGENT_RUN_STATUSES,
    AgentEventType,
    AgentName,
    AgentRunStatus,
    AgentSessionCreate,
    AgentSessionUpdate,
)
from app.services import project_workflow as project_workflow_service
from app.services._transaction import committing
from app.utils.identifiers import UUIDLike, as_uuid
from app.utils.jsonable import to_json_object
from app.utils.system import random_uid, utc_now

STALE_HEARTBEAT_AFTER = timedelta(seconds=60)
ACTIVE_DELETE_ERROR = 'An active run must be cancelled before deleting the session.'
GLOBAL_SCOPE_ERROR = 'Global sessions only support tool-generator.'
ACTIVE_RUN_ERROR = 'This session already has an active run.'

ALLOWED_RUN_TRANSITIONS = {
    AgentRunStatus.QUEUED: {
        AgentRunStatus.RUNNING,
        AgentRunStatus.CANCELLED,
        AgentRunStatus.FAILED,
    },
    AgentRunStatus.RUNNING: {
        AgentRunStatus.COMPLETED,
        AgentRunStatus.WAITING_INPUT,
        AgentRunStatus.CANCEL_REQUESTED,
        AgentRunStatus.CANCELLED,
        AgentRunStatus.FAILED,
    },
    AgentRunStatus.WAITING_INPUT: {
        AgentRunStatus.QUEUED,
        AgentRunStatus.CANCELLED,
    },
    AgentRunStatus.CANCEL_REQUESTED: {
        AgentRunStatus.CANCELLED,
        AgentRunStatus.FAILED,
    },
    AgentRunStatus.CANCELLED: set(),
    AgentRunStatus.COMPLETED: set(),
    AgentRunStatus.FAILED: set(),
}


class AgentServiceError(ValueError):
    """Base exception for durable web-agent operations."""


class AgentSessionNotFoundError(AgentServiceError):
    pass


class AgentRunNotFoundError(AgentServiceError):
    pass


class AgentProjectAccessError(AgentServiceError):
    pass


class AgentRunConflictError(AgentServiceError):
    pass


class AgentRunTransitionError(AgentServiceError):
    pass


class AgentScopeError(AgentServiceError):
    pass


def _require_owned_project(session: Session, project_id: int, owner_id: int) -> Project:
    project = session.get(Project, project_id)
    if project is None or project.owner_id != owner_id:
        raise AgentProjectAccessError
    return project


def get_owned_session(session: Session, session_uid: UUIDLike, owner_id: int) -> AgentSession:
    agent_session = agent_crud.get_session(session, as_uuid(session_uid))
    if agent_session is None or agent_session.owner_id != owner_id:
        raise AgentSessionNotFoundError
    return agent_session


def get_owned_run(session: Session, run_uid: UUIDLike, owner_id: int) -> AgentRun:
    run = agent_crud.get_run(session, as_uuid(run_uid))
    if run is None:
        raise AgentRunNotFoundError
    get_owned_session(session, run.session_uid, owner_id)
    reconcile_stale_run(session, run)
    return run


def create_session(session: Session, owner_id: int, data: AgentSessionCreate) -> AgentSession:
    if data.project_id is not None:
        _require_owned_project(session, data.project_id, owner_id)
    agent_session = AgentSession(owner_id=owner_id, project_id=data.project_id, title=data.title.strip())
    with committing(session):
        session.add(agent_session)
    session.refresh(agent_session)
    return agent_session


def list_sessions(
    session: Session,
    owner_id: int,
    project_id: int | None,
    cursor: datetime | None,
    limit: int,
) -> Sequence[AgentSession]:
    sessions = agent_crud.get_sessions_page(session, owner_id, project_id, cursor, limit)
    for agent_session in sessions:
        latest_run = agent_crud.get_latest_run(session, agent_session.uid)
        if latest_run is not None:
            reconcile_stale_run(session, latest_run)
    return sessions


def update_session(
    session: Session,
    agent_session: AgentSession,
    data: AgentSessionUpdate,
) -> AgentSession:
    with committing(session):
        agent_session.title = data.title.strip()
        agent_session.update_time = utc_now()
    session.refresh(agent_session)
    return agent_session


def delete_session(session: Session, agent_session: AgentSession) -> None:
    active = agent_crud.get_active_run(session, agent_session.uid)
    if active is not None:
        reconcile_stale_run(session, active)
        if AgentRunStatus(active.status) in ACTIVE_AGENT_RUN_STATUSES:
            raise AgentRunConflictError(ACTIVE_DELETE_ERROR)
    with committing(session):
        session.delete(agent_session)


def create_run(
    session: Session,
    agent_session: AgentSession,
    agent_name: AgentName,
    text: str,
    *,
    retry_of_uid: uuid.UUID | None = None,
) -> AgentRun:
    if agent_session.project_id is None and agent_name != AgentName.TOOL_GENERATOR:
        raise AgentScopeError(GLOBAL_SCOPE_ERROR)

    active = agent_crud.get_active_run(session, agent_session.uid)
    if active is not None:
        reconcile_stale_run(session, active)
        if AgentRunStatus(active.status) in ACTIVE_AGENT_RUN_STATUSES:
            raise AgentRunConflictError(ACTIVE_RUN_ERROR)

    run_uid = random_uid()
    now = utc_now()
    run = AgentRun(
        uid=run_uid,
        session_uid=agent_session.uid,
        project_id=agent_session.project_id,
        agent_name=agent_name.value,
        status=AgentRunStatus.QUEUED.value,
        checkpoint_thread_id=str(run_uid),
        retry_of_uid=retry_of_uid,
        create_time=now,
        update_time=now,
    )
    message = AgentMessage(
        session_uid=agent_session.uid,
        run_uid=run_uid,
        role='user',
        parts=[{'type': 'text', 'text': text}],
        create_time=now,
    )
    event = AgentEvent(
        run_uid=run_uid,
        event_type=AgentEventType.QUEUED.value,
        payload={'agent_name': agent_name.value},
        create_time=now,
    )
    try:
        with committing(session):
            session.add_all([run, message, event])
            if not agent_session.title:
                agent_session.title = text[:80]
            agent_session.update_time = now
    except IntegrityError as error:
        raise AgentRunConflictError(ACTIVE_RUN_ERROR) from error
    session.refresh(run)
    return run


def record_dispatch(session: Session, run: AgentRun, celery_task_id: str) -> AgentRun:
    with committing(session):
        run.celery_task_id = celery_task_id
        run.update_time = utc_now()
    session.refresh(run)
    return run


def mark_dispatch_failed(session: Session, run: AgentRun, message: str) -> AgentRun:
    return transition_run(
        session,
        run,
        AgentRunStatus.FAILED,
        event_type=AgentEventType.FAILED,
        payload={'code': 'DISPATCH_FAILED', 'message': message},
        error_code='DISPATCH_FAILED',
        error_message=message,
    )


def transition_run(
    session: Session,
    run: AgentRun,
    status: AgentRunStatus,
    *,
    event_type: AgentEventType,
    payload: dict[str, Any] | None = None,
    interrupt_payload: dict[str, Any] | None = None,
    result_payload: dict[str, Any] | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> AgentRun:
    session.refresh(run, with_for_update=True)
    current_status = AgentRunStatus(run.status)
    if status not in ALLOWED_RUN_TRANSITIONS[current_status]:
        message = f'Invalid agent run transition: {current_status.value} -> {status.value}.'
        raise AgentRunTransitionError(message)
    now = utc_now()
    with committing(session):
        run.status = status.value
        run.update_time = now
        run.interrupt_payload = to_json_object(interrupt_payload) if interrupt_payload is not None else None
        run.result_payload = to_json_object(result_payload) if result_payload is not None else None
        run.error_code = error_code
        run.error_message = error_message
        if status in {AgentRunStatus.CANCEL_REQUESTED, AgentRunStatus.CANCELLED}:
            run.cancel_requested = True
        if status == AgentRunStatus.RUNNING:
            run.start_time = run.start_time or now
            run.heartbeat_at = now
        if status in {AgentRunStatus.CANCELLED, AgentRunStatus.COMPLETED, AgentRunStatus.FAILED}:
            run.end_time = now
        session.add(
            AgentEvent(
                run_uid=run.uid,
                event_type=event_type.value,
                payload=to_json_object(payload or {}),
                create_time=now,
            )
        )
        run.agent_session.update_time = now
    session.refresh(run)
    return run


def append_event(
    session: Session,
    run: AgentRun,
    event_type: AgentEventType,
    payload: dict[str, Any],
) -> AgentEvent:
    now = utc_now()
    event = AgentEvent(
        run_uid=run.uid,
        event_type=event_type.value,
        payload=to_json_object(payload),
        create_time=now,
    )
    with committing(session):
        session.add(event)
        run.update_time = now
        run.agent_session.update_time = now
    session.refresh(event)
    return event


def add_assistant_message(session: Session, run: AgentRun, text: str) -> AgentMessage:
    message = AgentMessage(
        session_uid=run.session_uid,
        run_uid=run.uid,
        role='assistant',
        parts=[{'type': 'text', 'text': text}],
    )
    with committing(session):
        session.add(message)
        run.agent_session.update_time = utc_now()
    session.refresh(message)
    return message


def add_resume_message(session: Session, run: AgentRun, text: str) -> AgentMessage:
    message = AgentMessage(
        session_uid=run.session_uid,
        run_uid=run.uid,
        role='user',
        parts=[{'type': 'text', 'text': text.strip()}],
    )
    with committing(session):
        session.add(message)
        run.agent_session.update_time = utc_now()
    session.refresh(message)
    return message


def import_generated_workflows(session: Session, run: AgentRun, values: dict[str, Any]) -> None:
    """Import workflows produced by a workflow-builder run into its project."""
    if run.agent_name != AgentName.WORKFLOW_BUILDER.value or run.project_id is None:
        return

    for value in values.get('workflow_uids') or []:
        workflow_uid = as_uuid(value)
        if project_workflow_crud.exists(session, run.project_id, workflow_uid):
            continue
        try:
            project_workflow_service.add_workflow_to_project(session, run.project_id, workflow_uid)
        except (project_workflow_service.ProjectWorkflowConflictError, IntegrityError):
            # Another project-scoped run may have imported the same workflow concurrently.
            continue


def update_heartbeat(session: Session, run: AgentRun) -> bool:
    session.refresh(run)
    if run.cancel_requested:
        return False
    now = utc_now()
    with committing(session):
        run.heartbeat_at = now
        run.update_time = now
    return True


def reconcile_stale_run(session: Session, run: AgentRun) -> AgentRun:
    heartbeat_statuses = {
        AgentRunStatus.RUNNING.value,
        AgentRunStatus.CANCEL_REQUESTED.value,
    }
    if run.status not in heartbeat_statuses or run.heartbeat_at is None:
        return run
    if utc_now() - run.heartbeat_at <= STALE_HEARTBEAT_AFTER:
        return run
    return transition_run(
        session,
        run,
        AgentRunStatus.FAILED,
        event_type=AgentEventType.FAILED,
        payload={'code': 'WORKER_LOST', 'message': 'Agent worker heartbeat expired.'},
        error_code='WORKER_LOST',
        error_message='Agent worker heartbeat expired.',
    )


def list_messages(session: Session, agent_session: AgentSession) -> Sequence[AgentMessage]:
    return agent_crud.list_messages(session, agent_session.uid)


def list_runs(session: Session, agent_session: AgentSession) -> Sequence[AgentRun]:
    runs = agent_crud.list_runs(session, agent_session.uid)
    for run in runs:
        reconcile_stale_run(session, run)
    return runs


def list_events(session: Session, run: AgentRun, after_id: int = 0) -> Sequence[AgentEvent]:
    return agent_crud.list_events(session, run.uid, after_id)
