# ruff: noqa: D103

import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from loguru import logger
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import bioflow_ai.crud.agent as agent_crud
import bioflow_ai.crud.project_workflow as project_workflow_crud
from bioflow_ai.core.config import get_settings
from bioflow_ai.crud import project as project_crud
from bioflow_ai.crud import user as user_crud
from bioflow_ai.crud import workflow_run as run_crud
from bioflow_ai.infrastructure.database import engine
from bioflow_ai.models.agent import AgentEvent, AgentMessage, AgentRun, AgentRunTarget, AgentSession
from bioflow_ai.models.project import Project
from bioflow_ai.models.workflow import NodeRun, WorkflowRun
from bioflow_ai.schemas.agent import (
    ACTIVE_AGENT_RUN_STATUSES,
    MAX_IMAGES,
    TERMINAL_AGENT_RUN_STATUSES,
    AgentEventType,
    AgentName,
    AgentRunInput,
    AgentRunStatus,
    AgentSessionCreate,
    AgentSessionUpdate,
    ImageMessagePart,
    PDFMessagePart,
    WorkflowNodeMessagePart,
    WorkflowNodeReference,
)
from bioflow_ai.schemas.user_config import UserRole
from bioflow_ai.services._transaction import committing
from bioflow_ai.services.agents.artifacts import get_project_root
from bioflow_ai.services.agents.chat_attachment import (
    ImagePreparationError,
    resolve_image,
)
from bioflow_ai.services.agents.workflow_builder import MISSING_PLAN_ERROR, has_project_plan
from bioflow_ai.services.documents.pdf_index import PDFNotFoundError, PDFPreparationError, resolve_pdf
from bioflow_ai.services.projects import workflows as project_workflow_service
from bioflow_ai.services.workflow.subgraphs import SubgraphError, definition_id, enter_subgraphs
from bioflow_ai.utils.identifiers import UUIDLike, as_uuid
from bioflow_ai.utils.jsonable import to_json_object
from bioflow_ai.utils.system import random_uid, utc_now

WORKFLOW_INPUT_RUN_KEY = 'workflow_input_run_uid'
MESSAGE_ID_CONFLICT = 'The message ID has already been used for a different request.'
WORKFLOW_INPUT_ONLY = 'Only workflow-builder runs accept additional input.'
INPUT_STATUS_ERROR = 'Additional input requires a queued or running workflow build.'
INPUT_CLOSED_ERROR = 'This workflow build is finishing and no longer accepts input.'

ACTIVE_DELETE_ERROR = 'An active run must be cancelled before deleting the session.'
GLOBAL_SCOPE_ERROR = 'Global sessions only support tool-generator.'
ACTIVE_RUN_ERROR = 'This session already has an active run.'
ACTIVE_RUN_CONSTRAINT = 'uq_agent_runs_one_active_per_session'
WORKFLOW_RUN_TARGET_REQUIRED_ERROR = 'workflow-assistant and workflow-fixer require source_run_uid.'
WORKFLOW_RUN_TARGET_NOT_ALLOWED_ERROR = 'source_run_uid is only valid for workflow-assistant and workflow-fixer.'
NODE_AGENT_ERROR = 'Node attachments are only supported by workflow-assistant.'
NODE_STALE_ERROR = 'The workflow run has been re-executed. Remove the old node attachment and add it again.'
PROJECT_NOT_FOUND_ERROR = 'Project not found.'
AGENT_RESOURCE_NOT_FOUND_ERROR = 'Agent resource not found.'

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
    """Business error raised by agent session/run service operations."""

    def __init__(self, detail: str, *, status_code: int = 400) -> None:
        super().__init__(detail)
        self.status_code = status_code


class QueuedAgentInput(BaseModel):
    """A queued message and its monotonic position in the input journal."""

    event_id: int
    message_uid: uuid.UUID
    text: str


def _is_active_run_integrity_error(error: IntegrityError) -> bool:
    diagnostic = getattr(error.orig, 'diag', None)
    return getattr(diagnostic, 'constraint_name', None) == ACTIVE_RUN_CONSTRAINT


def _is_admin_actor(session: Session, user_id: int) -> bool:
    user = user_crud.get_by_id(session, user_id)
    return user is not None and user.role == UserRole.ADMIN


def _require_owned_project(session: Session, project_id: int, owner_id: int) -> Project:
    project = project_crud.get_by_id(session, project_id)
    if project is None or (project.owner_id != owner_id and not _is_admin_actor(session, owner_id)):
        raise AgentServiceError(PROJECT_NOT_FOUND_ERROR, status_code=404)
    return project


def get_owned_session(session: Session, session_uid: UUIDLike, owner_id: int) -> AgentSession:
    agent_session = agent_crud.get_session(session, as_uuid(session_uid))
    if agent_session is None or (agent_session.owner_id != owner_id and not _is_admin_actor(session, owner_id)):
        raise AgentServiceError(AGENT_RESOURCE_NOT_FOUND_ERROR, status_code=404)
    return agent_session


def get_owned_run(session: Session, run_uid: UUIDLike, owner_id: int) -> AgentRun:
    run = agent_crud.get_run(session, as_uuid(run_uid))
    if run is None:
        raise AgentServiceError(AGENT_RESOURCE_NOT_FOUND_ERROR, status_code=404)
    get_owned_session(session, run.session_uid, owner_id)
    reconcile_stale_run(session, run)
    return run


def create_session(session: Session, owner_id: int, data: AgentSessionCreate) -> AgentSession:
    if data.project_id is not None:
        _require_owned_project(session, data.project_id, owner_id)
    agent_session = AgentSession(owner_id=owner_id, project_id=data.project_id, title=data.title.strip())
    with committing(session):
        agent_crud.insert_session(session, agent_session)
    session.refresh(agent_session)
    return agent_session


def list_sessions(
    session: Session,
    owner_id: int | None,
    project_id: int | None,
    cursor: datetime | None,
    limit: int,
) -> Sequence[AgentSession]:
    return agent_crud.get_sessions_page(session, owner_id, project_id, cursor, limit)


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
            raise AgentServiceError(ACTIVE_DELETE_ERROR, status_code=409)
    with committing(session):
        agent_crud.delete_session(session, agent_session)


def create_run(
    session: Session,
    agent_session: AgentSession,
    agent_name: AgentName,
    text: str,
    *,
    retry_of_uid: uuid.UUID | None = None,
    source_run_uid: uuid.UUID | None = None,
    images: list[ImageMessagePart] | None = None,
    pdfs: list[PDFMessagePart] | None = None,
    node: WorkflowNodeReference | None = None,
) -> AgentRun:
    if len(images or []) + len(pdfs or []) > MAX_IMAGES:
        message = 'A message can contain at most 4 image or PDF attachments.'
        raise AgentServiceError(message, status_code=422)
    if not text.strip() and ((not images and not pdfs and node is None) or agent_name == AgentName.WORKFLOW_FIXER):
        message = 'A text request is required for workflow-fixer; other agents accept attachments alone.'
        raise AgentServiceError(message, status_code=422)
    if node is not None and agent_name != AgentName.WORKFLOW_ASSISTANT:
        raise AgentServiceError(NODE_AGENT_ERROR, status_code=422)
    if agent_session.project_id is None and agent_name != AgentName.TOOL_GENERATOR:
        raise AgentServiceError(GLOBAL_SCOPE_ERROR, status_code=422)
    prepared_node = None
    if agent_name in {AgentName.WORKFLOW_ASSISTANT, AgentName.WORKFLOW_FIXER}:
        if source_run_uid is None:
            raise AgentServiceError(WORKFLOW_RUN_TARGET_REQUIRED_ERROR, status_code=422)
        source_run = run_crud.get(session, source_run_uid)
        if (
            source_run is None
            or (source_run.owner_id != agent_session.owner_id and not _is_admin_actor(session, agent_session.owner_id))
            or source_run.project_id != agent_session.project_id
        ):
            raise AgentServiceError(PROJECT_NOT_FOUND_ERROR, status_code=404)
        if node is not None:
            prepared_node = _resolve_node(session, source_run, node)
    elif source_run_uid is not None:
        raise AgentServiceError(WORKFLOW_RUN_TARGET_NOT_ALLOWED_ERROR, status_code=422)
    if (
        agent_name == AgentName.WORKFLOW_BUILDER
        and agent_session.project_id is not None
        and not has_project_plan(get_project_root(agent_session.project_id))
    ):
        raise AgentServiceError(MISSING_PLAN_ERROR, status_code=422)

    active = agent_crud.get_active_run(session, agent_session.uid)
    if active is not None:
        reconcile_stale_run(session, active)
        if AgentRunStatus(active.status) in ACTIVE_AGENT_RUN_STATUSES:
            raise AgentServiceError(ACTIVE_RUN_ERROR, status_code=409)

    run_uid = random_uid()
    try:
        prepared_pdfs = [resolve_pdf(session, pdf) for pdf in pdfs or []]
        prepared = [
            resolve_image(session, image).model_copy(update={'id': str(random_uid())}) for image in images or []
        ]
    except (ImagePreparationError, PDFNotFoundError, PDFPreparationError, OSError, ValueError) as error:
        raise AgentServiceError(str(error), status_code=422) from error
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
        parts=[
            *([{'type': 'text', 'text': text}] if text else []),
            *(image.model_dump(mode='json') for image in prepared),
            *(pdf.model_dump(mode='json') for pdf in prepared_pdfs),
            *([prepared_node.model_dump(mode='json')] if prepared_node is not None else []),
        ],
        create_time=now,
    )
    event = AgentEvent(
        run_uid=run_uid,
        event_type=AgentEventType.QUEUED.value,
        payload={'agent_name': agent_name.value},
        create_time=now,
    )
    target = (
        AgentRunTarget(agent_run_uid=run_uid, source_run_uid=source_run_uid) if source_run_uid is not None else None
    )
    try:
        with committing(session):
            agent_crud.insert_run_records(session, run, message, event, target)
            if not agent_session.title:
                first = text if text.strip() else prepared_node.name if prepared_node else None
                agent_session.title = (first or (prepared or prepared_pdfs)[0].filename)[:80]
            agent_session.update_time = now
    except IntegrityError as error:
        if _is_active_run_integrity_error(error):
            raise AgentServiceError(ACTIVE_RUN_ERROR, status_code=409) from error
        raise
    session.refresh(run)
    return run


def _resolve_node(session: Session, run: WorkflowRun, node: WorkflowNodeReference) -> WorkflowNodeMessagePart:
    """Validate the node address against the run snapshot and replace client labels with saved ones."""
    if node.run_uid != run.uid:
        message = 'The attached node must belong to the selected workflow run.'
        raise AgentServiceError(message, status_code=422)
    if node.generation != run.generation:
        raise AgentServiceError(NODE_STALE_ERROR, status_code=422)
    try:
        graph, containers = enter_subgraphs(run.workflow, node.subgraph_path)
    except SubgraphError as error:
        raise AgentServiceError(str(error), status_code=422) from error
    saved = next((item for item in graph.nodes if item.id == node.node_id), None)
    if saved is None:
        message = 'The attached node does not exist in this run.'
        raise AgentServiceError(message, status_code=422)
    name = saved.data.get('name') or session.scalar(
        select(NodeRun.name).where(
            NodeRun.run_uid == run.uid,
            NodeRun.definition_node_id == definition_id((*node.subgraph_path, node.node_id)),
        )
    )
    return WorkflowNodeMessagePart(
        **node.model_dump(),
        name=str(name or saved.id),
        node_type=saved.type,
        subgraph_labels=[str(item.data.get('name') or item.id) for item in containers],
    )


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
    agent_crud.refresh_run_for_update(session, run)
    current_status = AgentRunStatus(run.status)
    if status not in ALLOWED_RUN_TRANSITIONS[current_status]:
        message = f'Invalid agent run transition: {current_status.value} -> {status.value}.'
        raise AgentServiceError(message, status_code=409)
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
        agent_crud.insert_event(
            session,
            AgentEvent(
                run_uid=run.uid,
                event_type=event_type.value,
                payload=to_json_object(payload or {}),
                create_time=now,
            ),
        )
        run.agent_session.update_time = now
    session.refresh(run)
    if status in TERMINAL_AGENT_RUN_STATUSES:
        _delete_checkpoint(run.checkpoint_thread_id)
    return run


def _delete_checkpoint(thread_id: str) -> None:
    """Drop a finished run's checkpoint; a failure only leaves an unused thread behind."""
    try:
        from bioflow_ai.llm.checkpoint import delete_checkpoint_thread

        delete_checkpoint_thread(thread_id)
    except Exception:
        logger.warning('Unable to delete checkpoint thread {}.', thread_id)


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
        agent_crud.insert_event(session, event)
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
        agent_crud.insert_message(session, message)
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
        agent_crud.insert_message(session, message)
        run.agent_session.update_time = utc_now()
    session.refresh(message)
    return message


def queue_input(session: Session, run: AgentRun, data: AgentRunInput) -> AgentMessage:
    """Serialize input acceptance with the build's final inbox check."""
    agent_crud.refresh_run_for_update(session, run)
    events = agent_crud.list_input_events(session, run.uid)
    existing = session.get(AgentMessage, data.message_uid)
    if existing is not None:
        is_input = any(event.payload.get('message_uid') == str(data.message_uid) for event in events)
        if existing.run_uid == run.uid and is_input and existing.parts == [{'type': 'text', 'text': data.text}]:
            session.commit()
            return existing
        raise AgentServiceError(MESSAGE_ID_CONFLICT, status_code=409)
    if run.agent_name != AgentName.WORKFLOW_BUILDER.value:
        raise AgentServiceError(WORKFLOW_INPUT_ONLY, status_code=409)
    if run.status not in {AgentRunStatus.QUEUED.value, AgentRunStatus.RUNNING.value} or run.cancel_requested:
        raise AgentServiceError(INPUT_STATUS_ERROR, status_code=409)
    if any(event.event_type == AgentEventType.INPUT_CLOSED.value for event in events):
        raise AgentServiceError(INPUT_CLOSED_ERROR, status_code=409)
    now = utc_now()
    message = AgentMessage(
        uid=data.message_uid,
        session_uid=run.session_uid,
        run_uid=run.uid,
        role='user',
        parts=[{'type': 'text', 'text': data.text}],
        create_time=now,
    )
    try:
        with committing(session):
            agent_crud.insert_message(session, message)
            agent_crud.insert_event(
                session,
                AgentEvent(
                    run_uid=run.uid,
                    event_type=AgentEventType.INPUT_QUEUED.value,
                    payload={'message_uid': str(message.uid)},
                    create_time=now,
                ),
            )
            run.update_time = now
            run.agent_session.update_time = now
    except IntegrityError as error:
        raise AgentServiceError(MESSAGE_ID_CONFLICT, status_code=409) from error
    session.refresh(message)
    return message


def _pending_inputs(session: Session, run_uid: uuid.UUID, cursor: int) -> list[QueuedAgentInput]:
    result = []
    for event in agent_crud.list_input_events(session, run_uid, cursor, [AgentEventType.INPUT_QUEUED]):
        message = session.get(AgentMessage, uuid.UUID(event.payload['message_uid']))
        if message is not None:
            result.append(
                QueuedAgentInput(
                    event_id=event.id,
                    message_uid=message.uid,
                    text=''.join(part.get('text', '') for part in message.parts if part.get('type') == 'text'),
                )
            )
    return result


def pending_inputs(run_uid: uuid.UUID, cursor: int) -> list[QueuedAgentInput]:
    """Replay from the graph's checkpoint cursor, including unacknowledged deliveries."""
    with Session(engine) as session:
        return _pending_inputs(session, run_uid, cursor)


def acknowledge_inputs(run_uid: uuid.UUID, cursor: int) -> None:
    """Publish delivery only after a model response confirms the input reached the model."""
    with Session(engine) as session, committing(session):
        events = agent_crud.list_input_events(
            session, run_uid, event_types=[AgentEventType.INPUT_QUEUED, AgentEventType.INPUT_APPLIED]
        )
        applied = {event.payload['message_uid'] for event in events if event.event_type == AgentEventType.INPUT_APPLIED}
        for event in events:
            if (
                event.event_type == AgentEventType.INPUT_QUEUED
                and event.id <= cursor
                and event.payload['message_uid'] not in applied
            ):
                agent_crud.insert_event(
                    session,
                    AgentEvent(
                        run_uid=run_uid,
                        event_type=AgentEventType.INPUT_APPLIED.value,
                        payload={'message_uid': event.payload['message_uid'], 'input_event_id': event.id},
                    ),
                )


def close_inputs(run_uid: uuid.UUID, cursor: int) -> list[QueuedAgentInput]:
    """Close an empty inbox atomically, or return the inputs that arrived since the cursor."""
    with Session(engine) as session, committing(session):
        run = agent_crud.get_run(session, run_uid)
        if run is None:
            return []
        agent_crud.refresh_run_for_update(session, run)
        if run.cancel_requested or run.status != AgentRunStatus.RUNNING.value:
            return []
        if inputs := _pending_inputs(session, run_uid, cursor):
            return inputs
        if not agent_crud.list_input_events(session, run_uid, event_types=[AgentEventType.INPUT_CLOSED]):
            agent_crud.insert_event(
                session,
                AgentEvent(run_uid=run_uid, event_type=AgentEventType.INPUT_CLOSED.value, payload={}),
            )
        return []


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
    if run.cancel_requested or run.status != AgentRunStatus.RUNNING.value:
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
    stale_after = timedelta(seconds=get_settings().celery.AGENT_HEARTBEAT_TIMEOUT)
    if utc_now() - run.heartbeat_at <= stale_after:
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
