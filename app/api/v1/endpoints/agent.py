# ruff: noqa: D103

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import suppress
from datetime import datetime
from typing import Annotated, Literal, NoReturn

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

import app.crud.agent as agent_crud
from app.core.auth import UserDep
from app.core.session import SessionDep, engine
from app.models.agent import AgentRun, AgentSession
from app.schemas.agent import (
    TERMINAL_AGENT_RUN_STATUSES,
    AgentEventPublic,
    AgentEventType,
    AgentMessagePublic,
    AgentName,
    AgentRunCreate,
    AgentRunPublic,
    AgentRunResume,
    AgentRunRetry,
    AgentRunStatus,
    AgentSessionCreate,
    AgentSessionPage,
    AgentSessionPublic,
    AgentSessionUpdate,
)
from app.schemas.agent_file import AgentFilePublic
from app.schemas.user_config import UserRole
from app.services import agent as agent_service
from app.services import agent_file as agent_file_service

session_router = APIRouter()
run_router = APIRouter()
SSE_HEARTBEAT_SECONDS = 15
RESUME_STATUS_ERROR = 'Only waiting_input runs can be resumed.'
CANCEL_STATUS_ERROR = 'A terminal run cannot be cancelled.'
RETRY_STATUS_ERROR = 'Only failed runs can be retried.'
RETRY_INPUT_ERROR = 'The original run input is unavailable.'


def _raise_service_error(error: Exception) -> NoReturn:
    if isinstance(error, (agent_service.AgentSessionNotFoundError, agent_service.AgentRunNotFoundError)):
        raise HTTPException(status_code=404, detail='Agent resource not found.') from error
    if isinstance(error, agent_service.AgentProjectAccessError):
        raise HTTPException(status_code=404, detail='Project not found.') from error
    if isinstance(error, (agent_service.AgentRunConflictError, agent_service.AgentRunTransitionError)):
        raise HTTPException(status_code=409, detail=str(error)) from error
    if isinstance(error, agent_service.AgentScopeError):
        raise HTTPException(status_code=422, detail=str(error)) from error
    raise error


def _session_public(session: SessionDep, agent_session: AgentSession) -> AgentSessionPublic:
    result = AgentSessionPublic.model_validate(agent_session)
    latest_run = agent_crud.get_latest_run(session, agent_session.uid)
    if latest_run is not None:
        result.latest_run = AgentRunPublic.model_validate(latest_run)
    return result


def _dispatch_run(session: SessionDep, run: AgentRun, language: str | None = None) -> AgentRun:
    from app.tasks.agent_run import execute_agent_run_task

    try:
        task = execute_agent_run_task.apply_async(args=(str(run.uid), language), queue='agent')
    except Exception as error:
        agent_service.mark_dispatch_failed(session, run, f'{type(error).__name__}: {error}')
        raise HTTPException(status_code=503, detail='Unable to dispatch agent run.') from error
    return agent_service.record_dispatch(session, run, task.id)


@session_router.post('', status_code=status.HTTP_201_CREATED)
def create_agent_session(
    data: AgentSessionCreate,
    session: SessionDep,
    current_user: UserDep,
) -> AgentSessionPublic:
    try:
        created = agent_service.create_session(session, current_user.id, data)
    except agent_service.AgentServiceError as error:
        _raise_service_error(error)
    return _session_public(session, created)


@session_router.get('')
def list_agent_sessions(
    session: SessionDep,
    current_user: UserDep,
    scope: Literal['global', 'project'] = 'global',
    project_id: int | None = None,
    cursor: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=20)] = 8,
) -> AgentSessionPage:
    if scope == 'project' and project_id is None:
        raise HTTPException(status_code=422, detail='project_id is required for project scope.')
    if scope == 'global' and project_id is not None:
        raise HTTPException(status_code=422, detail='project_id is not valid for global scope.')
    sessions = agent_service.list_sessions(
        session,
        None if current_user.role == UserRole.ADMIN else current_user.id,
        project_id if scope == 'project' else None,
        cursor,
        limit,
    )
    has_more = len(sessions) > limit
    page = sessions[:limit]
    return AgentSessionPage(
        limit=limit,
        has_more=has_more,
        next_cursor=page[-1].update_time if page else None,
        data=[_session_public(session, item) for item in page],
    )


@session_router.get('/{session_uid}')
def get_agent_session(
    session_uid: uuid.UUID,
    session: SessionDep,
    current_user: UserDep,
) -> AgentSessionPublic:
    try:
        agent_session = agent_service.get_owned_session(session, session_uid, current_user.id)
    except agent_service.AgentServiceError as error:
        _raise_service_error(error)
    return _session_public(session, agent_session)


@session_router.patch('/{session_uid}')
def update_agent_session(
    session_uid: uuid.UUID,
    data: AgentSessionUpdate,
    session: SessionDep,
    current_user: UserDep,
) -> AgentSessionPublic:
    try:
        agent_session = agent_service.get_owned_session(session, session_uid, current_user.id)
        updated = agent_service.update_session(session, agent_session, data)
    except agent_service.AgentServiceError as error:
        _raise_service_error(error)
    return _session_public(session, updated)


@session_router.delete('/{session_uid}', status_code=status.HTTP_204_NO_CONTENT)
def delete_agent_session(
    session_uid: uuid.UUID,
    session: SessionDep,
    current_user: UserDep,
) -> Response:
    try:
        agent_session = agent_service.get_owned_session(session, session_uid, current_user.id)
        agent_service.delete_session(session, agent_session)
    except agent_service.AgentServiceError as error:
        _raise_service_error(error)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@session_router.get('/{session_uid}/messages')
def get_agent_messages(
    session_uid: uuid.UUID,
    session: SessionDep,
    current_user: UserDep,
) -> list[AgentMessagePublic]:
    try:
        agent_session = agent_service.get_owned_session(session, session_uid, current_user.id)
    except agent_service.AgentServiceError as error:
        _raise_service_error(error)
    return [
        AgentMessagePublic.model_validate(message) for message in agent_service.list_messages(session, agent_session)
    ]


@session_router.get('/{session_uid}/runs')
def get_agent_runs(
    session_uid: uuid.UUID,
    session: SessionDep,
    current_user: UserDep,
) -> list[AgentRunPublic]:
    try:
        agent_session = agent_service.get_owned_session(session, session_uid, current_user.id)
    except agent_service.AgentServiceError as error:
        _raise_service_error(error)
    return [AgentRunPublic.model_validate(run) for run in agent_service.list_runs(session, agent_session)]


@session_router.post('/{session_uid}/runs', status_code=status.HTTP_202_ACCEPTED)
def create_agent_run(
    session_uid: uuid.UUID,
    data: AgentRunCreate,
    session: SessionDep,
    current_user: UserDep,
) -> AgentRunPublic:
    try:
        agent_session = agent_service.get_owned_session(session, session_uid, current_user.id)
        run = agent_service.create_run(
            session,
            agent_session,
            data.agent_name,
            data.text,
            source_run_uid=data.source_run_uid,
        )
        run = _dispatch_run(session, run, data.language.value if data.language else None)
    except agent_service.AgentServiceError as error:
        _raise_service_error(error)
    return AgentRunPublic.model_validate(run)


@run_router.get('/{run_uid}')
def get_agent_run(
    run_uid: uuid.UUID,
    session: SessionDep,
    current_user: UserDep,
) -> AgentRunPublic:
    try:
        run = agent_service.get_owned_run(session, run_uid, current_user.id)
    except agent_service.AgentServiceError as error:
        _raise_service_error(error)
    return AgentRunPublic.model_validate(run)


@run_router.get('/{run_uid}/artifacts')
def get_agent_run_artifacts(
    run_uid: uuid.UUID,
    session: SessionDep,
    current_user: UserDep,
) -> list[AgentFilePublic]:
    """List controlled files relevant to a completed project agent run."""
    try:
        run = agent_service.get_owned_run(session, run_uid, current_user.id)
        return agent_file_service.list_run_artifacts(session, run, current_user.id)
    except (agent_service.AgentServiceError, agent_file_service.AgentFileError) as error:
        raise HTTPException(status_code=404, detail='Agent resource not found.') from error


@run_router.get('/{run_uid}/events')
def get_agent_events(
    run_uid: uuid.UUID,
    session: SessionDep,
    current_user: UserDep,
    after_id: Annotated[int, Query(ge=0)] = 0,
) -> list[AgentEventPublic]:
    try:
        run = agent_service.get_owned_run(session, run_uid, current_user.id)
    except agent_service.AgentServiceError as error:
        _raise_service_error(error)
    return [AgentEventPublic.model_validate(event) for event in agent_service.list_events(session, run, after_id)]


@run_router.post('/{run_uid}/resume', status_code=status.HTTP_202_ACCEPTED)
def resume_agent_run(
    run_uid: uuid.UUID,
    data: AgentRunResume,
    session: SessionDep,
    current_user: UserDep,
) -> AgentRunPublic:
    try:
        run = agent_service.get_owned_run(session, run_uid, current_user.id)
        if run.status != AgentRunStatus.WAITING_INPUT.value:
            raise agent_service.AgentRunTransitionError(RESUME_STATUS_ERROR)
        if data.feedback:
            agent_service.add_resume_message(session, run, data.feedback)
        run = agent_service.transition_run(
            session,
            run,
            AgentRunStatus.QUEUED,
            event_type=AgentEventType.QUEUED,
            payload={'resume': True, 'approved': data.approved, 'feedback': data.feedback},
        )
        run.result_payload = {'resume': {'approved': data.approved, 'feedback': data.feedback}}
        session.commit()
        run = _dispatch_run(session, run, data.language.value if data.language else None)
    except agent_service.AgentServiceError as error:
        _raise_service_error(error)
    return AgentRunPublic.model_validate(run)


@run_router.post('/{run_uid}/cancel', status_code=status.HTTP_202_ACCEPTED)
def cancel_agent_run(
    run_uid: uuid.UUID,
    session: SessionDep,
    current_user: UserDep,
) -> AgentRunPublic:
    from script.worker import celery_app

    try:
        run = agent_service.get_owned_run(session, run_uid, current_user.id)
        current_status = AgentRunStatus(run.status)
        if current_status in TERMINAL_AGENT_RUN_STATUSES:
            raise agent_service.AgentRunTransitionError(CANCEL_STATUS_ERROR)
        if current_status in {AgentRunStatus.QUEUED, AgentRunStatus.WAITING_INPUT}:
            if run.celery_task_id:
                celery_app.control.revoke(run.celery_task_id, terminate=False)
            run = agent_service.transition_run(
                session,
                run,
                AgentRunStatus.CANCELLED,
                event_type=AgentEventType.CANCELLED,
            )
        else:
            run.cancel_requested = True
            session.commit()
            run = agent_service.transition_run(
                session,
                run,
                AgentRunStatus.CANCEL_REQUESTED,
                event_type=AgentEventType.CANCEL_REQUESTED,
            )
    except agent_service.AgentServiceError as error:
        _raise_service_error(error)
    return AgentRunPublic.model_validate(run)


@run_router.post('/{run_uid}/retry', status_code=status.HTTP_202_ACCEPTED)
def retry_agent_run(
    run_uid: uuid.UUID,
    session: SessionDep,
    current_user: UserDep,
    data: AgentRunRetry | None = None,
) -> AgentRunPublic:
    try:
        old_run = agent_service.get_owned_run(session, run_uid, current_user.id)
        if AgentRunStatus(old_run.status) != AgentRunStatus.FAILED:
            raise agent_service.AgentRunTransitionError(RETRY_STATUS_ERROR)
        messages = [
            message
            for message in agent_crud.list_messages(session, old_run.session_uid)
            if message.run_uid == old_run.uid and message.role == 'user'
        ]
        if not messages:
            raise agent_service.AgentRunTransitionError(RETRY_INPUT_ERROR)
        text = next(str(part['text']) for part in messages[0].parts if part.get('type') == 'text' and part.get('text'))
        agent_session = agent_service.get_owned_session(session, old_run.session_uid, current_user.id)
        target = agent_crud.get_run_target(session, old_run.uid)
        run = agent_service.create_run(
            session,
            agent_session,
            AgentName(old_run.agent_name),
            text,
            retry_of_uid=old_run.uid,
            source_run_uid=target.source_run_uid if target is not None else None,
        )
        language = data.language.value if data and data.language else None
        run = _dispatch_run(session, run, language)
    except agent_service.AgentServiceError as error:
        _raise_service_error(error)
    return AgentRunPublic.model_validate(run)


def _sse_event(event: AgentEventPublic) -> str:
    return (
        f'id: {event.id}\n'
        f'event: {event.event_type.value}\n'
        f'data: {json.dumps(event.model_dump(mode="json"), ensure_ascii=False)}\n\n'
    )


@run_router.get('/{run_uid}/stream')
async def stream_agent_events(
    request: Request,
    run_uid: uuid.UUID,
    session: SessionDep,
    current_user: UserDep,
    after_id: Annotated[int, Query(ge=0)] = 0,
) -> StreamingResponse:
    try:
        agent_service.get_owned_run(session, run_uid, current_user.id)
    except agent_service.AgentServiceError as error:
        _raise_service_error(error)
    owner_id = current_user.id
    is_admin = current_user.role == UserRole.ADMIN
    session.close()

    header_id = request.headers.get('last-event-id')
    if header_id and header_id.isdigit():
        after_id = max(after_id, int(header_id))

    async def generate() -> AsyncGenerator[str, None]:
        from app.tasks.agent_run import AGENT_EVENT_CHANNEL_PREFIX

        cursor = after_id
        last_heartbeat = time.monotonic()
        channel = f'{AGENT_EVENT_CHANNEL_PREFIX}{run_uid}'
        pubsub = None
        pubsub_available = False
        try:
            pubsub = request.app.state.redis.pubsub()
            await pubsub.subscribe(channel)
            pubsub_available = True
        except Exception:
            pubsub_available = False
        try:
            while not await request.is_disconnected():
                with Session(engine) as event_session:
                    run = agent_crud.get_run(event_session, run_uid)
                    if run is None or (run.agent_session.owner_id != owner_id and not is_admin):
                        return
                    events = agent_crud.list_events(event_session, run_uid, cursor)
                    status_value = AgentRunStatus(run.status)

                for stored_event in events:
                    event = AgentEventPublic.model_validate(stored_event)
                    cursor = event.id
                    yield _sse_event(event)

                if status_value in TERMINAL_AGENT_RUN_STATUSES or status_value == AgentRunStatus.WAITING_INPUT:
                    return

                if time.monotonic() - last_heartbeat >= SSE_HEARTBEAT_SECONDS:
                    yield ': heartbeat\n\n'
                    last_heartbeat = time.monotonic()

                if not pubsub_available or pubsub is None:
                    await asyncio.sleep(1)
                    continue
                try:
                    await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                except Exception:
                    pubsub_available = False
                    await asyncio.sleep(1)
        finally:
            if pubsub is not None:
                with suppress(Exception):
                    await pubsub.unsubscribe(channel)
                    await pubsub.aclose()

    return StreamingResponse(
        generate(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache, no-transform',
            'X-Accel-Buffering': 'no',
        },
    )
