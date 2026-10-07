"""Administrator-only temporary conversations with the built-in coding agent."""

import asyncio
import re
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Query
from starlette.responses import StreamingResponse

from bioflow_ai.core.auth import AdminDep
from bioflow_ai.infrastructure.database import SessionDep
from bioflow_ai.infrastructure.redis.client import RedisDep
from bioflow_ai.models.setting import ModelRole
from bioflow_ai.schemas.code import BASH_DEPENDENCIES_ERROR, CodeNodeType
from bioflow_ai.schemas.code_agent import (
    CodeAgentAvailability,
    CodeAgentProposalDecision,
    CodeAgentSessionCreate,
    CodeAgentSessionPublic,
    CodeAgentTurnCreate,
)
from bioflow_ai.services import llm_setting
from bioflow_ai.services.code.sessions import CodeSessionBusyError, CodeSessionNotFoundError, CodeSessionStore
from bioflow_ai.tasks.code import normalize_python_dependencies, normalize_r_dependencies
from bioflow_ai.tasks.code_editing import execute_code_turn

router = APIRouter()


def dependencies_for(node_type: CodeNodeType, dependencies: list[str]) -> list[str]:
    """Validate the editor baseline against the existing CODE dependency contract."""
    try:
        if node_type is CodeNodeType.PYTHON:
            return normalize_python_dependencies(dependencies)
        if node_type is CodeNodeType.R:
            return normalize_r_dependencies(dependencies)
    except (TypeError, ValueError) as error:
        raise HTTPException(422, str(error)) from error
    if dependencies:
        raise HTTPException(422, BASH_DEPENDENCIES_ERROR)
    return []


def session_error(error: CodeSessionNotFoundError | CodeSessionBusyError) -> HTTPException:
    """Keep missing and foreign sessions indistinguishable."""
    if isinstance(error, CodeSessionNotFoundError):
        return HTTPException(404, 'Coding session not found or expired.')
    return HTTPException(409, str(error))


@router.get('/availability')
def availability(session: SessionDep, _current_user: AdminDep) -> CodeAgentAvailability:
    """Inspect central Coding configuration without constructing or calling a model."""
    try:
        model = llm_setting.get_model(session, llm_setting.get_role_model_id(session, ModelRole.CODING))
    except llm_setting.LLMSettingServiceError:
        return CodeAgentAvailability(available=False)
    return CodeAgentAvailability(available=True, model_name=model.name)


@router.post('/sessions')
async def create_session(
    data: CodeAgentSessionCreate,
    redis: RedisDep,
    current_user: AdminDep,
) -> CodeAgentSessionPublic:
    """Create an idle session with a fixed code language and authenticated owner."""
    store = CodeSessionStore(redis)
    return store.public(await store.create(current_user.id, data.node_type))


@router.get('/sessions/{session_id}')
async def get_session(session_id: uuid.UUID, redis: RedisDep, current_user: AdminDep) -> CodeAgentSessionPublic:
    """Recover the current transcript, execution state, and pending proposal."""
    store = CodeSessionStore(redis)
    try:
        return store.public(await store.get(str(session_id), current_user.id))
    except (CodeSessionNotFoundError, CodeSessionBusyError) as error:
        raise session_error(error) from error


@router.delete('/sessions/{session_id}')
async def close_session(session_id: uuid.UUID, redis: RedisDep, current_user: AdminDep) -> CodeAgentSessionPublic:
    """Discard the conversation and immediately invalidate any active task."""
    store = CodeSessionStore(redis)
    try:
        return store.public(await store.close(str(session_id), current_user.id))
    except (CodeSessionNotFoundError, CodeSessionBusyError) as error:
        raise session_error(error) from error


@router.post('/sessions/{session_id}/turns')
async def submit_turn(
    session_id: uuid.UUID,
    data: CodeAgentTurnCreate,
    redis: RedisDep,
    current_user: AdminDep,
) -> CodeAgentSessionPublic:
    """Reserve a turn and dispatch one independently scheduled agent-queue task."""
    store = CodeSessionStore(redis)
    sid = str(session_id)
    try:
        state = await store.get(sid, current_user.id)
        # Validate, but retain the exact original baseline for the proposal fingerprint.
        dependencies_for(CodeNodeType(state['node_type']), data.dependencies)
        state = await store.submit(sid, current_user.id, data)
    except (CodeSessionNotFoundError, CodeSessionBusyError) as error:
        raise session_error(error) from error
    try:
        await asyncio.to_thread(execute_code_turn.delay, sid, state['turn_id'])
    except Exception as error:
        await store.fail(sid, state['turn_id'], 'Could not schedule the coding turn. Please retry.')
        raise HTTPException(503, 'Could not schedule the coding turn. Please retry.') from error
    return store.public(state)


@router.post('/sessions/{session_id}/stop')
async def stop_turn(session_id: uuid.UUID, redis: RedisDep, current_user: AdminDep) -> CodeAgentSessionPublic:
    """Cancel queued work or request safe cancellation of the active turn."""
    store = CodeSessionStore(redis)
    try:
        return store.public(await store.cancel(str(session_id), current_user.id))
    except (CodeSessionNotFoundError, CodeSessionBusyError) as error:
        raise session_error(error) from error


@router.post('/sessions/{session_id}/proposals/{proposal_id}')
async def decide_proposal(
    session_id: uuid.UUID,
    proposal_id: uuid.UUID,
    data: CodeAgentProposalDecision,
    redis: RedisDep,
    current_user: AdminDep,
) -> CodeAgentSessionPublic:
    """Record review feedback; accepted content is applied by the frontend form only."""
    store = CodeSessionStore(redis)
    try:
        return store.public(
            await store.decide(
                str(session_id),
                current_user.id,
                str(proposal_id),
                data.decision,
                data.source,
                data.dependencies,
            )
        )
    except (CodeSessionNotFoundError, CodeSessionBusyError) as error:
        raise session_error(error) from error


@router.get('/sessions/{session_id}/events')
async def events(
    session_id: uuid.UUID,
    redis: RedisDep,
    current_user: AdminDep,
    after: Annotated[str, Query(max_length=40)] = '0-0',
    last_event_id: Annotated[str | None, Header(max_length=40)] = None,
) -> StreamingResponse:
    """Replay from a validated cursor, then stream replies, tools, and state changes."""
    store = CodeSessionStore(redis)
    sid = str(session_id)
    cursor = last_event_id or after
    if re.fullmatch(r'\d{1,19}-\d{1,19}', cursor) is None:
        raise HTTPException(422, 'Invalid event cursor.')
    try:
        await store.get(sid, current_user.id)
    except (CodeSessionNotFoundError, CodeSessionBusyError) as error:
        raise session_error(error) from error

    async def stream() -> AsyncIterator[str]:
        try:
            async for event in store.read_events(sid, current_user.id, cursor):
                if event.type == 'heartbeat':
                    yield ': heartbeat\n\n'
                else:
                    yield f'id: {event.id}\nevent: {event.type}\ndata: {event.model_dump_json()}\n\n'
        except CodeSessionNotFoundError:
            yield 'event: session.closed\ndata: {"type":"session.closed","data":{}}\n\n'

    return StreamingResponse(
        stream(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )
