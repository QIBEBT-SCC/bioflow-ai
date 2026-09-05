"""Page-scoped CODE ACP sessions exposed through REST and resumable SSE."""

import json
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse

from app.core.auth import AdminDep
from app.core.redis_client import RedisDep
from app.schemas.code import CodeNodeType
from app.schemas.coding_agent import (
    CodeAgentConfigUpdate,
    CodeAgentProposalDecision,
    CodeAgentSessionCreate,
    CodeAgentSessionPublic,
    CodeAgentTurnCreate,
    CodingAgentAvailability,
    CodingAgentProviderAvailability,
)
from app.services.code_agent import account as account_service
from app.services.code_agent import session as session_store
from app.tasks.code import normalize_python_dependencies, normalize_r_dependencies

router = APIRouter()


def _session_error(error: Exception) -> HTTPException:
    if isinstance(error, session_store.CodeAgentSessionNotFoundError):
        return HTTPException(status_code=404, detail='Coding-agent session not found')
    if isinstance(error, session_store.CodeAgentConcurrencyError):
        return HTTPException(status_code=429, detail=str(error))
    return HTTPException(status_code=409, detail=str(error))


def _validate_dependencies(node_type: CodeNodeType, dependencies: list[str]) -> list[str]:
    if node_type is CodeNodeType.BASH:
        if dependencies:
            raise HTTPException(status_code=422, detail='Bash CODE definitions cannot declare runtime dependencies')
        return []
    try:
        if node_type is CodeNodeType.PYTHON:
            return normalize_python_dependencies(dependencies)
        return normalize_r_dependencies(dependencies)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get('/availability')
async def get_coding_agent_availability(
    _current_user: AdminDep,
) -> CodingAgentAvailability:
    """Expose the AI entry point when a supported provider has configured credentials."""
    providers = [
        CodingAgentProviderAvailability(
            provider=provider, name=name, available=account_service.credentials_configured(provider)
        )
        for provider, name in (('codex', 'Codex'), ('opencode', 'OpenCode'))
    ]
    return CodingAgentAvailability(available=any(item.available for item in providers), providers=providers)


@router.post('/sessions', status_code=status.HTTP_202_ACCEPTED)
async def create_code_agent_session(
    data: CodeAgentSessionCreate,
    redis: RedisDep,
    current_user: AdminDep,
) -> CodeAgentSessionPublic:
    """Create an ephemeral session and dispatch its long-lived ACP owner."""
    from app.tasks.code_agent import execute_code_agent_session

    if not account_service.credentials_configured(data.provider):
        raise HTTPException(status_code=503, detail=f'{data.provider} credentials are not configured')
    session_id = uuid.uuid4()
    try:
        await session_store.create_session(
            redis,
            session_id,
            owner_id=current_user.id,
            node_type=data.node_type.value,
            provider=data.provider,
        )
    except session_store.CodeAgentSessionError as error:
        raise _session_error(error) from error
    try:
        execute_code_agent_session.apply_async(args=(str(session_id),), queue='acp')
    except Exception as error:
        await session_store.release_session(redis, session_id, current_user.id)
        raise HTTPException(status_code=503, detail='Coding-agent worker is unavailable') from error
    return CodeAgentSessionPublic(id=session_id, status='starting')


@router.post('/sessions/{session_id}/turns', status_code=status.HTTP_202_ACCEPTED)
async def create_code_agent_turn(
    session_id: uuid.UUID,
    data: CodeAgentTurnCreate,
    redis: RedisDep,
    current_user: AdminDep,
) -> dict[str, str]:
    """Queue a turn with the exact editor source and dependency baseline."""
    try:
        state = await session_store.get_owned_session(redis, session_id, current_user.id)
        if state.get('status') != 'ready':
            message = 'Coding-agent session is not ready'
            raise session_store.CodeAgentSessionBusyError(message)
        dependencies = _validate_dependencies(CodeNodeType(state['node_type']), data.dependencies)
        await session_store.append_command(
            redis,
            session_id,
            'turn',
            {'prompt': data.prompt, 'source': data.source, 'dependencies': dependencies},
        )
    except session_store.CodeAgentSessionError as error:
        raise _session_error(error) from error
    return {'status': 'queued'}


@router.get('/sessions/{session_id}/events')
async def stream_code_agent_events(
    session_id: uuid.UUID,
    request: Request,
    redis: RedisDep,
    current_user: AdminDep,
    after_id: Annotated[str, Query(pattern=r'^(?:0|\d+)-(?:0|\d+)$')] = '0-0',
) -> StreamingResponse:
    """Stream normalized persisted events, resuming after a Redis stream ID."""
    try:
        await session_store.get_owned_session(redis, session_id, current_user.id)
    except session_store.CodeAgentSessionError as error:
        raise _session_error(error) from error
    last_event_id = request.headers.get('Last-Event-ID') or after_id

    async def generate() -> AsyncIterator[str]:
        async for event in session_store.read_events(redis, session_id, after_id=last_event_id):
            if await request.is_disconnected():
                return
            if event is None:
                yield ': heartbeat\n\n'
                continue
            payload = json.dumps(event.data, separators=(',', ':'))
            yield f'id: {event.id}\nevent: {event.type}\ndata: {payload}\n\n'
            if event.type == 'session.closed':
                return

    return StreamingResponse(
        generate(),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@router.post('/sessions/{session_id}/cancel', status_code=status.HTTP_202_ACCEPTED)
async def cancel_code_agent_turn(
    session_id: uuid.UUID,
    redis: RedisDep,
    current_user: AdminDep,
) -> dict[str, str]:
    """Request cancellation of the active ACP prompt."""
    try:
        await session_store.get_owned_session(redis, session_id, current_user.id)
        await session_store.append_command(redis, session_id, 'cancel', {})
    except session_store.CodeAgentSessionError as error:
        raise _session_error(error) from error
    return {'status': 'cancelling'}


async def _decide(
    session_id: uuid.UUID,
    proposal_id: uuid.UUID,
    decision: str,
    redis: RedisDep,
    current_user: AdminDep,
) -> CodeAgentProposalDecision:
    try:
        await session_store.get_owned_session(redis, session_id, current_user.id)
        await session_store.decide_proposal(redis, session_id, proposal_id, decision)
    except session_store.CodeAgentSessionError as error:
        raise _session_error(error) from error
    result = 'accepted' if decision == 'accept' else 'rejected'
    return CodeAgentProposalDecision(proposal_id=proposal_id, decision=result)


@router.post('/sessions/{session_id}/proposals/{proposal_id}/accept')
async def accept_code_agent_proposal(
    session_id: uuid.UUID,
    proposal_id: uuid.UUID,
    redis: RedisDep,
    current_user: AdminDep,
) -> CodeAgentProposalDecision:
    """Accept the whole turn into the temporary baseline."""
    return await _decide(session_id, proposal_id, 'accept', redis, current_user)


@router.post('/sessions/{session_id}/proposals/{proposal_id}/reject')
async def reject_code_agent_proposal(
    session_id: uuid.UUID,
    proposal_id: uuid.UUID,
    redis: RedisDep,
    current_user: AdminDep,
) -> CodeAgentProposalDecision:
    """Reject the whole turn and restore its pre-turn workspace snapshot."""
    return await _decide(session_id, proposal_id, 'reject', redis, current_user)


@router.delete('/sessions/{session_id}', status_code=status.HTTP_204_NO_CONTENT)
async def close_code_agent_session(
    session_id: uuid.UUID,
    redis: RedisDep,
    current_user: AdminDep,
) -> Response:
    """Request immediate adapter and workspace teardown."""
    try:
        await session_store.get_owned_session(redis, session_id, current_user.id)
        await session_store.append_command(redis, session_id, 'close', {})
    except session_store.CodeAgentSessionError as error:
        raise _session_error(error) from error
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.put('/sessions/{session_id}/config', status_code=status.HTTP_202_ACCEPTED)
async def update_code_agent_config(
    session_id: uuid.UUID,
    data: CodeAgentConfigUpdate,
    redis: RedisDep,
    current_user: AdminDep,
) -> dict[str, str]:
    """Queue an ACP configuration change while the session is idle."""
    try:
        state = await session_store.get_owned_session(redis, session_id, current_user.id)
        if state.get('status') != 'ready':
            message = 'Coding-agent session is not ready'
            raise session_store.CodeAgentSessionBusyError(message)
        await session_store.append_command(redis, session_id, 'config', data.model_dump())
    except session_store.CodeAgentSessionError as error:
        raise _session_error(error) from error
    return {'status': 'queued'}
