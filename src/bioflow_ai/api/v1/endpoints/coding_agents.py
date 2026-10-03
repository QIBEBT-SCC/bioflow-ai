"""Provider-specific administration APIs for ACP coding agents."""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from bioflow_ai.core.auth import AdminDep
from bioflow_ai.infrastructure.redis.client import RedisDep
from bioflow_ai.schemas.coding_agent import (
    CodexAgentSettings,
    CodexLoginPublic,
    OpenCodeCredentialsCreate,
    OpenCodeSettings,
)
from bioflow_ai.services.code_agent import account as account_service
from bioflow_ai.services.code_agent import login as login_store
from bioflow_ai.services.code_agent import session as session_store
from bioflow_ai.services.code_agent import settings as settings_service

router = APIRouter()


@router.post('/providers/codex/login', status_code=status.HTTP_202_ACCEPTED)
async def start_codex_login(
    redis: RedisDep,
    current_user: AdminDep,
) -> CodexLoginPublic:
    """Start the only supported sign-in method: ChatGPT device code."""
    from bioflow_ai.tasks.code_agent import login_codex_account

    if await session_store.active_session_count(redis):
        raise HTTPException(status_code=409, detail='Close active coding-agent sessions before signing in')
    if account_service.credentials_configured():
        raise HTTPException(status_code=409, detail='Codex is already connected')
    login_id = uuid.uuid4()
    try:
        await login_store.create_login(redis, login_id, current_user.id)
    except login_store.CodingAgentLoginBusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    try:
        login_codex_account.apply_async(args=(str(login_id),), queue='acp')
    except Exception as error:
        await login_store.finish_login(redis, login_id, 'failed')
        raise HTTPException(status_code=503, detail='Coding-agent worker is unavailable') from error
    return CodexLoginPublic(id=login_id, status='starting')


@router.get('/providers/codex/login/{login_id}/events')
async def stream_codex_login_events(
    login_id: uuid.UUID,
    request: Request,
    redis: RedisDep,
    current_user: AdminDep,
    after_id: Annotated[str, Query(pattern=r'^(?:0|\d+)-(?:0|\d+)$')] = '0-0',
) -> StreamingResponse:
    """Stream the device URL/code and final result to the initiating administrator."""
    try:
        await login_store.get_owned_login(redis, login_id, current_user.id)
    except login_store.CodingAgentLoginNotFoundError as error:
        raise HTTPException(status_code=404, detail='Codex login not found') from error
    last_event_id = request.headers.get('Last-Event-ID') or after_id

    async def generate() -> AsyncIterator[str]:
        async for event in login_store.read_events(redis, login_id, after_id=last_event_id):
            if await request.is_disconnected():
                return
            if event is None:
                yield ': heartbeat\n\n'
                continue
            payload = json.dumps(event.data, separators=(',', ':'))
            yield f'id: {event.id}\nevent: {event.type}\ndata: {payload}\n\n'

    return StreamingResponse(
        generate(),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@router.post('/providers/codex/login/{login_id}/cancel', status_code=status.HTTP_202_ACCEPTED)
async def cancel_codex_login(
    login_id: uuid.UUID,
    redis: RedisDep,
    current_user: AdminDep,
) -> dict[str, str]:
    """Ask the ACP owner to cancel an outstanding device-code login."""
    try:
        await login_store.get_owned_login(redis, login_id, current_user.id)
    except login_store.CodingAgentLoginNotFoundError as error:
        raise HTTPException(status_code=404, detail='Codex login not found') from error
    await login_store.request_cancel(redis, login_id)
    return {'status': 'cancelling'}


@router.get('/providers/codex/settings')
async def get_codex_settings(_current_user: AdminDep) -> CodexAgentSettings:
    """Return native Codex defaults for new sessions."""
    return await asyncio.to_thread(settings_service.read_settings)


@router.put('/providers/codex/settings')
async def update_codex_settings(data: CodexAgentSettings, _current_user: AdminDep) -> CodexAgentSettings:
    """Save site-wide policy without changing active sessions."""
    return await asyncio.to_thread(settings_service.write_settings, data)


@router.put('/providers/opencode/credentials')
async def save_opencode_credentials(
    data: OpenCodeCredentialsCreate,
    _current_user: AdminDep,
) -> OpenCodeSettings:
    """Save credentials without ever returning the secret to the browser."""
    key = data.api_key.get_secret_value().strip()
    if not key:
        raise HTTPException(status_code=422, detail='API key must not be blank')
    try:
        return await asyncio.to_thread(
            account_service.write_opencode_credentials,
            key,
            data.model_provider,
            str(data.base_url) if data.base_url is not None else None,
            data.model_id,
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get('/providers/opencode/settings')
async def get_opencode_settings(_current_user: AdminDep) -> OpenCodeSettings:
    """Return the selected provider and non-secret custom endpoint fields."""
    return await asyncio.to_thread(account_service.read_opencode_settings)
