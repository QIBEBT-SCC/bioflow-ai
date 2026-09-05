"""Redis coordination for the single administrator-managed Codex device login."""

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from redis.asyncio import Redis

from app.core.config import get_settings
from app.schemas.coding_agent import CodeAgentEvent

LOGIN_PREFIX = 'bioflow:code-agent:login'
ACTIVE_LOGIN_KEY = f'{LOGIN_PREFIX}:active'


class CodingAgentLoginError(ValueError):
    """Base error for invalid device-login operations."""


class CodingAgentLoginBusyError(CodingAgentLoginError):
    """Raised when another administrator login is still active."""


class CodingAgentLoginNotFoundError(CodingAgentLoginError):
    """Raised when a login is absent or belongs to another administrator."""


def login_key(login_id: uuid.UUID | str) -> str:
    """Return the Redis hash key for a login attempt."""
    return f'{LOGIN_PREFIX}:{login_id}'


def event_key(login_id: uuid.UUID | str) -> str:
    """Return the Redis event stream key for a login attempt."""
    return f'{login_key(login_id)}:events'


async def create_login(redis: Redis, login_id: uuid.UUID, owner_id: int) -> None:
    """Reserve the one global login slot and persist the initiating administrator."""
    ttl = get_settings().code_agent.LOGIN_TTL_SECONDS
    reserved = await redis.set(ACTIVE_LOGIN_KEY, str(login_id), ex=ttl, nx=True)
    if not reserved:
        active = await redis.get(ACTIVE_LOGIN_KEY)
        if active and await redis.exists(login_key(active)):
            message = 'A Codex device login is already in progress'
            raise CodingAgentLoginBusyError(message)
        await redis.delete(ACTIVE_LOGIN_KEY)
        reserved = await redis.set(ACTIVE_LOGIN_KEY, str(login_id), ex=ttl, nx=True)
        if not reserved:
            message = 'A Codex device login is already in progress'
            raise CodingAgentLoginBusyError(message)
    await redis.hset(  # pyrefly: ignore [not-async]
        login_key(login_id),
        mapping={'owner_id': str(owner_id), 'status': 'starting', 'cancel_requested': '0'},
    )
    await redis.expire(login_key(login_id), ttl)


async def get_owned_login(redis: Redis, login_id: uuid.UUID, owner_id: int) -> dict[str, str]:
    """Return one login only to the administrator who started it."""
    state = await redis.hgetall(login_key(login_id))  # pyrefly: ignore [not-async]
    if not state or state.get('owner_id') != str(owner_id):
        raise CodingAgentLoginNotFoundError
    return state


async def set_status(redis: Redis, login_id: uuid.UUID | str, status: str) -> None:
    """Update the temporary login status and refresh its TTL."""
    await redis.hset(login_key(login_id), 'status', status)  # pyrefly: ignore [not-async]
    await redis.expire(login_key(login_id), get_settings().code_agent.LOGIN_TTL_SECONDS)


async def request_cancel(redis: Redis, login_id: uuid.UUID) -> None:
    """Set the cancellation flag consumed by the Celery login owner."""
    await redis.hset(login_key(login_id), 'cancel_requested', '1')  # pyrefly: ignore [not-async]


async def cancel_requested(redis: Redis, login_id: uuid.UUID | str) -> bool:
    """Return whether the administrator cancelled this login."""
    return await redis.hget(login_key(login_id), 'cancel_requested') == '1'  # pyrefly: ignore [not-async]


async def append_event(redis: Redis, login_id: uuid.UUID | str, event_type: str, data: dict[str, Any]) -> str:
    """Persist a normalized login SSE event."""
    key = event_key(login_id)
    event_id = await redis.xadd(key, {'type': event_type, 'data': json.dumps(data, separators=(',', ':'))}, maxlen=100)
    await redis.expire(key, get_settings().code_agent.LOGIN_TTL_SECONDS)
    return str(event_id)


async def read_events(
    redis: Redis,
    login_id: uuid.UUID,
    *,
    after_id: str,
) -> AsyncIterator[CodeAgentEvent | None]:
    """Yield persisted and new login events with heartbeat markers."""
    cursor = after_id
    while True:
        rows = await redis.xread({event_key(login_id): cursor}, count=20, block=15_000)
        if not rows:
            if not await redis.exists(login_key(login_id)):
                return
            yield None
            continue
        for _stream, events in rows:
            for event_id, fields in events:
                cursor = str(event_id)
                raw_data = fields.get('data', '{}')
                try:
                    data = json.loads(raw_data)
                except json.JSONDecodeError:
                    data = {'message': raw_data}
                event_type = fields.get('type', 'login.failed')
                yield CodeAgentEvent(id=cursor, type=event_type, data=data)
                if event_type in {'login.completed', 'login.failed', 'login.cancelled'}:
                    return


async def finish_login(redis: Redis, login_id: uuid.UUID | str, status: str) -> None:
    """Release the global slot while retaining final state/events for reconnecting SSE clients."""
    await set_status(redis, login_id, status)
    active = await redis.get(ACTIVE_LOGIN_KEY)
    if active == str(login_id):
        await redis.delete(ACTIVE_LOGIN_KEY)
