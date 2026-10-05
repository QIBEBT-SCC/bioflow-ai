"""Redis-backed lifecycle primitives for ephemeral CODE ACP sessions."""

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from redis.asyncio import Redis

from bioflow_ai.core.config import get_settings
from bioflow_ai.schemas.coding_agent import CodeAgentEvent

SESSION_PREFIX = 'bioflow:code-agent:session'
GLOBAL_SESSIONS_KEY = f'{SESSION_PREFIX}:active'
ADMISSION_LOCK_KEY = f'{SESSION_PREFIX}:admission-lock'


class CodeAgentSessionError(ValueError):
    """Base error for invalid ephemeral session operations."""


class CodeAgentSessionNotFoundError(CodeAgentSessionError):
    """Raised when a session is absent or belongs to a different user."""


class CodeAgentSessionBusyError(CodeAgentSessionError):
    """Raised when a turn or proposal prevents the requested operation."""


class CodeAgentConcurrencyError(CodeAgentSessionError):
    """Raised when configured session concurrency is exhausted."""


def session_key(session_id: uuid.UUID | str) -> str:
    """Return the Redis session hash key."""
    return f'{SESSION_PREFIX}:{session_id}'


def command_key(session_id: uuid.UUID | str) -> str:
    """Return the Redis command stream key."""
    return f'{session_key(session_id)}:commands'


def event_key(session_id: uuid.UUID | str) -> str:
    """Return the Redis event stream key."""
    return f'{session_key(session_id)}:events'


def proposal_key(session_id: uuid.UUID | str, proposal_id: uuid.UUID | str) -> str:
    """Return a pending proposal key."""
    return f'{session_key(session_id)}:proposal:{proposal_id}'


def user_sessions_key(owner_id: int) -> str:
    """Return a user's active-session set key."""
    return f'{SESSION_PREFIX}:user:{owner_id}'


async def _prune_set(redis: Redis, key: str) -> int:
    members = await redis.smembers(key)  # ty: ignore[invalid-await]
    stale = [member for member in members if not await redis.exists(session_key(member))]
    if stale:
        await redis.srem(key, *stale)  # ty: ignore[invalid-await]
    return len(members) - len(stale)


async def create_session(
    redis: Redis,
    session_id: uuid.UUID,
    *,
    owner_id: int,
    node_type: str,
    provider: str = 'codex',
) -> None:
    """Reserve concurrency and persist one starting session atomically enough for Redis failure recovery."""
    config = get_settings().code_agent
    lock = redis.lock(ADMISSION_LOCK_KEY, timeout=10, blocking_timeout=5, thread_local=False)
    async with lock:
        user_key = user_sessions_key(owner_id)
        user_count = await _prune_set(redis, user_key)
        system_count = await _prune_set(redis, GLOBAL_SESSIONS_KEY)
        if user_count >= config.MAX_SESSIONS_PER_USER or system_count >= config.MAX_SESSIONS_SYSTEM:
            message = 'Coding-agent session limit reached'
            raise CodeAgentConcurrencyError(message)
        key = session_key(session_id)
        async with redis.pipeline(transaction=True) as pipeline:
            pipeline.hset(
                key,
                mapping={
                    'owner_id': str(owner_id),
                    'node_type': node_type,
                    'provider': provider,
                    'status': 'starting',
                },
            )
            pipeline.expire(key, config.SESSION_TTL_SECONDS)
            pipeline.sadd(user_key, str(session_id))
            pipeline.expire(user_key, config.SESSION_TTL_SECONDS)
            pipeline.sadd(GLOBAL_SESSIONS_KEY, str(session_id))
            pipeline.expire(GLOBAL_SESSIONS_KEY, config.SESSION_TTL_SECONDS)
            await pipeline.execute()


async def active_session_count(redis: Redis) -> int:
    """Return the number of live coding sessions after pruning expired entries."""
    return await _prune_set(redis, GLOBAL_SESSIONS_KEY)


async def get_owned_session(redis: Redis, session_id: uuid.UUID, owner_id: int) -> dict[str, str]:
    """Return session state only to its owning administrator."""
    state = await redis.hgetall(session_key(session_id))  # ty: ignore[invalid-await]
    if not state or state.get('owner_id') != str(owner_id):
        raise CodeAgentSessionNotFoundError
    await touch_session(redis, session_id)
    return state


async def touch_session(redis: Redis, session_id: uuid.UUID | str) -> None:
    """Refresh ephemeral keys while a client or runner remains active."""
    config = get_settings().code_agent
    await redis.expire(session_key(session_id), config.SESSION_TTL_SECONDS)
    await redis.expire(command_key(session_id), config.SESSION_TTL_SECONDS)
    await redis.expire(event_key(session_id), config.EVENT_TTL_SECONDS)
    await redis.expire(GLOBAL_SESSIONS_KEY, config.SESSION_TTL_SECONDS)
    owner_id = await redis.hget(session_key(session_id), 'owner_id')  # ty: ignore[invalid-await]
    if owner_id is not None:
        await redis.expire(user_sessions_key(int(owner_id)), config.SESSION_TTL_SECONDS)


async def set_status(redis: Redis, session_id: uuid.UUID | str, status: str) -> None:
    """Update the page-visible session state."""
    await redis.hset(session_key(session_id), 'status', status)  # ty: ignore[invalid-await]
    await touch_session(redis, session_id)


async def append_event(redis: Redis, session_id: uuid.UUID | str, event_type: str, data: dict[str, Any]) -> str:
    """Persist one normalized event and return its Redis stream ID."""
    key = event_key(session_id)
    event_id = await redis.xadd(key, {'type': event_type, 'data': json.dumps(data, separators=(',', ':'))}, maxlen=1000)
    await redis.expire(key, get_settings().code_agent.EVENT_TTL_SECONDS)
    return str(event_id)


async def append_command(redis: Redis, session_id: uuid.UUID, command: str, data: dict[str, Any]) -> str:
    """Append a runner command to the session stream."""
    key = command_key(session_id)
    command_id = await redis.xadd(key, {'command': command, 'data': json.dumps(data, separators=(',', ':'))})
    await touch_session(redis, session_id)
    return str(command_id)


async def read_events(
    redis: Redis,
    session_id: uuid.UUID,
    *,
    after_id: str,
) -> AsyncIterator[CodeAgentEvent | None]:
    """Yield persisted and new events with resumable Redis stream IDs."""
    cursor = after_id
    while True:
        rows = await redis.xread({event_key(session_id): cursor}, count=100, block=15_000)
        if not rows:
            await touch_session(redis, session_id)
            if not await redis.exists(session_key(session_id)):
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
                yield CodeAgentEvent(id=cursor, type=fields.get('type', 'tool.updated'), data=data)


async def save_proposal(
    redis: Redis, session_id: uuid.UUID | str, proposal_id: uuid.UUID, payload: dict[str, Any]
) -> None:
    """Persist one pending whole-turn proposal."""
    key = proposal_key(session_id, proposal_id)
    await redis.set(key, json.dumps(payload, separators=(',', ':')), ex=get_settings().code_agent.SESSION_TTL_SECONDS)
    await redis.hset(  # ty: ignore[invalid-await]
        session_key(session_id), mapping={'proposal_id': str(proposal_id), 'status': 'proposal'}
    )


async def decide_proposal(
    redis: Redis,
    session_id: uuid.UUID,
    proposal_id: uuid.UUID,
    decision: str,
) -> None:
    """Atomically mark the current proposal and release the editor session."""
    state = await redis.hgetall(session_key(session_id))  # ty: ignore[invalid-await]
    if state.get('status') != 'proposal' or state.get('proposal_id') != str(proposal_id):
        message = 'Proposal is no longer pending'
        raise CodeAgentSessionBusyError(message)
    key = proposal_key(session_id, proposal_id)
    raw = await redis.get(key)
    if raw is None:
        message = 'Proposal expired'
        raise CodeAgentSessionBusyError(message)
    payload = json.loads(raw)
    payload['decision'] = decision
    async with redis.pipeline(transaction=True) as pipeline:
        pipeline.set(key, json.dumps(payload, separators=(',', ':')), ex=get_settings().code_agent.SESSION_TTL_SECONDS)
        pipeline.hset(session_key(session_id), mapping={'status': 'ready', 'proposal_id': ''})
        await pipeline.execute()
    await append_command(redis, session_id, decision, {'proposal_id': str(proposal_id)})


async def release_session(redis: Redis, session_id: uuid.UUID | str, owner_id: int) -> None:
    """Release concurrency state while retaining the short-lived event stream."""
    async with redis.pipeline(transaction=True) as pipeline:
        pipeline.srem(user_sessions_key(owner_id), str(session_id))
        pipeline.srem(GLOBAL_SESSIONS_KEY, str(session_id))
        pipeline.delete(session_key(session_id), command_key(session_id))
        await pipeline.execute()
    await redis.expire(event_key(session_id), get_settings().code_agent.EVENT_TTL_SECONDS)
