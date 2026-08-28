import hashlib
import secrets
import time
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError
from redis.asyncio import Redis

from app.core.config import get_settings

SESSION_COOKIE = 'session_id'
CSRF_COOKIE = 'csrf_token'
LEGACY_AUTH_COOKIES = ('access_token', 'refresh_token')


class SessionRecord(BaseModel):
    """Server-side browser session stored in Redis."""

    user_id: int
    csrf_token: str
    issued_at: int
    absolute_expires_at: int


@dataclass(frozen=True, slots=True)
class IssuedSession:
    session_id: str
    csrf_token: str
    absolute_expires_at: int


@dataclass(frozen=True, slots=True)
class AuthSession:
    digest: str
    record: SessionRecord


def _session_digest(session_id: str) -> str:
    return hashlib.sha256(session_id.encode()).hexdigest()


def _session_key(digest: str) -> str:
    return f'auth-session:{digest}'


def _user_sessions_key(user_id: int) -> str:
    return f'user-sessions:{user_id}'


def _idle_ttl() -> int:
    return get_settings().security.SESSION_IDLE_TIMEOUT_HOURS * 3600


def _absolute_ttl() -> int:
    return get_settings().security.SESSION_ABSOLUTE_TIMEOUT_DAYS * 86400


async def create_session(redis: Redis, user_id: int) -> IssuedSession:
    """Create an opaque, revocable browser session."""
    now = int(time.time())
    session_id = secrets.token_urlsafe(32)
    csrf_token = secrets.token_urlsafe(32)
    digest = _session_digest(session_id)
    absolute_expires_at = now + _absolute_ttl()
    record = SessionRecord(
        user_id=user_id,
        csrf_token=csrf_token,
        issued_at=now,
        absolute_expires_at=absolute_expires_at,
    )
    async with redis.pipeline(transaction=True) as pipeline:
        pipeline.set(_session_key(digest), record.model_dump_json(), ex=min(_idle_ttl(), _absolute_ttl()))
        pipeline.sadd(_user_sessions_key(user_id), digest)
        pipeline.expire(_user_sessions_key(user_id), _absolute_ttl())
        await pipeline.execute()
    return IssuedSession(
        session_id=session_id,
        csrf_token=csrf_token,
        absolute_expires_at=absolute_expires_at,
    )


async def get_session(redis: Redis, session_id: str, *, touch: bool = True) -> AuthSession | None:
    """Load a session and apply idle sliding expiration within its absolute lifetime."""
    digest = _session_digest(session_id)
    key = _session_key(digest)
    raw = await redis.get(key)
    if not raw:
        return None
    try:
        record = SessionRecord.model_validate_json(raw)
    except ValidationError:
        await redis.delete(key)
        return None

    now = int(time.time())
    remaining = record.absolute_expires_at - now
    if remaining <= 0:
        await delete_session_by_digest(redis, record.user_id, digest)
        return None
    if touch:
        await redis.expire(key, min(_idle_ttl(), remaining))
    return AuthSession(digest=digest, record=record)


async def delete_session_by_digest(redis: Redis, user_id: int, digest: str) -> None:
    """Delete one session and remove it from the user's session index."""
    async with redis.pipeline(transaction=True) as pipeline:
        pipeline.delete(_session_key(digest))
        pipeline.srem(_user_sessions_key(user_id), digest)
        await pipeline.execute()


async def delete_session(redis: Redis, session_id: str) -> None:
    """Delete an opaque session by its browser value when it exists."""
    session = await get_session(redis, session_id, touch=False)
    if session is not None:
        await delete_session_by_digest(redis, session.record.user_id, session.digest)


async def revoke_user_sessions(redis: Redis, user_id: int, *, keep_digest: str | None = None) -> None:
    """Revoke all sessions for a user, optionally preserving one current session."""
    index_key = _user_sessions_key(user_id)
    digests = set(await redis.smembers(index_key))
    to_delete = digests - ({keep_digest} if keep_digest else set())
    async with redis.pipeline(transaction=True) as pipeline:
        if to_delete:
            pipeline.delete(*(_session_key(digest) for digest in to_delete))
            pipeline.srem(index_key, *to_delete)
        if keep_digest:
            pipeline.expire(index_key, _absolute_ttl())
        else:
            pipeline.delete(index_key)
        await pipeline.execute()


def _login_failure_key(email: str) -> str:
    digest = hashlib.sha256(email.strip().casefold().encode()).hexdigest()
    return f'login-failures:{digest}'


async def login_is_rate_limited(redis: Redis, email: str) -> bool:
    """Return whether the normalized account exceeded failed-login limits."""
    count = await redis.get(_login_failure_key(email))
    return bool(count and int(count) >= get_settings().security.LOGIN_MAX_FAILURES)


async def record_login_failure(redis: Redis, email: str) -> None:
    """Record one failed login in the fixed rate-limit window."""
    key = _login_failure_key(email)
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, get_settings().security.LOGIN_FAILURE_WINDOW_MINUTES * 60)


async def clear_login_failures(redis: Redis, email: str) -> None:
    """Clear an account's failed-login counter after successful login."""
    await redis.delete(_login_failure_key(email))
