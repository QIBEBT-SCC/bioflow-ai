import secrets
from collections.abc import Callable, Coroutine
from typing import Annotated, Any, TypeAlias
from urllib.parse import urlsplit

from fastapi import Depends, HTTPException, Request
from starlette import status

from bioflow_ai.core.config import get_settings
from bioflow_ai.crud import user as user_crud
from bioflow_ai.infrastructure.database import SessionDep
from bioflow_ai.infrastructure.redis.client import RedisDep
from bioflow_ai.schemas.user import CurrentUser
from bioflow_ai.schemas.user_config import UserRole
from bioflow_ai.services.identity import auth as auth_service

_UNSAFE_METHODS = {'POST', 'PUT', 'PATCH', 'DELETE'}


def validate_request_origin(request: Request) -> None:
    """Reject unsafe browser requests from origins outside the configured allowlist."""
    if request.method not in _UNSAFE_METHODS:
        return
    source = request.headers.get('origin')
    if source is None:
        referer = request.headers.get('referer')
        if referer:
            parsed = urlsplit(referer)
            source = f'{parsed.scheme}://{parsed.netloc}'
    if source is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Missing request origin')

    own_origin = f'{request.url.scheme}://{request.url.netloc}'
    allowed = {origin.rstrip('/') for origin in get_settings().network.CORS_ORIGINS}
    allowed.add(own_origin.rstrip('/'))
    if source.rstrip('/') not in allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Invalid request origin')


async def get_current_session(request: Request, redis: RedisDep) -> auth_service.AuthSession:
    """Resolve an opaque browser session and enforce CSRF on unsafe requests."""
    session_id = request.cookies.get(auth_service.SESSION_COOKIE)
    if not session_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required')
    auth_session = await auth_service.get_session(redis, session_id)
    if auth_session is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid or expired session')

    if request.method in _UNSAFE_METHODS:
        validate_request_origin(request)
        cookie_token = request.cookies.get(auth_service.CSRF_COOKIE)
        header_token = request.headers.get('X-CSRF-Token')
        expected = auth_session.record.csrf_token
        if not (
            cookie_token
            and header_token
            and secrets.compare_digest(cookie_token, header_token)
            and secrets.compare_digest(cookie_token, expected)
        ):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Invalid CSRF token')
    request.state.auth_session = auth_session
    return auth_session


async def get_current_user(
    auth_session: Annotated[auth_service.AuthSession, Depends(get_current_session)],
    session: SessionDep,
    redis: RedisDep,
) -> CurrentUser:
    """Return the database-backed user projection for an authenticated session."""
    user_id = auth_session.record.user_id
    cache_key = f'user-cache:{user_id}'
    cached = await redis.get(cache_key)
    if cached:
        return CurrentUser.model_validate_json(cached)

    user = user_crud.get_by_id(session, user_id)
    if user is None:
        await auth_service.delete_session_by_digest(redis, user_id, auth_session.digest)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid session user')
    projection = CurrentUser.model_validate(user)
    await redis.set(
        cache_key,
        projection.model_dump_json(),
        ex=get_settings().security.USER_CACHE_TTL_SECONDS,
    )
    return projection


async def get_current_active_user(current_user: Annotated[CurrentUser, Depends(get_current_user)]) -> CurrentUser:
    """Ensure the current user is active."""
    if not current_user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Inactive user')
    return current_user


def require_role(required_role: UserRole) -> Callable[..., Coroutine[Any, Any, CurrentUser]]:
    """Create a dependency that enforces a minimum role."""

    async def role_checker(current_user: Annotated[CurrentUser, Depends(get_current_active_user)]) -> CurrentUser:
        if current_user.role < required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f'Requires {required_role.name} or higher permission',
            )
        return current_user

    return role_checker


AuthSessionDep: TypeAlias = Annotated[auth_service.AuthSession, Depends(get_current_session)]
UserDep: TypeAlias = Annotated[CurrentUser, Depends(get_current_active_user)]
MemberDep: TypeAlias = Annotated[CurrentUser, Depends(require_role(UserRole.MEMBER))]
AdminDep: TypeAlias = Annotated[CurrentUser, Depends(require_role(UserRole.ADMIN))]
