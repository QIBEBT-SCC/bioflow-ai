import secrets
import time

from fastapi import APIRouter, HTTPException, Request, Response, status

from bioflow_ai.core.auth import AuthSessionDep, UserDep, validate_request_origin
from bioflow_ai.core.config import get_settings
from bioflow_ai.infrastructure.database import SessionDep
from bioflow_ai.infrastructure.redis.client import RedisDep
from bioflow_ai.schemas.auth import LoginRequest
from bioflow_ai.schemas.user import PasswordUpdate, UserProfileUpdate, UserPublic, UserUpdate
from bioflow_ai.services.identity import auth as auth_service
from bioflow_ai.services.identity import user as user_service

router = APIRouter()

_COOKIE_PATH = '/'


def _cookie_kwargs(*, httponly: bool) -> dict:
    config = get_settings().security
    return {
        'httponly': httponly,
        'secure': config.COOKIE_SECURE,
        'samesite': config.COOKIE_SAMESITE,
        'domain': config.COOKIE_DOMAIN,
        'path': _COOKIE_PATH,
    }


def _set_auth_cookies(response: Response, issued: auth_service.IssuedSession) -> None:
    max_age = max(0, issued.absolute_expires_at - int(time.time()))
    response.set_cookie(
        auth_service.SESSION_COOKIE,
        issued.session_id,
        max_age=max_age,
        **_cookie_kwargs(httponly=True),
    )
    response.set_cookie(
        auth_service.CSRF_COOKIE,
        issued.csrf_token,
        max_age=max_age,
        **_cookie_kwargs(httponly=False),
    )
    _delete_legacy_cookies(response)


def _delete_legacy_cookies(response: Response) -> None:
    config = get_settings().security
    for name in auth_service.LEGACY_AUTH_COOKIES:
        response.delete_cookie(name, path=_COOKIE_PATH, domain=config.COOKIE_DOMAIN)


def _clear_auth_cookies(response: Response) -> None:
    config = get_settings().security
    response.delete_cookie(auth_service.SESSION_COOKIE, path=_COOKIE_PATH, domain=config.COOKIE_DOMAIN)
    response.delete_cookie(auth_service.CSRF_COOKIE, path=_COOKIE_PATH, domain=config.COOKIE_DOMAIN)
    _delete_legacy_cookies(response)


@router.post('/login')
async def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    session: SessionDep,
    redis: RedisDep,
) -> UserPublic:
    """Authenticate credentials and create an opaque Redis-backed session."""
    validate_request_origin(request)
    if await auth_service.login_is_rate_limited(redis, body.email):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail='Too many login attempts')

    user = user_service.authenticate_user(session, body.email, body.password.get_secret_value())
    if user is None or not user.is_active:
        await auth_service.record_login_failure(redis, body.email)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Incorrect email or password')

    await auth_service.clear_login_failures(redis, body.email)
    if old_session_id := request.cookies.get(auth_service.SESSION_COOKIE):
        await auth_service.delete_session(redis, old_session_id)
    issued = await auth_service.create_session(redis, user.id)
    _set_auth_cookies(response, issued)
    return UserPublic.model_validate(user)


@router.get('/me')
async def read_users_me(current_user: UserDep) -> UserPublic:
    """Return the current authenticated user's information."""
    return UserPublic.model_validate(current_user)


@router.patch('/me')
async def update_users_me(
    body: UserProfileUpdate,
    session: SessionDep,
    redis: RedisDep,
    current_user: UserDep,
) -> UserPublic:
    """Update the current authenticated user's profile."""
    try:
        user = user_service.update_user(session, current_user.id, UserUpdate(username=body.username))
    except user_service.UserNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='User not found') from error
    await user_service.invalidate_user_cache(redis, user.id)
    return UserPublic.model_validate(user)


@router.post('/logout', status_code=status.HTTP_204_NO_CONTENT)
async def logout(request: Request, response: Response, redis: RedisDep) -> None:
    """Idempotently revoke the current session and clear authentication cookies."""
    validate_request_origin(request)
    session_id = request.cookies.get(auth_service.SESSION_COOKIE)
    if session_id:
        auth_session = await auth_service.get_session(redis, session_id, touch=False)
        if auth_session is not None:
            cookie_token = request.cookies.get(auth_service.CSRF_COOKIE)
            header_token = request.headers.get('X-CSRF-Token')
            if not (
                cookie_token
                and header_token
                and secrets.compare_digest(cookie_token, header_token)
                and secrets.compare_digest(cookie_token, auth_session.record.csrf_token)
            ):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Invalid CSRF token')
            await auth_service.delete_session_by_digest(redis, auth_session.record.user_id, auth_session.digest)
    _clear_auth_cookies(response)


@router.post('/sessions/revoke-others', status_code=status.HTTP_204_NO_CONTENT)
async def revoke_other_sessions(current_user: UserDep, auth_session: AuthSessionDep, redis: RedisDep) -> None:
    """Revoke every session for the user except the current browser session."""
    await auth_service.revoke_user_sessions(redis, current_user.id, keep_digest=auth_session.digest)


@router.put('/password', status_code=status.HTTP_204_NO_CONTENT)
async def update_password(
    body: PasswordUpdate,
    response: Response,
    session: SessionDep,
    redis: RedisDep,
    current_user: UserDep,
    _auth_session: AuthSessionDep,
) -> None:
    """Change the current user's password and rotate the current session."""
    try:
        user = user_service.change_password(
            session,
            current_user.id,
            body.current_password.get_secret_value(),
            body.new_password.get_secret_value(),
        )
    except user_service.InvalidCurrentPasswordError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Current password is incorrect') from error
    except user_service.UserNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='User not found') from error

    await user_service.invalidate_user_cache(redis, user.id)
    await auth_service.revoke_user_sessions(redis, user.id)
    issued = await auth_service.create_session(redis, user.id)
    _set_auth_cookies(response, issued)
