import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from jwt import InvalidTokenError

from app.core.auth import UserDep, get_current_active_user, get_token_from_header_or_cookie
from app.core.config import get_settings
from app.core.redis_client import RedisDep
from app.core.session import SessionDep
from app.schemas.auth import Token
from app.schemas.user import CurrentUser, PasswordUpdate, UserPublic
from app.services import auth as auth_service
from app.services import user as user_service

router = APIRouter()

_COOKIE_PATH = '/'


def _cookie_kwargs() -> dict:
    config = get_settings().security
    return {
        'httponly': True,
        'secure': config.COOKIE_SECURE,
        'samesite': config.COOKIE_SAMESITE,
        'domain': config.COOKIE_DOMAIN,
        'path': _COOKIE_PATH,
    }


def _set_auth_cookies(response: Response, access_token: str, refresh_token: str) -> None:
    config = get_settings().security
    cookie_options = _cookie_kwargs()
    response.set_cookie(
        'access_token',
        access_token,
        max_age=config.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        **cookie_options,
    )
    response.set_cookie(
        'refresh_token',
        refresh_token,
        max_age=config.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        **cookie_options,
    )


@router.post('/token')
async def login_for_access_token(
    response: Response,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: SessionDep,
    redis: RedisDep,
) -> Token:
    """Authenticate a user and issue an access/refresh token pair."""
    user = user_service.authenticate_user(session, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Incorrect email or password',
            headers={'WWW-Authenticate': 'Bearer'},
        )

    access_token = auth_service.issue_access_token(user.email)
    refresh_token = auth_service.issue_refresh_token(user.email)
    await auth_service.store_refresh_token(redis, refresh_token.jti, refresh_token.subject)

    _set_auth_cookies(response, access_token.value, refresh_token.value)
    return Token(access_token=access_token.value, token_type='bearer', refresh_token=refresh_token.value)  # noqa: S106


@router.post('/refresh')
async def refresh_access_token(request: Request, response: Response, redis: RedisDep) -> Token:
    """Consume a refresh token and rotate the token pair."""
    refresh_token_value = request.cookies.get('refresh_token')
    if not refresh_token_value:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Missing refresh token')

    try:
        subject = await auth_service.consume_refresh_token(redis, refresh_token_value)
    except auth_service.InvalidRefreshTokenError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid or expired refresh token',
            headers={'WWW-Authenticate': 'Bearer'},
        ) from error

    access_token = auth_service.issue_access_token(subject)
    refresh_token = auth_service.issue_refresh_token(subject)
    await auth_service.store_refresh_token(redis, refresh_token.jti, refresh_token.subject)

    _set_auth_cookies(response, access_token.value, refresh_token.value)
    return Token(access_token=access_token.value, token_type='bearer', refresh_token=refresh_token.value)  # noqa: S106


@router.get('/me')
async def read_users_me(current_user: Annotated[CurrentUser, Depends(get_current_active_user)]) -> UserPublic:
    """Return the current authenticated user's information."""
    return UserPublic.model_validate(current_user)


@router.post('/logout')
async def logout(
    request: Request,
    response: Response,
    token: Annotated[str | None, Depends(get_token_from_header_or_cookie)],
    redis: RedisDep,
) -> dict[str, str]:
    """Revoke the current token pair and clear authentication cookies."""
    config = get_settings().security

    if token:
        try:
            payload = auth_service.decode_token(token, expected_type='access')
            remaining = payload.exp - int(time.time())
            if remaining > 0:
                await auth_service.revoke_access_token(redis, payload.jti, remaining)
            await redis.delete(f'user-cache:{payload.sub}')
        except InvalidTokenError:
            pass

    refresh_token = request.cookies.get('refresh_token')
    if refresh_token:
        try:
            payload = auth_service.decode_token(refresh_token, expected_type='refresh')
            await auth_service.delete_refresh_token(redis, payload.jti)
        except InvalidTokenError:
            pass

    response.delete_cookie('access_token', path=_COOKIE_PATH, domain=config.COOKIE_DOMAIN)
    response.delete_cookie('refresh_token', path=_COOKIE_PATH, domain=config.COOKIE_DOMAIN)
    response.delete_cookie('csrf_token', path=_COOKIE_PATH, domain=config.COOKIE_DOMAIN)
    return {'message': 'Successfully logged out'}


@router.put('/password', status_code=status.HTTP_204_NO_CONTENT)
async def update_password(
    body: PasswordUpdate,
    session: SessionDep,
    redis: RedisDep,
    current_user: UserDep,
) -> None:
    """Update the current user's password."""
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
    await user_service.invalidate_user_cache(redis, user.email)
