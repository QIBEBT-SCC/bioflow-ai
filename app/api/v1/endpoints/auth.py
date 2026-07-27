import time
from datetime import timedelta
from typing import Annotated

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm

from app.core.config import get_settings
from app.core.redis_client import RedisDep
from app.core.session import SessionDep
from app.schemas.auth import Token
from app.schemas.user import CurrentUser, PasswordUpdate, UserPublic
from app.services import user as user_service
from app.utils.security import (
    UserDep,
    create_access_token,
    create_refresh_token,
    get_current_active_user,
    get_token_from_header_or_cookie,
    revoke_token,
    store_refresh_token,
    validate_and_rotate_refresh_token,
)

router = APIRouter()

_COOKIE_PATH = '/'


def _cookie_kwargs() -> dict:
    cfg = get_settings().security
    return {
        'httponly': True,
        'secure': cfg.COOKIE_SECURE,
        'samesite': cfg.COOKIE_SAMESITE,
        'domain': cfg.COOKIE_DOMAIN,
        'path': _COOKIE_PATH,
    }


def _set_auth_cookies(response: Response, access_token: str, refresh_token: str) -> None:
    cfg = get_settings().security
    kw = _cookie_kwargs()
    response.set_cookie('access_token', access_token, max_age=cfg.ACCESS_TOKEN_EXPIRE_MINUTES * 60, **kw)
    response.set_cookie(
        'refresh_token',
        refresh_token,
        max_age=cfg.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        **kw,
    )


@router.post('/token')
async def login_for_access_token(
    response: Response,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: SessionDep,
    redis: RedisDep,
) -> Token:
    """Authenticate user and return access + refresh tokens (cookie + JSON body)."""
    user = user_service.authenticate_user(session, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Incorrect email or password',
            headers={'WWW-Authenticate': 'Bearer'},
        )
    cfg = get_settings().security
    access_token = create_access_token(
        data={'sub': user.email},
        expires_delta=timedelta(minutes=cfg.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    refresh_token = create_refresh_token(sub=user.email)

    refresh_payload = jwt.decode(
        refresh_token,
        cfg.SECRET_KEY.get_secret_value(),
        algorithms=[cfg.ALGORITHM],
    )
    await store_refresh_token(redis, refresh_payload['jti'], user.email)

    _set_auth_cookies(response, access_token, refresh_token)
    return Token(access_token=access_token, token_type='bearer', refresh_token=refresh_token)  # noqa: S106


@router.post('/refresh')
async def refresh_access_token(request: Request, response: Response, redis: RedisDep) -> Token:
    """Exchange a valid refresh token for a new access + refresh token pair (token rotation)."""
    refresh_token = request.cookies.get('refresh_token')
    if not refresh_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Missing refresh token')

    sub, _ = await validate_and_rotate_refresh_token(redis, refresh_token)

    cfg = get_settings().security
    new_access = create_access_token(
        data={'sub': sub},
        expires_delta=timedelta(minutes=cfg.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    new_refresh = create_refresh_token(sub=sub)

    new_refresh_payload = jwt.decode(
        new_refresh,
        cfg.SECRET_KEY.get_secret_value(),
        algorithms=[cfg.ALGORITHM],
    )
    await store_refresh_token(redis, new_refresh_payload['jti'], sub)

    _set_auth_cookies(response, new_access, new_refresh)
    return Token(access_token=new_access, token_type='bearer', refresh_token=new_refresh)  # noqa: S106


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
) -> dict:
    """Invalidate the current access + refresh tokens and clear cookies."""
    cfg = get_settings().security

    if token:
        try:
            payload = jwt.decode(
                token,
                cfg.SECRET_KEY.get_secret_value(),
                algorithms=[cfg.ALGORITHM],
            )
            jti: str | None = payload.get('jti')
            exp: int | None = payload.get('exp')
            email: str | None = payload.get('sub')
            if jti and exp:
                remaining = exp - int(time.time())
                if remaining > 0:
                    await revoke_token(redis, jti, remaining)
            if email:
                await redis.delete(f'user-cache:{email}')
        except jwt.InvalidTokenError:
            pass

    refresh_token = request.cookies.get('refresh_token')
    if refresh_token:
        try:
            payload = jwt.decode(
                refresh_token,
                cfg.SECRET_KEY.get_secret_value(),
                algorithms=[cfg.ALGORITHM],
            )
            rjti: str | None = payload.get('jti')
            if rjti:
                await redis.delete(f'refresh-token:{rjti}')
        except jwt.InvalidTokenError:
            pass

    response.delete_cookie('access_token', path=_COOKIE_PATH, domain=cfg.COOKIE_DOMAIN)
    response.delete_cookie('refresh_token', path=_COOKIE_PATH, domain=cfg.COOKIE_DOMAIN)
    response.delete_cookie('csrf_token', path=_COOKIE_PATH, domain=cfg.COOKIE_DOMAIN)
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
