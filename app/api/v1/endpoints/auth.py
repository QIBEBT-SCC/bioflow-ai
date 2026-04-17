import time
from datetime import timedelta
from typing import Annotated

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.core.config import get_settings
from app.core.session import RedisDep, SessionDep
from app.models.user import User
from app.schemas.auth import Token
from app.schemas.user import UserPublic
from app.utils.security import (
    authenticate_user,
    create_access_token,
    get_current_active_user,
    oauth2_scheme,
    revoke_token,
)

router = APIRouter()


@router.post('/token')
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()], session: SessionDep
) -> Token:
    """Authenticate user and return an access token."""
    user = authenticate_user(session, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Incorrect email or password',
            headers={'WWW-Authenticate': 'Bearer'},
        )
    access_token_expires = timedelta(minutes=get_settings().security.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={'sub': user.email}, expires_delta=access_token_expires)
    return Token(access_token=access_token, token_type='bearer')  # noqa: S106


@router.get('/me')
async def read_users_me(current_user: Annotated[User, Depends(get_current_active_user)]) -> UserPublic:
    """Return the current authenticated user's information."""
    return current_user


@router.post('/logout')
async def logout(token: Annotated[str, Depends(oauth2_scheme)], redis: RedisDep) -> dict:
    """Invalidate the current access token."""
    try:
        payload = jwt.decode(
            token,
            get_settings().security.SECRET_KEY.get_secret_value(),
            algorithms=[get_settings().security.ALGORITHM],
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
    return {'message': 'Successfully logged out'}
