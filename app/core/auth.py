from collections.abc import Callable, Coroutine
from typing import Annotated, Any, TypeAlias

from fastapi import Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from jwt import InvalidTokenError
from starlette import status

from app.core.config import get_settings
from app.core.redis_client import RedisDep
from app.core.session import SessionDep
from app.crud import user as user_crud
from app.schemas.user import CurrentUser
from app.schemas.user_config import UserRole
from app.services import auth as auth_service

oauth2_scheme = OAuth2PasswordBearer(tokenUrl='/api/v1/auth/token', auto_error=False)


async def get_token_from_header_or_cookie(
    request: Request,
    header_token: Annotated[str | None, Depends(oauth2_scheme)] = None,
) -> str | None:
    """Extract JWT from Authorization header first, then fall back to access_token cookie."""
    if header_token:
        return header_token
    return request.cookies.get('access_token')


async def get_current_user(
    token: Annotated[str | None, Depends(get_token_from_header_or_cookie)],
    session: SessionDep,
    redis: RedisDep,
) -> CurrentUser:
    """Validate an access token and return its user projection."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail='Could not validate credentials',
        headers={'WWW-Authenticate': 'Bearer'},
    )
    if not token:
        raise credentials_exception
    try:
        payload = auth_service.decode_token(token, expected_type='access')
    except InvalidTokenError as error:
        raise credentials_exception from error

    if await auth_service.is_access_token_revoked(redis, payload.jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Token has been revoked',
            headers={'WWW-Authenticate': 'Bearer'},
        )

    cache_key = f'user-cache:{payload.sub}'
    cached = await redis.get(cache_key)
    if cached:
        return CurrentUser.model_validate_json(cached)

    user = user_crud.get(session, payload.sub)
    if user is None:
        raise credentials_exception
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
        raise HTTPException(status_code=400, detail='Inactive user')
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


UserDep: TypeAlias = Annotated[CurrentUser, Depends(get_current_active_user)]
MemberDep: TypeAlias = Annotated[CurrentUser, Depends(require_role(UserRole.MEMBER))]
AdminDep: TypeAlias = Annotated[CurrentUser, Depends(require_role(UserRole.ADMIN))]
