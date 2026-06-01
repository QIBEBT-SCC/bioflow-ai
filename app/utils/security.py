from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, TypeAlias, cast

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from jwt import InvalidTokenError
from redis.asyncio import Redis
from starlette import status
from uuid_extensions import uuid7str

from app.core.config import get_settings
from app.core.redis_client import RedisDep
from app.core.session import SessionDep
from app.crud import user as user_crud
from app.models.user import User, UserRole
from app.schemas.user import CurrentUser

_USER_CACHE_TTL = 300
config = get_settings().security


oauth2_scheme = OAuth2PasswordBearer(tokenUrl='/api/v1/auth/token', auto_error=False)


async def get_token_from_header_or_cookie(
    request: Request,
    header_token: Annotated[str | None, Depends(oauth2_scheme)] = None,
) -> str | None:
    """Extract JWT from Authorization header first, then fall back to access_token cookie."""
    if header_token:
        return header_token
    return request.cookies.get('access_token')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password by comparing plain text with bcrypt hashed password."""
    password_byte = plain_password.encode('utf-8')
    hashed_password_byte = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password=password_byte, hashed_password=hashed_password_byte)


def get_password_hash(password: str) -> str:
    """Generate bcrypt hash for the given password."""
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hash_bytes = bcrypt.hashpw(password=pwd_bytes, salt=salt)
    return hash_bytes.decode('utf-8')


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Generate JWT access token with the provided claims and expiration time."""
    to_encode = data.copy()
    expire = datetime.now(UTC) + expires_delta if expires_delta else datetime.now(UTC) + timedelta(minutes=15)
    to_encode.update({'exp': expire, 'jti': uuid7str(), 'type': 'access'})
    return jwt.encode(to_encode, config.SECRET_KEY.get_secret_value(), algorithm=config.ALGORITHM)


def create_refresh_token(sub: str) -> str:
    """Generate JWT refresh token for the given subject (user email)."""
    expire = datetime.now(UTC) + timedelta(days=config.REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {'sub': sub, 'exp': expire, 'jti': uuid7str(), 'type': 'refresh'}
    return jwt.encode(payload, config.SECRET_KEY.get_secret_value(), algorithm=config.ALGORITHM)


async def revoke_token(redis: Redis, jti: str, expire_seconds: int) -> None:
    """Add token JTI to blacklist in Redis with expiration time."""
    await redis.set(f'token-blacklist:{jti}', '1', ex=expire_seconds)


async def store_refresh_token(redis: Redis, jti: str, sub: str) -> None:
    """Write the refresh token whitelist."""
    ttl = config.REFRESH_TOKEN_EXPIRE_DAYS * 86400
    await redis.set(f'refresh-token:{jti}', sub, ex=ttl)


async def validate_and_rotate_refresh_token(redis: Redis, token: str) -> tuple[str, str]:
    """Validate the refresh token, and after rotation return (sub, old_jti).

    Raises HTTPException 401 if invalid or revoked.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail='Invalid or expired refresh token',
        headers={'WWW-Authenticate': 'Bearer'},
    )
    try:
        payload = jwt.decode(token, config.SECRET_KEY.get_secret_value(), algorithms=[config.ALGORITHM])
        if payload.get('type') != 'refresh':
            raise credentials_exception
        jti: str | None = payload.get('jti')
        sub: str | None = payload.get('sub')
        if not jti or not sub:
            raise credentials_exception
    except InvalidTokenError as e:
        raise credentials_exception from e

    stored = await redis.get(f'refresh-token:{jti}')
    if not stored:
        raise credentials_exception

    await redis.delete(f'refresh-token:{jti}')
    return sub, jti


def authenticate_user(session: SessionDep, email: str, password: str) -> User | None:
    """Authenticate user by email and password. Returns User if valid, None otherwise."""
    user = user_crud.get(session, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


async def get_current_user(
    token: Annotated[str | None, Depends(get_token_from_header_or_cookie)],
    session: SessionDep,
    redis: RedisDep,
) -> User:
    """Validate JWT token and return corresponding user from database. Caches user in Redis."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail='Could not validate credentials',
        headers={'WWW-Authenticate': 'Bearer'},
    )
    if not token:
        raise credentials_exception
    try:
        payload = jwt.decode(token, config.SECRET_KEY.get_secret_value(), algorithms=[config.ALGORITHM])
        email: str = payload.get('sub')
        jti: str | None = payload.get('jti')
        if email is None:
            raise credentials_exception
    except InvalidTokenError as e:
        raise credentials_exception from e

    if jti and await redis.exists(f'token-blacklist:{jti}'):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Token has been revoked',
            headers={'WWW-Authenticate': 'Bearer'},
        )

    cache_key = f'user-cache:{email}'
    cached = await redis.get(cache_key)
    if cached:
        return User.model_validate_json(cached)

    user = user_crud.get(session, email)
    if user is None:
        raise credentials_exception
    await redis.set(cache_key, user.model_dump_json(), ex=_USER_CACHE_TTL)
    return user


async def get_current_active_user(current_user: Annotated[User, Depends(get_current_user)]) -> CurrentUser:
    """Ensure current user is active, raise 400 if inactive."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail='Inactive user')
    return cast('CurrentUser', current_user)


def require_role(required_role: UserRole) -> Callable[..., Coroutine[Any, Any, CurrentUser]]:
    """Create a dependency that enforces minimum role requirement."""

    async def role_checker(current_user: Annotated[CurrentUser, Depends(get_current_active_user)]) -> CurrentUser:
        """Verify current user has required role level, raise 403 if insufficient permission."""
        if current_user.role < required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=f'Requires {required_role.name} or higher permission'
            )
        return current_user

    return role_checker


UserDep: TypeAlias = Annotated[CurrentUser, Depends(get_current_active_user)]
MemberDep: TypeAlias = Annotated[CurrentUser, Depends(require_role(UserRole.MEMBER))]
AdminDep: TypeAlias = Annotated[CurrentUser, Depends(require_role(UserRole.ADMIN))]
