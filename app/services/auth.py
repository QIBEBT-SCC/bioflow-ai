from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

import jwt
from jwt import InvalidTokenError
from pydantic import BaseModel, ValidationError
from redis.asyncio import Redis
from uuid_extensions import uuid7str

from app.core.config import get_settings

TokenType = Literal['access', 'refresh']


class TokenPayload(BaseModel):
    """Validated claims shared by access and refresh tokens."""

    sub: str
    exp: int
    jti: str
    type: TokenType


@dataclass(frozen=True, slots=True)
class IssuedToken:
    """Encoded token plus claims needed by server-side token storage."""

    value: str
    subject: str
    jti: str
    expires_at: datetime
    type: TokenType


class AuthServiceError(ValueError):
    """Base error for authentication-state operations."""


class InvalidRefreshTokenError(AuthServiceError):
    """Raised when a refresh token is invalid, expired, revoked, or already consumed."""


def _issue_token(subject: str, token_type: TokenType, expires_at: datetime) -> IssuedToken:
    config = get_settings().security
    jti = uuid7str()
    payload = {'sub': subject, 'exp': expires_at, 'jti': jti, 'type': token_type}
    value = jwt.encode(payload, config.SECRET_KEY.get_secret_value(), algorithm=config.ALGORITHM)
    return IssuedToken(value=value, subject=subject, jti=jti, expires_at=expires_at, type=token_type)


def issue_access_token(subject: str) -> IssuedToken:
    """Issue an access token using the configured lifetime."""
    config = get_settings().security
    expires_at = datetime.now(UTC) + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    return _issue_token(subject, 'access', expires_at)


def issue_refresh_token(subject: str) -> IssuedToken:
    """Issue a refresh token using the configured lifetime."""
    config = get_settings().security
    expires_at = datetime.now(UTC) + timedelta(days=config.REFRESH_TOKEN_EXPIRE_DAYS)
    return _issue_token(subject, 'refresh', expires_at)


def decode_token(token: str, *, expected_type: TokenType | None = None) -> TokenPayload:
    """Verify a JWT signature and return validated claims."""
    config = get_settings().security
    try:
        raw_payload = jwt.decode(
            token,
            config.SECRET_KEY.get_secret_value(),
            algorithms=[config.ALGORITHM],
        )
        payload = TokenPayload.model_validate(raw_payload)
    except (InvalidTokenError, ValidationError) as error:
        message = 'Invalid token payload.'
        raise InvalidTokenError(message) from error
    if expected_type is not None and payload.type != expected_type:
        message = f'Expected a {expected_type} token.'
        raise InvalidTokenError(message)
    return payload


async def store_refresh_token(redis: Redis, jti: str, subject: str) -> None:
    """Store a refresh-token grant until its configured expiration."""
    ttl = get_settings().security.REFRESH_TOKEN_EXPIRE_DAYS * 86400
    await redis.set(f'refresh-token:{jti}', subject, ex=ttl)


async def consume_refresh_token(redis: Redis, token: str) -> str:
    """Validate and atomically consume one refresh-token grant."""
    try:
        payload = decode_token(token, expected_type='refresh')
    except InvalidTokenError as error:
        raise InvalidRefreshTokenError from error

    key = f'refresh-token:{payload.jti}'
    stored_subject = await redis.getdel(key)
    if stored_subject != payload.sub:
        raise InvalidRefreshTokenError
    return payload.sub


async def revoke_access_token(redis: Redis, jti: str, expire_seconds: int) -> None:
    """Add an access-token ID to the revocation list."""
    await redis.set(f'token-blacklist:{jti}', '1', ex=expire_seconds)


async def is_access_token_revoked(redis: Redis, jti: str) -> bool:
    """Return whether an access-token ID has been revoked."""
    return bool(await redis.exists(f'token-blacklist:{jti}'))


async def delete_refresh_token(redis: Redis, jti: str) -> None:
    """Delete one server-side refresh-token grant."""
    await redis.delete(f'refresh-token:{jti}')
