from redis.asyncio import Redis
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.secret_cipher import hash_password, verify_password
from app.crud import user as user_crud
from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate
from app.schemas.user_config import UserRole
from app.services._transaction import committing


class UserServiceError(ValueError):
    """Base error for user application operations."""


class UserAlreadyExistsError(UserServiceError):
    """Raised when a user email violates the database uniqueness contract."""


class UserNotFoundError(UserServiceError):
    """Raised when a requested user does not exist."""


class InvalidCurrentPasswordError(UserServiceError):
    """Raised when a password change supplies the wrong current password."""


def authenticate_user(session: Session, email: str, password: str) -> User | None:
    """Return the matching user only when the supplied password is valid."""
    user = user_crud.get(session, email)
    if user is None or not verify_password(password, user.hashed_password):
        return None
    return user


def create_user(
    session: Session,
    data: UserCreate,
    *,
    role: UserRole = UserRole.VISITOR,
) -> User:
    """Create and commit a user account."""
    user = User(
        email=data.email,
        username=data.username,
        hashed_password=hash_password(data.password.get_secret_value()),
        is_active=True,
        role=role,
    )
    try:
        with committing(session):
            user_crud.insert(session, user)
    except IntegrityError as error:
        raise UserAlreadyExistsError from error
    return user


def update_user(session: Session, user_id: int, data: UserUpdate) -> User:
    """Update and commit a user account."""
    with committing(session):
        user = user_crud.update(session, user_id, data)
        if user is None:
            raise UserNotFoundError
    return user


def change_password(session: Session, user_id: int, current_password: str, new_password: str) -> User:
    """Validate the current password, then commit a replacement hash."""
    user = user_crud.get_by_id(session, user_id)
    if user is None:
        raise UserNotFoundError
    if not verify_password(current_password, user.hashed_password):
        raise InvalidCurrentPasswordError

    with committing(session):
        updated = user_crud.update(session, user_id, UserUpdate(hashed_password=hash_password(new_password)))
        if updated is None:  # Defensive against an unexpected concurrent delete.
            raise UserNotFoundError
    return updated


async def invalidate_user_cache(redis: Redis, email: str) -> None:
    """Invalidate the authenticated-user projection after account updates."""
    await redis.delete(f'user-cache:{email}')
