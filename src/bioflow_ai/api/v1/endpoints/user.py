from fastapi import APIRouter, HTTPException, status

from bioflow_ai.core.auth import AdminDep
from bioflow_ai.crud import user as user_crud
from bioflow_ai.infrastructure.database import SessionDep
from bioflow_ai.infrastructure.redis.client import RedisDep
from bioflow_ai.schemas.user import (
    AdminUserCreate,
    PasswordReset,
    RoleUpdate,
    StatusUpdate,
    UserDetail,
    UserPublic,
    UserUpdate,
)
from bioflow_ai.schemas.user_config import UserRole
from bioflow_ai.services.identity import auth as auth_service
from bioflow_ai.services.identity import user as user_service

router = APIRouter()


def _ensure_active_admin_remains(session: SessionDep, user_id: int, update: UserUpdate) -> None:
    target = user_crud.get_by_id(session, user_id)
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='User not found')
    removes_active_admin = target.role == UserRole.ADMIN and target.is_active and (
        (update.role is not None and update.role != UserRole.ADMIN) or update.is_active is False
    )
    if removes_active_admin and user_crud.count_active_admins(session, lock=True) <= 1:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail='At least one active administrator is required')


async def _after_security_update(redis: RedisDep, user_id: int) -> None:
    await user_service.invalidate_user_cache(redis, user_id)
    await auth_service.revoke_user_sessions(redis, user_id)


@router.post('', status_code=status.HTTP_201_CREATED)
def new_user(user_create: AdminUserCreate, session: SessionDep, _admin: AdminDep) -> UserPublic:
    """Create a user account. Requires administrator permission."""
    try:
        user = user_service.create_user(session, user_create, role=user_create.role)
    except user_service.UserAlreadyExistsError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail='User already exists') from error
    return UserPublic.model_validate(user)


@router.get('')
def list_users(session: SessionDep, _admin: AdminDep) -> list[UserDetail]:
    """Return all users with token cost and run-instance statistics."""
    return user_crud.get_all_with_stats(session)


@router.patch('/{user_id}/role')
async def update_user_role(
    user_id: int,
    body: RoleUpdate,
    session: SessionDep,
    redis: RedisDep,
    _admin: AdminDep,
) -> UserPublic:
    """Change a user's role and revoke their sessions."""
    update = UserUpdate(role=body.role)
    _ensure_active_admin_remains(session, user_id, update)
    try:
        user = user_service.update_user(session, user_id, update)
    except user_service.UserNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='User not found') from error
    await _after_security_update(redis, user.id)
    return UserPublic.model_validate(user)


@router.patch('/{user_id}/status')
async def update_user_status(
    user_id: int,
    body: StatusUpdate,
    session: SessionDep,
    redis: RedisDep,
    _admin: AdminDep,
) -> UserPublic:
    """Activate or deactivate a user and revoke their sessions."""
    update = UserUpdate(is_active=body.is_active)
    _ensure_active_admin_remains(session, user_id, update)
    try:
        user = user_service.update_user(session, user_id, update)
    except user_service.UserNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='User not found') from error
    await _after_security_update(redis, user.id)
    return UserPublic.model_validate(user)


@router.put('/{user_id}/password', status_code=status.HTTP_204_NO_CONTENT)
async def reset_user_password(
    user_id: int,
    body: PasswordReset,
    session: SessionDep,
    redis: RedisDep,
    _admin: AdminDep,
) -> None:
    """Reset a user's password and revoke all of their sessions."""
    try:
        user = user_service.reset_password(session, user_id, body.password.get_secret_value())
    except user_service.UserNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='User not found') from error
    await _after_security_update(redis, user.id)
