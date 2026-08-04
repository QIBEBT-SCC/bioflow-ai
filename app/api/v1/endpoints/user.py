from fastapi import APIRouter, HTTPException, status

from app.core.auth import AdminDep
from app.core.redis_client import RedisDep
from app.core.session import SessionDep
from app.crud import user as user_crud
from app.schemas.user import RoleUpdate, UserCreate, UserDetail, UserPublic, UserUpdate
from app.services import user as user_service

router = APIRouter()


@router.post('')
def new_user(user_create: UserCreate, session: SessionDep) -> UserPublic:
    """Create a new user account."""
    try:
        user = user_service.create_user(session, user_create)
    except user_service.UserAlreadyExistsError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail='User already exists') from error
    return UserPublic.model_validate(user)


@router.get('')
def list_users(session: SessionDep, _admin: AdminDep) -> list[UserDetail]:
    """Return all users with token cost and run-instance statistics. Requires admin permission."""
    return user_crud.get_all_with_stats(session)


@router.patch('/{user_id}/role')
async def update_user_role(
    user_id: int,
    body: RoleUpdate,
    session: SessionDep,
    redis: RedisDep,
    _admin: AdminDep,
) -> UserPublic:
    """Change the role of any user. Requires admin permission."""
    try:
        user = user_service.update_user(session, user_id, UserUpdate(role=body.role))
    except user_service.UserNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='User not found') from error
    await user_service.invalidate_user_cache(redis, user.email)
    return UserPublic.model_validate(user)
