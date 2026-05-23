from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, SecretStr
from starlette import status

from app.core.session import SessionDep
from app.crud import user as user_crud
from app.models.user import User, UserRole
from app.schemas.user import RoleUpdate, UserDetail, UserPublic
from app.utils.security import AdminDep, get_password_hash

router = APIRouter()


class UserCreate(BaseModel):
    email: str
    username: str
    password: SecretStr


@router.post('')
def new_user(user_create: UserCreate, session: SessionDep) -> UserPublic:
    """Create a new user account."""
    user = User(
        email=user_create.email,
        username=user_create.username,
        hashed_password=get_password_hash(user_create.password.get_secret_value()),
        is_active=True,
        role=UserRole.VISITOR,
    )

    user_crud.insert(session, user)
    return user  # type: ignore


@router.get('')
def list_users(session: SessionDep, _admin: AdminDep) -> list[UserDetail]:
    """Return all users with token cost and run-instance statistics. Requires admin permission."""
    return user_crud.get_all_with_stats(session)


@router.patch('/{user_id}/role')
def update_user_role(user_id: int, body: RoleUpdate, session: SessionDep, _admin: AdminDep) -> UserPublic:
    """Change the role of any user. Requires admin permission."""
    target = user_crud.get_by_id(session, user_id)
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='User not found')
    return user_crud.update(session, target, role=body.role)  # type: ignore
