from fastapi import APIRouter

from app.core.session import SessionDep
from app.crud import user as user_crud
from app.models.user import User, UserRole
from app.schemas.user import RoleUpdate, UserCreate, UserDetail, UserPublic, UserUpdate
from app.utils.security import AdminDep, get_password_hash

router = APIRouter()


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
    # pyrefly: ignore [bad-return]
    return user


@router.get('')
def list_users(session: SessionDep, _admin: AdminDep) -> list[UserDetail]:
    """Return all users with token cost and run-instance statistics. Requires admin permission."""
    return user_crud.get_all_with_stats(session)


@router.patch('/{user_id}/role')
def update_user_role(user_id: int, body: RoleUpdate, session: SessionDep, _admin: AdminDep) -> UserPublic:
    """Change the role of any user. Requires admin permission."""
    update_data = UserUpdate(role=body.role)
    # pyrefly: ignore [bad-return]
    return user_crud.update(session, user_id, update_data)
