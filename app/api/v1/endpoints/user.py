from fastapi import APIRouter
from pydantic import BaseModel, SecretStr

from app.core.session import SessionDep
from app.crud import user as user_crud
from app.models.user import User, UserRole
from app.schemas.user import UserPublic
from app.utils.security import get_password_hash

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
    return user
