from typing import Optional

from pydantic import BaseModel

from app.models.user import UserRole


class UserPublic(BaseModel):
    username: str
    email: str
    role: UserRole


class UserUpdate(BaseModel):
    username: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None
