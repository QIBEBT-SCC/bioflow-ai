from typing import Optional

from pydantic import BaseModel, ConfigDict

from app.models.user import UserRole


class UserPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    username: str
    email: str
    role: UserRole


class UserUpdate(BaseModel):
    username: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None
