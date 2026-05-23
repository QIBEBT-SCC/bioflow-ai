from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, SecretStr

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


class RoleUpdate(BaseModel):
    role: UserRole


class PasswordUpdate(BaseModel):
    current_password: SecretStr
    new_password: SecretStr


class UserDetail(BaseModel):
    id: int
    username: str
    email: str
    role: UserRole
    is_active: bool
    total_cost: Decimal
    monthly_cost: Decimal
    run_count: int
