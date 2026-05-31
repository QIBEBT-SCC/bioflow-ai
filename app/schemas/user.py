from decimal import Decimal

from pydantic import BaseModel, ConfigDict, SecretStr

from app.models.user import UserRole


class UserCreate(BaseModel):
    email: str
    username: str
    password: SecretStr


class UserPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    username: str
    email: str
    role: UserRole


class UserUpdate(BaseModel):
    username: str | None = None
    role: UserRole | None = None
    is_active: bool | None = None
    hashed_password: str | None = None


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
