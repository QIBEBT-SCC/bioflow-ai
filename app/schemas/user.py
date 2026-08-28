from decimal import Decimal

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

from app.schemas.auth import EMAIL_REQUIRED_ERROR, validate_password
from app.schemas.user_config import UserRole

USERNAME_REQUIRED_ERROR = 'Username must not be empty'


class UserCreate(BaseModel):
    email: str
    username: str
    password: SecretStr

    @field_validator('email')
    @classmethod
    def normalize_email(cls, value: str) -> str:
        """Normalize the account email identifier."""
        value = value.strip().casefold()
        if not value:
            raise ValueError(EMAIL_REQUIRED_ERROR)
        return value

    @field_validator('username')
    @classmethod
    def validate_username(cls, value: str) -> str:
        """Trim usernames and reject empty display names."""
        value = value.strip()
        if not value:
            raise ValueError(USERNAME_REQUIRED_ERROR)
        return value

    @field_validator('password')
    @classmethod
    def password_boundary(cls, value: SecretStr) -> SecretStr:
        """Enforce password input boundaries."""
        return validate_password(value)


class AdminUserCreate(UserCreate):
    role: UserRole = UserRole.VISITOR


class CurrentUser(BaseModel):
    """Serializable authenticated-user projection stored in the Redis cache."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str
    role: UserRole
    is_active: bool


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

    @field_validator('current_password', 'new_password')
    @classmethod
    def password_boundary(cls, value: SecretStr) -> SecretStr:
        """Enforce password input boundaries."""
        return validate_password(value)


class PasswordReset(BaseModel):
    password: SecretStr

    @field_validator('password')
    @classmethod
    def password_boundary(cls, value: SecretStr) -> SecretStr:
        """Enforce password input boundaries."""
        return validate_password(value)


class StatusUpdate(BaseModel):
    is_active: bool


class UserDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str
    role: UserRole
    is_active: bool
    total_cost: Decimal
    monthly_cost: Decimal
    run_count: int
