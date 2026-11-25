from decimal import Decimal

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

from bioflow_ai.schemas.auth import EMAIL_REQUIRED_ERROR, validate_password
from bioflow_ai.schemas.user_config import UserRole

USERNAME_REQUIRED_ERROR = 'Username must not be empty'
USERNAME_MAX_LENGTH = 64
USERNAME_TOO_LONG_ERROR = f'Username must be at most {USERNAME_MAX_LENGTH} characters'


def validate_username(value: str) -> str:
    """Trim usernames and reject empty or overly long display names."""
    value = value.strip()
    if not value:
        raise ValueError(USERNAME_REQUIRED_ERROR)
    if len(value) > USERNAME_MAX_LENGTH:
        raise ValueError(USERNAME_TOO_LONG_ERROR)
    return value


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
        return validate_username(value)

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


class UserProfileUpdate(BaseModel):
    username: str

    @field_validator('username')
    @classmethod
    def validate_username_value(cls, value: str) -> str:
        """Trim usernames and enforce display-name bounds."""
        return validate_username(value)


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
