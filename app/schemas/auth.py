from pydantic import BaseModel, SecretStr, field_validator

MAX_PASSWORD_BYTES = 128
PASSWORD_REQUIRED_ERROR = 'Password must not be empty'  # noqa: S105
PASSWORD_TOO_LONG_ERROR = 'Password must not exceed 128 UTF-8 bytes'  # noqa: S105
EMAIL_REQUIRED_ERROR = 'Email must not be empty'


def validate_password(value: SecretStr) -> SecretStr:
    """Accept weak passwords while enforcing safe input boundaries."""
    raw = value.get_secret_value()
    if not raw:
        raise ValueError(PASSWORD_REQUIRED_ERROR)
    if len(raw.encode()) > MAX_PASSWORD_BYTES:
        raise ValueError(PASSWORD_TOO_LONG_ERROR)
    return value


class LoginRequest(BaseModel):
    email: str
    password: SecretStr

    @field_validator('email')
    @classmethod
    def normalize_email(cls, value: str) -> str:
        """Normalize email identifiers before authentication and rate limiting."""
        value = value.strip().casefold()
        if not value:
            raise ValueError(EMAIL_REQUIRED_ERROR)
        return value

    @field_validator('password')
    @classmethod
    def password_boundary(cls, value: SecretStr) -> SecretStr:
        """Enforce password input boundaries without a strength policy."""
        return validate_password(value)
