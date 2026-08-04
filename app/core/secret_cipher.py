import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import get_settings

SECRET_CIPHERTEXT_PREFIX = 'bioflow:v1:'  # noqa: S105


class SecretDecryptionError(ValueError):
    """Raised when a stored secret cannot be decrypted."""


def _cipher(purpose: str) -> Fernet:
    if not purpose:
        msg = 'Secret encryption purpose must not be empty.'
        raise ValueError(msg)
    secret = get_settings().security.SECRET_KEY.get_secret_value().encode()
    derived_key = hashlib.sha256(b'bioflow-' + purpose.encode() + b'\0' + secret).digest()
    return Fernet(base64.urlsafe_b64encode(derived_key))


def encrypt_secret(value: str, *, purpose: str) -> str:
    """Encrypt a secret for database storage within one purpose domain."""
    ciphertext = _cipher(purpose).encrypt(value.encode()).decode()
    return f'{SECRET_CIPHERTEXT_PREFIX}{ciphertext}'


def decrypt_secret(value: str, *, purpose: str) -> str:
    """Decrypt a stored secret from one purpose domain."""
    ciphertext = value.removeprefix(SECRET_CIPHERTEXT_PREFIX)
    try:
        return _cipher(purpose).decrypt(ciphertext.encode()).decode()
    except InvalidToken as error:
        message = f'Secret cannot be decrypted for purpose {purpose!r} with the configured security key.'
        raise SecretDecryptionError(message) from error
