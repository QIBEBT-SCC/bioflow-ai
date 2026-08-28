import base64
import hashlib

import bcrypt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from cryptography.fernet import Fernet, InvalidToken

from app.core.config import get_settings

SECRET_CIPHERTEXT_PREFIX = 'bioflow:v1:'  # noqa: S105
_password_hasher = PasswordHasher()


class SecretDecryptionError(ValueError):
    """Raised when a stored secret cannot be decrypted."""


def hash_password(password: str) -> str:
    """Generate an Argon2id password hash."""
    return _password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Verify Argon2id hashes and legacy bcrypt hashes."""
    valid, _replacement = verify_and_update_password(password, password_hash)
    return valid


def verify_and_update_password(password: str, password_hash: str) -> tuple[bool, str | None]:
    """Verify a password and return a modern replacement for legacy hashes."""
    if password_hash.startswith('$argon2'):
        try:
            valid = _password_hasher.verify(password_hash, password)
        except (InvalidHashError, VerificationError, VerifyMismatchError):
            return False, None
        replacement = _password_hasher.hash(password) if _password_hasher.check_needs_rehash(password_hash) else None
        return valid, replacement

    try:
        valid = bcrypt.checkpw(password.encode(), password_hash.encode())
    except (ValueError, TypeError):
        return False, None
    return valid, hash_password(password) if valid else None


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
