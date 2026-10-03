import base64
import hashlib
from pathlib import Path


def calculate_file_sha256(file_path: str | Path, chunk_size: int = 8192) -> str:
    """Return the SHA-256 hex digest for the given file path.

    Args:
        file_path: Absolute or relative path pointing to the file to hash.
        chunk_size: Size of the read buffer to limit memory usage during hashing.

    Returns:
        Hexadecimal digest string if the file is read successfully
    """
    sha256_hash = hashlib.sha256()

    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                sha256_hash.update(chunk)
    except FileNotFoundError:
        return 'Error: File not found.'
    except Exception as e:
        return f'Error: {str(e)}'

    return sha256_hash.hexdigest()


def calculate_file_md5(file_path: str | Path, chunk_size: int = 8192) -> str:
    """Return the MD5 hex digest for the given file path.

    Args:
        file_path: Absolute or relative path pointing to the file to hash.
        chunk_size: Size of the read buffer to limit memory usage during hashing.

    Returns:
        Hexadecimal digest string if the file is read successfully
    """
    md5_hash = hashlib.md5()

    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5_hash.update(chunk)
    except FileNotFoundError:
        return ''
    except Exception as e:
        return ''

    return md5_hash.hexdigest()


def calculate_base64_sha256(encoded_str: str) -> str:
    """Compute the SHA-256 hex digest of the decoded base64 input string."""
    file_bytes = base64.b64decode(encoded_str)
    return hashlib.sha256(file_bytes).hexdigest()
