import os
import subprocess
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from uuid_extensions import uuid7str


@lru_cache
def get_cpu_cores() -> int:
    """Get CPU core count."""
    return os.cpu_count() or 1


def _format_size(size_bytes: int) -> str:
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == 'TB':  # noqa: PLR2004
            return f'{size:.2f} {unit}'
        size /= 1024
    return ''


def get_file_size(file_path: str | Path) -> int:
    """Get file size."""
    try:
        return Path(file_path).stat().st_size
    except (FileNotFoundError, OSError):
        return 0


def get_folder_size_du(path: str) -> str:
    """Get folder size."""
    result = subprocess.run(['/usr/bin/du', '-sb', path], check=False, stdout=subprocess.PIPE, text=True)
    size_in_bytes = int(result.stdout.split()[0])
    return _format_size(size_in_bytes)


def random_uid() -> uuid.UUID:
    """Generate a random UID."""
    return uuid.UUID(uuid7str())


def utc_now() -> datetime:
    """Get current UTC time."""
    return datetime.now(UTC)
