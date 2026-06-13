import mimetypes
import os
import subprocess
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse, Response
from starlette.status import HTTP_413_REQUEST_ENTITY_TOO_LARGE
from uuid_extensions import uuid7str

from app.schemas.run_instance import OutputFileItem, OutputFileNode, OutputFolderItem

_TEXT_EXTENSIONS = {
    '.json',
    '.csv',
    '.tsv',
    '.fasta',
    '.fastq',
    '.vcf',
    '.bed',
    '.gff',
    '.gtf',
    '.sam',
    '.log',
    '.md',
    '.sh',
    '.yaml',
    '.yml',
    '.toml',
    '.ini',
    '.conf',
}
_MAX_TEXT_BYTES = 10 * 1024 * 1024  # 10 MB


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
    result = subprocess.run(['/usr/bin/du', '-sb', path], check=False, stdout=subprocess.PIPE, text=True)  # noqa: S603
    size_in_bytes = int(result.stdout.split()[0])
    return _format_size(size_in_bytes)


def get_current_user_identity() -> str:
    """获取当前用户的UID:GID.

    Returns:
        str: 格式为 "UID:GID" 的字符串
    """
    uid = os.getuid()
    gid = os.getgid()
    return f'{uid}:{gid}'


def random_uid() -> uuid.UUID:
    """Generate a random UID."""
    return uuid.UUID(uuid7str())


def utc_now() -> datetime:
    """Get current UTC time."""
    return datetime.now(UTC)


def build_file_tree(base_dir: Path, current_dir: Path) -> list[OutputFileNode]:
    """Recursively build a file tree from a directory."""
    nodes: list[OutputFileNode] = []
    try:
        entries = sorted(current_dir.iterdir(), key=lambda e: (e.is_file(), e.name))
    except PermissionError:
        return nodes

    for entry in entries:
        rel_path = str(entry.relative_to(base_dir))
        if entry.is_dir():
            children = build_file_tree(base_dir, entry)
            nodes.append(OutputFolderItem(path=rel_path, name=entry.name, children=children))
        else:
            nodes.append(OutputFileItem(path=rel_path, name=entry.name))

    return nodes


def serve_file(file_path: Path) -> Response:
    """Return an appropriate FastAPI Response for a file based on its MIME type.

    Text-based files (including bioinformatics formats) are read inline up to
    10 MB. Binary files (images, PDFs, etc.) are streamed via FileResponse.

    Args:
        file_path: Absolute path to the file. Must already be validated as safe.

    Returns:
        Response: FastAPI response with the file content and correct Content-Type.

    Raises:
        HTTPException: 413 if a text file exceeds the size limit.
    """
    mime_type, _ = mimetypes.guess_type(file_path.name)
    suffix = file_path.suffix.lower()

    is_text = (
        (mime_type and mime_type.startswith('text/')) or suffix in _TEXT_EXTENSIONS or mime_type == 'application/json'
    )

    if is_text:
        size = file_path.stat().st_size
        if size > _MAX_TEXT_BYTES:
            raise HTTPException(
                HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f'File too large to preview ({size:,} bytes). Download the file instead.',
            )
        effective_mime = (mime_type or 'text/plain') + '; charset=utf-8'
        return Response(content=file_path.read_bytes(), media_type=effective_mime)

    return FileResponse(
        path=str(file_path),
        media_type=mime_type or 'application/octet-stream',
        filename=file_path.name,
    )
