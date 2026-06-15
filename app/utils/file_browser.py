import asyncio
import mimetypes
from os import PathLike
from pathlib import Path

from fastapi import HTTPException
from starlette.responses import FileResponse, Response
from starlette.status import HTTP_413_REQUEST_ENTITY_TOO_LARGE

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


type FileSystemPath = str | PathLike[str]


def _to_path(path: FileSystemPath) -> Path:
    return Path(str(path))


def _build_file_tree_sync(base_dir: Path, current_dir: Path) -> list[OutputFileNode]:
    """Recursively build a file tree from a directory."""
    nodes: list[OutputFileNode] = []
    try:
        entries = sorted(current_dir.iterdir(), key=lambda e: (e.is_file(), e.name))
    except PermissionError:
        return nodes

    for entry in entries:
        rel_path = str(entry.relative_to(base_dir))
        if entry.is_dir():
            children = _build_file_tree_sync(base_dir, entry)
            nodes.append(OutputFolderItem(path=rel_path, name=entry.name, children=children))
        else:
            nodes.append(OutputFileItem(path=rel_path, name=entry.name))

    return nodes


async def build_file_tree(base_dir: FileSystemPath, current_dir: FileSystemPath) -> list[OutputFileNode]:
    """Recursively build a file tree from a directory without blocking the event loop."""
    return await asyncio.to_thread(_build_file_tree_sync, _to_path(base_dir), _to_path(current_dir))


async def serve_file(file_path: FileSystemPath) -> Response:
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
    file_path = _to_path(file_path)
    mime_type, _ = mimetypes.guess_type(file_path.name)
    suffix = file_path.suffix.lower()

    is_text = (
        (mime_type and mime_type.startswith('text/')) or suffix in _TEXT_EXTENSIONS or mime_type == 'application/json'
    )

    if is_text:
        size = await asyncio.to_thread(lambda: file_path.stat().st_size)
        if size > _MAX_TEXT_BYTES:
            raise HTTPException(
                HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f'File too large to preview ({size:,} bytes). Download the file instead.',
            )
        effective_mime = (mime_type or 'text/plain') + '; charset=utf-8'
        content = await asyncio.to_thread(file_path.read_bytes)
        return Response(content=content, media_type=effective_mime)

    return FileResponse(
        path=str(file_path),
        media_type=mime_type or 'application/octet-stream',
        filename=file_path.name,
    )
