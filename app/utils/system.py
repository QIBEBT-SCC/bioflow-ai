import os
import subprocess
from functools import lru_cache
from pathlib import Path

from app.schemas.run_instance import OutputFileItem, OutputFileNode, OutputFolderItem


@lru_cache
def get_cpu_cores() -> int:
    """Get CPU core count."""
    return os.cpu_count()


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
