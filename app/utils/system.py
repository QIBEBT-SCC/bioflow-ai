import os
import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache
def get_cpu_cores() -> int:
    return os.cpu_count()


def format_size(size_bytes: int) -> str:
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == 'TB':
            return f'{size:.2f} {unit}'
        size /= 1024
    return ''


def get_file_size(file_path: str | Path) -> int:
    """获取文件大小（字节）

    Args:
        file_path: 文件路径

    Returns:
        文件大小（字节），如果文件不存在返回 0
    """
    try:
        return Path(file_path).stat().st_size
    except (FileNotFoundError, OSError):
        return 0


def get_folder_size_du(path: str) -> str:
    result = subprocess.run(['du', '-sb', path], stdout=subprocess.PIPE, text=True)
    size_in_bytes = int(result.stdout.split()[0])
    return format_size(size_in_bytes)


def get_current_user_identity() -> str:
    """获取当前用户的UID:GID

    Returns:
        str: 格式为 "UID:GID" 的字符串
    """
    uid = os.getuid()
    gid = os.getgid()
    return f'{uid}:{gid}'
