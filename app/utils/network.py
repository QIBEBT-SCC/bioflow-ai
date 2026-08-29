import gzip
import shutil
import tempfile
import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import httpx
from loguru import logger
from requests import ReadTimeout
from requests.exceptions import ChunkedEncodingError, SSLError
from urllib3.exceptions import ResponseError

from app.core.config import get_settings


def retry(retries: int = 3, delay: float = 1) -> Callable:
    """为函数提供重试逻辑的装饰器.

    Args:
        retries: 最大重试次数, 默认为3
        delay: 两次重试之间的延迟时间 (秒), 默认为1

    Returns:
        Callable: 被装饰的函数

    Raises:
        ValueError: 如果retries小于1或delay小于等于0, 则抛出此异常
    """
    if retries < 1 or delay <= 0:
        raise ValueError

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for i in range(1, retries + 1):
                try:
                    return func(*args, **kwargs)
                except (ResponseError, SSLError, ReadTimeout, ChunkedEncodingError, ConnectionError) as e:
                    if i == retries:
                        logger.error(f'Error: {e!r}.')
                        logger.error(f'"{func.__name__}()" failed after {retries} retries.')
                        break
                    logger.debug(f'Error: {e!r} -> Retrying...')
                    time.sleep(delay)
            return None

        return wrapper

    return decorator


def download_gz_files(
    files_spec: dict[str, dict[str, str]],
    target_dir: Path,
) -> dict[str, str]:
    """从 URL 下载并解压参考文件, 已存在则跳过.

    Args:
        files_spec: 文件规格字典, key 为逻辑名称, value 包含 'url' 和 'filename'.
        target_dir: 解压后文件的目标目录.

    Returns:
        逻辑名称到本地文件路径的映射.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    proxy = get_settings().network.get_proxy_url() if get_settings().network.USE_PROXY else None
    result = {}

    for key, info in files_spec.items():
        target = target_dir / info['filename']
        if target.exists():
            logger.debug(f'{key} already exists: {target}')
            result[key] = str(target)
            continue

        logger.debug(f'Downloading {key} from {info["url"]}')
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / f'{info["filename"]}.gz'
            with httpx.Client(timeout=3600, proxy=proxy) as client, client.stream('GET', info['url']) as resp:
                resp.raise_for_status()
                with gz_path.open('wb') as f:
                    f.writelines(resp.iter_bytes(chunk_size=8192))

            logger.debug(f'Decompressing {gz_path.name}')
            with gzip.open(gz_path, 'rb') as f_in, target.open('wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        result[key] = str(target)
        logger.debug(f'{key} ready: {target}')

    return result
