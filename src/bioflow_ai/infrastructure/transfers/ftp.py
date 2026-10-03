import asyncio
import shutil
from ftplib import FTP
from pathlib import Path
from typing import Optional, Self

import aiohttp
import anyio
from loguru import logger
from pydantic import BaseModel
from tqdm.asyncio import tqdm


class DownloadInfo(BaseModel):
    remote_link: str
    local_path: str | Path


class FTPClient:
    def __init__(self, host: str, port: int = 21, username: str = '', password: str = '', timeout: int = 30):
        """初始化FTP客户端.

        Args:
            host: FTP服务器地址
            port: FTP服务器端口, 默认为21
            username: 用户名, 默认为空字符串(匿名登录)
            password: 密码, 默认为空字符串
            timeout: 连接超时时间, 默认为30秒
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.timeout = timeout

        self.ftp: FTP = None  # pyrefly: ignore [bad-assignment]
        self._is_connected = False

    def __enter__(self) -> Self:  # noqa: D105
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001, D105
        self.disconnect()

    def connect(self) -> bool:
        """连接到FTP服务器.

        Returns:
            bool: 连接是否成功
        """
        try:
            self.ftp = FTP()  # noqa: S321
            self.ftp.connect(self.host, self.port, self.timeout)
            self.ftp.login(self.username, self.password)
            self._is_connected = True
            logger.info(f'成功连接到FTP服务器: {self.host}:{self.port}')
        except Exception as e:
            logger.error(f'连接FTP服务器失败: {e!s}')
            self._is_connected = False
            return False
        else:
            return True

    def disconnect(self) -> None:
        """断开与FTP服务器的连接."""
        if self.ftp and self._is_connected:
            try:
                self.ftp.quit()
                logger.info(f'已断开与FTP服务器的连接: {self.host}:{self.port}')
            except Exception as e:
                logger.warning(f'断开FTP连接时发生异常: {e!s}')
            finally:
                self._is_connected = False
                self.ftp = None  # pyrefly: ignore [bad-assignment]

    def list_files(self, remote_path: str = '.') -> list[str]:
        """获取指定路径下的文件和目录列表.

        Args:
            remote_path: 远程服务器上的路径, 默认为当前目录

        Returns:
            文件名的列表
        """
        if not self._is_connected and not self.connect():
            logger.error('无法获取文件列表: 未连接到FTP服务器')
            return []

        try:
            self.ftp.cwd(remote_path)

            file_list = []

            def _process_line(line: str) -> None:
                parts = line.split()
                if len(parts) >= 9:  # noqa: PLR2004
                    is_dir = parts[0].startswith('d')
                    name = ' '.join(parts[8:])
                    if name not in ('.', '..') and not is_dir:
                        file_list.append(name)

            self.ftp.retrlines('LIST', _process_line)

            logger.info(f'成功获取目录 {remote_path} 下的文件列表, 共 {len(file_list)} 个项目')
        except Exception as e:
            logger.error(f'获取文件列表失败: {e!s}')
            return []
        else:
            return file_list


class PositionManager:
    def __init__(self, max_concurrency: int):
        self.max_concurrency = max_concurrency
        self.available_positions = list(range(max_concurrency))
        self.used_positions = {}
        self.lock = asyncio.Lock()

    async def get_position(self, task_id: str) -> int:
        """获取一个可用位置."""
        async with self.lock:
            if not self.available_positions:
                return max(self.used_positions.values()) + 1 if self.used_positions else 1
            position = self.available_positions.pop(0)
            self.used_positions[task_id] = position
            return position + 1

    async def release_position(self, task_id: str) -> None:
        """释放位置."""
        async with self.lock:
            if task_id in self.used_positions:
                position = self.used_positions.pop(task_id)
                if position < self.max_concurrency:  # 只回收原始并发范围内的位置
                    self.available_positions.append(position)


class HTTPDownloader:
    def __init__(
        self,
        max_concurrency: int = 5,
        timeout: int = 30,
        chunk_size: int = 1024 * 1024,  # 1MB
    ):
        """初始化HTTP下载器.

        Args:
            max_concurrency: 最大并发下载数, 默认为5
            timeout: 连接超时时间, 默认为30秒
            chunk_size: 下载分块大小, 默认为1MB
        """
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.download_results: dict[str, bool] = {}
        self.failed_files: set[DownloadInfo] = set()
        self.session: aiohttp.ClientSession = None  # pyrefly: ignore [bad-assignment]
        self.position_manager = PositionManager(max_concurrency)

        self.global_pbar = None
        self.total_files = 0
        self.completed_files = 0

    async def _ensure_session(self) -> None:
        """确保aiohttp会话已创建."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))

    async def download_file(self, url: str, local_path: str | Path, task_id: str) -> Optional[bool]:
        """异步下载单个文件.

        Args:
            url: 文件URL
            local_path: 本地保存路径
            task_id: 任务ID, 用于位置管理

        Returns:
            bool: 下载是否成功
        """
        download_file = local_path if isinstance(local_path, Path) else Path(local_path)

        filename = download_file.name
        async with self.semaphore:
            await self._ensure_session()

            download_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file = download_file.parent / f'{filename}.download'

            position = await self.position_manager.get_position(task_id)

            try:
                async with self.session.get(url, allow_redirects=True) as response:
                    if response.status != 200:  # noqa: PLR2004
                        logger.error(f'下载文件失败: {url}, 状态码: {response.status}')

                        if self.global_pbar:
                            self.completed_files += 1
                            self.global_pbar.update(1)
                        return False

                    file_size = int(response.headers.get('Content-Length', 0))
                    downloaded = 0

                    desc = f'下载 {filename}'

                    with tqdm(
                        desc=desc,
                        total=file_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        miniters=1,
                        position=position,
                        leave=False,
                    ) as pbar:
                        async with await anyio.open_file(temp_file, 'wb') as f:
                            async for chunk in response.content.iter_chunked(self.chunk_size):
                                if chunk:
                                    await f.write(chunk)
                                    downloaded += len(chunk)
                                    pbar.update(len(chunk))

                shutil.move(temp_file, download_file)

                if self.global_pbar:
                    self.completed_files += 1
                    self.global_pbar.update(1)
            except Exception:
                if temp_file.exists():
                    temp_file.unlink(missing_ok=True)

                if self.global_pbar:
                    self.completed_files += 1
                    self.global_pbar.update(1)

                return False
            else:
                return True
            finally:
                await self.position_manager.release_position(task_id)

    async def download_files(
        self, file_list: list[DownloadInfo], local_dir: str, *, keep_path: bool = True
    ) -> dict[str, bool]:
        """批量下载文件.

        Args:
            file_list: 要下载的文件列表
            local_dir: 本地保存根目录
            keep_path: 是否保留远程目录结构

        Returns:
            Dict[str, bool]: 文件下载结果字典, 键为文件名, 值为是否下载成功
        """
        logger.info(f'开始下载 {len(file_list)} 个文件, 最大并发数: {self.max_concurrency}')

        self.total_files = len(file_list)
        self.completed_files = 0

        # 全局固定在位置0
        self.global_pbar = tqdm(desc='总进度', total=self.total_files, unit='个文件', position=0, leave=True)

        tasks = []
        for i, download_info in enumerate(file_list):
            local_file = Path(download_info.local_path)
            local_path = Path(local_dir) / local_file if keep_path else Path(local_dir) / local_file.name
            task_id = f'task_{i}'

            task = asyncio.create_task(self.download_file(download_info.remote_link, local_path, task_id))
            tasks.append((str(local_file), task))

        results = {}
        for remote_file, task in tasks:
            try:
                results[remote_file] = await task
            except Exception as e:
                logger.error(f'下载文件 {remote_file} 时发生错误: {e!s}')
                results[remote_file] = False

        if self.session and not self.session.closed:
            await self.session.close()

        return results  # pyrefly: ignore [bad-return]

    async def retry_failed_files(
        self, local_dir: str, *, keep_path: bool = True, max_retries: int = 3
    ) -> dict[str, bool]:
        """重试下载失败的文件.

        Args:
            local_dir: 本地保存目录
            keep_path: 是否保留远程目录结构
            max_retries: 最大重试次数

        Returns:
            Dict[str, bool]: 重试结果字典
        """
        retry_results = {}

        for retry in range(max_retries):
            if not self.failed_files:
                break

            logger.info(f'第 {retry + 1} 次重试下载失败的文件，共 {len(self.failed_files)} 个')

            current_failed = list(self.failed_files)
            self.failed_files.clear()

            results = await self.download_files(current_failed, local_dir, keep_path=keep_path)
            retry_results.update(results)

            if not self.failed_files:
                logger.info('所有文件重试下载成功')
                break

            logger.info(f'第 {retry + 1} 次重试后仍有 {len(self.failed_files)} 个文件下载失败')

        return retry_results

    async def close(self) -> None:
        """关闭HTTP会话."""
        if self.session and not self.session.closed:
            await self.session.close()

        if self.global_pbar:
            self.global_pbar.clear()
            self.global_pbar.close()
            print()  # noqa: T201


async def download_http_files(
    file_list: list[DownloadInfo],
    local_dir: str,
    *,
    max_concurrency: int = 5,
    retry_count: int = 3,
    timeout: int = 30,
    keep_path: bool = True,
) -> Optional[dict[str, bool]]:
    """下载HTTP文件.

    Args:
        file_list: 要下载的文件列表
        local_dir: 本地保存目录
        max_concurrency: 最大并发下载数
        retry_count: 失败重试次数
        timeout: 超时时间 (秒)
        keep_path: 是否保持远端的目录结构

    Returns:
        Dict[str, bool]: 文件下载结果字典
    """
    downloader = HTTPDownloader(max_concurrency=max_concurrency, timeout=timeout)

    try:
        results = await downloader.download_files(file_list, local_dir, keep_path=keep_path)

        if downloader.failed_files and retry_count > 0:
            retry_results = await downloader.retry_failed_files(local_dir, keep_path=keep_path, max_retries=retry_count)
            results.update(retry_results)

        return results
    finally:
        await downloader.close()
