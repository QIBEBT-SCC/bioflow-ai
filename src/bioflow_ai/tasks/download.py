from pathlib import Path
from urllib.parse import unquote, urlparse

import aiofiles

from bioflow_ai.infrastructure.transfers.ftp import DownloadInfo, download_http_files


async def download_raw_data(
    urls_file: str, save_dir: str, max_concurrent: int = 5, timeout: int = 60
) -> None:
    """
    从文本文件中读取URL并异步下载所有文件

    Args:
        urls_file: 包含下载链接的文本文件路径
        save_dir: 文件保存目录
        max_concurrent: 最大并发下载数
        timeout: 文件超时阈值
    """
    # 读取URL列表
    try:
        async with aiofiles.open(urls_file, 'r') as f:
            content = await f.read()
            urls = [line.strip() for line in content.split('\n') if line.strip()]
    except FileNotFoundError:
        print(f'找不到URL文件: {urls_file}')
        return
    except Exception as e:
        print(f'读取URL文件时出错: {str(e)}')
        return

    download_list = []
    for origin_url in urls:
        parsed_url = urlparse(origin_url)
        decoded_path = Path(unquote(parsed_url.path))

        parts = decoded_path.parts

        if len(parts) >= 2:
            local_file = Path(parts[-2]) / parts[-1]
        else:
            local_file = parts[-1] if parts else 'unknown_file'

        download_list.append(DownloadInfo(remote_link=origin_url, local_path=local_file))

    await download_http_files(
        file_list=download_list, local_dir=save_dir, max_concurrency=max_concurrent, timeout=timeout
    )
