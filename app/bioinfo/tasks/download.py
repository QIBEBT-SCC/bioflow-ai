import os
import asyncio
import aiohttp
import aiofiles
from urllib.parse import urlparse, unquote
from tqdm.asyncio import tqdm
import re


async def download_file(
    session: aiohttp.ClientSession, url: str, save_dir: str, total_pbar: tqdm, position: int
) -> None:
    """
    异步下载单个文件并保持最后一级文件夹结构

    Args:
        session: aiohttp会话
        url: 下载链接
        save_dir: 保存目录
        total_pbar: 总进度条对象
        position: 进度条位置
    """
    try:
        parsed_url = urlparse(url)
        # 解码URL
        decoded_path = unquote(parsed_url.path)

        # 分割路径并获取最后两个部分（文件夹和文件名）
        path_parts = [p for p in decoded_path.split('/') if p]
        if len(path_parts) >= 2:
            # 取最后两个部分
            final_path = os.path.join(path_parts[-2], path_parts[-1])
        else:
            # 如果路径部分少于2个，就只使用文件名
            final_path = path_parts[-1] if path_parts else 'unknown_file'

        # 移除URL参数（如果存在）
        final_path = re.sub(r'\?.*$', '', final_path)

        # 构建保存路径
        save_path = os.path.join(save_dir, final_path)

        # 创建必要的目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        async with session.get(url) as response:
            if response.status == 200:
                # 获取文件大小
                total = int(response.headers.get('content-length', 0))

                # 创建单文件进度条
                desc = f'下载 {os.path.basename(final_path)}'
                with tqdm(
                    total=total,
                    desc=desc,
                    unit='B',
                    unit_scale=True,
                    position=position,
                    leave=False,
                ) as pbar:
                    async with aiofiles.open(save_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                            pbar.update(len(chunk))

                # 更新总进度条
                total_pbar.update(1)
            else:
                print(f'\r下载失败 {url}, 状态码: {response.status}')
                total_pbar.update(1)
    except Exception as e:
        print(f'\r下载出错 {url}: {str(e)}')
        total_pbar.update(1)


async def download_raw_data(urls_file: str, save_dir: str, max_concurrent: int = 5) -> None:
    """
    从文本文件中读取URL并异步下载所有文件

    Args:
        urls_file: 包含下载链接的文本文件路径
        save_dir: 文件保存目录
        max_concurrent: 最大并发下载数
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

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

    # 创建限制并发数的信号量
    semaphore = asyncio.Semaphore(max_concurrent)

    # 创建总进度条
    total_files = len(urls)

    # 使用position参数确保总进度条在最下方
    with tqdm(
        total=total_files, desc='总体进度', unit='files', position=max_concurrent, leave=True
    ) as total_pbar:

        async def bounded_download(url: str, position: int):
            async with semaphore:
                async with aiohttp.ClientSession() as session:
                    await download_file(session, url, save_dir, total_pbar, position)

        # 创建所有下载任务
        tasks = [bounded_download(url, i % max_concurrent) for i, url in enumerate(urls)]

        # 执行所有下载任务
        try:
            await asyncio.gather(*tasks)
            print('\n所有文件下载完成！')
        except Exception as e:
            print(f'\n下载过程中发生错误: {str(e)}')


if __name__ == '__main__':
    asyncio.run(
        download_raw_data(
            '/mnt/sda/yeyu/scc/250312sc/linklist.txt', '/mnt/sda/yeyu/scc/250312sc/raw_data'
        )
    )
