import io
import zipfile
from pathlib import Path
from typing import Optional

import httpx
import nanoid
from anyio import Path as AsyncPath
from loguru import logger

from bioflow_ai.core.config import get_settings
from bioflow_ai.schemas.documents import ImageMap, PDFParseResult

PARSE_OPTION = {
    'backend': 'hybrid-auto-engine',
    'parse_method': 'auto',
    'formula_enable': True,
    'table_enable': True,
    'return_md': True,
    'return_content_list': False,
    'return_images': True,
    'response_format_zip': True,
}


def _get_async_client() -> httpx.AsyncClient:
    timeout = get_settings().pdf_parser.open_mineru.TIMEOUT

    return httpx.AsyncClient(timeout=timeout)


async def parse(file_path: str | Path, output_dir: str | Path) -> PDFParseResult:
    """调用 MinerU 解析 PDF, 返回保存后的 Markdown 路径和重命名的图片路径列表."""
    pdf_path = await AsyncPath(file_path).expanduser()
    pdf_path = await pdf_path.absolute()
    if not await pdf_path.exists():
        err_msg = f'文件不存在: {pdf_path}'
        raise FileNotFoundError(err_msg)

    out_dir = await AsyncPath(output_dir).expanduser()
    out_dir = await out_dir.absolute()
    await out_dir.mkdir(parents=True, exist_ok=True)

    endpoint = f'{get_settings().pdf_parser.open_mineru.API_URL}/file_parse'

    pdf_content = await pdf_path.read_bytes()
    async with _get_async_client() as client:
        logger.info(f'Calling MinerU at {endpoint} to parse {pdf_path.name}')
        files = [('files', (pdf_path.name, pdf_content, 'application/pdf'))]
        response = await client.post(endpoint, data=PARSE_OPTION, files=files)

    response.raise_for_status()
    content_type = response.headers.get('content-type', '')
    if 'application/zip' not in content_type and 'application/octet-stream' not in content_type:
        err_msg = f'Incorrect response type {content_type}'
        raise ValueError(err_msg)

    md_path: Optional[AsyncPath] = None
    image_paths: list[AsyncPath] = []

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        names = zf.namelist()

        for name in names:
            target = out_dir / name
            await target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src:
                await target.write_bytes(src.read())

            lower = name.lower()
            if lower.endswith(('.md', '.markdown')):
                md_path = target
            if lower.endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                image_paths.append(target)

    if md_path is None:
        err_msg = 'The parsing result is empty'
        raise ValueError(err_msg)

    # 重命名图片为 nanoid
    image_mapper: list[ImageMap] = []
    for img_path in image_paths:
        img_uid = nanoid.generate(size=8)
        new_path = img_path.with_stem(img_uid)
        image_mapper.append(ImageMap(old_name=f'images/{img_path.name}', file_path=str(new_path)))
        await img_path.rename(new_path)

    return PDFParseResult(markdown=str(md_path), images=image_mapper)


__all__ = ['parse']
