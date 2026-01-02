import io
import zipfile
from pathlib import Path
from typing import Literal, Optional

import httpx
import uuid_extensions
from loguru import logger
from pydantic import BaseModel, Field

from app.core.config import get_settings


class MineruParseOptions(BaseModel):
    backend: Literal[
        'pipeline',
        'vlm-transformers',
        'vlm-mlx-engine',
        'vlm-vllm-async-engine',
        'vlm-lmdeploy-engine',
        'vlm-http-client',
    ] = Field(
        'pipeline',
        description='解析后端，pipeline / vlm-transformers / vlm-mlx-engine / vlm-vllm-async-engine / vlm-lmdeploy-engine / vlm-http-client',
    )
    parse_method: str = Field('auto', description='解析方式，auto / txt / ocr，仅 pipeline 后端有效')
    start_page_id: int = Field(0, description='起始页，0 基')
    end_page_id: int = Field(99999, description='终止页，0 基，默认解析到末尾')
    lang_list: list[str] = Field(default_factory=lambda: ['ch', 'en'], description='语言列表，仅 pipeline 后端有效')
    formula_enable: bool = Field(True, description='是否解析公式')
    table_enable: bool = Field(True, description='是否解析表格')
    return_md: bool = Field(True, description='返回 Markdown')
    return_content_list: bool = Field(False, description='返回 content_list JSON')
    return_images: bool = Field(False, description='返回提取的图片')
    return_middle_json: bool = Field(False, description='返回中间 JSON')
    return_model_output: bool = Field(False, description='返回模型原始输出')
    response_format_zip: bool = Field(False, description='返回 ZIP 文件而非 JSON')
    output_dir: str = Field('./output', description='服务端输出目录')
    server_url: Optional[str] = Field(None, description='当 backend=vlm-http-client 时的 OpenAI 兼容服务地址')


def _get_async_client() -> httpx.AsyncClient:
    timeout = get_settings().pdf_parser.open_mineru.TIMEOUT

    return httpx.AsyncClient(timeout=timeout)


async def parse(file_path: str | Path, output_dir: str | Path) -> dict[str, list[str] | str]:
    """调用 MinerU 解析 PDF，返回保存后的 Markdown 路径和重命名的图片路径列表"""

    pdf_path = Path(file_path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f'文件不存在: {pdf_path}')

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    options = MineruParseOptions(
        backend='vlm-vllm-async-engine', return_md=True, return_images=True, response_format_zip=True
    )
    endpoint = f'{get_settings().pdf_parser.open_mineru.API_URL}/file_parse'

    async with _get_async_client() as client:
        logger.info(f'Calling MinerU at {endpoint} to parse {pdf_path.name}')
        with pdf_path.open('rb') as f:
            files = [('files', (pdf_path.name, f, 'application/pdf'))]
            response = await client.post(endpoint, data=options.model_dump(), files=files)

    response.raise_for_status()
    content_type = response.headers.get('content-type', '')
    if 'application/zip' not in content_type and 'application/octet-stream' not in content_type:
        raise ValueError(f'Incorrect response type {content_type}')

    md_path: Optional[Path] = None
    image_paths: list[Path] = []
    name_map: dict[str, str] = {}

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        names = zf.namelist()

        for name in names:
            target = out_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(target, 'wb') as dst:
                dst.write(src.read())

            lower = name.lower()
            if lower.endswith(('.md', '.markdown')):
                md_path = target
            if lower.endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                image_paths.append(target)

    if md_path is None:
        raise ValueError('The parsing result is empty')

    # 重命名图片为 UUID
    new_image_paths: list[str] = []
    for img_path in image_paths:
        img_uid = uuid_extensions.uuid7str()
        new_path = img_path.with_stem(img_uid)
        name_map[img_path.name] = new_path.name
        img_path.rename(new_path)
        new_image_paths.append(str(new_path))

    # 更新 Markdown 中的图片引用
    md_text = md_path.read_text(encoding='utf-8')
    for old_name, new_name in name_map.items():
        md_text = md_text.replace(f'images/{old_name}', new_name)
    md_path.write_text(md_text, encoding='utf-8')

    return {'markdown': str(md_path), 'images': new_image_paths}


__all__ = ['MineruParseOptions', 'parse']
