from functools import partial
from pathlib import Path

import anydoc
from anyio import Path as AsyncPath
from anyio import to_thread

from bioflow_ai.schemas.documents import PDFParseResult


async def parse(file_path: str | Path, output_dir: str | Path) -> PDFParseResult:
    """Convert a PDF locally and write Markdown into the caller's temporary directory."""
    pdf_path = await AsyncPath(file_path).expanduser()
    pdf_path = await pdf_path.absolute()
    if not await pdf_path.is_file():
        message = f'PDF 文件不存在: {pdf_path}'
        raise FileNotFoundError(message)

    try:
        convert = partial(anydoc.to_markdown, ocr='reject')
        markdown = await to_thread.run_sync(convert, str(pdf_path))
    except anydoc.NeedsOcrError as error:
        message = 'local 无法提取需要 OCR 的页面文本, 请配置 pdf_parser.engine = "open_mineru" 后重试'
        raise ValueError(message) from error
    except anydoc.EncryptedError as error:
        message = '无法解析加密 PDF, 请先解密文件后重试'
        raise ValueError(message) from error
    except anydoc.ConvertError as error:
        message = f'本地 PDF 解析失败: {error}'
        raise ValueError(message) from error

    if not markdown.strip():
        message = '本地 PDF 解析结果为空'
        raise ValueError(message)

    out_dir = await AsyncPath(output_dir).expanduser()
    out_dir = await out_dir.absolute()
    await out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / pdf_path.with_suffix('.md').name
    await md_path.write_text(markdown, encoding='utf-8')
    return PDFParseResult(markdown=str(md_path))
