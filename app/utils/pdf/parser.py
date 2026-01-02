import base64
import tempfile
from pathlib import Path

from loguru import logger
from sqlmodel import Session

from app.core.config import get_settings
from app.core.session import engine
from app.crud.resource import file as file_crud
from app.models.resource import File
from app.utils.file import storage
from app.utils.file.hash import calculate_base64_sha256
from app.utils.pdf import open_mineru


async def parse_pdf(base64_data: str, file_name: str) -> str:
    """解析 PDF 并上传到对象存储，带查重功能

    Args:
        base64_data: PDF 文件的 base64 编码
        file_name: 原始文件名

    Returns:
        ParseResult: 包含 PDF 和 Markdown 在对象存储中的 key

    Raises:
        ValueError: 解析引擎不支持或解析失败
    """

    with Session(engine) as session:
        pdf_hash = calculate_base64_sha256(base64_data)
        existing_file = file_crud.get_by_sha256(session, pdf_hash)
        if existing_file:
            return existing_file.s3_path

    logger.info('PDF not found in database, starting parse process')

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        pdf_data = base64.b64decode(base64_data)
        pdf_path = tmp_path / file_name
        pdf_path.write_bytes(pdf_data)
        logger.debug(f'Saved PDF to temporary location: {pdf_path}')

        match get_settings().pdf_parser.ENGINE:
            case 'open_mineru':
                # 调用本地部署 MinerU 解析
                parse_output_dir = tmp_path / 'parsed'
                parse_result = await open_mineru.parse(pdf_path, parse_output_dir)

                md_path = Path(parse_result['markdown'])
                image_paths = [Path(p) for p in parse_result['images']]

            case _:
                raise ValueError(f'不支持的解析引擎: {engine}')

        # 上传文件到对象存储（所有文件使用 PDF 的 hash 来生成路径，确保在同一目录下）
        pdf_key = storage.upload_file(pdf_path, pdf_hash, prefix='documents/')
        storage.upload_file(md_path, pdf_hash, prefix='documents/')
        for img_path in image_paths:
            storage.upload_file(img_path, pdf_hash, prefix='documents/')

        # 保存到数据库
        with Session(engine) as session:
            new_file = File(sha256=pdf_hash, file_name=file_name, s3_path=pdf_key)
            file_crud.insert(session, new_file)

        return pdf_key
