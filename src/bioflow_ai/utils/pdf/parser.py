import base64
import tempfile
from pathlib import Path

from loguru import logger
from sqlalchemy.orm import Session

from bioflow_ai.core.config import get_settings
from bioflow_ai.crud.resource import file as file_crud
from bioflow_ai.infrastructure.database import engine
from bioflow_ai.infrastructure.external import open_mineru
from bioflow_ai.infrastructure.object_storage import files as storage
from bioflow_ai.llm.utils.retrievers import insert_documents
from bioflow_ai.models.resource import File
from bioflow_ai.services.resource import file as file_service
from bioflow_ai.utils.file.hash import calculate_base64_sha256


async def parse_pdf(base64_data: str, file_name: str) -> str:
    """Parse a PDF, deduplicate it, and upload its generated artifacts.

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
                parse_result: open_mineru.OpenMinerUParseResult = await open_mineru.parse(pdf_path, parse_output_dir)

                md_path = Path(parse_result.markdown)
                img_mappers = parse_result.images

            case _:
                message = f'不支持的解析引擎: {engine}'
                raise ValueError(message)

        # Upload all generated files below a shared path derived from the PDF hash.
        # 同时更新markdown中的图片引用路径
        pdf_key = storage.upload_file(pdf_path, pdf_hash, prefix='documents')

        md_text = md_path.read_text(encoding='utf-8')  # noqa: ASYNC240
        for mapper in img_mappers:
            img_key = storage.upload_file(mapper.file_path, pdf_hash, prefix='documents')
            md_text = md_text.replace(mapper.old_name, img_key)

        md_path.write_text(md_text, encoding='utf-8')  # noqa: ASYNC240
        storage.upload_file(md_path, pdf_hash, prefix='documents')

        # Persist parent documents in PostgreSQL and child documents in Milvus.
        await insert_documents(md_text, pdf_key)

        # 保存到数据库
        with Session(engine) as session:
            new_file = File(sha256=pdf_hash, file_name=file_name, s3_path=pdf_key)
            file_service.create_file(session, new_file)

        return pdf_key
