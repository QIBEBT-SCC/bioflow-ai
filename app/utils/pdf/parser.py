import base64
import tempfile
from pathlib import Path

from langchain_milvus import Milvus
from langchain_text_splitters import MarkdownHeaderTextSplitter
from loguru import logger
from sqlmodel import Session

from app.core.config import get_settings
from app.core.session import engine
from app.crud.resource import file as file_crud
from app.models.resource import File
from app.utils.file import storage
from app.utils.file.hash import calculate_base64_sha256
from app.utils.pdf import open_mineru
from llm.models import qwen3_embedding


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
                parse_result: open_mineru.OpenMinerUParseResult = await open_mineru.parse(pdf_path, parse_output_dir)

                md_path = Path(parse_result.markdown)
                img_mappers = parse_result.images

            case _:
                raise ValueError(f'不支持的解析引擎: {engine}')

        # 上传文件到对象存储（所有文件使用 PDF 的 hash 来生成路径，确保在同一目录下）
        # 同时更新markdown中的图片引用路径
        pdf_key = storage.upload_file(pdf_path, pdf_hash, prefix='documents')

        md_text = md_path.read_text(encoding='utf-8')
        for mapper in img_mappers:
            img_key = storage.upload_file(mapper.file_path, pdf_hash, prefix='documents')
            md_text = md_text.replace(mapper.old_name, img_key)

        md_path.write_text(md_text, encoding='utf-8')
        storage.upload_file(md_path, pdf_hash, prefix='documents')

        # 将文件存入向量数据库
        headers_to_split_on = [('#', 'Title'), ('##', 'Section')]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
        md_docs = markdown_splitter.split_text(md_text)
        for doc in md_docs:
            doc.metadata['doc_id'] = pdf_key

        embedding_model = qwen3_embedding()
        vectorstore = Milvus(
            collection_name='Papers',
            connection_args={'uri': get_settings().knowledge_base.MILVUS_URL},
            embedding_function=embedding_model,
            index_params={'index_type': 'HNSW', 'metric_type': 'L2'},
            consistency_level='Strong',
            drop_old=False,
            auto_id=True,
        )
        batch_size = 5
        for i in range(0, len(md_docs), batch_size):
            batch = md_docs[i : i + batch_size]
            await vectorstore.aadd_documents(batch)

        # 保存到数据库
        with Session(engine) as session:
            new_file = File(sha256=pdf_hash, file_name=file_name, s3_path=pdf_key)
            file_crud.insert(session, new_file)

        return pdf_key
