import asyncio
from typing import Literal

import click
from anyio import Path
from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from bioflow_ai.utils.init.database import (
    AdminInitResult,
    PresetImportResult,
    init_admin,
    init_docs,
    init_images,
    init_preset_database_records,
    init_tools,
    rebuild_auxiliary_node_index,
)


def _log_admin_result(result: AdminInitResult) -> None:
    """Log default administrator initialization result."""
    if not result.created:
        logger.info('Default admin account already exists, skipped.')
        return

    logger.info('Administrator account created with the credentials supplied in the prompt.')


def _log_import_result(result: PresetImportResult) -> None:
    """Log preset import result."""
    source = 'downloaded' if result.downloaded else 'local'
    logger.info(f'Initialized {result.name} from {result.preset_file} ({source}).')
    if result.deleted_count is not None:
        logger.info(f'Delete {result.deleted_count} old {result.name} records.')
    logger.info(f'Inserted {result.inserted_count} {result.name} records.')
    if result.validation_error_count:
        logger.warning(f'{result.name} validation failed for {result.validation_error_count} records.')
    if result.warning_count:
        logger.warning(f'{result.name} completed with {result.warning_count} warnings.')


async def rebuild_image_index() -> None:
    """重建镜像向量索引."""
    from langchain_core.documents import Document
    from tqdm import tqdm

    from bioflow_ai.core.config import get_settings
    from bioflow_ai.core.session import engine
    from bioflow_ai.llm.models import qwen3_embedding
    from bioflow_ai.llm.vectorstores import MilvusVectorStore
    from bioflow_ai.models.tools import ToolImage

    embedding = qwen3_embedding()
    vectorstore = MilvusVectorStore(
        embedding_function=embedding,
        collection_name='ImageCollection',
        uri=get_settings().knowledge_base.MILVUS_URL,
        drop_old=True,
    )

    with Session(engine) as session:
        statement = select(
            ToolImage.uid,
            ToolImage.name,
            ToolImage.description,
        )
        results = session.execute(statement).all()

    images_to_add = []
    empty_images = []
    batch_size = 10

    for image_data in tqdm(results, total=len(results)):
        if image_data.description:
            images_to_add.append(
                Document(
                    page_content=image_data.description,
                    metadata={
                        'image_uid': str(image_data.uid),
                        'name': image_data.name,
                    },
                )
            )

            if len(images_to_add) >= batch_size:
                async with vectorstore:
                    await vectorstore.aadd_documents(images_to_add)
                images_to_add = []
        else:
            empty_images.append(str(image_data.uid))

    if images_to_add:
        async with vectorstore:
            await vectorstore.aadd_documents(images_to_add)

    if empty_images:
        empty_file = Path('empty_images.txt')
        await empty_file.write_text('\n'.join(empty_images) + '\n')


async def rebuild_tool_index() -> None:
    """重建工具库向量索引."""
    from langchain_core.documents import Document
    from tqdm import tqdm

    from bioflow_ai.core.config import get_settings
    from bioflow_ai.core.session import engine
    from bioflow_ai.llm.models import qwen3_embedding
    from bioflow_ai.llm.vectorstores import MilvusVectorStore
    from bioflow_ai.models.tools import ToolDefine

    embedding = qwen3_embedding()
    vectorstore = MilvusVectorStore(
        collection_name='ToolCollection',
        embedding_function=embedding,
        uri=get_settings().knowledge_base.MILVUS_URL,
        hybrid=True,
        drop_old=True,
    )

    with Session(engine) as session:
        statement = select(ToolDefine)
        results = session.scalars(statement).all()

    docs_to_add = []
    empty_tools = []
    batch_size = 10

    for tool_data in tqdm(results, total=len(results)):
        if tool_data.description:
            docs_to_add.append(
                Document(
                    page_content=tool_data.description,
                    metadata={
                        'image_uid': str(tool_data.image_uid),
                        'tool_uid': str(tool_data.uid),
                        'name': tool_data.name,
                    },
                )
            )

            if len(docs_to_add) >= batch_size:
                async with vectorstore:
                    await vectorstore.aadd_documents(docs_to_add)
                docs_to_add = []
        else:
            empty_tools.append(str(tool_data.uid))

    if docs_to_add:
        async with vectorstore:
            await vectorstore.aadd_documents(docs_to_add)

    if empty_tools:
        empty_file = Path('empty_tools.txt')
        await empty_file.write_text('\n'.join(empty_tools) + '\n')


@click.group()
def cli() -> None:
    """Database initialization and management tool."""
    from bioflow_ai.utils.logger import init_logging

    init_logging()


@cli.command()
def system_init_cmd() -> None:
    """Initialize all system information, including the administrator user, database, and vector index."""

    async def _run() -> None:
        result = await init_preset_database_records(drop_old=True)
        _log_admin_result(result.admin)
        _log_import_result(result.images)
        _log_import_result(result.docs)
        _log_import_result(result.tools)
        logger.info('Rebuild auxiliary node index...')
        await rebuild_auxiliary_node_index()
        logger.info('System initialization completed.')

    asyncio.run(_run())


@cli.command()
def create_admin_cmd() -> None:
    """Interactively create the initial administrator account."""
    _log_admin_result(asyncio.run(init_admin()))


@cli.command()
@click.option('--drop-old', is_flag=True, default=False, help='Drop existing data before importing (default: keep)')
@click.option(
    '--databases',
    '-d',
    type=click.Choice(['all', 'image', 'doc', 'tool']),
    default='all',
    show_default=True,
    help='Target database(s) to initialize',
)
def init_db_cmd(*, drop_old: bool, databases: Literal['all', 'image', 'doc', 'tool']) -> None:
    """Initialize database records from preset files."""

    async def _run() -> None:
        if databases == 'all':
            result = await init_preset_database_records(drop_old=drop_old)
            _log_admin_result(result.admin)
            _log_import_result(result.images)
            _log_import_result(result.docs)
            _log_import_result(result.tools)
            return

        logger.warning(
            'Partial preset initialization does not rebuild dependent tables. '
            'Use `--databases all` for the safe image -> doc -> tool order.'
        )

        if databases in ('all', 'image'):
            _log_import_result(await init_images(drop_old=drop_old))

        if databases in ('all', 'doc'):
            _log_import_result(await init_docs(drop_old=drop_old))

        if databases in ('all', 'tool'):
            _log_import_result(await init_tools(drop_old=drop_old))

    asyncio.run(_run())


@cli.command()
@click.option(
    '--collections',
    '-c',
    type=click.Choice(['all', 'tool', 'processor', 'image']),
    default='all',
    show_default=True,
    help='Target vector index collection(s) to rebuild',
)
def rebuild_index_cmd(collections: Literal['all', 'tool', 'processor', 'image']) -> None:
    """Rebuild vector store indexes."""

    async def _run() -> None:
        if collections in ('all', 'image'):
            logger.info('Rebuild image index...')
            await rebuild_image_index()

        if collections in ('all', 'tool'):
            logger.info('Rebuild tool index...')
            await rebuild_tool_index()

        if collections in ('all', 'processor'):
            logger.info('Rebuild auxiliary node index...')
            await rebuild_auxiliary_node_index()

    asyncio.run(_run())


if __name__ == '__main__':
    cli()
