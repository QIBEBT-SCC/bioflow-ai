import asyncio
from pathlib import Path
from typing import Literal

import click
from loguru import logger


async def init_images(*, drop_old: bool = True) -> None:
    """初始化镜像数据库."""
    import json

    import polars as pl
    from langchain_core.documents import Document
    from langchain_milvus import Milvus
    from pydantic import ValidationError
    from sqlmodel import Session
    from tqdm import tqdm

    import app.crud.tools.image as image_crud
    from app.core.config import get_settings
    from app.core.session import engine
    from app.models.tools import ToolImage
    from llm.models import qwen3_embedding

    parquet_file = 'presets/images.parquet'
    if Path(parquet_file).exists():
        logger.info('Initialize the image database with presets/images.parquet.')
    else:
        logger.info(
            'No local configuration file is found; a remote configuration file is used to initialize the mirror database.'
        )
        from huggingface_hub import download_bucket_files

        download_bucket_files(
            'Aye10032/bioflow-ai-prebuild',
            files=[('images.parquet', 'presets/images.parquet')],
        )

    images_df = pl.read_parquet(parquet_file)
    images: list[ToolImage] = []

    for row in images_df.iter_rows(named=True):
        row['image'] = json.loads(row['image'])

        try:
            images.append(ToolImage.model_validate(row))
        except ValidationError as e:
            logger.error(e)

    embedding = qwen3_embedding()
    vectorstore = Milvus(
        embedding_function=embedding,
        collection_name='ImageCollection',
        connection_args={'uri': get_settings().knowledge_base.MILVUS_URL},
        index_params={'index_type': 'HNSW', 'metric_type': 'L2'},
        consistency_level='Strong',
        drop_old=drop_old,
        auto_id=True,
    )

    batch_size = 50

    with Session(engine) as session:
        if drop_old:
            delete_count = image_crud.drop_all(session)
            logger.info(f'Delete {delete_count} data')

        images = image_crud.insert_all(session, images)

    for i in tqdm(range(0, len(images), batch_size)):
        batch_images = images[i : i + batch_size]

        docs = [
            Document(page_content=img.description, metadata={'image_uid': str(img.uid), 'name': img.name})
            for img in batch_images
        ]

        await vectorstore.aadd_documents(docs)


async def init_docs(*, drop_old: bool = True) -> None:
    """初始化工具文档库."""
    import json

    import polars as pl
    from pydantic import ValidationError
    from sqlmodel import Session

    import app.crud.tools.tool_doc as doc_crud
    from app.core.session import engine
    from app.models.tools import ToolHelpDoc

    parquet_file = 'presets/tool_docs.parquet'
    if Path(parquet_file).exists():
        logger.info('Initialize the image database with presets/tool_docs.parquet.')
    else:
        logger.info(
            'No local configuration file is found; a remote configuration file is used to initialize the mirror database.'
        )
        from huggingface_hub import download_bucket_files

        download_bucket_files(
            'Aye10032/bioflow-ai-prebuild',
            files=[('tool_docs.parquet', 'presets/tool_docs.parquet')],
        )

    docs_df = pl.read_parquet(parquet_file)
    docs = []

    for row in docs_df.iter_rows(named=True):
        try:
            docs.append(ToolHelpDoc.model_validate(row))
        except ValidationError as e:
            logger.error(e)

    with Session(engine) as session:
        if drop_old:
            delete_count = doc_crud.drop_all(session)
            logger.info(f'Delete {delete_count} data')

        doc_crud.insert_all(session, docs)


async def init_tools(*, drop_old: bool = True) -> None:
    """初始化工具数据库."""
    import json

    import polars as pl
    from langchain_core.documents import Document
    from langchain_milvus import Milvus
    from pydantic import ValidationError
    from sqlmodel import Session
    from tqdm import tqdm

    import app.crud.tools.tool as tool_crud
    from app.core.config import get_settings
    from app.core.session import engine
    from app.models.tools import ToolDefine
    from llm.models import qwen3_embedding

    parquet_file = 'presets/tools.parquet'
    if Path(parquet_file).exists():
        logger.info('Initialize the image database with presets/tools.parquet.')
    else:
        logger.info(
            'No local configuration file is found; a remote configuration file is used to initialize the mirror database.'
        )
        from huggingface_hub import download_bucket_files

        download_bucket_files(
            'Aye10032/bioflow-ai-prebuild',
            files=[('tools.parquet', 'presets/tools.parquet')],
        )

    tools_df = pl.read_parquet(parquet_file)
    tools = []

    for row in tools_df.iter_rows(named=True):
        for field in ('dynamic_params', 'file_mounts'):
            if row[field]:
                row[field] = json.loads(row[field])
            else:
                row[field] = []

        try:
            tools.append(ToolDefine.model_validate(row))
        except ValidationError as e:
            logger.error(e)

    embedding = qwen3_embedding()
    vectorstore = Milvus(
        embedding_function=embedding,
        collection_name='ToolCollection',
        connection_args={'uri': get_settings().knowledge_base.MILVUS_URL},
        index_params={'index_type': 'HNSW', 'metric_type': 'L2'},
        consistency_level='Strong',
        drop_old=drop_old,
        auto_id=True,
    )
    batch_size = 50

    with Session(engine) as session:
        if drop_old:
            delete_count = tool_crud.drop_all(session)
            logger.info(f'Delete {delete_count} data')

        tools = tool_crud.insert_all(session, tools)

        for i in tqdm(range(0, len(tools), batch_size)):
            batch_tools = tools[i : i + batch_size]

            docs = [
                Document(
                    page_content=tool.description,
                    metadata={'image_uid': str(tool.uid), 'tool_uid': str(tool.image.uid), 'name': tool.name},
                )
                for tool in batch_tools
            ]

            await vectorstore.aadd_documents(docs)


async def rebuild_image_index() -> None:
    """重建镜像向量索引."""
    from langchain_core.documents import Document
    from langchain_milvus import Milvus
    from sqlmodel import Session, select
    from tqdm import tqdm

    from app.core.config import get_settings
    from app.core.session import engine
    from app.models.tools import ToolImage
    from llm.models import qwen3_embedding

    embedding = qwen3_embedding()
    vectorstore = Milvus(
        embedding_function=embedding,
        collection_name='ImageCollection',
        connection_args={'uri': get_settings().knowledge_base.MILVUS_URL},
        index_params={'index_type': 'HNSW', 'metric_type': 'L2'},
        search_params={'M': 64, 'efConstruction': 100},
        consistency_level='Strong',
        drop_old=True,
        auto_id=True,
    )

    with Session(engine) as session:
        statement = select(
            ToolImage.uid,
            ToolImage.name,
            ToolImage.description,
        )
        results = session.exec(statement).all()

    images_to_add = []
    empty_images = []
    batch_size = 10

    for image_data in tqdm(results, total=len(results)):
        if image_data.description:  # type: ignore[attr-defined]
            images_to_add.append(
                Document(
                    page_content=image_data.description,  # type: ignore[attr-defined]
                    metadata={
                        'image_uid': str(image_data.uid),  # type: ignore[attr-defined]
                        'name': image_data.name,  # type: ignore[attr-defined]
                    },
                )
            )

            if len(images_to_add) >= batch_size:
                await vectorstore.aadd_documents(images_to_add)
                images_to_add = []
        else:
            empty_images.append(str(image_data.uid))  # type: ignore[attr-defined]

    if images_to_add:
        await vectorstore.aadd_documents(images_to_add)

    if empty_images:
        empty_file = Path('empty_images.txt')
        empty_file.write_text('\n'.join(empty_images) + '\n')


async def rebuild_tool_index() -> None:
    """重建工具库向量索引."""
    from langchain_core.documents import Document
    from langchain_milvus import BM25BuiltInFunction, Milvus
    from sqlmodel import Session, select
    from tqdm import tqdm

    from app.core.config import get_settings
    from app.core.session import engine
    from app.models.tools import ToolDefine
    from llm.models import qwen3_embedding

    embedding = qwen3_embedding()
    vectorstore = Milvus(
        collection_name='ToolCollection',
        connection_args={'uri': get_settings().knowledge_base.MILVUS_URL},
        embedding_function=embedding,
        builtin_function=BM25BuiltInFunction(),
        vector_field=['dense', 'sparse'],
        index_params=[{'index_type': 'HNSW', 'metric_type': 'L2'}, {}],
        consistency_level='Strong',
        drop_old=True,
        auto_id=True,
    )

    with Session(engine) as session:
        statement = select(
            ToolDefine.uid,
            ToolDefine.description,
            ToolDefine.image_uid,
            ToolDefine.name,
        )
        results = session.exec(statement).all()

    docs_to_add = []
    empty_tools = []
    batch_size = 10

    for tool_data in tqdm(results, total=len(results)):
        if tool_data.description:  # type: ignore[attr-defined]
            docs_to_add.append(
                Document(
                    page_content=tool_data.description,  # type: ignore[attr-defined]
                    metadata={
                        'image_uid': str(tool_data.image_uid),  # type: ignore[attr-defined]
                        'tool_uid': str(tool_data.uid),  # type: ignore[attr-defined]
                        'name': tool_data.name,  # type: ignore[attr-defined]
                    },
                )
            )

            if len(docs_to_add) >= batch_size:
                await vectorstore.aadd_documents(docs_to_add)
                docs_to_add = []
        else:
            empty_tools.append(str(tool_data.uid))  # type: ignore[attr-defined]

    if docs_to_add:
        await vectorstore.aadd_documents(docs_to_add)

    if empty_tools:
        empty_file = Path('empty_tools.txt')
        empty_file.write_text('\n'.join(empty_tools) + '\n')


async def rebuild_processor_index() -> None:
    """重建处理器索引."""
    from langchain_core.documents import Document
    from langchain_milvus import Milvus
    from tqdm import tqdm

    from app.core.config import get_settings
    from llm.models import qwen3_embedding
    from llm.utils.node_schema import PROCESSORS

    embedding = qwen3_embedding()
    vectorstore = Milvus(
        collection_name='ProcessorCollection',
        connection_args={'uri': get_settings().knowledge_base.MILVUS_URL},
        embedding_function=embedding,
        index_params={'index_type': 'HNSW', 'metric_type': 'L2'},
        consistency_level='Strong',
        drop_old=True,
        auto_id=True,
    )

    for process_name in tqdm(PROCESSORS, total=len(PROCESSORS)):
        description = PROCESSORS[process_name]['description']
        docs = [
            Document(
                page_content=description,
                metadata={
                    'name': process_name,
                },
            ),
        ]
        await vectorstore.aadd_documents(docs)


@click.group()
def cli() -> None:
    """数据库初始化和管理工具."""


@cli.command()
@click.option('--drop-old', is_flag=True, default=False, help='删除旧数据 (默认: 不删除)')
@click.option(
    '--databases',
    '-d',
    type=click.Choice(['all', 'image', 'tool', 'doc']),
    default='all',
    show_default=True,
    help='指定要初始化的数据库',
)
def init_db_cmd(*, drop_old: bool, databases: Literal['all', 'image', 'tool', 'doc']) -> None:
    """初始化数据库数据."""

    async def _run() -> None:
        if databases in ('all', 'image'):
            await init_images(drop_old=drop_old)

        if databases in ('all', 'doc'):
            await init_docs(drop_old=drop_old)

        if databases in ('all', 'tool'):
            await init_tools(drop_old=drop_old)

    asyncio.run(_run())


@cli.command()
@click.option(
    '--collections',
    '-c',
    type=click.Choice(['all', 'tool', 'processor', 'image']),
    default='all',
    show_default=True,
    help='指定要重建的向量索引集合',
)
def rebuild_index_cmd(collections: Literal['all', 'tool', 'processor', 'image']) -> None:
    """重建工具向量存储."""

    async def _run() -> None:
        if collections in ('all', 'image'):
            logger.info('Rebuild image index...')
            await rebuild_image_index()

        if collections in ('all', 'tool'):
            logger.info('Rebuild tool index...')
            await rebuild_tool_index()

        if collections in ('all', 'processor'):
            logger.info('Rebuild processor index...')
            await rebuild_processor_index()

    asyncio.run(_run())


if __name__ == '__main__':
    cli()
