from collections.abc import Callable
from dataclasses import dataclass

from anyio import Path

from app.utils.init import tui

PREBUILD_REPO = 'Aye10032/bioflow-ai-prebuild'
BATCH_SIZE = 50


@dataclass(frozen=True)
class AdminInitResult:
    """Default administrator initialization result."""

    created: bool


@dataclass(frozen=True)
class PresetFileResult:
    """Preset parquet file resolution result."""

    path: str
    downloaded: bool


@dataclass(frozen=True)
class PresetImportResult:
    """Preset SQL/vector import result."""

    name: str
    preset_file: str
    downloaded: bool
    inserted_count: int
    deleted_count: int | None = None
    validation_error_count: int = 0
    warning_count: int = 0


@dataclass(frozen=True)
class PresetDatabaseInitResult:
    """Preset database initialization result."""

    admin: AdminInitResult
    images: PresetImportResult
    docs: PresetImportResult
    tools: PresetImportResult


async def init_preset_database_records(
    *,
    drop_old: bool = True,
    show_progress: bool = True,
) -> PresetDatabaseInitResult:
    """Initialize preset database records in dependency order."""
    return PresetDatabaseInitResult(
        admin=await init_admin(),
        images=await init_images(drop_old=drop_old, show_progress=show_progress),
        docs=await init_docs(drop_old=drop_old),
        tools=await init_tools(drop_old=drop_old, show_progress=show_progress),
    )


async def init_admin() -> AdminInitResult:
    """当 User 表为空时, 创建默认管理员账户."""
    from sqlmodel import Session, select

    import app.crud.user as user_crud
    from app.core.session import engine
    from app.models.user import User, UserRole
    from app.utils.security import get_password_hash

    with Session(engine) as session:
        existing = session.exec(select(User)).first()
        if existing is not None:
            return AdminInitResult(created=False)

        admin = User(
            username='admin',
            email='admin@email.com',
            role=UserRole.ADMIN,
            hashed_password=get_password_hash('admin'),
            is_active=True,
        )
        user_crud.insert(session, admin)
        return AdminInitResult(created=True)


async def _ensure_preset_file(filename: str) -> PresetFileResult:
    """Ensure a preset parquet file exists locally."""
    parquet_file = f'presets/{filename}'
    if await Path(parquet_file).exists():
        return PresetFileResult(path=parquet_file, downloaded=False)

    from huggingface_hub import download_bucket_files

    download_bucket_files(
        PREBUILD_REPO,
        files=[(filename, parquet_file)],
    )
    return PresetFileResult(path=parquet_file, downloaded=True)


async def init_images(*, drop_old: bool = True, show_progress: bool = True) -> PresetImportResult:
    """初始化镜像数据库和镜像向量索引."""
    import orjson
    import polars as pl
    from langchain_core.documents import Document
    from pydantic import ValidationError
    from sqlmodel import Session
    from tqdm import tqdm

    import app.crud.tools.image as image_crud
    from app.core.config import get_settings
    from app.core.session import engine
    from app.models.tools import ToolImage
    from llm.models import qwen3_embedding
    from llm.vectorstores import MilvusVectorStore

    preset_file = await _ensure_preset_file('images.parquet')
    images_df = pl.read_parquet(preset_file.path)
    images: list[ToolImage] = []
    validation_error_count = 0

    for row in images_df.iter_rows(named=True):
        row['image'] = orjson.loads(row['image'])

        try:
            images.append(ToolImage.model_validate(row))
        except ValidationError:
            validation_error_count += 1

    embedding = qwen3_embedding()
    vectorstore = MilvusVectorStore(
        embedding_function=embedding,
        collection_name='ImageCollection',
        uri=get_settings().knowledge_base.MILVUS_URL,
        drop_old=drop_old,
    )

    delete_count = None
    with Session(engine) as session:
        if drop_old:
            delete_count = image_crud.drop_all(session)

        images = image_crud.insert_all(session, images)

    for i in tqdm(range(0, len(images), BATCH_SIZE), disable=not show_progress):
        batch_images = images[i : i + BATCH_SIZE]

        docs = [
            Document(page_content=img.description, metadata={'image_uid': str(img.uid), 'name': img.name})
            for img in batch_images
        ]

        async with vectorstore:
            await vectorstore.aadd_documents(docs)

    return PresetImportResult(
        name='images',
        preset_file=preset_file.path,
        downloaded=preset_file.downloaded,
        inserted_count=len(images),
        deleted_count=delete_count,
        validation_error_count=validation_error_count,
    )


async def init_docs(*, drop_old: bool = True) -> PresetImportResult:
    """初始化工具文档库."""
    import polars as pl
    from pydantic import ValidationError
    from sqlmodel import Session

    import app.crud.tools.tool_doc as doc_crud
    from app.core.session import engine
    from app.models.tools import ToolHelpDoc

    preset_file = await _ensure_preset_file('tool_docs.parquet')
    docs_df = pl.read_parquet(preset_file.path)
    docs = []
    validation_error_count = 0

    for row in docs_df.iter_rows(named=True):
        try:
            docs.append(ToolHelpDoc.model_validate(row))
        except ValidationError:
            validation_error_count += 1

    delete_count = None
    with Session(engine) as session:
        if drop_old:
            delete_count = doc_crud.drop_all(session)

        doc_crud.insert_all(session, docs)

    return PresetImportResult(
        name='docs',
        preset_file=preset_file.path,
        downloaded=preset_file.downloaded,
        inserted_count=len(docs),
        deleted_count=delete_count,
        validation_error_count=validation_error_count,
    )


async def init_tools(*, drop_old: bool = True, show_progress: bool = True) -> PresetImportResult:
    """初始化工具数据库和工具向量索引."""
    import orjson
    import polars as pl
    from langchain_core.documents import Document
    from pydantic import ValidationError
    from sqlmodel import Session
    from tqdm import tqdm

    import app.crud.tools.tool as tool_crud
    import app.crud.tools.tool_tag as tag_crud
    from app.core.config import get_settings
    from app.core.session import engine
    from app.models.tools import ToolDefine
    from llm.models import qwen3_embedding
    from llm.vectorstores import MilvusVectorStore

    preset_file = await _ensure_preset_file('tools.parquet')
    tools_df = pl.read_parquet(preset_file.path)
    tools = []
    tag_ids_per_tool: list[list[int]] = []
    validation_error_count = 0

    for row in tools_df.iter_rows(named=True):
        for field in ('dynamic_params', 'file_mounts'):
            if row[field]:
                row[field] = orjson.loads(row[field])
            else:
                row[field] = []

        raw_tags = orjson.loads(row['tags']) if row['tags'] else []
        tag_ids = [t['id'] for t in raw_tags if t.get('id') is not None]
        row['tags'] = []

        try:
            tools.append(ToolDefine.model_validate(row))
            tag_ids_per_tool.append(tag_ids)
        except ValidationError:
            validation_error_count += 1

    embedding = qwen3_embedding()
    vectorstore = MilvusVectorStore(
        collection_name='ToolCollection',
        embedding_function=embedding,
        uri=get_settings().knowledge_base.MILVUS_URL,
        hybrid=True,
        drop_old=drop_old,
    )

    delete_count = None
    warning_count = 0
    with Session(engine) as session:
        if drop_old:
            delete_count = tool_crud.drop_all(session)

        unique_tag_ids = {tid for ids in tag_ids_per_tool for tid in ids}
        if unique_tag_ids:
            existing_tags = tag_crud.get_all_by_id(session, list(unique_tag_ids))
            tag_map = {t.id: t for t in existing_tags}
            missing = unique_tag_ids - tag_map.keys()
            if missing:
                warning_count += 1

            for tool, ids in zip(tools, tag_ids_per_tool, strict=False):
                tool.tags = [tag_map[i] for i in ids if i in tag_map]

        tools = tool_crud.insert_all(session, tools)

        for i in tqdm(range(0, len(tools), BATCH_SIZE), disable=not show_progress):
            batch_tools = tools[i : i + BATCH_SIZE]

            docs = [
                Document(
                    page_content=tool.description,
                    # pyrefly: ignore [missing-attribute]
                    metadata={'tool_uid': str(tool.uid), 'image_uid': str(tool.image.uid), 'name': tool.name},
                )
                for tool in batch_tools
            ]

            async with vectorstore:
                await vectorstore.aadd_documents(docs)

    return PresetImportResult(
        name='tools',
        preset_file=preset_file.path,
        downloaded=preset_file.downloaded,
        inserted_count=len(tools),
        deleted_count=delete_count,
        validation_error_count=validation_error_count,
        warning_count=warning_count,
    )


async def rebuild_processor_index(*, show_progress: bool = True) -> None:
    """重建处理器索引."""
    from langchain_core.documents import Document
    from tqdm import tqdm

    from app.core.config import get_settings
    from llm.models import qwen3_embedding
    from llm.utils.node_schema import PROCESSORS
    from llm.vectorstores import MilvusVectorStore

    embedding = qwen3_embedding()
    vectorstore = MilvusVectorStore(
        collection_name='ProcessorCollection',
        embedding_function=embedding,
        uri=get_settings().knowledge_base.MILVUS_URL,
        drop_old=True,
    )

    for process_name in tqdm(PROCESSORS, total=len(PROCESSORS), disable=not show_progress):
        description = PROCESSORS[process_name]['description']
        docs = [
            Document(
                page_content=description,
                metadata={
                    'name': process_name,
                },
            ),
        ]
        async with vectorstore:
            await vectorstore.aadd_documents(docs)


async def initialize_preset_database(confirm: Callable[..., bool]) -> bool | None:
    """Prompt for and initialize preset SQL records and vector collections."""
    tui.render_section('系统预置数据', '初始化管理员账户、工具数据库以及工具/镜像/处理器向量索引。')
    if not confirm('是否初始化系统预置数据和向量索引?', default=True):
        tui.render_notice('已跳过预置数据初始化。', style='#94a3b8')
        return None

    try:
        from app.core.session import create_db_and_tables

        create_db_and_tables()
        result = await init_preset_database_records(drop_old=True, show_progress=False)
        await rebuild_processor_index(show_progress=False)
    except Exception as exc:
        tui.render_notice(f'预置数据初始化失败: {exc}', style='#ef4444')
        return False

    tui.render_preset_database_summary(result)
    return True
