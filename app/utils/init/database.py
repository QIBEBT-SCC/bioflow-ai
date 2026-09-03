from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from anyio import Path
from sqlalchemy.orm import Session

from app.utils.init import tui

MAX_PASSWORD_BYTES = 128

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
    """Prompt for and create an administrator when the user table is empty."""
    from pydantic import SecretStr
    from sqlalchemy import select

    from app.core.session import engine
    from app.models.user import User
    from app.schemas.user import UserCreate
    from app.schemas.user_config import UserRole
    from app.services import user as user_service

    with Session(engine) as session:
        existing = session.scalar(select(User))
        if existing is not None:
            return AdminInitResult(created=False)

        def required(value: str) -> bool | str:
            return True if value.strip() else 'This field is required.'

        email = (await tui.prompt_text_async('Administrator email', validate=required)).strip().casefold()
        username = (await tui.prompt_text_async('Administrator username', default='admin', validate=required)).strip()
        password = await tui.prompt_password_async(
            'Administrator password (any strength, required, maximum 128 UTF-8 bytes)',
            validate=lambda value: (
                required(value)
                if len(value.encode()) <= MAX_PASSWORD_BYTES
                else 'The password cannot exceed 128 UTF-8 bytes.'
            ),
        )
        user_service.create_user(
            session,
            UserCreate(username=username, email=email, password=SecretStr(password)),
            role=UserRole.ADMIN,
        )
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
    """Initialize the image database and image vector index."""
    import orjson
    import polars as pl
    from langchain_core.documents import Document
    from pydantic import ValidationError
    from tqdm import tqdm

    import app.crud.tools.image as image_crud
    from app.core.config import get_settings
    from app.core.session import engine
    from app.models.tools import ToolImage
    from app.schemas.tools import ToolImageImport
    from llm.models import qwen3_embedding
    from llm.vectorstores import MilvusVectorStore

    preset_file = await _ensure_preset_file('images.parquet')
    images_df = pl.read_parquet(preset_file.path)
    images: list[ToolImage] = []
    validation_error_count = 0

    for row in images_df.iter_rows(named=True):
        row['image'] = orjson.loads(row['image'])

        try:
            image_data = ToolImageImport.model_validate(row)
            image_kwargs = image_data.model_dump(exclude={'uid'})
            image_kwargs['image'] = image_data.image
            if image_data.uid is not None:
                image_kwargs['uid'] = image_data.uid
            images.append(ToolImage(**image_kwargs))
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
        session.commit()
        for image in images:
            session.refresh(image)

    for i in tqdm(range(0, len(images), BATCH_SIZE), disable=not show_progress):
        batch_images = images[i : i + BATCH_SIZE]

        docs = [
            Document(page_content=img.description, metadata={'image_uid': str(img.uid), 'name': img.name})
            for img in batch_images
            if img.description
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
    """Initialize the tool documentation database."""
    import polars as pl
    from pydantic import ValidationError

    import app.crud.tools.tool_doc as doc_crud
    from app.core.session import engine
    from app.models.tools import ToolHelpDoc
    from app.schemas.tools import ToolHelpDocImport

    preset_file = await _ensure_preset_file('tool_docs.parquet')
    docs_df = pl.read_parquet(preset_file.path)
    docs = []
    validation_error_count = 0

    for row in docs_df.iter_rows(named=True):
        try:
            doc_data = ToolHelpDocImport.model_validate(row)
            doc_kwargs = doc_data.model_dump(exclude={'uid'})
            if doc_data.uid is not None:
                doc_kwargs['uid'] = doc_data.uid
            docs.append(ToolHelpDoc(**doc_kwargs))
        except ValidationError:
            validation_error_count += 1

    delete_count = None
    with Session(engine) as session:
        if drop_old:
            delete_count = doc_crud.drop_all(session)

        doc_crud.insert_all(session, docs)
        session.commit()

    return PresetImportResult(
        name='docs',
        preset_file=preset_file.path,
        downloaded=preset_file.downloaded,
        inserted_count=len(docs),
        deleted_count=delete_count,
        validation_error_count=validation_error_count,
    )


async def init_tools(*, drop_old: bool = True, show_progress: bool = True) -> PresetImportResult:
    """Initialize the tool database and tool vector index."""
    import orjson
    import polars as pl
    from langchain_core.documents import Document
    from pydantic import ValidationError
    from tqdm import tqdm

    import app.crud.tools.tool as tool_crud
    import app.crud.tools.tool_tag as tag_crud
    from app.core.config import get_settings
    from app.core.session import engine
    from app.models.tools import ToolDefine
    from app.schemas.tools import ToolDefinitionImport
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
            tool_data = ToolDefinitionImport.model_validate(row)
            tool_kwargs = tool_data.model_dump(exclude={'uid'})
            tool_kwargs['dynamic_params'] = tool_data.dynamic_params
            tool_kwargs['file_mounts'] = tool_data.file_mounts
            if tool_data.uid is not None:
                tool_kwargs['uid'] = tool_data.uid
            tools.append(ToolDefine(**tool_kwargs))
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
        session.commit()

        for i in tqdm(range(0, len(tools), BATCH_SIZE), disable=not show_progress):
            batch_tools = tools[i : i + BATCH_SIZE]

            docs = [
                Document(
                    page_content=tool.description,
                    # pyrefly: ignore [missing-attribute]
                    metadata={'tool_uid': str(tool.uid), 'image_uid': str(tool.image.uid), 'name': tool.name},
                )
                for tool in batch_tools
                if tool.description
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


async def rebuild_auxiliary_node_index(*, show_progress: bool = True) -> None:
    """Rebuild the shared static-processor and reusable-CODE index."""
    from langchain_core.documents import Document
    from sqlalchemy import select
    from tqdm import tqdm

    from app.core.config import get_settings
    from app.core.session import engine
    from app.models.code import CodeDefine
    from app.services.code_catalog import (
        AUXILIARY_NODE_COLLECTION,
        PROCESSOR_NODE_KIND,
        code_index_document,
    )
    from llm.models import qwen3_embedding
    from llm.utils.node_schema import PROCESSORS
    from llm.vectorstores import MilvusVectorStore

    embedding = qwen3_embedding()
    vectorstore = MilvusVectorStore(
        collection_name=AUXILIARY_NODE_COLLECTION,
        embedding_function=embedding,
        uri=get_settings().knowledge_base.MILVUS_URL,
        drop_old=True,
    )

    processor_docs = [
        Document(
            page_content=PROCESSORS[processor_name]['description'],
            metadata={
                'kind': PROCESSOR_NODE_KIND,
                'name': processor_name,
                'node_type': processor_name,
            },
        )
        for processor_name in PROCESSORS
    ]

    with Session(engine) as session:
        codes = session.scalars(select(CodeDefine).order_by(CodeDefine.name)).all()

    documents = [*processor_docs, *(code_index_document(code) for code in codes)]
    for index in tqdm(range(0, len(documents), BATCH_SIZE), disable=not show_progress):
        async with vectorstore:
            await vectorstore.aadd_documents(documents[index : index + BATCH_SIZE])


async def initialize_preset_database(confirm: Callable[..., Awaitable[bool]]) -> bool | None:
    """Prompt for and initialize preset SQL records and vector collections."""
    tui.render_section(
        'System Preset Data',
        'Initialize the administrator account, tool database, and tool/image/auxiliary-node vector indexes.',
    )
    if not await confirm('Initialize system preset data and vector indexes?', default=True):
        tui.render_notice('Preset data initialization skipped.', style='#94a3b8')
        return None

    try:
        from app.core.session import create_db_and_tables

        create_db_and_tables()
        result = await init_preset_database_records(drop_old=True, show_progress=False)
        await rebuild_auxiliary_node_index(show_progress=False)
    except Exception as exc:
        tui.render_notice(f'Preset data initialization failed: {exc}', style='#ef4444')
        return False

    tui.render_preset_database_summary(result)
    return True
