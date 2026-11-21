from typing import TYPE_CHECKING

from anyio import Path
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from bioflow_ai.core.paths import PRESETS_DIR
from bioflow_ai.services.progress import ProgressIterator, with_progress

MAX_PASSWORD_BYTES = 128

PREBUILD_REPO = 'Aye10032/bioflow-ai-prebuild'
BATCH_SIZE = 50
TOOL_MEMORIES_PRESET = 'tool_memories.parquet'


class AdminInitResult(BaseModel):
    """Default administrator initialization result."""

    model_config = ConfigDict(frozen=True)

    created: bool


class PresetFileResult(BaseModel):
    """Preset parquet file resolution result."""

    model_config = ConfigDict(frozen=True)

    path: str
    downloaded: bool


class PresetImportResult(BaseModel):
    """Preset SQL/vector import result."""

    model_config = ConfigDict(frozen=True)

    name: str
    preset_file: str
    downloaded: bool
    inserted_count: int
    deleted_count: int | None = None
    validation_error_count: int = 0
    warning_count: int = 0


class PresetDatabaseInitResult(BaseModel):
    """Preset database initialization result."""

    model_config = ConfigDict(frozen=True)

    admin: AdminInitResult
    images: PresetImportResult
    docs: PresetImportResult
    tools: PresetImportResult
    tool_memories: PresetImportResult


async def init_preset_database_records(
    *,
    admin: AdminInitResult,
    drop_old: bool = True,
    progress: ProgressIterator | None = None,
) -> PresetDatabaseInitResult:
    """Initialize preset database records in dependency order."""
    # Resolve the dependent preset before any tool catalog truncation can cascade to memories.
    memory_preset = await _ensure_preset_file(TOOL_MEMORIES_PRESET)
    return PresetDatabaseInitResult(
        admin=admin,
        images=await init_images(drop_old=drop_old, progress=progress),
        docs=await init_docs(drop_old=drop_old),
        tools=await init_tools(drop_old=drop_old, progress=progress),
        tool_memories=await init_tool_memories(drop_old=drop_old, preset_file=memory_preset),
    )


async def _ensure_preset_file(filename: str) -> PresetFileResult:
    """Ensure a preset parquet file exists locally."""
    parquet_file = Path(PRESETS_DIR / filename)
    if await parquet_file.exists():
        return PresetFileResult(path=str(parquet_file), downloaded=False)

    from huggingface_hub import download_bucket_files

    await parquet_file.parent.mkdir(parents=True, exist_ok=True)
    download_bucket_files(
        PREBUILD_REPO,
        files=[(filename, str(parquet_file))],
    )
    return PresetFileResult(path=str(parquet_file), downloaded=True)


async def init_images(*, drop_old: bool = True, progress: ProgressIterator | None = None) -> PresetImportResult:
    """Initialize the image database and image vector index."""
    import orjson
    import polars as pl
    from langchain_core.documents import Document
    from pydantic import ValidationError

    import bioflow_ai.crud.tools.image as image_crud
    from bioflow_ai.core.config import get_settings
    from bioflow_ai.infrastructure.database import engine
    from bioflow_ai.llm.models import qwen3_embedding
    from bioflow_ai.llm.vectorstores import MilvusVectorStore
    from bioflow_ai.models.tools import ToolImage
    from bioflow_ai.schemas.tools import ToolImageImport

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

    for i in with_progress(range(0, len(images), BATCH_SIZE), progress):
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

    import bioflow_ai.crud.tools.tool_doc as doc_crud
    from bioflow_ai.infrastructure.database import engine
    from bioflow_ai.models.tools import ToolHelpDoc
    from bioflow_ai.schemas.tools import ToolHelpDocImport

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


async def init_tools(*, drop_old: bool = True, progress: ProgressIterator | None = None) -> PresetImportResult:
    """Initialize the tool database and tool vector index."""
    import orjson
    import polars as pl

    import bioflow_ai.crud.tools.tool as tool_crud
    import bioflow_ai.crud.tools.tool_tag as tag_crud
    from bioflow_ai.infrastructure.database import engine
    from bioflow_ai.models.tools import ToolDefine
    from bioflow_ai.schemas.tools import ToolDefinitionImport
    from bioflow_ai.services.tools.cache import invalidate_all_tool_definitions_sync
    from bioflow_ai.services.tools.index import index_tools
    from bioflow_ai.services.workflow.validation.passthrough import validate_file_mounts

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
            validate_file_mounts(tool_data.file_mounts, tool_data.tool_type)
            tool_kwargs = tool_data.model_dump(exclude={'uid'})
            tool_kwargs['dynamic_params'] = tool_data.dynamic_params
            tool_kwargs['file_mounts'] = tool_data.file_mounts
            if tool_data.uid is not None:
                tool_kwargs['uid'] = tool_data.uid
            tools.append(ToolDefine(**tool_kwargs))
            tag_ids_per_tool.append(tag_ids)
        except ValueError:
            validation_error_count += 1

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
        invalidate_all_tool_definitions_sync()
        await index_tools(tools, drop_old=drop_old, progress=progress)

    return PresetImportResult(
        name='tools',
        preset_file=preset_file.path,
        downloaded=preset_file.downloaded,
        inserted_count=len(tools),
        deleted_count=delete_count,
        validation_error_count=validation_error_count,
        warning_count=warning_count,
    )


async def init_tool_memories(
    *, drop_old: bool = True, preset_file: PresetFileResult | None = None
) -> PresetImportResult:
    """Restore shared memories after their referenced tools have been imported."""
    import polars as pl
    from pydantic import ValidationError
    from sqlalchemy import delete, func, select
    from sqlalchemy.dialects.postgresql import insert

    from bioflow_ai.infrastructure.database import engine
    from bioflow_ai.models.tools import ToolDefine, ToolMemory
    from bioflow_ai.schemas.node_memory import ToolMemoryImport

    if preset_file is None:
        preset_file = await _ensure_preset_file(TOOL_MEMORIES_PRESET)
    records = []
    validation_error_count = 0
    for row in pl.read_parquet(preset_file.path).iter_rows(named=True):
        try:
            records.append(ToolMemoryImport.model_validate(row))
        except ValidationError:
            validation_error_count += 1

    deleted_count = None
    inserted_count = 0
    warning_count = 0
    with Session(engine) as session, session.begin():
        if drop_old:
            deleted_count = session.scalar(select(func.count()).select_from(ToolMemory)) or 0
            session.execute(delete(ToolMemory))

        for offset in range(0, len(records), BATCH_SIZE):
            batch = records[offset : offset + BATCH_SIZE]
            tool_uids = {record.tool_uid for record in batch}
            existing_tools = set(session.scalars(select(ToolDefine.uid).where(ToolDefine.uid.in_(tool_uids))))
            values = [record.model_dump() for record in batch if record.tool_uid in existing_tools]
            warning_count += len(batch) - len(values)
            if values:
                stmt = insert(ToolMemory).on_conflict_do_nothing(index_elements=[ToolMemory.uid])
                inserted_count += len(session.scalars(stmt.returning(ToolMemory.uid), values).all())

    return PresetImportResult(
        name='tool_memories',
        preset_file=preset_file.path,
        downloaded=preset_file.downloaded,
        inserted_count=inserted_count,
        deleted_count=deleted_count,
        validation_error_count=validation_error_count,
        warning_count=warning_count,
    )


async def rebuild_auxiliary_node_index(*, progress: ProgressIterator | None = None) -> None:
    """Rebuild the shared static-processor and reusable-CODE index."""
    from langchain_core.documents import Document
    from sqlalchemy import select

    from bioflow_ai.core.config import get_settings
    from bioflow_ai.infrastructure.database import engine
    from bioflow_ai.llm.models import qwen3_embedding
    from bioflow_ai.llm.vectorstores import MilvusVectorStore
    from bioflow_ai.models.code import CodeDefine
    from bioflow_ai.schemas.workflow_nodes import PROCESSORS
    from bioflow_ai.services.code.catalog import AUXILIARY_NODE_COLLECTION, PROCESSOR_NODE_KIND, code_index_document

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
    for index in with_progress(range(0, len(documents), BATCH_SIZE), progress):
        async with vectorstore:
            await vectorstore.aadd_documents(documents[index : index + BATCH_SIZE])


def administrator_exists() -> bool:
    """Return whether initialization already has an administrator record."""
    from sqlalchemy import select

    from bioflow_ai.infrastructure.database import engine
    from bioflow_ai.models.user import User

    with Session(engine) as session:
        return session.scalar(select(User)) is not None


async def init_admin(data: 'UserCreate | None' = None) -> AdminInitResult:
    """Create the initial administrator from explicitly supplied credentials."""
    from sqlalchemy import select

    from bioflow_ai.infrastructure.database import engine
    from bioflow_ai.models.user import User
    from bioflow_ai.schemas.user_config import UserRole
    from bioflow_ai.services.identity import user as user_service

    with Session(engine) as session:
        if session.scalar(select(User)) is not None:
            return AdminInitResult(created=False)
        if data is None:
            message = 'Administrator credentials are required for an empty user table.'
            raise ValueError(message)
        user_service.create_user(session, data, role=UserRole.ADMIN)
        return AdminInitResult(created=True)


if TYPE_CHECKING:
    from bioflow_ai.schemas.user import UserCreate
