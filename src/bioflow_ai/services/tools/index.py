"""Vector index projection of the tool catalog."""

from collections.abc import Sequence

from langchain_core.documents import Document

from bioflow_ai.core.config import get_settings
from bioflow_ai.llm.models import qwen3_embedding
from bioflow_ai.llm.vectorstores import MilvusVectorStore
from bioflow_ai.models.tools import ToolDefine
from bioflow_ai.services.progress import ProgressIterator, with_progress
from bioflow_ai.utils.identifiers import UUIDLike

TOOL_COLLECTION = 'ToolCollection'
_INDEX_BATCH_SIZE = 50


def tool_vectorstore(*, drop_old: bool = False) -> MilvusVectorStore:
    """Return the hybrid vector store holding tool description projections."""
    return MilvusVectorStore(
        collection_name=TOOL_COLLECTION,
        embedding_function=qwen3_embedding(),
        uri=get_settings().knowledge_base.MILVUS_URL,
        hybrid=True,
        drop_old=drop_old,
    )


def _tool_index_document(tool: ToolDefine) -> Document:
    """Build the vector document used to discover a tool by function."""
    return Document(
        page_content=tool.description,
        metadata={
            'image_uid': str(tool.image_uid),
            'tool_uid': str(tool.uid),
            'name': tool.name,
        },
    )


async def index_tools(
    tools: Sequence[ToolDefine],
    *,
    drop_old: bool = False,
    progress: ProgressIterator | None = None,
) -> list[ToolDefine]:
    """Add tool descriptions to the vector index and return the tools skipped for lacking a description."""
    indexable = [tool for tool in tools if tool.description]
    vectorstore = tool_vectorstore(drop_old=drop_old)
    async with vectorstore:
        for start in with_progress(range(0, len(indexable), _INDEX_BATCH_SIZE), progress):
            batch = indexable[start : start + _INDEX_BATCH_SIZE]
            await vectorstore.aadd_documents([_tool_index_document(tool) for tool in batch])
    return [tool for tool in tools if not tool.description]


async def remove_tool_index(tool_uid: UUIDLike) -> None:
    """Remove a tool's description projection from the vector index."""
    await tool_vectorstore().adelete(
        filter_expression='tool_uid == {uid}',
        filter_params={'uid': str(tool_uid)},
    )


async def refresh_tool_index(tool: ToolDefine) -> None:
    """Replace a tool's description projection after its description changes."""
    await remove_tool_index(tool.uid)
    await index_tools([tool])
