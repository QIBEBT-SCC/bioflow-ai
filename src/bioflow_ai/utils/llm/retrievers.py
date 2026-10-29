from typing import cast
from uuid import UUID

from langchain_core.documents import Document
from sqlalchemy.orm import Session

import bioflow_ai.crud.tools.tool as tool_crud
from bioflow_ai.core.config import get_settings
from bioflow_ai.infrastructure.database import engine
from bioflow_ai.llm.models import qwen3_embedding
from bioflow_ai.llm.vectorstores import MilvusVectorStore
from bioflow_ai.schemas.tools import ToolNodePublic
from bioflow_ai.services.code.catalog import AUXILIARY_NODE_COLLECTION
from bioflow_ai.utils.uid_mapper import BidirectionalMapper, HasUid


async def retrieve_tools(
    query: str, image_uid: UUID | None = None
) -> tuple[list[ToolNodePublic], BidirectionalMapper | None]:
    """Search existing tools from the vector database based on functionality."""
    embedding = qwen3_embedding()
    vectorstore = MilvusVectorStore(
        collection_name='ToolCollection',
        embedding_function=embedding,
        uri=get_settings().knowledge_base.MILVUS_URL,
        hybrid=True,
    )

    task = 'Provide a search query to search for tools with relevant functions'
    query = f'Instruct: {task}\nQuery:{query}'

    async with vectorstore:
        results = await vectorstore.asimilarity_search(
            query,
            k=10,
            filter_expression=f'image_uid == "{image_uid!s}"' if image_uid else '',
            weights=(0.85, 0.15),
        )

    if not results:
        return [], None

    with Session(engine) as session:
        exist_tools = list(tool_crud.get_all_by_uid(session, [t.metadata['tool_uid'] for t in results]))
        mappers = BidirectionalMapper(datas=cast('list[HasUid]', exist_tools))

    tool_nodes = [ToolNodePublic.model_validate(exist_tool) for exist_tool in exist_tools]

    return tool_nodes, mappers


async def retrieve_auxiliary_nodes(query: str) -> list[Document]:
    """Search reusable CODE definitions and static processors by function."""
    embedding = qwen3_embedding()
    vectorstore = MilvusVectorStore(
        collection_name=AUXILIARY_NODE_COLLECTION,
        embedding_function=embedding,
        uri=get_settings().knowledge_base.MILVUS_URL,
    )

    task = 'Provide a search query to search for reusable code or data processors with relevant functions'
    query = f'Instruct: {task}\nQuery:{query}'

    async with vectorstore:
        return await vectorstore.amax_marginal_relevance_search(query, k=10)
