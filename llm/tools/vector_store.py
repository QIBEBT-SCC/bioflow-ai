from typing import cast

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field
from sqlmodel import Session

import app.crud.tools.tool as tool_crud
from app.core.config import get_settings
from app.core.session import engine
from app.schemas.tools import ToolNodePublic
from llm.models import qwen3_embedding
from llm.utils.formatters import search_nodes_formatter
from llm.utils.retrievers import retrieve_documents
from llm.utils.uid_mapper import BidirectionalMapper, merge_mappers
from llm.vectorstores import MilvusVectorStore

TOOL_DESCRIPTION = """Search existing tools from the vector database based on functionality
**This tool prohibits parallel calls! You are only allowed to make one search at a time!**

## Return Results
The search results will include the tool node information encapsulated in the database. In addition to the name and description of the node's function, it also includes the following information:
- Command: The actual command-line instruction corresponding to this node, where some variables are templates to be filled.
- ImmutableParams: Parameters in the tool execution command that users **cannot modify**, which generally directly change the tool's working mode or output results.
- ModifiableParams: Users can modify or add parameters by editing this field, but they cannot conflict with `ImmutableParams`.
"""


class SearchToolArgsSchema(BaseModel):
    query: str = Field(description='Briefly describe the tool or function to be searched for in ENGLISH')


@tool(description=TOOL_DESCRIPTION, args_schema=SearchToolArgsSchema)
async def search_tools_by_function(query: str, runtime: ToolRuntime) -> Command:
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
    image_uid = runtime.state.get('image_uid', None)

    async with vectorstore:
        results = await vectorstore.asimilarity_search(
            query,
            k=10,
            filter_expression=f'image_uid == "{image_uid!s}"' if image_uid else '',
            weights=(0.85, 0.15),
        )

    if not results:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='Based on your search query, no corresponding results found',
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    with Session(engine) as session:
        exist_tools = list(tool_crud.get_all_by_uid(session, [t.metadata['tool_uid'] for t in results]))
        new_mapper = BidirectionalMapper(datas=exist_tools)

    tool_nodes = [ToolNodePublic.model_validate(exist_tool) for exist_tool in exist_tools]
    exist_mapper = cast('BidirectionalMapper', runtime.state.get('id_mapper'))
    merged_mapper = merge_mappers(exist_mapper, new_mapper)

    return Command(
        update={
            'id_mapper': merged_mapper,
            'messages': [
                ToolMessage(
                    content=search_nodes_formatter(tool_nodes, merged_mapper),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
async def get_paper_snippets(query: str, runtime: ToolRuntime) -> str:
    """Get relevant paper content from the vector database based on the query."""
    if paper_id := runtime.state.get('paper_id'):
        documents = await retrieve_documents(query, doc_id=paper_id, k=5)

        if not documents:
            return 'Based on your query, no relevant paper content was found.'

        # 格式化返回的父文档内容
        content_blocks = '\n\n'.join(
            [
                f'<Section title="{doc.metadata.get("Title", "")}" subsection="{doc.metadata.get("Section", "")}">\n'
                f'{doc.page_content}\n'
                f'</Section>'
                for doc in documents
            ]
        )
        return f'Relevant paper content based on your query:\n{content_blocks}'

    return 'In the current environment, there are no references; please make decisions based on your own judgment.'


@tool(parse_docstring=True)
async def search_processors(query: str) -> str:
    """Search for the processor node corresponding to the function.

    Args:
        query: Briefly describe the processor or function to be searched for
    """
    embedding = qwen3_embedding()
    vectorstore = MilvusVectorStore(
        collection_name='ProcessorCollection',
        embedding_function=embedding,
        uri=get_settings().knowledge_base.MILVUS_URL,
    )

    task = 'Provide a search query to search for data processors with relevant functions'
    query = f'Instruct: {task}\nQuery:{query}'

    async with vectorstore:
        results = await vectorstore.amax_marginal_relevance_search(query, k=10)

    if not results:
        return 'Based on your search query, no corresponding results found'

    processors_content = '\n'.join(
        [
            f'  <Processor>\n'
            f'    <Name>{t.metadata["name"]}</Name>\n'
            f'    <Description>{t.page_content}</Description>\n'
            f'  </Processor>'
            for t in results
        ]
    )
    return f'<Processors>\n{processors_content}\n</Processors>'
