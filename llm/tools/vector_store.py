from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_milvus import Milvus
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field
from sqlmodel import Session

import app.crud.tools.tool as tool_crud
from app.core.config import get_settings
from app.core.session import engine
from app.models.tools import ToolDefine
from app.schemas.tools import ToolNodePublic
from llm.models import qwen3_embedding
from llm.utils.formatters import handle_formatter
from llm.utils.retrievers import retrieve_documents
from llm.utils.uid_mapper import BidirectionalMapper

TOOL_DESCRIPTION = """Search existing tools from the vector database based on functionality
**This tool prohibits parallel calls! You are only allowed to make one search at a time!**

## Return Results
The search results will include the tool node information encapsulated in the database. In addition to the name and description of the node's function, it also includes the following information:
- Command: The actual command-line instruction corresponding to this node, where some variables are templates to be filled.
- ImmutableParams: Parameters in the tool execution command that users **cannot modify**, which generally directly change the tool's working mode or output results.
- ModifiableParams: Users can modify or add parameters by editing this field, but they cannot conflict with `ImmutableParams`.
"""


class SearchToolArgsSchema(BaseModel):
    query: str = Field(description='Briefly describe the tool or function to be searched for')


@tool(description=TOOL_DESCRIPTION, args_schema=SearchToolArgsSchema)
async def search_tools_by_function(query: str, runtime: ToolRuntime):
    """Search existing tools from the vector database based on functionality"""

    embedding = qwen3_embedding()
    vectorstore = Milvus(
        collection_name='ToolCollection',
        connection_args={'uri': get_settings().knowledge_base.MILVUS_URL},
        embedding_function=embedding,
        search_params={'M': 64, 'efConstruction': 100},
        consistency_level='Strong',
    )

    task = 'Provide a search query to search for tools with relevant functions'
    query = embedding.get_detailed_instruct(task, query)
    image_uid = runtime.state.get('image_uid', None)
    if image_uid:
        results = await vectorstore.amax_marginal_relevance_search(query, k=10, expr=f'image_uid == "{str(image_uid)}"')
    else:
        results = await vectorstore.amax_marginal_relevance_search(query, k=10)

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
        exist_tools: list[ToolDefine] = tool_crud.get_all_by_uid(session, [t.metadata['tool_uid'] for t in results])
        mappers = BidirectionalMapper(datas=exist_tools)

    tool_nodes = [ToolNodePublic.model_validate(exist_tool.model_dump()) for exist_tool in exist_tools]

    tool_blocks = [
        (
            f'  <tool>\n'
            f'    <name>{t.name}</name>\n'
            f'    <description>{t.description}</description>\n'
            f'    <input_handles>\n'
            f'{handle_formatter(t.input_handles)}\n'
            f'    </input_handles>\n'
            f'    <output_handles>\n'
            f'{handle_formatter(t.output_handles)}\n'
            f'    </output_handles>\n'
            f'    <data>\n'
            f'      <tool_id>{mappers.get_index_by_uid(str(t.uid))}</tool_id>\n'
            f'      <args editable=False>{t.immutable_static_params}</args>\n'
            f'      <args editable=True>{t.modifiable_static_params}</args>\n'
            f'    </data>\n'
            f'    <command>{t.complete_command}</command>'
            f'  </tool>'
        )
        for t in tool_nodes
    ]
    return Command(
        update={
            'id_mapper': mappers,
            'messages': [
                ToolMessage(
                    content=f'<Tools>\n{"\n".join(tool_blocks)}\n</Tools>',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
async def get_paper_snippets(query: str, runtime: ToolRuntime) -> str:
    """Get relevant paper content from the vector database based on the query"""

    paper_id = runtime.state.get('paper_id', None)

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


@tool(parse_docstring=True)
async def search_processors(query: str) -> str:
    """Search for the processor node corresponding to the function

    Args:
        query: Briefly describe the processor or function to be searched for
    """

    embedding = qwen3_embedding()
    vectorstore = Milvus(
        collection_name='ProcessorCollection',
        connection_args={'uri': get_settings().knowledge_base.MILVUS_URL},
        embedding_function=embedding,
        search_params={'M': 64, 'efConstruction': 100},
        consistency_level='Strong',
    )

    task = 'Provide a search query to search for data processors with relevant functions'
    query = embedding.get_detailed_instruct(task, query)

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
