from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_milvus import Milvus
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from sqlmodel import Session

import app.crud.tools.tool as tool_crud
from app.core.config import get_settings
from app.core.session import engine
from app.models.tools import ToolDefine
from app.schemas.tools import ToolNodePublic
from llm.models import qwen3_embedding
from llm.utils.uid_mapper import BidirectionalMapper


@tool(parse_docstring=True)
async def search_tools_by_function(query: str, runtime: ToolRuntime):
    """Search existing tools from the vector database based on functionality
    **This tool prohibits parallel calls! You are only allowed to make one search at a time!**

    Args:
        query: Functionality of the tool to be searched
    """

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
                        content='No tool was found locally that satisfies this function, it seems a new one needs to be created.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    with Session(engine) as session:
        exist_tools: list[ToolDefine] = tool_crud.get_all_by_uid(session, [t.metadata['tool_uid'] for t in results])
        mappers = BidirectionalMapper(datas=exist_tools)

    tool_nodes = [ToolNodePublic.model_validate(exist_tool.model_dump()) for exist_tool in exist_tools]

    tools_content = '\n'.join(
        [
            f'  <Tool>\n'
            f'    <Id>{mappers.get_index_by_uid(str(t.uid))}</Id>\n'
            f'    <Name>{t.name}</Name>\n'
            f'    <Description>{t.description}</Description>\n'
            f'    <Command>{t.complete_command}</Command>\n'
            f'    <Args>{t.modifiable_static_params}</Args>\n'
            f'  </Tool>'
            for t in tool_nodes
        ]
    )
    return Command(
        update={
            'id_mapper': mappers,
            'messages': [
                ToolMessage(
                    content=f'<Tools>\n{tools_content}\n</Tools>',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
