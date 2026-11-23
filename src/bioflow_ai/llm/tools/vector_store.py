from typing import cast

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field

from bioflow_ai.utils.formatters import search_nodes_formatter
from bioflow_ai.utils.llm.retrievers import retrieve_tools
from bioflow_ai.utils.uid_mapper import BidirectionalMapper, merge_mappers

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


@tool('search_tools', description=TOOL_DESCRIPTION, args_schema=SearchToolArgsSchema)
async def search_tools_by_function(query: str, runtime: ToolRuntime) -> Command:
    """Search existing tools from the vector database based on functionality."""
    tool_nodes, new_mapper = await retrieve_tools(query, image_uid=runtime.state.get('image_uid'))
    if new_mapper is None:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='Based on your search query, no corresponding results found',
                        tool_call_id=runtime.tool_call_id,
                        artifact={'result_count': 0},
                    )
                ],
            }
        )

    exist_mapper = cast('BidirectionalMapper', runtime.state.get('id_mapper'))
    merged_mapper = merge_mappers(exist_mapper, new_mapper)

    return Command(
        update={
            'id_mapper': merged_mapper,
            'messages': [
                ToolMessage(
                    content=search_nodes_formatter(tool_nodes, merged_mapper),
                    tool_call_id=runtime.tool_call_id,
                    artifact={'result_count': len(tool_nodes)},
                )
            ],
        }
    )
