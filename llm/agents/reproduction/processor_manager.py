from typing import Annotated, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import StateGraph
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from loguru import logger

from app.schemas.react_flow import NodeData
from app.schemas.tools import Handle
from llm.models import gpt_5_mini
from llm.states import BuildEdge, BuildNode, ReproductionState
from llm.tools.human import ask_human
from llm.utils.formatters import node_formatter
from llm.utils.uid_mapper import generate_char_index


class ProcessorManager:
    def __init__(self):
        self.name = 'other_manager'

        self.tools = [
            search_processor,
            add_data_process_node,
            need_other_node,
            ask_human,
        ]

        self.llm = gpt_5_mini(temperature=0.2).bind_tools(self.tools, tool_choice='auto')

    def build(self):
        graph = StateGraph(state_schema=ReproductionState)

        compiled_graph = graph.compile(name=self.name)
        return compiled_graph



@tool
async def search_processor(query: str):
    """Find the data_process node for the relevant function

    Args:
        query: Briefly describe the function you need to search for nodes
    """
    logger.debug(query)
    return (
        'Based on your needs, no suitable data_process node was found. '
        'You can consider changing your search query or using the `ask_human` tool to report this situation to a human.'
    )


@tool
async def add_data_process_node(
    processor_id: int,
    pos_x: int,
    pos_y: int,
    input_edges: list[BuildEdge],
    state: Annotated[ReproductionState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Add a data_process node to the workflow

    Args:
        processor_id: program language(R or python)
        pos_x: The x-coordinate of the node's layout on the panel, only positive integer
        pos_y: The y-coordinate of the node's layout on the panel
        input_edges: Code node only have one edge `files`, just need to connect the files generated during the preliminary analysis, which are required by the code here, to this handle.
    """
    node_count = len(state.exist_nodes)
    node_id = generate_char_index(node_count + 1)

    # TODO: 这里实际应该是处理器名称
    new_node = BuildNode(
        id=node_id,
        type=f'processor_{processor_id}',
        input_handles=[Handle(name='files', description='The files required by this code')],
        output_handles=[],
        data=NodeData(args='', resource_uid=''),
        x=pos_x,
        y=pos_y,
    )

    for edge in input_edges:
        edge.target_node = node_id

    return Command(
        update={
            'exist_nodes': [new_node],
            'exist_edges': input_edges,
            'messages': [
                ToolMessage(
                    content=f'add new data_process node:\n{node_formatter(new_node)}', tool_call_id=tool_call_id
                )
            ],
        }
    )

@tool
async def need_other_node(node_type: Literal['tool', 'resource', 'code', 'note'], requirement: str):
    """

    Args:
        node_type:
        requirement:
    """

    return ''