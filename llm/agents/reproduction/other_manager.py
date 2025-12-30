from typing import Annotated, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import StateGraph
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from app.schemas.react_flow import NodeData
from app.schemas.tools import Handle
from llm.models import get_agent_model
from llm.states import BuildEdge, BuildNode, ReproductionState
from llm.tools.human import ask_human
from llm.utils.formatters import node_formatter
from llm.utils.uid_mapper import generate_char_index


class OtherManager:
    def __init__(self):
        self.name = 'other_manager'

        self.tools = [
            add_code_node,
            add_note_node,
            need_other_node,
            ask_human,
        ]

        self.llm = get_agent_model().bind_tools(self.tools, tool_choice='auto')

    def build(self):
        graph = StateGraph(state_schema=ReproductionState)

        compiled_graph = graph.compile(name=self.name)
        return compiled_graph


@tool
async def add_code_node(
    language: Literal['R', 'python'],
    pos_x: int,
    pos_y: int,
    program_request: str,
    input_edges: list[BuildEdge],
    state: Annotated[ReproductionState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Add a code node to the workflow

    Args:
        language: program language(R or python)
        pos_x: The x-coordinate of the node's layout on the panel, only positive integer
        pos_y: The y-coordinate of the node's layout on the panel
        program_request: What functionality needs to be implemented in the code. The description needs to be **accurate**, as subsequent work will directly base code writing on your description.
        input_edges: Code node only have one edge `files`, just need to connect the files generated during the preliminary analysis, which are required by the code here, to this handle.
    """
    node_count = len(state.exist_nodes)
    node_id = generate_char_index(node_count + 1)

    new_node = BuildNode(
        id=node_id,
        type=f'code_{language}',
        input_handles=[Handle(name='input_files', description='The files required by this code')],
        output_handles=[Handle(name='output_folder', description='The files required by this code')],
        data=NodeData(args=f'{{"description":"{program_request}","code":""}}'),
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
                ToolMessage(content=f'add new code node:\n{node_formatter(new_node)}', tool_call_id=tool_call_id)
            ],
        }
    )


@tool
async def add_note_node(
    note_content: str,
    pos_x: int,
    pos_y: int,
    input_edges: list[BuildEdge],
    state: Annotated[ReproductionState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Add a note node to the workflow

    Args:
        note_content: Specific content of the notes (as brief and precise as possible)
        pos_x: The x-coordinate of the node's layout on the panel, only positive integer
        pos_y: The y-coordinate of the node's layout on the panel
        input_edges: Code node only have one edge `files`, just need to connect the files generated during the preliminary analysis, which are required by the code here, to this handle.
    """
    node_count = len(state.exist_nodes)
    node_id = generate_char_index(node_count + 1)

    new_node = BuildNode(
        id=node_id, type='note', input_handles=[], output_handles=[], data=NodeData(args=note_content), x=pos_x, y=pos_y
    )

    for edge in input_edges:
        edge.target_node = node_id

    return Command(
        update={
            'exist_nodes': [new_node],
            'exist_edges': input_edges,
            'messages': [ToolMessage(content=f'add new note:\n{node_formatter(new_node)}', tool_call_id=tool_call_id)],
        }
    )


@tool
async def need_other_node(node_type: Literal['tool', 'resource', 'processor'], requirement: str):
    """

    Args:
        node_type:
        requirement:
    """

    return ''
