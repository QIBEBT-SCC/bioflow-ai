from typing import Annotated, Literal

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from sqlmodel import Session

from app.core.session import engine
from app.crud.tools import tool as tool_crud
from app.schemas.react_flow import NodeData
from app.schemas.tools import ToolNodePublic
from llm.models import gpt_5_mini
from llm.agents.tool_manager.create_tool_agent import CreateToolAgent
from llm.states import BuildEdge, BuildNode, CreateToolState, ReproductionState
from llm.tools.database import search_tool
from llm.tools.human import ask_human
from llm.utils.formatters import node_formatter
from llm.utils.uid_mapper import generate_char_index


class ToolManager:
    def __init__(self):
        self.name = 'tool_manager'

        self.tools = [
            search_tool,
            add_tool_node,
            create_tool_node,
            need_other_node,
            ask_human,
        ]

        self.llm = gpt_5_mini(temperature=0.2).bind_tools(self.tools, tool_choice='auto')

    def build(self):
        graph = StateGraph(state_schema=ReproductionState)

        compiled_graph = graph.compile(name=self.name)
        return compiled_graph


@tool
async def create_tool_node(request: str, tool_call_id: Annotated[str, InjectedToolCallId]):
    """Create new tool node

    Args:
        request: Node creation requirements. Format: "Using XXX (tool name) to XXX (specific task description, including what you want to do with this tool, what the tool's input is, and what the main expected output is)"
    """

    writer = get_stream_writer()
    writer({'info': f'Creating new tool: {request}'})

    sub_agent = CreateToolAgent().build()
    sub_agent_input = CreateToolState(messages=[HumanMessage(content=request)])
    response = await sub_agent.ainvoke(sub_agent_input)

    return Command(
        update={
            'id_mapper': response['id_mapper'],
            'messages': [ToolMessage(response['messages'][-1].content, tool_call_id=tool_call_id)],
        }
    )


@tool
async def add_tool_node(
    pos_x: int,
    pos_y: int,
    selected_id: int,
    input_edges: list[BuildEdge],
    state: Annotated[ReproductionState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Add tool nodes to the workflow by tool ID

    Args:
        pos_x: The x-coordinate of the node's layout on the panel, only positive integer
        pos_y: The y-coordinate of the node's layout on the panel
        selected_id: The tool ID of the corresponding node source
        input_edges: Input edge connection list for this node
    """
    node_count = len(state.exist_nodes)
    node_id = generate_char_index(node_count + 1)
    tool_uid = state.id_mapper.get_uid_by_index(selected_id)

    with Session(engine) as session:
        db_tool = tool_crud.get(session, tool_uid)
        db_node = ToolNodePublic.model_validate(db_tool.model_dump())
        new_node = BuildNode(
            id=node_id,
            type='tool',
            input_handles=db_node.input_handles,
            output_handles=db_node.output_handles,
            data=NodeData(resource_uid=tool_uid, args=db_node.static_params),
            command=db_node.complete_command,
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
                ToolMessage(content=f'add new tool node:\n{node_formatter(new_node)}', tool_call_id=tool_call_id)
            ],
        }
    )


@tool
async def need_other_node(node_type: Literal['resource', 'processor', 'code', 'note'], requirement: str):
    """

    Args:
        node_type:
        requirement:
    """

    return ''
