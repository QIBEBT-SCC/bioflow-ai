import datetime

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from sqlmodel import Session

import app.crud.tools.tool as tool_crud
import app.crud.workflow as wf_crud
from app.core.session import engine
from app.models.workflow import Workflow, WorkflowType
from app.schemas.react_flow import Edge, Node, NodeData, Position, WorkflowDefinition
from app.schemas.tools import ToolNodePublic
from llm.states import BuildEdge, BuildNode, BuildSchema, UserContext
from llm.utils.formatters import edge_formatter, node_formatter, simple_node_formatter
from llm.utils.uid_mapper import generate_char_index


@tool
async def view_workflow(runtime: ToolRuntime):
    """View the current workflow structure."""

    nodes = []
    for n in runtime.state.get('exist_nodes', []):
        nodes.append(f'    {simple_node_formatter(n)}')
    node_content = f'  <Nodes>\n{"\n".join(nodes)}\n  </Nodes>'

    edges = []
    for e in runtime.state.get('exist_edges', []):
        edges.append(f'    {edge_formatter(e)}')
    edge_content = f'  <Edges>\n{"\n".join(edges)}\n  </Edges>'

    return f'<Workflow>\n{node_content}\n\n{edge_content}\n</Workflow>'


@tool(parse_docstring=True)
async def view_node_details(node_id: str, runtime: ToolRuntime):
    """View the details of a specific node in the workflow.

    Args:
        node_id: The ID of the node to view details for.
    """
    for n in runtime.state.get('exist_nodes', []):
        n: BuildNode
        if n.id == node_id:
            return node_formatter(n)

    return f'Node with ID {node_id} not found.'


@tool(parse_docstring=True)
async def add_tool_node(
    pos_x: int, pos_y: int, selected_id: int, input_edges: list[BuildEdge], runtime: ToolRuntime[UserContext]
):
    """Add tool nodes to the workflow by tool ID

    Args:
        pos_x: The x-coordinate of the node's layout on the panel, only positive integer
        pos_y: The y-coordinate of the node's layout on the panel
        selected_id: The tool ID of the corresponding node source
        input_edges: Input edge connection list for this node
    """
    node_count = len(runtime.state.get('exist_nodes', []))
    node_id = generate_char_index(node_count + 1)
    tool_uid = runtime.state['id_mapper'].get_uid_by_index(selected_id)

    with Session(engine) as session:
        db_tool = tool_crud.get(session, tool_uid)
        db_node = ToolNodePublic.model_validate(db_tool.model_dump())
        new_node = BuildNode(
            id=node_id,
            name=db_node.name,
            type='tool',
            input_handles=db_node.input_handles,
            output_handles=db_node.output_handles,
            data=NodeData(resource_uid=tool_uid, args=db_node.modifiable_static_params),
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
                ToolMessage(
                    content=f'add new tool node:\n{node_formatter(new_node)}', tool_call_id=runtime.tool_call_id
                )
            ],
        }
    )


@tool(parse_docstring=True)
async def add_note_node(
    note_content: str, pos_x: int, pos_y: int, input_edges: list[BuildEdge], runtime: ToolRuntime[UserContext]
):
    """Add a note node to the workflow

    Args:
        note_content: Specific content of the notes (as brief and precise as possible)
        pos_x: The x-coordinate of the node's layout on the panel, only positive integer
        pos_y: The y-coordinate of the node's layout on the panel
        input_edges: Code node only have one edge `files`, just need to connect the files generated during the preliminary analysis, which are required by the code here, to this handle.
    """
    node_count = len(runtime.state.get('exist_nodes', []))
    node_id = generate_char_index(node_count + 1)

    new_node = BuildNode(
        id=node_id,
        name='note',
        type='note',
        input_handles=[],
        output_handles=[],
        data=NodeData(args=note_content),
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
                ToolMessage(content=f'add new note:\n{node_formatter(new_node)}', tool_call_id=runtime.tool_call_id)
            ],
        }
    )


@tool(parse_docstring=True)
def save_workflow(name: str, runtime: ToolRuntime[UserContext, BuildSchema]):
    """Save the completed workflow into the database

    Args:
        name: Workflow name, brief description of this workflow's tasks
    """
    nodes = []
    for n in runtime.state.get('exist_nodes', []):
        node = Node(
            id=n.id,
            type=n.type,
            data=n.data,
            position=Position(x=n.x * 500, y=n.y * 400),
            dragHandle='.nodeDragable',
            zIndex=-10 if n.type == 'note' else 20,
        )
        nodes.append(node)

    edges = []
    for e in runtime.state.get('exist_edges', []):
        source_handle = f'{e.source_node}-out-{e.source_handle}'
        target_handle = f'{e.target_node}-in-{e.target_handle}'
        edge = Edge(
            id=f'xy-edge__{source_handle}-{target_handle}',
            animated=True,
            source=e.source_node,
            target=e.target_node,
            sourceHandle=source_handle,
            targetHandle=target_handle,
        )
        edges.append(edge)

    workflow_define = WorkflowDefinition(nodes=nodes, edges=edges)

    now = datetime.datetime.now()
    with Session(engine) as session:
        workflow = Workflow(
            name=f'{now.strftime("%y%m%d")}_{name}(AI)',
            owner_id=runtime.context.user_id,
            wf_type=WorkflowType.TEMPLATE,
            workflow=workflow_define,
            public=True,
        )

        workflow = wf_crud.insert(session, workflow)

    # 更新工作流列表并清空面板
    return Command(
        update={
            'workflow_uid': [workflow.uid],
            'messages': [
                ToolMessage(content='The workflow has been saved to the database.', tool_call_id=runtime.tool_call_id)
            ],
            'exist_nodes': [],
            'exist_edges': [],
        }
    )
