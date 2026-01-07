import datetime
from typing import Any, Literal, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException, tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import ValidationError
from sqlmodel import Session

import app.crud.resource.bio_db as db_crud
import app.crud.tools.tool as tool_crud
import app.crud.workflow as wf_crud
from app.core.session import engine
from app.models.resource import BioDb
from app.models.workflow import Workflow, WorkflowType
from app.schemas.react_flow import Edge, Node, NodeData, Position, WorkflowDefinition
from app.schemas.tools import Handle, ToolNodePublic
from llm.states import BuildEdge, BuildNode, BuildSchema, GlobalMarker, UserContext
from llm.utils.formatters import (
    edge_formatter,
    global_mark_formatter,
    handle_formatter,
    node_formatter,
    simple_node_formatter,
)
from llm.utils.node_schema import PROCESSORS, RESOURCES
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
    selected_id: int,
    update_args: Optional[str],
    input_edges: list[BuildEdge],
    pos_x: int,
    pos_y: int,
    runtime: ToolRuntime[UserContext],
):
    """Add tool nodes to the workflow by tool ID

    Args:
        selected_id: The tool ID of the corresponding node source
        update_args: Updated parameters for this tool node. Leave blank means not modifying parameters. Note that if you pass an empty string, it also means modifying parameters.
        input_edges: Input edge connection list for this node
        pos_x: The x-coordinate of the node's layout on the panel, only positive integer
        pos_y: The y-coordinate of the node's layout on the panel
    """
    node_count = len(runtime.state.get('exist_nodes', []))
    node_id = generate_char_index(node_count + 1)
    tool_uid = runtime.state['id_mapper'].get_uid_by_index(selected_id)

    with Session(engine) as session:
        db_tool = tool_crud.get(session, tool_uid)
        db_node = ToolNodePublic.model_validate(db_tool.model_dump())

        tool_arg = db_node.modifiable_static_params if update_args is None else update_args
        new_node = BuildNode(
            id=node_id,
            name=db_node.name,
            type='tool',
            input_handles=db_node.input_handles,
            output_handles=db_node.output_handles,
            data=NodeData(resource_uid=tool_uid, args=tool_arg),
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
async def view_resource_node(node_name: str):
    """View the details of a specific resource node by name.

    Args:
        node_name: The name of the resource node to view details for.
    """
    if node_name not in RESOURCES:
        raise ToolException(f'Error: Unknown resource type: {node_name}.')

    resource_schema = RESOURCES[node_name]

    input_handles = handle_formatter(resource_schema['input_handles'], 4)
    output_handles = handle_formatter(resource_schema['output_handles'], 4)
    return (
        f'<resource_node name="{node_name}">\n'
        f'  <description>{resource_schema["description"]}</description>\n'
        f'  <type>{node_name}</type>\n'
        f'  <input_handles>\n'
        f'{input_handles}\n'
        f'  </input_handles>\n'
        f'  <output_handles>\n'
        f'{output_handles}\n'
        f'  </output_handles>\n'
        f'</resource_node>'
    )


@tool(parse_docstring=True)
async def add_resource_node(
    resource_type: Literal['sequence', 'file', 'db', 'genome', 'value_string', 'global_mark'],
    args: Optional[dict[str, Any]],
    input_edges: list[BuildEdge],
    pos_x: int,
    pos_y: int,
    runtime: ToolRuntime[UserContext],
):
    """Add a resource node to the workflow

    Args:
        resource_type: Resource node type
        args: Parameters of the resource node. Not all resource nodes need to have parameters set; the specific situation depends on user instructions.
        input_edges: Input edge connection list for this node
        pos_x: The x-coordinate of the node's layout on the panel, only positive integer
        pos_y: The y-coordinate of the node's layout on the panel
    """
    node_count = len(runtime.state.get('exist_nodes', []))
    node_id = generate_char_index(node_count + 1)

    if resource_type not in RESOURCES:
        return f'Error: Unknown resource type: {resource_type}.'

    resource_schema = RESOURCES[resource_type]

    # 验证参数
    if args is None:
        args = {}
    try:
        resource_schema['args'].model_validate(args)
    except ValidationError as e:
        error_details = []
        for error in e.errors():
            field = ' -> '.join(str(loc) for loc in error['loc'])
            msg = error['msg']
            error_details.append(f'  - {field}: {msg}')
        return f"Error: Invalid arguments for resource '{resource_type}':\n" + '\n'.join(error_details)

    # 准备节点数据
    match resource_type:
        case 'db':
            with Session(engine) as session:
                selected_id = args['db_id']
                db_db: BioDb = db_crud.get(session, selected_id)
                node_data = NodeData(args=f'db:{db_db.name}', resource_uid=f'{selected_id}')
        case 'sequence':
            node_data = NodeData(args={'r1': '', 'r2': ''})
        case 'global_mark':
            node_data = NodeData(args=args)
        case _:
            node_data = NodeData(args='')

    # 创建节点
    new_node = BuildNode(
        id=node_id,
        name=resource_schema['name'],
        type=f'resource_{resource_type}',
        input_handles=resource_schema['input_handles'],
        output_handles=resource_schema['output_handles'],
        data=node_data,
        x=pos_x,
        y=pos_y,
    )

    for edge in input_edges:
        edge.target_node = node_id

    if resource_type == 'global_mark':
        return Command(
            update={
                'global_marks': [GlobalMarker.model_validate(args)],
                'exist_nodes': [new_node],
                'exist_edges': input_edges,
                'messages': [
                    ToolMessage(
                        content=f'add new resource node:\n{node_formatter(new_node)}', tool_call_id=runtime.tool_call_id
                    )
                ],
            }
        )

    return Command(
        update={
            'exist_nodes': [new_node],
            'exist_edges': input_edges,
            'messages': [
                ToolMessage(
                    content=f'add new resource node:\n{node_formatter(new_node)}', tool_call_id=runtime.tool_call_id
                )
            ],
        }
    )


@tool(parse_docstring=True)
async def view_processor_node(node_name: str):
    """View the details of a specific processor node by name.

    Args:
        node_name: The name of the processor node to view details for.
    """
    if node_name not in PROCESSORS:
        raise ToolException(f'Error: Unknown processor type: {node_name}.')

    processor_schema = PROCESSORS[node_name]

    input_handles = handle_formatter(processor_schema['input_handles'], 4)
    output_handles = handle_formatter(processor_schema['output_handles'], 4)
    return (
        f'<processor_node name="{node_name}">\n'
        f'  <description>{processor_schema["description"]}</description>\n'
        f'  <type>{node_name}</type>\n'
        f'  <input_handles>\n'
        f'{input_handles}\n'
        f'  </input_handles>\n'
        f'  <output_handles>\n'
        f'{output_handles}\n'
        f'  </output_handles>\n'
        f'</processor_node>'
    )


@tool(parse_docstring=True)
async def add_processor_node(
    processor_name: str,
    args: Optional[dict[str, Any]],
    input_edges: list[BuildEdge],
    pos_x: int,
    pos_y: int,
    runtime: ToolRuntime[UserContext],
):
    """Add a processor node to the workflow

    Args:
        processor_name: Processor name
        args: Parameters of the resource node.
        pos_x: The x-coordinate of the node's layout on the panel, only positive integer
        pos_y: The y-coordinate of the node's layout on the panel
        input_edges: Input edge connection list for this node
    """
    node_count = len(runtime.state.get('exist_nodes', []))
    node_id = generate_char_index(node_count + 1)

    if processor_name not in PROCESSORS:
        return 'Error: Unknown processor name.'

    # 验证参数
    processor_schema = PROCESSORS[processor_name]['args']
    try:
        if args is None:
            args = {}
        processor_schema.model_validate(args)
    except ValidationError as e:
        error_details = []
        for error in e.errors():
            field = ' -> '.join(str(loc) for loc in error['loc'])
            msg = error['msg']
            error_details.append(f'  - {field}: {msg}')
        return f"Error: Invalid arguments for processor '{processor_name}':\n" + '\n'.join(error_details)

    new_node = BuildNode(
        id=node_id,
        name=PROCESSORS[processor_name]['name'],
        type=processor_name,
        input_handles=PROCESSORS[processor_name]['input_handles'],
        output_handles=PROCESSORS[processor_name]['output_handles'],
        data=NodeData(args=args, resource_uid=''),
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
                    content=f'add new processor node:\n{node_formatter(new_node)}', tool_call_id=runtime.tool_call_id
                )
            ],
        }
    )


@tool(parse_docstring=True)
async def add_code_node(
    language: Literal['R', 'python', 'bash'],
    pos_x: int,
    pos_y: int,
    program_request: str,
    input_edges: list[BuildEdge],
    runtime: ToolRuntime[UserContext],
):
    """Add a code node to the workflow

    Args:
        language: program language(R, python, linux bash)
        pos_x: The x-coordinate of the node's layout on the panel, only positive integer
        pos_y: The y-coordinate of the node's layout on the panel
        program_request: What functionality needs to be implemented in the code. The description needs to be **accurate**, as subsequent work will directly base code writing on your description.
        input_edges: Code node only have one edge `files`, just need to connect the files generated during the preliminary analysis, which are required by the code here, to this handle.
    """
    node_count = len(runtime.state.get('exist_nodes', []))
    node_id = generate_char_index(node_count + 1)

    new_node = BuildNode(
        id=node_id,
        type=f'code_{language}',
        input_handles=[Handle(name='input_files', description='The files required by this code')],
        output_handles=[Handle(name='output_folder', description='The files generated by this code')],
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
                ToolMessage(
                    content=f'add new code node:\n{node_formatter(new_node)}', tool_call_id=runtime.tool_call_id
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
async def assist_tool(
    action: str,
    parameter: Optional[dict[str, Any]],
    runtime: ToolRuntime,
):
    """Assist tool for various auxiliary functions.
    This tool is designed for accessibility in SKILL. Do not call it if you do not see information on how to use it.

    Args:
        action: The action to perform.
        parameter: Additional parameter for the action, if needed.
    """
    if action == 'view_global_marks':
        global_marks = runtime.state.get('global_marks', [])
        return global_mark_formatter(global_marks)
    else:
        return f'Action "{action}" is not supported.'


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
