import datetime
import json
from typing import Any, Literal, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException, tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command, Overwrite
from pydantic import ValidationError
from sqlmodel import Session

import app.crud.resource.bio_db as db_crud
import app.crud.tools.tool as tool_crud
import app.crud.workflow as wf_crud
from app.core.session import engine
from app.models.workflow import Workflow, WorkflowType
from app.schemas.react_flow import Edge, Node, Position, WorkflowDefinition
from app.schemas.tools import Handle, ToolNodePublic
from llm.states import BuildEdge, BuildNode, BuildSchema, GlobalMarker, UserContext
from llm.utils.formatters import (
    global_mark_formatter,
    node_formatter,
)
from llm.utils.node_schema import PROCESSORS, RESOURCES
from llm.utils.uid_mapper import generate_char_index


@tool
async def view_workflow(runtime: ToolRuntime) -> str:
    """View the current workflow structure."""
    if runtime.state.get('exist_nodes', []):
        nodes = [
            {'id': n.id, 'name': n.name, 'type': n.type, 'x': n.x, 'y': n.y}
            for n in runtime.state.get('exist_nodes', [])
        ]
        edges = [
            {
                'source_node': e.source_node,
                'source_handle': e.source_handle,
                'target_node': e.target_node,
                'target_handle': e.target_handle,
            }
            for e in runtime.state.get('exist_edges', [])
        ]
        return json.dumps({'nodes': nodes, 'edges': edges}, ensure_ascii=False)

    return 'The current workflow panel is empty.'


@tool(parse_docstring=True)
async def view_node_details(node_id: str, runtime: ToolRuntime) -> str:
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
async def delete_node(node_id: str, runtime: ToolRuntime) -> Command:
    """Remove existing node from the panel.

    **This tool prohibits parallel calls! You are only allowed to make one search at a time!**

    Args:
        node_id: Node ID to be deleted
    """
    exist_nodes: list[BuildNode] = runtime.state.get('exist_nodes', [])
    exist_edges: list[BuildEdge] = runtime.state.get('exist_edges', [])

    target_node = next((n for n in exist_nodes if n.id == node_id), None)
    if target_node is None:
        return Command(
            update={
                'messages': [
                    ToolMessage(content=f'Error: Node "{node_id}" not found.', tool_call_id=runtime.tool_call_id)
                ],
            }
        )

    removed_edges = [e for e in exist_edges if node_id in (e.source_node, e.target_node)]
    remaining_edges = [e for e in exist_edges if node_id not in (e.source_node, e.target_node)]
    remaining_nodes = [n for n in exist_nodes if n.id != node_id]

    # Reassign IDs sequentially and build remapping table
    id_map: dict[str, str] = {}
    for i, node in enumerate(remaining_nodes):
        new_id = generate_char_index(i + 1)
        if node.id != new_id:
            id_map[node.id] = new_id
            node.id = new_id

    # Update edge references with new IDs
    for edge in remaining_edges:
        if edge.source_node in id_map:
            edge.source_node = id_map[edge.source_node]
        if edge.target_node in id_map:
            edge.target_node = id_map[edge.target_node]

    # Build deletion report
    if removed_edges:
        edge_lines = []
        for e in removed_edges:
            if e.source_node == node_id:
                edge_lines.append(
                    f'  - outgoing → node {e.target_node} (handle: {e.source_handle} → {e.target_handle})'
                )
            else:
                edge_lines.append(
                    f'  - incoming ← node {e.source_node} (handle: {e.source_handle} → {e.target_handle})'
                )
        edge_summary = f'{len(removed_edges)} edge(s) removed:\n' + '\n'.join(edge_lines)
    else:
        edge_summary = 'No edges were connected to this node.'

    id_remap_summary = ', '.join(f'{old}→{new}' for old, new in id_map.items()) if id_map else 'no changes'

    msg = (
        f'Deleted node {node_id} ("{target_node.name}", type={target_node.type}, '
        f'pos={{x:{target_node.x}, y:{target_node.y}}}).\n'
        f'{edge_summary}\n'
        f'Node ID reassignment: {id_remap_summary}.'
    )

    return Command(
        update={
            'exist_nodes': Overwrite(remaining_nodes),
            'exist_edges': Overwrite(remaining_edges),
            'messages': [ToolMessage(content=msg, tool_call_id=runtime.tool_call_id)],
        }
    )


@tool(parse_docstring=True)
async def add_tool_node(
    selected_id: int,
    input_edges: list[BuildEdge],
    pos_x: int,
    pos_y: int,
    runtime: ToolRuntime[UserContext],
) -> Command:
    """Add tool nodes to the workflow by tool ID.

    **This tool prohibits parallel calls! You are only allowed to make one search at a time!**

    Args:
        selected_id: The tool ID of the corresponding node source
        input_edges: Input edge connection list for this node. Each input handle allows only one edge to be connected.
        pos_x: The x-coordinate of the node's layout on the panel, only positive integer
        pos_y: The y-coordinate of the node's layout on the panel
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
            data={'tool_uid': tool_uid, 'args': db_node.modifiable_static_params},
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
async def view_resource_node(node_name: str) -> str:
    """View the details of a specific resource node by name.

    Args:
        node_name: The name of the resource node to view details for.
    """
    if node_name == 'genome':
        return (
            'The system currently includes the following nodes for reference genomes (available node_name):\n'
            '- `ncbi_genome`: Loads reference genomes and annotation files for any species from NCBI RefSeq based on species name or ID.\n'
            '- `GRCh38`: Loads the human reference genome released by GENCODE.\n'
            '- `GRCm39`: Loads the mouse reference genome released by GENCODE.\n\n'
            'In addition to providing genome and annotation files, all these nodes also offer indexes for common alignment tools. '
            'Generally, you can use them directly rather than manually building indexes. '
            'Refer to the respective nodes for more details.'
        )

    if node_name not in RESOURCES:
        return f'Error: Unknown resource type: {node_name}.'

    resource_schema = RESOURCES[node_name]

    node_obj = {
        'name': node_name,
        'type': node_name,
        'description': resource_schema['description'],
        'input_handles': [{'name': h.name, 'description': h.description} for h in resource_schema['input_handles']],
        'output_handles': [{'name': h.name, 'description': h.description} for h in resource_schema['output_handles']],
    }
    return json.dumps(node_obj)


@tool(parse_docstring=True)
async def add_resource_node(
    resource_type: str,
    input_edges: list[BuildEdge],
    pos_x: int,
    pos_y: int,
    runtime: ToolRuntime[UserContext],
    arguments: Optional[dict[str, Any]] = None,
) -> Command:
    """Add a resource node to the workflow.

    **This tool prohibits parallel calls! You are only allowed to make one search at a time!**

    Args:
        resource_type: Resource node type
        input_edges: Input edge connection list for this node
        pos_x: The x-coordinate of the node's layout on the panel, only positive integer
        pos_y: The y-coordinate of the node's layout on the panel
        arguments: Parameters of the resource node. Not all resource nodes need to have parameters set; the specific situation depends on user instructions.
    """
    node_count = len(runtime.state.get('exist_nodes', []))
    node_id = generate_char_index(node_count + 1)

    if resource_type not in RESOURCES:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content=f'Error: Unknown resource type: {resource_type}.', tool_call_id=runtime.tool_call_id
                    )
                ],
            }
        )

    resource_schema = RESOURCES[resource_type]

    # 验证参数
    if arguments is None:
        arguments = {}

    try:
        resource_schema['args'].model_validate(arguments)
    except ValidationError as e:
        error_details = []
        for error in e.errors():
            field = ' -> '.join(str(loc) for loc in error['loc'])
            msg = error['msg']
            error_details.append(f'  - {field}: {msg}')

        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content=f"Error: Invalid arguments for resource '{resource_type}':\n"
                        + '\n'.join(error_details),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    # 准备节点数据
    node_data = {}
    match resource_type:
        case 'value_string':
            new_node = BuildNode(
                id=node_id,
                name=resource_schema['name'],
                type='value_string',
                input_handles=[],
                output_handles=resource_schema['output_handles'],
                data=arguments,
                x=pos_x,
                y=pos_y,
            )
            return Command(
                update={
                    'exist_nodes': [new_node],
                    'messages': [
                        ToolMessage(
                            content=f'add new resource node:\n{node_formatter(new_node)}',
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )
        case 'genome':
            return Command(
                update={
                    'messages': [
                        ToolMessage(
                            content='You should use a specific reference genome node type rather than "genome" as the value for `resource_type`.',
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )
        case 'sequence':
            node_data = {'r1': '', 'r2': ''}
        case 'file':
            node_data = {'file_path': ''}
        case 'ncbi_genome' | 'GRCh38' | 'GRCm39':
            node_data = {'required_index': []}
        case 'db':
            with Session(engine) as session:
                bio_db = db_crud.get(session, arguments['db_id'])
                node_data = {**arguments, 'db_name': bio_db.name}
        case _:
            node_data = arguments

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
                'global_marks': [GlobalMarker.model_validate(arguments)],
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
async def view_processor_node(node_name: str) -> str:
    """View the details of a specific processor node by name.

    Args:
        node_name: The name of the processor node to view details for.
    """
    if node_name not in PROCESSORS:
        exception_msg = f'Error: Unknown processor type: {node_name}.'
        raise ToolException(exception_msg)

    processor_schema = PROCESSORS[node_name]

    node_obj = {
        'name': node_name,
        'type': node_name,
        'description': processor_schema['description'],
        'input_handles': [{'name': h.name, 'description': h.description} for h in processor_schema['input_handles']],
        'output_handles': [{'name': h.name, 'description': h.description} for h in processor_schema['output_handles']],
    }
    return json.dumps(node_obj)


@tool(parse_docstring=True)
async def add_processor_node(
    processor_name: str,
    arguments: Optional[dict[str, Any]],
    input_edges: list[BuildEdge],
    pos_x: int,
    pos_y: int,
    runtime: ToolRuntime[UserContext],
) -> Command:
    """Add a processor node to the workflow.

    **This tool prohibits parallel calls! You are only allowed to make one search at a time!**

    Args:
        processor_name: Processor name
        arguments: Parameters of the resource node.
        pos_x: The x-coordinate of the node's layout on the panel, only positive integer
        pos_y: The y-coordinate of the node's layout on the panel
        input_edges: Input edge connection list for this node
    """
    node_count = len(runtime.state.get('exist_nodes', []))
    node_id = generate_char_index(node_count + 1)

    if processor_name not in PROCESSORS:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='Error: Unknown processor name.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    # 验证参数
    processor_schema = PROCESSORS[processor_name]['args']
    try:
        if arguments is None:
            arguments = {}
        processor_schema.model_validate(arguments)
    except ValidationError as e:
        error_details = []
        for error in e.errors():
            field = ' -> '.join(str(loc) for loc in error['loc'])
            msg = error['msg']
            error_details.append(f'  - {field}: {msg}')
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content=f"Error: Invalid arguments for processor '{processor_name}':\n"
                        + '\n'.join(error_details),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    new_node = BuildNode(
        id=node_id,
        name=PROCESSORS[processor_name]['name'],
        type=processor_name,
        input_handles=PROCESSORS[processor_name]['input_handles'],
        output_handles=PROCESSORS[processor_name]['output_handles'],
        data=arguments,
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
    code: Optional[str] = None,
) -> Command:
    """Add a code node to the workflow.

    **This tool prohibits parallel calls! You are only allowed to make one search at a time!**

    Args:
        language: program language(R, python, linux bash)
        pos_x: The x-coordinate of the node's layout on the panel, only positive integer
        pos_y: The y-coordinate of the node's layout on the panel
        program_request: What functionality needs to be implemented in the code. The description needs to be **accurate**, as subsequent work will directly base code writing on your description.
        input_edges: Code node only have one edge `files`, just need to connect the files generated during the preliminary analysis, which are required by the code here, to this handle.
        code: Specific code content; this parameter is required only when the `language` value is bash.
    """
    node_count = len(runtime.state.get('exist_nodes', []))
    node_id = generate_char_index(node_count + 1)

    new_node = BuildNode(
        id=node_id,
        name=f'{language} Code',
        type=f'code_{language}',
        input_handles=[Handle(name='input_files', description='The files required by this code')],
        output_handles=[Handle(name='output_folder', description='The files generated by this code')],
        data={'description': program_request, 'code': code},
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
async def add_note_node(note_content: str, pos_x: int, pos_y: int, runtime: ToolRuntime[UserContext]) -> Command:
    """Add a note node to the workflow.

    **This tool prohibits parallel calls! You are only allowed to make one search at a time!**

    Args:
        note_content: Specific content of the notes (as brief and precise as possible)
        pos_x: The x-coordinate of the node's layout on the panel, only positive integer
        pos_y: The y-coordinate of the node's layout on the panel
    """
    node_count = len(runtime.state.get('exist_nodes', []))
    node_id = generate_char_index(node_count + 1)

    new_node = BuildNode(
        id=node_id,
        name='note',
        type='note',
        input_handles=[],
        output_handles=[],
        data={'content': note_content},
        x=pos_x,
        y=pos_y,
    )

    return Command(
        update={
            'exist_nodes': [new_node],
            'messages': [
                ToolMessage(content=f'add new note:\n{node_formatter(new_node)}', tool_call_id=runtime.tool_call_id)
            ],
        }
    )


@tool(parse_docstring=True)
async def assist_tool(
    action: str,
    parameter: Optional[dict[str, Any]],  # noqa: ARG001
    runtime: ToolRuntime,
) -> str:
    """Assist tool for various auxiliary functions.

    This tool is designed for accessibility in SKILL. Do not call it if you do not see information on how to use it.

    Args:
        action: The action to perform.
        parameter: Additional parameter for the action, if needed.
    """
    if action == 'view_global_marks':
        global_marks = runtime.state.get('global_marks', [])
        return global_mark_formatter(global_marks)

    return f'Action "{action}" is not supported.'


@tool(parse_docstring=True)
def save_workflow(name: str, runtime: ToolRuntime[UserContext, BuildSchema]) -> Command:
    """Save the completed workflow into the database.

    Args:
        name: Workflow name, brief description of this workflow's tasks
    """
    nodes = []
    exist_nodes = runtime.state.get('exist_nodes', [])
    exist_edges = runtime.state.get('exist_edges', [])
    for n in exist_nodes:
        if n.type in {'resource_ncbi_genome', 'resource_GRCh38', 'resource_GRCm39'}:
            required_index = list(dict.fromkeys(e.source_handle for e in exist_edges if e.source_node == n.id))
            node = Node(
                id=n.id,
                type=n.type,
                data={'required_index': required_index},
                position=Position(x=n.x * 500, y=n.y * 400),
                dragHandle='.nodeDragable',
                zIndex=-10 if n.type == 'note' else 20,
            )
        else:
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
    for e in exist_edges:
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

    now = datetime.datetime.now(datetime.UTC)
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
