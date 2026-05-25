import datetime
import json
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_milvus import BM25BuiltInFunction, Milvus
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command, Overwrite
from pydantic import BaseModel, Field, ValidationError
from sqlmodel import Session

import app.crud.workflow as wf_crud
from app.core.config import get_settings
from app.core.session import engine
from app.models.workflow import Workflow, WorkflowType
from app.schemas.react_flow import Edge, Node, Position, WorkflowDefinition
from llm.models import qwen3_embedding
from llm.states import BuildContext, BuildEdge, BuildNode, UserContext
from llm.utils.edge_validation import check_orphan_edges
from llm.utils.formatters import format_validation_error, manager_help_command, node_formatter
from llm.utils.uid_mapper import generate_char_index


class HelpSchema(BaseModel):
    action: str = Field(description='The action name to get detailed schema for')


class ViewNodeSchema(BaseModel):
    """View details of a specific node."""

    node_id: str = Field(description='The ID of the node to view details for')


class DeleteNodeSchema(BaseModel):
    """Remove existing node from the panel."""

    node_id: str = Field(description='Node ID to be deleted')


class FixEdgeItem(BaseModel):
    source_node: str = Field(description='ID of the existing node that produces the data')
    source_handle: str = Field(description='Output handle name on the source node')
    target_node: str = Field(description='ID of the existing node that consumes the data')
    target_handle: str = Field(description='Input handle name on the target node')


class FixEdgeSchema(BaseModel):
    """Replace ALL orphan edges with the provided complete edge list in one call."""

    edges: list[FixEdgeItem] = Field(description='Complete list of edges to install in place of the orphan edges')


class SaveWorkflowSchema(BaseModel):
    name: str = Field(description="Workflow name, brief description of this workflow's tasks in English")
    description: str = Field(
        description='Description of the workflow, including the primary functions and the principal tools utilized.'
    )


VALID_OPERATES: dict[str, type[BaseModel] | None] = {
    'view_workflow': None,
    'view_node': ViewNodeSchema,
    'delete_node': DeleteNodeSchema,
    'save_workflow': SaveWorkflowSchema,
}

HELP_MESSAGE = """other_node_manager contains the following available actions:
- `view_workflow`: View the current workflow structure
- `view_node`: View the details of a specific node in the workflow
- `delete_node`: Remove existing node from the panel
- `save_workflow`: Save the completed workflow into the database.

Use action="help" with params={"action": "<action_name>"} to get the parameter schema for a specific action.
(e.g. workflow_manager(action="help" params={"action": "view_workflow"}))
"""


@tool(parse_docstring=True)
async def workflow_manager(action: str, params: dict[str, Any], runtime: ToolRuntime[BuildContext]) -> Command:
    """Workflow manager, execute operations related to workflow panel.

    Args:
        action: Action to be performed
        params: Parameters used in the action
    """
    match action:
        case 'help':
            return manager_help_command(params, HelpSchema, VALID_OPERATES, HELP_MESSAGE, runtime.tool_call_id)

        case 'view_workflow':
            return _view_workflow(runtime)

        case 'view_node':
            try:
                validate_param = ViewNodeSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('view_node', e, ViewNodeSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return _view_node(validate_param.node_id, runtime)

        case 'delete_node':
            try:
                validate_param = DeleteNodeSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('view_node', e, DeleteNodeSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return _delete_node(validate_param.node_id, runtime)

        case 'fix_edge':
            try:
                validate_param = FixEdgeSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('fix_edge', e, FixEdgeSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return _fix_edge(validate_param, runtime)

        case 'save_workflow':
            try:
                validate_param = SaveWorkflowSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('save_workflow', e, SaveWorkflowSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return await _save_workflow(validate_param.name, validate_param.description, runtime)

        case _:
            return Command(
                update={
                    'messages': [
                        ToolMessage(
                            content=f'Unknown action: "{action}". Use action="help" with params={{}} to see available actions.',
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )


def _view_workflow(runtime: ToolRuntime[BuildContext]) -> Command:
    if runtime.state.get('exist_nodes', []):
        nodes = [
            {
                'id': n.id,
                'name': n.name,
                'type': n.type,
                'x': n.x,
                'y': n.y,
            }
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
        content = json.dumps({'nodes': nodes, 'edges': edges}, ensure_ascii=False)
    else:
        content = 'The current workflow panel is empty.'

    return Command(update={'messages': [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)]})


def _view_node(node_id: str, runtime: ToolRuntime[BuildContext]) -> Command:
    for n in runtime.state.get('exist_nodes', []):
        if n.id == node_id:
            return Command(
                update={'messages': [ToolMessage(content=node_formatter(n), tool_call_id=runtime.tool_call_id)]}
            )

    return Command(
        update={
            'messages': [
                ToolMessage(
                    content=f'Node with ID {node_id} not found.',
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )


def _delete_node(node_id: str, runtime: ToolRuntime[BuildContext]) -> Command:
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


def _orphan_edges_report(orphans: list[BuildEdge], exist_nodes: list[BuildNode], tool_call_id: str) -> Command:
    node_names = {n.id: n.name for n in exist_nodes}
    remaining_ids = sorted(node_names)
    affected_sources = sorted({e.source_node for e in orphans})

    lines = [
        f'Cannot save workflow: {len(orphans)} orphan edge(s) detected (target_node is missing).',
        '',
        'Orphan edges:',
    ]
    for i, e in enumerate(orphans, 1):
        src_name = node_names.get(e.source_node, '<unknown>')
        lines.append(f'  {i}. {e.source_node}({src_name}) [{e.source_handle}] -> <missing> [{e.target_handle}]')

    lines += [
        '',
        f'Source nodes referenced by orphans: {affected_sources}',
        f'Current nodes on the panel: {remaining_ids}',
        '',
        'Use action="fix_edge" with params={"edges": [...]} to replace ALL orphan edges in one call. '
        'Pass the complete list of edges that should replace the orphans; '
        'every orphan edge is removed atomically and the provided edges are appended. '
        'After fixing, retry save_workflow.',
        '',
        'fix_edge parameters:',
        '  - edges (list): each item has source_node, source_handle, target_node, target_handle (all str)',
    ]
    return Command(update={'messages': [ToolMessage(content='\n'.join(lines), tool_call_id=tool_call_id)]})


def _fix_edge(params: FixEdgeSchema, runtime: ToolRuntime[BuildContext]) -> Command:
    exist_nodes: list[BuildNode] = runtime.state.get('exist_nodes', [])
    exist_edges: list[BuildEdge] = runtime.state.get('exist_edges', [])

    nodes_by_id = {n.id: n for n in exist_nodes}
    errors: list[str] = []
    for i, item in enumerate(params.edges, 1):
        src = nodes_by_id.get(item.source_node)
        tgt = nodes_by_id.get(item.target_node)
        if src is None:
            errors.append(f"  edge {i}: source node '{item.source_node}' does not exist.")
            continue
        if tgt is None:
            errors.append(f"  edge {i}: target node '{item.target_node}' does not exist.")
            continue
        src_handles = {h.name for h in src.output_handles}
        if item.source_handle not in src_handles:
            errors.append(
                f"  edge {i}: source '{item.source_node}' has no output handle "
                f"'{item.source_handle}'. Available: {sorted(src_handles)}."
            )
        tgt_handles = {h.name for h in tgt.input_handles}
        if item.target_handle not in tgt_handles:
            errors.append(
                f"  edge {i}: target '{item.target_node}' has no input handle "
                f"'{item.target_handle}'. Available: {sorted(tgt_handles)}."
            )

    if errors:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='fix_edge rejected, no changes applied:\n' + '\n'.join(errors),
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    cleaned_edges = [e for e in exist_edges if e.target_node is not None]
    removed = len(exist_edges) - len(cleaned_edges)
    new_edges = [
        BuildEdge(
            source_node=item.source_node,
            source_handle=item.source_handle,
            target_node=item.target_node,
            target_handle=item.target_handle,
        )
        for item in params.edges
    ]
    cleaned_edges.extend(new_edges)

    added_lines = '\n'.join(
        f'  + {item.source_node}[{item.source_handle}] -> {item.target_node}[{item.target_handle}]'
        for item in params.edges
    )
    msg = f'Removed {removed} orphan edge(s). Added {len(new_edges)} new edge(s):\n{added_lines}'

    return Command(
        update={
            'exist_edges': Overwrite(cleaned_edges),
            'messages': [ToolMessage(content=msg, tool_call_id=runtime.tool_call_id)],
        }
    )


async def _save_workflow(name: str, description: str, runtime: ToolRuntime[UserContext]) -> Command:
    nodes = []
    exist_nodes = runtime.state.get('exist_nodes', [])
    exist_edges = runtime.state.get('exist_edges', [])

    orphans = check_orphan_edges(exist_edges)
    if orphans:
        return _orphan_edges_report(orphans, exist_nodes, runtime.tool_call_id)

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
            description=description,
            owner_id=runtime.context.user_id,
            wf_type=WorkflowType.TEMPLATE,
            workflow=workflow_define,
            public=True,
            auto_summary=True,
        )

        workflow = wf_crud.insert(session, workflow)

        embedding_model = qwen3_embedding()
        vectorstore = Milvus(
            collection_name='WorkflowCollection',
            connection_args={'uri': get_settings().knowledge_base.MILVUS_URL},
            embedding_function=embedding_model,
            builtin_function=BM25BuiltInFunction(),
            vector_field=['dense', 'sparse'],
            index_params=[{'index_type': 'HNSW', 'metric_type': 'L2'}, {}],
            consistency_level='Strong',
            drop_old=False,
            auto_id=True,
        )

        docs = [
            Document(
                page_content=workflow.description,
                metadata={
                    'wf_uid': str(workflow.uid),
                    'name': workflow.name,
                },
            ),
        ]
        await vectorstore.aadd_documents(docs)

    # 更新工作流列表并清空面板
    return Command(
        update={
            'messages': [
                ToolMessage(content='The workflow has been saved to the database.', tool_call_id=runtime.tool_call_id)
            ],
            'exist_nodes': Overwrite([]),
            'exist_edges': Overwrite([]),
            'workflow_uids': [workflow.uid],
            'updated': True,
        }
    )
