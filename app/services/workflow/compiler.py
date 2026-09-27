"""Validation and durable compilation of workflow definitions."""

import re
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.models.workflow import NodeDependency, NodeRun, WorkflowOutbox, WorkflowRun
from app.schemas.react_flow import Edge, Node, WorkflowDefinition
from app.schemas.workflow_run import NodeRunStatus, RunStatus
from app.services.workflow.batch_graph import lower_batch_subgraphs
from app.services.workflow.executors import (
    PRESENTATION_NODE_TYPES,
    ExecutorContext,
    prepare_node,
    supported_node_types,
)
from app.services.workflow.subgraphs import flatten_workflow, validate_interface_handles


class WorkflowCompileError(ValueError):
    """The supplied workflow graph is not a valid executable DAG."""


@dataclass(frozen=True, slots=True)
class CompileContext:
    """Run-scoped inputs needed to freeze all node configurations."""

    project_id: int | None
    sample_uid: uuid.UUID | None
    sample_name: str | None
    output_base: str


def _parse_source_handle(edge: Edge) -> str:
    prefix = f'{edge.source}-out-'
    if not edge.sourceHandle.startswith(prefix) or not (value := edge.sourceHandle.removeprefix(prefix)):
        message = f'Edge {edge.id}: invalid sourceHandle {edge.sourceHandle!r}'
        raise WorkflowCompileError(message)
    return value


def _parse_target_handle(edge: Edge) -> str:
    prefix = f'{edge.target}-in-'
    if not edge.targetHandle.startswith(prefix) or not (value := edge.targetHandle.removeprefix(prefix)):
        message = f'Edge {edge.id}: invalid targetHandle {edge.targetHandle!r}'
        raise WorkflowCompileError(message)
    return value


def _validate_graph(
    workflow: WorkflowDefinition, *, require_inputs: bool = False, allow_internal_frozen_paths: bool = False
) -> tuple[dict[str, Node], list[Edge], list[str]]:
    try:
        workflow = flatten_workflow(workflow, require_inputs=require_inputs)
    except ValueError as error:
        raise WorkflowCompileError(str(error)) from error
    node_map: dict[str, Node] = {}
    for node in workflow.nodes:
        if not node.id:
            message = 'Node IDs must not be empty'
            raise WorkflowCompileError(message)
        if node.id in node_map:
            message = f'Duplicate node ID: {node.id}'
            raise WorkflowCompileError(message)
        if node.type not in supported_node_types() | PRESENTATION_NODE_TYPES:
            message = f'Node {node.id}: unsupported executor type {node.type!r}'
            raise WorkflowCompileError(message)
        if node.type == 'resource_frozen_path' and (
            not allow_internal_frozen_paths or not node.id.startswith('@foreach/input/')
        ):
            message = f'Node {node.id}: fixed-path sources are reserved for ForEach child runs'
            raise WorkflowCompileError(message)
        if node.type == 'collect_file_collection':
            pattern = node.data.get('pattern')
            prefix = node.data.get('subfolder_prefix', 'item')
            if not isinstance(pattern, str) or not pattern.strip():
                message = f'Node {node.id}: pattern must be a non-empty glob string'
                raise WorkflowCompileError(message)
            if not isinstance(prefix, str) or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', prefix) is None:
                message = f'Node {node.id}: subfolder prefix must be a safe directory name'
                raise WorkflowCompileError(message)
        if node.type == 'foreach':
            child = WorkflowDefinition.model_validate(node.data['workflow'])
            item_port_id = node.data.get('item_port_id')
            if child.interface is None or item_port_id not in {port.id for port in child.interface.inputs}:
                message = f'Node {node.id}: child workflow needs a current-file input'
                raise WorkflowCompileError(message)

            def check_descendants(graph: WorkflowDefinition, owner_id: str) -> None:
                for nested in graph.nodes:
                    if nested.type in {'foreach', 'collect_file_collection', 'sample_mark', 'project_mark'}:
                        message = f'Node {owner_id}: {nested.type} is unavailable inside ForEach'
                        raise WorkflowCompileError(message)
                    if nested.type == 'subgraph':
                        check_descendants(WorkflowDefinition.model_validate(nested.data['workflow']), owner_id)

            check_descendants(child, node.id)
            _validate_graph(child)
        node_map[node.id] = node

    executable_order = [node_id for node_id, node in node_map.items() if node.type not in PRESENTATION_NODE_TYPES]
    executable_ids = set(executable_order)
    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree = dict.fromkeys(executable_ids, 0)
    bindings: set[tuple[str, str, str, str]] = set()
    validated_edges: list[Edge] = []

    for edge in workflow.edges:
        if edge.source not in node_map or edge.target not in node_map:
            message = f'Edge {edge.id}: source or target node does not exist'
            raise WorkflowCompileError(message)
        if edge.source not in executable_ids or edge.target not in executable_ids:
            message = f'Edge {edge.id}: presentation-only nodes cannot carry data dependencies'
            raise WorkflowCompileError(message)
        source_key = _parse_source_handle(edge)
        target_key = _parse_target_handle(edge)
        binding = (edge.source, source_key, edge.target, target_key)
        if binding in bindings:
            message = f'Edge {edge.id}: duplicate parameter binding'
            raise WorkflowCompileError(message)
        bindings.add(binding)
        validated_edges.append(edge)
        if edge.target not in adjacency[edge.source]:
            adjacency[edge.source].append(edge.target)
            indegree[edge.target] += 1

    queue = deque(node_id for node_id in executable_order if indegree[node_id] == 0)
    topological_order: list[str] = []
    while queue:
        node_id = queue.popleft()
        topological_order.append(node_id)
        for successor in adjacency[node_id]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                queue.append(successor)

    if len(topological_order) != len(executable_ids):
        message = 'Workflow graph contains a cycle'
        raise WorkflowCompileError(message)
    return node_map, validated_edges, topological_order


def validate_workflow(workflow: WorkflowDefinition) -> list[str]:
    """Validate a workflow and return its stable topological definition order."""
    return _validate_graph(lower_batch_subgraphs(workflow))[2]


def compile_run(
    session: Session,
    run: WorkflowRun,
    workflow: WorkflowDefinition,
    context: CompileContext,
    *,
    allow_internal_frozen_paths: bool = False,
) -> list[NodeRun]:
    """Compile a workflow into NodeRun, dependency and root Outbox rows.

    The caller owns the surrounding transaction.  Nothing is published here.
    """
    try:
        compiled_workflow = lower_batch_subgraphs(workflow)
        validate_interface_handles(compiled_workflow, session)
    except ValueError as error:
        raise WorkflowCompileError(str(error)) from error
    node_map, edges, order = _validate_graph(
        compiled_workflow, require_inputs=True, allow_internal_frozen_paths=allow_internal_frozen_paths
    )
    node_runs: dict[str, NodeRun] = {}

    for position_index, node_id in enumerate(order, start=1):
        node = node_map[node_id]
        node_runs[node_id] = NodeRun(
            run_uid=run.uid,
            generation=run.generation,
            definition_node_id=node.id,
            node_type=node.type,
            name=node.type,
            position_index=position_index,
            status=NodeRunStatus.PENDING,
        )
        session.add(node_runs[node_id])
    session.flush()

    incoming: dict[str, dict[str, dict[str, str] | list[dict[str, str]]]] = defaultdict(dict)
    source_nodes_by_target: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        source_key = _parse_source_handle(edge)
        target_key = _parse_target_handle(edge)
        source_run = node_runs[edge.source]
        target_run = node_runs[edge.target]
        reference = {'source_uid': str(source_run.uid), 'source_k': source_key}
        existing = incoming[edge.target].get(target_key)
        if existing is None:
            incoming[edge.target][target_key] = reference
        elif isinstance(existing, list):
            existing.append(reference)
        else:
            incoming[edge.target][target_key] = [existing, reference]
        source_nodes_by_target[edge.target].add(edge.source)
        session.add(
            NodeDependency(
                run_uid=run.uid,
                generation=run.generation,
                source_node_uid=source_run.uid,
                target_node_uid=target_run.uid,
                source_handle=source_key,
                target_handle=target_key,
            )
        )

    for node_id in order:
        node_run = node_runs[node_id]
        prepared = prepare_node(
            node_map[node_id],
            incoming[node_id],
            ExecutorContext(
                session=session,
                run_uid=run.uid,
                generation=run.generation,
                project_id=context.project_id,
                sample_uid=context.sample_uid,
                sample_name=context.sample_name,
                output_base=context.output_base,
                position_index=node_run.position_index,
            ),
        )
        node_run.name = prepared.name
        node_run.track_progress = prepared.track_progress
        node_run.tool_uid = prepared.tool_uid
        node_run.base_dir = prepared.base_dir
        node_run.prepared_data = prepared.as_json()
        if not source_nodes_by_target[node_id]:
            node_run.status = NodeRunStatus.READY
            if node_run.node_type != 'foreach':
                session.add(
                    WorkflowOutbox(
                        run_uid=run.uid,
                        node_run_uid=node_run.uid,
                        generation=run.generation,
                    )
                )

    if not node_runs:
        run.status = RunStatus.SUCCEEDED
        run.settled = True
    session.flush()
    return list(node_runs.values())
