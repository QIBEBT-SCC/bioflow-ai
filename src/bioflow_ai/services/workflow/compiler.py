"""Validation and durable compilation of workflow definitions."""

from collections import defaultdict

from sqlalchemy.orm import Session

from bioflow_ai.crud import node_dependency as dependency_crud
from bioflow_ai.crud import node_run as node_run_crud
from bioflow_ai.models.workflow import NodeDependency, NodeRun, WorkflowRun
from bioflow_ai.schemas.workflow_run import NodeRunStatus, RunStatus
from bioflow_ai.services.workflow.batch_graph import lower_batch_subgraphs
from bioflow_ai.services.workflow.executors import (
    ExecutorContext,
    prepare_node,
)
from bioflow_ai.services.workflow.subgraphs import flatten_workflow
from bioflow_ai.services.workflow.validation import validate_workflow
from bioflow_ai.services.workflow.validation._topology import order_nodes
from bioflow_ai.services.workflow.validation.registry import PRESENTATION_NODE_TYPES
from bioflow_ai.services.workflow.validation.resolution import resolve_workflow
from bioflow_ai.utils.system import utc_now


class WorkflowCompileError(ValueError):
    """The submitted workflow failed the shared validation operation."""


def compile_run(session: Session, run: WorkflowRun, *, sample_name: str | None) -> None:
    """Compile ``run.workflow`` into NodeRun and dependency rows; root nodes start READY.

    The caller owns the surrounding transaction.  Nothing is published here.
    """
    document = resolve_workflow(
        run.workflow,
        session,
        fresh=True,
        project_id=run.project_id,
        execution_scope='sample_level' if run.sample_uid is not None else 'project_level',
    )
    result = validate_workflow(document)
    if not result.valid:
        raise WorkflowCompileError(result.message)
    compiled_workflow = flatten_workflow(lower_batch_subgraphs(run.workflow))
    node_map = {node.id: node for node in compiled_workflow.nodes if node.type not in PRESENTATION_NODE_TYPES}
    edges = [edge for edge in compiled_workflow.edges if edge.target in node_map]
    order = order_nodes(list(node_map), [(edge.source, edge.target) for edge in edges])
    context = ExecutorContext(run=run, sample_name=sample_name, tool_definitions=dict(document.catalog or {}))
    node_runs: dict[str, NodeRun] = {}

    for position_index, node_id in enumerate(order, start=1):
        node = node_map[node_id]
        node_runs[node_id] = NodeRun(
            run_uid=run.uid,
            definition_node_id=node.id,
            node_type=node.type,
            name=node.type,
            position_index=position_index,
            status=NodeRunStatus.PENDING,
        )
        node_run_crud.insert(session, node_runs[node_id])
    session.flush()

    incoming: dict[str, dict[str, dict[str, str] | list[dict[str, str]]]] = defaultdict(dict)
    source_nodes_by_target: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        source_key = edge.sourceHandle.removeprefix(f'{edge.source}-out-')
        target_key = edge.targetHandle.removeprefix(f'{edge.target}-in-')
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
        dependency_crud.insert(
            session,
            NodeDependency(
                run_uid=run.uid,
                source_node_uid=source_run.uid,
                target_node_uid=target_run.uid,
                source_handle=source_key,
                target_handle=target_key,
            ),
        )

    for node_id in order:
        node_run = node_runs[node_id]
        prepared = prepare_node(node_map[node_id], incoming[node_id], node_run.position_index, context)
        node_run.name = prepared.name
        node_run.track_progress = prepared.track_progress
        node_run.tool_uid = prepared.tool_uid
        node_run.base_dir = prepared.base_dir
        node_run.prepared_data = {'executor_name': node_run.node_type, 'kwargs': prepared.kwargs}
        if not source_nodes_by_target[node_id]:
            node_run.status = NodeRunStatus.READY

    if not node_runs:
        run.status = RunStatus.SUCCEEDED
        run.settled = True
        run.end_time = utc_now()
    session.flush()


__all__ = ['WorkflowCompileError', 'compile_run']
