"""Validation and durable compilation of workflow definitions."""

import uuid
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy.orm import Session

from bioflow_ai.models.workflow import NodeDependency, NodeRun, WorkflowOutbox, WorkflowRun
from bioflow_ai.schemas.react_flow import WorkflowDefinition
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


class WorkflowCompileError(ValueError):
    """The submitted workflow failed the shared validation operation."""


@dataclass(frozen=True, slots=True)
class CompileContext:
    """Run-scoped inputs needed to freeze all node configurations."""

    project_id: int | None
    sample_uid: uuid.UUID | None
    sample_name: str | None
    output_base: str


def compile_run(
    session: Session,
    run: WorkflowRun,
    workflow: WorkflowDefinition,
    context: CompileContext,
) -> list[NodeRun]:
    """Compile a workflow into NodeRun, dependency and root Outbox rows.

    The caller owns the surrounding transaction.  Nothing is published here.
    """
    document = resolve_workflow(
        workflow,
        session,
        fresh=True,
        project_id=context.project_id,
        execution_scope='sample_level' if context.sample_uid is not None else 'project_level',
    )
    result = validate_workflow(document)
    if not result.valid:
        raise WorkflowCompileError(result.message)
    compiled_workflow = flatten_workflow(lower_batch_subgraphs(workflow))
    node_map = {node.id: node for node in compiled_workflow.nodes if node.type not in PRESENTATION_NODE_TYPES}
    edges = [edge for edge in compiled_workflow.edges if edge.target in node_map]
    order = order_nodes(list(node_map), [(edge.source, edge.target) for edge in edges])
    tool_definitions = dict(document.catalog or {})
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
        session.add(node_runs[node_id])
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
        session.add(
            NodeDependency(
                run_uid=run.uid,
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
                project_id=context.project_id,
                sample_uid=context.sample_uid,
                sample_name=context.sample_name,
                output_base=context.output_base,
                position_index=node_run.position_index,
                tool_definitions=tool_definitions,
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
                    )
                )

    if not node_runs:
        run.status = RunStatus.SUCCEEDED
        run.settled = True
    session.flush()
    return list(node_runs.values())


__all__ = ['WorkflowCompileError', 'compile_run']
