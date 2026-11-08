"""Read-only workflow execution evidence and optional diagnosis report output."""

import json
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from sqlalchemy import select
from sqlalchemy.orm import Session, load_only

from bioflow_ai.infrastructure.database import engine
from bioflow_ai.llm.context import WorkflowAssistantContext
from bioflow_ai.llm.tools.container import resolve_workflow_run
from bioflow_ai.models.workflow import NodeRun, WorkflowRun
from bioflow_ai.schemas.react_flow import WorkflowDefinition
from bioflow_ai.schemas.workflow_run import NodeRunStatus, RunStatus
from bioflow_ai.services.workflow.node_details import describe_node
from bioflow_ai.services.workflow.subgraphs import SubgraphError, definition_id, enter_subgraphs

MAX_CHILD_PAGE = 100


# ============ Workflow Reader =============
def _parents_and_children(workflow: WorkflowDefinition) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    parents: dict[str, set[str]] = defaultdict(set)
    children: dict[str, set[str]] = defaultdict(set)
    for edge in workflow.edges:
        parents[edge.target].add(edge.source)
        children[edge.source].add(edge.target)
    return parents, children


def _upstream_scope(node_ids: list[str], parents: dict[str, set[str]], depth: int) -> list[str]:
    """List targets first, then their ancestors by shortest upstream distance."""
    ordered = list(dict.fromkeys(node_ids))
    seen = set(ordered)
    frontier = ordered
    for _ in range(depth):
        upstream = sorted({parent for node_id in frontier for parent in parents.get(node_id, ())} - seen)
        if not upstream:
            break
        ordered.extend(upstream)
        seen.update(upstream)
        frontier = upstream
    return ordered


def _workflow_payload(
    context: WorkflowAssistantContext,
    action: str,
    include_node_id: str | None = None,
    subgraph_path: list[str] | None = None,
    child_run_uid: str | None = None,
    offset: int = 0,
    limit: int = 20,
    *,
    failed_only: bool = True,
    upstream_depth: int = 0,
) -> dict[str, Any]:
    if offset < 0 or not 1 <= limit <= MAX_CHILD_PAGE:
        return {'error': 'offset must be nonnegative and limit must be between 1 and 100.'}
    if upstream_depth < 0:
        return {'error': 'upstream_depth must be nonnegative.'}
    if upstream_depth and action != 'overview':
        return {'error': 'upstream_depth is only supported for action=overview.'}
    if upstream_depth and not include_node_id and not failed_only:
        return {'error': 'upstream_depth requires node_id or failed_only=true.'}
    if action in {'node', 'children'} and not include_node_id:
        return {'error': f'node_id is required for action={action}.'}
    with Session(engine) as session:
        try:
            run = resolve_workflow_run(session, context, child_run_uid)
        except ValueError as error:
            return {'error': str(error)}
        path = tuple(subgraph_path or ())
        try:
            graph, _ = enter_subgraphs(run.workflow, list(path))
        except SubgraphError as error:
            return {'error': str(error)}
        graph_nodes = {node.id: node for node in graph.nodes}
        if include_node_id is not None and include_node_id not in graph_nodes:
            return {'error': f'Node not found: {include_node_id}'}
        target_runtime_id = definition_id((*path, include_node_id)) if include_node_id is not None else None
        if action == 'children':
            control = session.scalar(
                select(NodeRun)
                .where(
                    NodeRun.run_uid == run.uid,
                    NodeRun.definition_node_id == target_runtime_id,
                )
                .options(load_only(NodeRun.node_type))
            )
            if control is None or control.node_type != 'foreach':
                return {'error': 'No ForEach control task exists for this node.'}
            filters = [WorkflowRun.parent_node_uid == control.uid]
            if failed_only:
                filters.append(WorkflowRun.status == RunStatus.FAILED)
            children = list(
                session.scalars(
                    select(WorkflowRun)
                    .where(*filters)
                    .options(
                        load_only(
                            WorkflowRun.item_index,
                            WorkflowRun.item_name,
                            WorkflowRun.generation,
                            WorkflowRun.status,
                            WorkflowRun.base_dir,
                        )
                    )
                    .order_by(WorkflowRun.item_index)
                    .offset(offset)
                    .limit(limit + 1)
                )
            )
            return {
                'parent_run_uid': str(run.uid),
                'parent_generation': run.generation,
                'subgraph_path': list(path),
                'node_id': include_node_id,
                **({'next_offset': offset + limit} if len(children) > limit else {}),
                'children': [
                    {
                        'run_uid': str(item.uid),
                        **({'item_index': item.item_index} if item.item_index is not None else {}),
                        **({'item_name': item.item_name} if item.item_name is not None else {}),
                        'generation': item.generation,
                        'status': item.status.value,
                        'base_dir': item.base_dir,
                    }
                    for item in children[:limit]
                ],
            }
        node_runs = list(
            session.scalars(
                select(NodeRun)
                .where(NodeRun.run_uid == run.uid)
                .options(
                    load_only(
                        NodeRun.definition_node_id,
                        NodeRun.node_type,
                        NodeRun.name,
                        NodeRun.status,
                    )
                )
            )
        )
        by_id = {node.definition_node_id: node for node in node_runs}
        parents, _ = _parents_and_children(graph)
        node_statuses = {}
        for node in graph.nodes:
            runtime_id = definition_id((*path, node.id))
            task = by_id.get(runtime_id)
            if task is None and node.type == 'subgraph':
                prefix = definition_id((*path, node.id, ''))
                internal = [item for key, item in by_id.items() if key.startswith(prefix)]
                task_status = (
                    'failed'
                    if any(item.status == NodeRunStatus.FAILED for item in internal)
                    else (
                        'blocked'
                        if any(item.status == NodeRunStatus.BLOCKED for item in internal)
                        else 'running'
                        if any(item.status == NodeRunStatus.RUNNING for item in internal)
                        else 'succeeded'
                        if internal and all(item.status == NodeRunStatus.SUCCEEDED for item in internal)
                        else 'pending'
                    )
                )
            else:
                task_status = task.status.value if task else 'presentation_only'
            node_statuses[node.id] = task_status

        failed_node_ids = [node_id for node_id, status in node_statuses.items() if status == 'failed']
        targets = (
            [include_node_id] if include_node_id is not None else failed_node_ids if failed_only else list(graph_nodes)
        )
        scoped_ids = [
            node_id for node_id in _upstream_scope(targets, parents, upstream_depth) if node_id in graph_nodes
        ]
        shown_ids = scoped_ids[offset : offset + limit] if action == 'overview' else scoped_ids
        if action == 'node':
            detail = session.scalar(
                select(NodeRun)
                .where(
                    NodeRun.run_uid == run.uid,
                    NodeRun.definition_node_id == target_runtime_id,
                )
                .execution_options(populate_existing=True)
            )
            if detail is not None:
                by_id[detail.definition_node_id] = detail

        nodes = []
        for node_id in shown_ids:
            node = graph_nodes[node_id]
            runtime_id = definition_id((*path, node.id))
            task = by_id.get(runtime_id)
            payload: dict[str, Any] = {
                'node_id': node.id,
                'name': node.data.get('name') or (task.name if task else node.id),
                'type': node.type,
                'status': node_statuses[node.id],
            }
            if node.type == 'subgraph':
                payload['child_path'] = [*path, node.id]
                payload['batch'] = task is not None and task.node_type == 'foreach'
            if action == 'node':
                payload.update(describe_node(session, graph, node, path, by_id))
            nodes.append(payload)
        result = {
            'run': {
                'workflow_uid': str(run.workflow_uid) if run.workflow_uid else None,
                'status': run.status.value,
                'base_dir': run.base_dir,
                **({'item_index': run.item_index} if run.item_index is not None else {}),
                **({'item_name': run.item_name} if run.item_name is not None else {}),
            },
            'subgraph_path': list(path),
            'nodes': nodes,
        }
        if action == 'overview':
            shown_set = set(shown_ids)
            result['edges'] = [
                edge.model_dump(include={'source', 'sourceHandle', 'target', 'targetHandle'})
                for edge in graph.edges
                if edge.source in shown_set and edge.target in shown_set
            ]
            if offset + limit < len(scoped_ids):
                result['next_offset'] = offset + limit
        return result


@tool('workflow_reader', parse_docstring=True)
def workflow_reader(
    action: Literal['overview', 'node', 'children'],
    runtime: ToolRuntime[WorkflowAssistantContext],
    node_id: str | None = None,
    subgraph_path: list[str] | None = None,
    child_run_uid: str | None = None,
    offset: int = 0,
    limit: int = 20,
    *,
    failed_only: bool = True,
    upstream_depth: int = 0,
) -> str:
    """Read compact failure summaries, a bounded upstream graph, or one node's full evidence.

    Start diagnosis with ``overview`` to list only failed nodes. For
    a named node, use ``node`` directly. Expand relevant context with
    ``overview(node_id=..., upstream_depth=1)``; increase the depth only when needed.
    Successful ancestors are included in the scope. Overview never returns full node
    configuration or scripts; use ``node`` for port descriptions, connected sources,
    the assembled command template, recorded output results and CODE source. A Tool
    node may also list ``memories``: IDs and short descriptions of shared usage lessons.

    Args:
        action: ``overview`` returns paginated compact node summaries and edges within
            the returned page; ``node`` returns configuration, commands and input/output
            handles with their sources or recorded results, without internal task metadata;
            ``children`` lists ForEach child runs (failed by default).
        node_id: Local node ID, required for ``node`` and ``children``. For ``overview``,
            focus on this node and its ancestors instead of filtering by failure status.
        subgraph_path: Nested subgraph node IDs to enter in the selected run.
        child_run_uid: ForEach child run UID returned by ``children``.
        offset: First scoped node or child item to list. Follow ``next_offset`` when
            present; its absence means all matching results have been returned.
        limit: Maximum scoped nodes or child items to list (1 to 100, default 20).
        failed_only: Default true. Without ``node_id``, overview starts from failed
            nodes; children lists failed runs. Set false explicitly to list all nodes
            or child runs. Ignored by ``node`` and by an overview with ``node_id``.
        upstream_depth: Overview only. Number of upstream edge hops (default 0).
            One hop includes all direct parents, two includes their parents as well.
            Start from ``node_id`` or all failed nodes; targets come first, then ancestors
            ordered by shortest distance. All scopes support pagination, including
            scopes inside ordinary subgraphs or ForEach child runs.
    """
    if action == 'node' and not node_id:
        return '`node_id` is required for action=node.'
    return json.dumps(
        _workflow_payload(
            runtime.context,
            action,
            node_id,
            subgraph_path,
            child_run_uid,
            offset,
            limit,
            failed_only=failed_only,
            upstream_depth=upstream_depth,
        ),
        ensure_ascii=False,
    )


@tool('write_diagnosis', parse_docstring=True)
def write_diagnosis(content: str, runtime: ToolRuntime[WorkflowAssistantContext]) -> Command:
    """Write the completed workflow diagnosis to ``DIAGNOSIS.md``.

    Use this tool when the user requests diagnosis, troubleshooting, or an investigation
    into a workflow failure, following the diagnosis skill. Do not call it for ordinary
    questions, status or result explanations, or follow-up questions about an existing report.

    Args:
        content: Complete Markdown diagnosis report in the language required by the system prompt.
    """
    report = content.strip()
    if not report:
        message = 'The diagnosis report cannot be empty.'
        raise ValueError(message)

    project_root = Path(runtime.context.workdir).expanduser().resolve()
    agent_run_uid = uuid.UUID(runtime.context.agent_run_uid)
    output_dir = project_root / 'fixes' / str(agent_run_uid)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'DIAGNOSIS.md'
    temporary_path = output_dir / f'.DIAGNOSIS.{uuid.uuid4().hex}.tmp'
    metadata = (
        '---\n'
        'schema_version: 1\n'
        f'diagnosis_run_uid: {agent_run_uid}\n'
        f'source_run_uid: {runtime.context.source_run_uid}\n'
        f'workflow_uid: {runtime.context.workflow_uid or "null"}\n'
        f'project_id: {runtime.context.project_id}\n'
        '---\n\n'
    )
    try:
        temporary_path.write_text(f'{metadata}{report}\n', encoding='utf-8')
        temporary_path.replace(report_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    relative_path = report_path.relative_to(project_root).as_posix()
    return Command(
        update={
            'diagnosis_path': relative_path,
            'source_run_uid': runtime.context.source_run_uid,
            'workflow_uid': runtime.context.workflow_uid,
            'messages': [
                ToolMessage(
                    content=f'The diagnosis report has been saved to {relative_path}.',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
