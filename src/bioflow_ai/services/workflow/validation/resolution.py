"""Database/cache orchestration outside the pure workflow diagnostic kernel."""

import uuid
from collections.abc import Generator
from typing import Any

from sqlalchemy.orm import Session

from bioflow_ai.crud import project_file_mapping as mapping_crud
from bioflow_ai.crud.resource import sample as sample_crud
from bioflow_ai.schemas.tools import WorkflowToolDefinition
from bioflow_ai.services.workflow.subgraphs import MAX_SUBGRAPH_DEPTH
from bioflow_ai.services.workflow.validation.adapters import copy_graph
from bioflow_ai.services.workflow.validation.types import (
    ExecutionScopeName,
    SubgraphPurpose,
    ValidationContext,
    WorkflowDocument,
)


def graph_nodes(graph: Any) -> Generator[dict[str, Any], None, None]:
    """Visit available nodes without requiring an otherwise valid graph."""
    pending = [(copy_graph(graph), 0)]
    while pending:
        value, depth = pending.pop()
        if not isinstance(value, dict) or depth > MAX_SUBGRAPH_DEPTH:
            continue
        nodes = value.get('nodes', [])
        if not isinstance(nodes, list):
            continue
        for node in nodes:
            if not isinstance(node, dict):
                continue
            yield node
            data = node.get('data')
            if node.get('type') in {'subgraph', 'foreach'} and isinstance(data, dict):
                pending.append((data.get('workflow'), depth + 1))


def collect_tool_uids(graph: Any) -> list[str]:
    """Collect valid, unique tool UIDs; invalid IDs are diagnosed by the kernel."""
    uids = {}
    for node in graph_nodes(graph):
        if node.get('type') != 'tool' or not isinstance(node.get('data'), dict):
            continue
        try:
            uid = str(uuid.UUID(str(node['data'].get('tool_uid', ''))))
        except ValueError:
            continue
        uids[uid] = None
    return list(uids)


def mark_storage_conflicts(
    session: Session,
    project_id: int,
    graph: Any,
    workflow_uid: uuid.UUID | None = None,
) -> dict[str, str]:
    """Report marks already used by static resources or by another workflow's dynamic outputs.

    ``workflow_uid`` identifies the saved workflow being checked; ``None`` means a new workflow, so every
    dynamic mark of an existing workflow conflicts. Marks left by deleted workflows can be replaced.
    """
    conflicts = {}
    checked = set()
    for node in graph_nodes(graph):
        kind, data = node.get('type'), node.get('data', {})
        if kind not in {'project_mark', 'sample_mark'} or not isinstance(data, dict):
            continue
        name = str(data.get('mark_name', ''))
        key = f'{kind}:{name}'
        if key in checked:
            continue
        checked.add(key)
        if kind == 'project_mark':
            mapping = mapping_crud.get_by_keyword(session, project_id, name)
            if mapping is not None and not mapping.is_dynamic:
                conflicts[key] = (
                    f'Cannot add mark `{name}` because `proj:{name}` already exists as an initial project resource.'
                )
            elif owned := mapping_crud.get_dynamic_owned_elsewhere(session, project_id, name, workflow_uid):
                conflicts[key] = (
                    f'Cannot add mark `{name}` because `proj:{name}` is already produced by workflow '
                    f'{owned.producer_workflow_uid}. Choose another mark name, or delete that workflow first.'
                )
            continue
        matches = sample_crud.get_files_by_tag_in_project(session, project_id, name, is_dynamic=False)
        if matches:
            names = ', '.join(sample_name for _, sample_name in matches)
            conflicts[key] = (
                f'Cannot add mark `{name}` because `sample:{name}` already exists on project sample(s): {names}.'
            )
        elif owned_files := sample_crud.get_dynamic_files_owned_elsewhere(session, project_id, name, workflow_uid):
            producer = owned_files[0][0].producer_workflow_uid
            names = ', '.join(sample_name for _, sample_name in owned_files)
            conflicts[key] = (
                f'Cannot add mark `{name}` because workflow {producer} already registers `sample:{name}` on '
                f'project sample(s): {names}. Choose another mark name, or delete that workflow first.'
            )
    return conflicts


def resolve_workflow(
    graph: Any,
    session: Session,
    *,
    execution_scope: ExecutionScopeName = 'sample_level',
    subgraph_purpose: SubgraphPurpose = 'ordinary',
    project_id: int | None = None,
    catalog: dict[str, WorkflowToolDefinition] | None = None,
    fresh: bool = False,
    workflow_uid: uuid.UUID | None = None,
) -> WorkflowDocument:
    """Build a resolved document; data freshness does not select validation rules."""
    from bioflow_ai.services.tools.cache import get_tool_definitions_sync

    definitions = {} if fresh else dict(catalog or {})
    uids = collect_tool_uids(graph)
    missing = uids if fresh else [uid for uid in uids if uid not in definitions]
    definitions.update(get_tool_definitions_sync(session, missing, fresh=fresh))
    conflicts = mark_storage_conflicts(session, project_id, graph, workflow_uid) if project_id is not None else {}
    return WorkflowDocument(
        graph=graph,
        catalog=definitions,
        context=ValidationContext(
            execution_scope=execution_scope, subgraph_purpose=subgraph_purpose, mark_conflicts=conflicts
        ),
    )
