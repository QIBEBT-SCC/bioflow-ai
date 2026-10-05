"""Validation, expansion and presentation helpers for snapshot subgraphs."""

from copy import deepcopy
from typing import Any
from urllib.parse import quote

from bioflow_ai.schemas.react_flow import WorkflowDefinition

MAX_SUBGRAPH_DEPTH = 32


class SubgraphError(ValueError):
    """A subgraph interface cannot be resolved."""


def definition_id(path: tuple[str, ...]) -> str:
    """Keep legacy root IDs; namespace descendants without delimiter collisions."""
    return path[0] if len(path) == 1 else '@subgraph/' + '/'.join(quote(part, safe='') for part in path)


def flatten_workflow(workflow: WorkflowDefinition) -> WorkflowDefinition:
    """Expand an already-validated document into executable node identities."""
    from bioflow_ai.services.workflow.validation._expansion import _flatten_workflow

    return _flatten_workflow(workflow)


def aggregate_status(statuses: list[str]) -> str:
    """Aggregate executable descendants; failure remains visible during other work."""
    for status in ('failed', 'blocked'):
        if status in statuses:
            return status
    if not statuses or all(status == 'succeeded' for status in statuses):
        return 'succeeded'
    if 'running' in statuses or 'succeeded' in statuses:
        return 'running'
    return next((status for status in ('queued', 'ready', 'pending') if status in statuses), 'pending')


def inject_run_data(
    workflow: WorkflowDefinition,
    by_definition: dict[str, dict[str, Any]],
    preview_by_definition: dict[str, dict[str, Any]] | None = None,
) -> list[dict]:
    """Decorate the immutable run snapshot, including all nested descendants."""

    def visit(graph: dict, path: tuple[str, ...]) -> list[str]:
        statuses = []
        for node in graph['nodes']:
            node_path = (*path, node['id'])
            node['data'].pop('run_data', None)
            node['data'].pop('preview_data', None)
            if preview := (preview_by_definition or {}).get(definition_id(node_path)):
                node['data']['preview_data'] = preview
            if node['type'] == 'subgraph' and definition_id(node_path) not in by_definition:
                descendants = visit(node['data']['workflow'], node_path)
                node['data']['run_data'] = {'status': aggregate_status(descendants)}
                statuses.extend(descendants)
            elif data := by_definition.get(definition_id(node_path)):
                node['data']['run_data'] = data
                statuses.append(data['status'])
        return statuses

    graph = deepcopy(workflow.model_dump(mode='json'))
    visit(graph, ())
    return graph['nodes']
