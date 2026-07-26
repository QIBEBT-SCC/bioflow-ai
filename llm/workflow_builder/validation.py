"""Context-aware validation performed before adding a node to a workflow draft."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llm.workflow_builder.drafts import get_workflow_drafts

if TYPE_CHECKING:
    from collections.abc import Mapping

    from llm.states import BuildNode

MARK_NODE_TYPES = {'project_mark', 'sample_mark'}


def validate_node_addition(
    state: Mapping[str, Any],
    *,
    node_type: str,
    arguments: Mapping[str, Any],
) -> str | None:
    """Return an actionable error when a node is invalid for the active draft."""
    active_workflow_id = state.get('active_workflow_id')
    drafts = get_workflow_drafts(state)
    draft = drafts.get(active_workflow_id) if active_workflow_id else None
    if draft is None:
        return 'Create or select a workflow draft before adding nodes.'

    scope = draft.execution_scope
    if scope == 'sample_level' and node_type == 'project_mark':
        return 'Cannot add `project_mark` to a sample-level workflow. Use `sample_mark` instead.'
    if scope == 'sample_level' and node_type == 'resource_sample_mark_collection':
        return (
            'Cannot add `sample_mark_collection` to a sample-level workflow. '
            'Create a project-level aggregation workflow for cross-sample collection.'
        )
    if scope == 'project_level' and node_type == 'sample_mark':
        return 'Cannot add `sample_mark` to a project-level workflow. Use `project_mark` instead.'

    if node_type in MARK_NODE_TYPES:
        mark_name = str(arguments.get('mark_name', ''))
        existing_nodes: list[BuildNode] = state.get('exist_nodes', [])
        conflict = next(
            (
                node
                for node in existing_nodes
                if node.type in MARK_NODE_TYPES and str(node.data.get('mark_name', '')) == mark_name
            ),
            None,
        )
        if conflict is not None:
            return (
                f'Cannot add duplicate mark `{mark_name}`. Node {conflict.id} '
                f'(`{conflict.type}`) already uses this name in the active workflow.'
            )

    if (
        scope == 'project_level'
        and node_type == 'resource_file'
        and str(arguments.get('file_path', '')).startswith('sample:')
    ):
        return (
            'Cannot add a sample-bound `sample:` file reference to a project-level workflow. '
            'Use `sample_mark_collection` for cross-sample aggregation.'
        )

    return None
