"""Draft-state helpers for the workflow-builder agent."""

from collections.abc import Mapping
from typing import Any, Literal

from bioflow_ai.llm.states import BuildEdge, BuildNode, WorkflowDraft
from bioflow_ai.utils.llm.uid_mapper import generate_char_index

WorkflowScope = Literal['sample_level', 'project_level']


def get_workflow_drafts(state: Mapping[str, Any]) -> dict[str, WorkflowDraft]:
    """Return workflow drafts from state, normalizing serialized dictionaries."""
    drafts = state.get('workflow_drafts', {})
    normalized = {}
    for draft_id, value in drafts.items():
        draft = value if isinstance(value, WorkflowDraft) else WorkflowDraft.model_validate(value)
        minimum_next_index = len(draft.nodes) + 1
        if draft.next_node_index < minimum_next_index:
            draft = draft.model_copy(update={'next_node_index': minimum_next_index}, deep=True)
        normalized[draft_id] = draft
    return normalized


def next_draft_id(drafts: Mapping[str, WorkflowDraft]) -> str:
    """Generate the next stable draft ID."""
    index = len(drafts) + 1
    while True:
        draft_id = f'wf_{generate_char_index(index)}'
        if draft_id not in drafts:
            return draft_id
        index += 1


def snapshot_active_workflow(state: Mapping[str, Any]) -> dict[str, WorkflowDraft]:
    """Copy the active panel nodes and edges into the active workflow draft."""
    drafts = get_workflow_drafts(state)
    active_workflow_id = state.get('active_workflow_id')
    if not active_workflow_id:
        return drafts

    draft = drafts.get(active_workflow_id)
    if draft is None:
        draft = WorkflowDraft(draft_id=active_workflow_id, name=active_workflow_id)

    drafts[active_workflow_id] = draft.model_copy(
        update={
            'nodes': state.get('exist_nodes', []),
            'edges': state.get('exist_edges', []),
            'next_node_index': max(
                int(state.get('next_node_index', draft.next_node_index)),
                len(state.get('exist_nodes', [])) + 1,
            ),
        },
        deep=True,
    )
    return drafts


def build_workflow_draft(
    *,
    draft_id: str,
    name: str,
    description: str,
    execution_scope: WorkflowScope,
    nodes: list[BuildNode] | None = None,
    edges: list[BuildEdge] | None = None,
    next_node_index: int | None = None,
) -> WorkflowDraft:
    """Create a WorkflowDraft with optional panel contents."""
    draft_nodes = nodes or []
    return WorkflowDraft(
        draft_id=draft_id,
        name=name,
        description=description,
        execution_scope=execution_scope,
        nodes=draft_nodes,
        edges=edges or [],
        next_node_index=max(next_node_index or 1, len(draft_nodes) + 1),
    )
