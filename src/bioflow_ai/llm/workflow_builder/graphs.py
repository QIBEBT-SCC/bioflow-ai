"""Adapt builder state into the document consumed by the workflow validator."""

from collections.abc import Mapping
from typing import Any

from sqlalchemy.orm import Session

from bioflow_ai.infrastructure.database import engine
from bioflow_ai.llm.workflow_builder.drafts import snapshot_active_workflow
from bioflow_ai.schemas.tools import WorkflowToolDefinition
from bioflow_ai.schemas.workflow_build import BuildEdge, BuildNode
from bioflow_ai.services.workflow.validation.adapters import build_graph
from bioflow_ai.services.workflow.validation.resolution import extract_draft_marks, resolve_workflow
from bioflow_ai.services.workflow.validation.types import WorkflowDocument
from bioflow_ai.utils.identifiers import as_uuid


def workflow_document(
    state: Mapping[str, Any],
    *,
    graph: Any | None = None,
    nodes: list[BuildNode] | None = None,
    edges: list[BuildEdge] | None = None,
    project_id: int | None = None,
    catalog: dict[str, WorkflowToolDefinition] | None = None,
    fresh: bool = False,
    outputs: list[dict[str, Any]] | None = None,
) -> WorkflowDocument:
    """Copy the full candidate graph, preserving all editing identities."""
    supplied_graph = graph is not None
    graph = (
        graph
        if graph is not None
        else build_graph(
            list(state.get('exist_nodes', [])) if nodes is None else nodes,
            list(state.get('exist_edges', [])) if edges is None else edges,
        )
    )
    active = state.get('workflow_drafts', {}).get(state.get('active_workflow_id'))
    scope = getattr(active, 'execution_scope', None) or (active or {}).get('execution_scope', 'sample_level')
    interface = active.get('interface') if isinstance(active, dict) else getattr(active, 'interface', None)
    if interface is not None and not supplied_graph:
        graph['interface'] = interface.model_dump(mode='json') if hasattr(interface, 'model_dump') else interface
    choices = state.get('subgraph_outputs', []) if outputs is None else outputs
    if not supplied_graph and (choices or outputs is not None):
        graph['interface'] = {
            'inputs': [],
            'outputs': [
                {
                    'id': choice['id'],
                    'name': choice['name'],
                    'description': choice.get('description', ''),
                    'source': choice['source']
                    if 'source' in choice
                    else {'node_id': choice.get('source_node'), 'handle': choice.get('source_handle')},
                }
                for choice in choices
            ],
        }
    draft_marks = extract_draft_marks(
        [
            *(build_graph(draft.nodes, draft.edges) for draft in snapshot_active_workflow(state).values()),
            graph,
        ]
    )
    target_workflow_uid = state.get('target_workflow_uid')
    with Session(engine) as session:
        return resolve_workflow(
            graph,
            session,
            execution_scope=scope,
            subgraph_purpose=state.get('subgraph_purpose', 'ordinary'),
            project_id=project_id,
            catalog=catalog,
            fresh=fresh,
            workflow_uid=as_uuid(target_workflow_uid) if target_workflow_uid is not None else None,
            draft_marks=draft_marks,
        )
