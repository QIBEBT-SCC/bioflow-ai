"""Persistence helpers for workflow-builder drafts."""

import datetime
import uuid

from sqlalchemy.orm import Session

from app.core.session import engine
from app.models.workflow import Workflow
from app.schemas.workflow_config import WorkflowType
from app.services import workflow_catalog
from llm.states import WorkflowDraft
from llm.workflow_builder.node_ids import canonicalize_build_graph, has_canonical_node_ids
from llm.workflow_builder.serialization import build_workflow_definition, to_execution_scope


async def save_workflow_drafts(drafts: list[WorkflowDraft], user_id: int) -> list[Workflow]:
    """Persist workflow drafts and index their descriptions for retrieval."""
    now = datetime.datetime.now(datetime.UTC)
    workflows = []
    with Session(engine) as session:
        for draft in drafts:
            if has_canonical_node_ids(draft.nodes):
                nodes = draft.nodes
                edges = draft.edges
            else:
                nodes, edges, _ = canonicalize_build_graph(draft.nodes, draft.edges)
            workflow_define = build_workflow_definition(nodes, edges)
            workflow = Workflow(
                name=f'{now.strftime("%y%m%d")}_{draft.name}(AI)',
                description=draft.description,
                owner_id=user_id,
                wf_type=WorkflowType.TEMPLATE,
                execution_scope=to_execution_scope(draft.execution_scope),
                workflow=workflow_define,
                public=True,
                auto_summary=True,
            )
            workflows.append(workflow)

        return await workflow_catalog.create_workflows(session, workflows)


def workflow_uids(workflows: list[Workflow]) -> list[uuid.UUID]:
    """Return saved workflow UIDs with a concrete UUID type."""
    return [workflow.uid for workflow in workflows]
