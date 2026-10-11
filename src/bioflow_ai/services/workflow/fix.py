"""Prepare the current template and optional diagnosis for a workflow-fixer run."""

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from sqlalchemy.orm import Session

import bioflow_ai.crud.workflow as workflow_crud
from bioflow_ai.crud import node_run as node_run_crud
from bioflow_ai.crud import workflow_run as run_crud
from bioflow_ai.llm.workflow_builder.loading import load_workflow_draft
from bioflow_ai.models.agent import AgentRun
from bioflow_ai.models.workflow import WorkflowRun
from bioflow_ai.schemas.react_flow import WorkflowDefinition
from bioflow_ai.services.workflow.diagnosis import load_latest_diagnosis_report

MISSING_TEMPLATE_ERROR = 'The source run is not linked to a workflow template, so it cannot be fixed automatically.'
TEMPLATE_NOT_FOUND_ERROR = 'The source workflow template no longer exists.'
SOURCE_PATH_NOT_FOUND_ERROR = 'The source batch subgraph no longer matches the current workflow template.'


@dataclass(frozen=True)
class WorkflowFixPreparation:
    """Inputs loaded once before starting a Fix graph."""

    state: dict[str, Any]
    diagnosis_report: str | None
    diagnosis_run_uid: uuid.UUID | None
    workflow_uid: uuid.UUID
    node_id_mapping: dict[str, str]


def prepare_workflow_fix(
    session: Session,
    *,
    fix_run: AgentRun,
    source_run: WorkflowRun,
    project_root: Path,
) -> WorkflowFixPreparation:
    """Load the current template and any usable same-session diagnosis report."""
    diagnosis = load_latest_diagnosis_report(
        session,
        session_uid=fix_run.session_uid,
        source_run_uid=source_run.uid,
        project_root=project_root,
    )
    diagnosis_report = diagnosis.content if diagnosis is not None else None
    diagnosis_run_uid = diagnosis.run_uid if diagnosis is not None else None
    template_run = source_run
    source_subgraph_path: list[str] = []
    if source_run.parent_node_uid is not None:
        parent_node = node_run_crud.get(session, source_run.parent_node_uid)
        parent_run = run_crud.get(session, parent_node.run_uid) if parent_node else None
        if parent_node is None or parent_run is None or parent_node.node_type != 'foreach':
            raise ValueError(SOURCE_PATH_NOT_FOUND_ERROR)
        template_run = parent_run
        runtime_id = parent_node.definition_node_id
        source_subgraph_path = (
            [unquote(part) for part in runtime_id.removeprefix('@subgraph/').split('/')]
            if runtime_id.startswith('@subgraph/')
            else [runtime_id]
        )
    if template_run.workflow_uid is None:
        raise ValueError(MISSING_TEMPLATE_ERROR)
    workflow = workflow_crud.get(session, template_run.workflow_uid)
    if workflow is None:
        raise ValueError(TEMPLATE_NOT_FOUND_ERROR)
    if workflow.owner_id != fix_run.agent_session.owner_id:
        raise ValueError(TEMPLATE_NOT_FOUND_ERROR)
    graph = workflow.workflow
    run_graph = template_run.workflow
    for node_id in source_subgraph_path:
        target = next((node for node in graph.nodes if node.id == node_id and node.type == 'subgraph'), None)
        run_target = next((node for node in run_graph.nodes if node.id == node_id and node.type == 'subgraph'), None)
        if target is None or run_target is None or target.data.get('source_uid') != run_target.data.get('source_uid'):
            raise ValueError(SOURCE_PATH_NOT_FOUND_ERROR)
        graph = WorkflowDefinition.model_validate(target.data['workflow'])
        run_graph = WorkflowDefinition.model_validate(run_target.data['workflow'])

    loaded = load_workflow_draft(
        workflow.workflow,
        name=workflow.name,
        description=workflow.description,
        execution_scope=workflow.execution_scope,
        session=session,
    )
    draft = loaded.draft
    state: dict[str, Any] = {
        'source_run_uid': str(source_run.uid),
        'target_workflow_uid': str(workflow.uid),
        'source_subgraph_path': source_subgraph_path,
        'original_workflow': workflow.workflow.model_dump(mode='json'),
        'original_node_id_mapping': loaded.node_id_mapping,
        'subgraph_stack': [],
        'subgraph_source_baselines': {},
        'changed_subgraph_paths': [],
        'changed_workflow_uids': [],
        'active_workflow_id': draft.draft_id,
        'workflow_drafts': {draft.draft_id: draft},
        'exist_nodes': draft.nodes,
        'exist_edges': draft.edges,
        'next_node_index': draft.next_node_index,
        'fix_completed': False,
        'workflow_changed': False,
        'update_path': None,
        'updated': False,
    }
    if diagnosis_run_uid is not None:
        state['diagnosis_run_uid'] = str(diagnosis_run_uid)
        state['diagnosis_report'] = diagnosis_report
    return WorkflowFixPreparation(
        state=state,
        diagnosis_report=diagnosis_report,
        diagnosis_run_uid=diagnosis_run_uid,
        workflow_uid=workflow.uid,
        node_id_mapping=loaded.node_id_mapping,
    )
