"""Prepare the current template and optional diagnosis for a workflow-fixer run."""

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from loguru import logger
from sqlalchemy.orm import Session

import app.crud.agent as agent_crud
import app.crud.workflow as workflow_crud
from app.models.agent import AgentRun
from app.models.workflow import NodeRun, WorkflowRun
from app.schemas.react_flow import WorkflowDefinition
from llm.workflow_builder.loading import load_workflow_draft

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


def _report_path(project_root: Path, diagnosis: AgentRun) -> Path | None:
    relative_path = (diagnosis.result_payload or {}).get('diagnosis_path')
    if not isinstance(relative_path, str) or not relative_path.strip():
        return None
    report_path = (project_root / relative_path).resolve()
    try:
        report_path.relative_to(project_root)
    except ValueError:
        return None
    if not report_path.is_file():
        return None
    return report_path


def prepare_workflow_fix(
    session: Session,
    *,
    fix_run: AgentRun,
    source_run: WorkflowRun,
    project_root: Path,
) -> WorkflowFixPreparation:
    """Load the current template and any usable same-session diagnosis report."""
    diagnosis = agent_crud.get_latest_completed_diagnosis(session, fix_run.session_uid, source_run.uid)
    report_path = _report_path(project_root, diagnosis) if diagnosis is not None else None
    diagnosis_report: str | None = None
    if report_path is not None:
        try:
            diagnosis_report = report_path.read_text(encoding='utf-8')
        except (OSError, UnicodeError):
            logger.warning('Workflow diagnosis report {} is unavailable; continuing Fix without it.', report_path)
    diagnosis_run_uid = diagnosis.uid if diagnosis is not None and diagnosis_report is not None else None
    template_run = source_run
    source_subgraph_path: list[str] = []
    if source_run.parent_node_uid is not None:
        parent_node = session.get(NodeRun, source_run.parent_node_uid)
        parent_run = session.get(WorkflowRun, parent_node.run_uid) if parent_node else None
        if (
            parent_node is None
            or parent_run is None
            or parent_node.generation != source_run.parent_generation
            or parent_run.generation != source_run.parent_generation
            or parent_node.node_type != 'foreach'
        ):
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
