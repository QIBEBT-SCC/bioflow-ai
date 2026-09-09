"""Prepare the diagnosis and current template used by a workflow-fixer run."""

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

import app.crud.agent as agent_crud
import app.crud.workflow as workflow_crud
from app.models.agent import AgentRun
from app.models.workflow import WorkflowRun
from llm.workflow_builder.loading import load_workflow_draft

MISSING_DIAGNOSIS_ERROR = 'No completed workflow diagnosis exists for this run in the current conversation.'
MISSING_DIAGNOSIS_PATH_ERROR = 'The latest completed diagnosis has no diagnosis report. Run diagnosis again first.'
MISSING_DIAGNOSIS_FILE_ERROR = 'The latest diagnosis report no longer exists. Run diagnosis again first.'
MISSING_TEMPLATE_ERROR = 'The source run is not linked to a workflow template, so it cannot be fixed automatically.'
TEMPLATE_NOT_FOUND_ERROR = 'The source workflow template no longer exists.'


@dataclass(frozen=True)
class WorkflowFixPreparation:
    """Inputs loaded once before starting a Fix graph."""

    state: dict[str, Any]
    diagnosis_report: str
    diagnosis_run_uid: uuid.UUID
    workflow_uid: uuid.UUID
    node_id_mapping: dict[str, str]


def _report_path(project_root: Path, diagnosis: AgentRun) -> Path:
    relative_path = (diagnosis.result_payload or {}).get('diagnosis_path')
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError(MISSING_DIAGNOSIS_PATH_ERROR)
    report_path = (project_root / relative_path).resolve()
    try:
        report_path.relative_to(project_root)
    except ValueError as error:
        raise ValueError(MISSING_DIAGNOSIS_PATH_ERROR) from error
    if not report_path.is_file():
        raise ValueError(MISSING_DIAGNOSIS_FILE_ERROR)
    return report_path


def prepare_workflow_fix(
    session: Session,
    *,
    fix_run: AgentRun,
    source_run: WorkflowRun,
    project_root: Path,
) -> WorkflowFixPreparation:
    """Load the latest same-session diagnosis and the current persisted template."""
    diagnosis = agent_crud.get_latest_completed_diagnosis(session, fix_run.session_uid, source_run.uid)
    if diagnosis is None:
        raise ValueError(MISSING_DIAGNOSIS_ERROR)

    report_path = _report_path(project_root, diagnosis)
    diagnosis_report = report_path.read_text(encoding='utf-8')
    if source_run.workflow_uid is None:
        raise ValueError(MISSING_TEMPLATE_ERROR)
    workflow = workflow_crud.get(session, source_run.workflow_uid)
    if workflow is None:
        raise ValueError(TEMPLATE_NOT_FOUND_ERROR)
    if workflow.owner_id != fix_run.agent_session.owner_id:
        raise ValueError(TEMPLATE_NOT_FOUND_ERROR)

    loaded = load_workflow_draft(
        workflow.workflow,
        name=workflow.name,
        description=workflow.description,
        execution_scope=workflow.execution_scope,
        session=session,
    )
    draft = loaded.draft
    state: dict[str, Any] = {
        'diagnosis_run_uid': str(diagnosis.uid),
        'target_workflow_uid': str(workflow.uid),
        'diagnosis_report': diagnosis_report,
        'original_workflow': workflow.workflow.model_dump(mode='json'),
        'original_node_id_mapping': loaded.node_id_mapping,
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
    return WorkflowFixPreparation(
        state=state,
        diagnosis_report=diagnosis_report,
        diagnosis_run_uid=diagnosis.uid,
        workflow_uid=workflow.uid,
        node_id_mapping=loaded.node_id_mapping,
    )
