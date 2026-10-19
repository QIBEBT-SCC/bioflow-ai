"""Read optional diagnosis reports for workflow questions and repair."""

import uuid
from pathlib import Path, PurePosixPath

from loguru import logger
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

import bioflow_ai.crud.agent as agent_crud
from bioflow_ai.models.agent import AgentRun


class WorkflowDiagnosisReport(BaseModel):
    """A readable report belonging to a completed assistant run."""

    model_config = ConfigDict(frozen=True)

    run_uid: uuid.UUID
    content: str


def _report_path(project_root: Path, diagnosis: AgentRun) -> Path | None:
    relative_path = (diagnosis.result_payload or {}).get('diagnosis_path')
    if not isinstance(relative_path, str) or not relative_path.strip():
        return None
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or '..' in relative.parts or relative.name != 'DIAGNOSIS.md':
        return None
    root = project_root.resolve()
    report_path = (root / Path(*relative.parts)).resolve()
    if not report_path.is_relative_to(root) or not report_path.is_file():
        return None
    return report_path


def load_latest_diagnosis_report(
    session: Session,
    *,
    session_uid: uuid.UUID,
    source_run_uid: uuid.UUID,
    project_root: Path,
) -> WorkflowDiagnosisReport | None:
    """Find the latest readable report, skipping questions and unavailable files."""
    for diagnosis in agent_crud.list_completed_diagnoses(session, session_uid, source_run_uid):
        try:
            report_path = _report_path(project_root, diagnosis)
            if report_path is None:
                continue
            content = report_path.read_text(encoding='utf-8')
        except (OSError, UnicodeError, ValueError):
            logger.warning('Workflow diagnosis report for {} is unavailable; checking older reports.', diagnosis.uid)
            continue
        if content.strip():
            return WorkflowDiagnosisReport(run_uid=diagnosis.uid, content=content)
    return None
