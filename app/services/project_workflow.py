"""Project workflow orchestration services."""

import io
import uuid
import zipfile
from collections.abc import Generator
from pathlib import Path

from anyio import Path as AsyncPath
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.crud import project_workflow as project_workflow_crud
from app.crud import run_instance as run_crud
from app.crud import workflow as wf_crud
from app.models.workflow import ProjectWorkflow, RunInstance, TaskInstance, TaskMonitor
from app.schemas.project_workflow import ProjectWorkflowPublic
from app.schemas.workflow_config import ExecutionScope
from app.services._transaction import committing

_AGENTS_MD = Path(__file__).parents[2] / 'presets' / 'download_resources' / 'Agent.md'
_PACKAGE_SUBDIR_MAP = {'summary': 'summary', 'downstream': 'raw_data'}
_WORKFLOW_NOT_FOUND = 'Workflow not found'
_WORKFLOW_CONFLICT = 'Workflow already exists in this project'
_PROJECT_WORKFLOW_NOT_FOUND = 'Workflow not found in this project'
_RUN_NOT_FOUND = 'No run instance exists for this workflow'


class ProjectWorkflowError(Exception):
    """Base exception for project-workflow application services."""


class ProjectWorkflowNotFoundError(ProjectWorkflowError):
    """A requested workflow or project association does not exist."""


class ProjectWorkflowConflictError(ProjectWorkflowError):
    """A workflow is already associated with the project."""


def add_workflow_to_project(
    session: Session,
    project_id: int,
    workflow_uid: uuid.UUID,
) -> ProjectWorkflowPublic:
    """Associate an existing workflow template with a project."""
    workflow = wf_crud.get(session, workflow_uid)
    if workflow is None:
        raise ProjectWorkflowNotFoundError(_WORKFLOW_NOT_FOUND)
    if project_workflow_crud.exists(session, project_id, workflow_uid):
        raise ProjectWorkflowConflictError(_WORKFLOW_CONFLICT)

    with committing(session):
        association = project_workflow_crud.insert(
            session,
            ProjectWorkflow(project_id=project_id, workflow_uid=workflow_uid),
        )

    return ProjectWorkflowPublic(
        workflow_uid=workflow.uid,
        workflow_name=workflow.name,
        execution_scope=workflow.execution_scope,
        import_time=association.import_time,
        enabled=association.enabled,
        auto_summary=workflow.auto_summary,
    )


def list_project_workflows(
    session: Session,
    project_id: int,
    limit: int = 20,
    offset: int = 0,
) -> list[ProjectWorkflowPublic]:
    """Return workflow templates imported into a project."""
    return [
        ProjectWorkflowPublic(
            workflow_uid=association.workflow_uid,
            workflow_name=workflow.name,
            execution_scope=workflow.execution_scope,
            import_time=association.import_time,
            enabled=association.enabled,
            auto_summary=workflow.auto_summary,
        )
        for association, workflow in project_workflow_crud.list_with_workflows(
            session,
            project_id,
            limit,
            offset,
        )
    ]


def remove_workflow_from_project(session: Session, project_id: int, workflow_uid: uuid.UUID) -> None:
    """Remove a workflow from a project and clean up its run data."""
    if not project_workflow_crud.exists(session, project_id, workflow_uid):
        raise ProjectWorkflowNotFoundError(_PROJECT_WORKFLOW_NOT_FOUND)

    with committing(session):
        _delete_workflow_runs(session, project_id, workflow_uid)
        project_workflow_crud.delete(session, project_id, workflow_uid)


def _delete_workflow_runs(session: Session, project_id: int, workflow_uid: uuid.UUID) -> None:
    """Delete run instances and dependent task data for a project workflow."""
    run_instance_uids = session.scalars(
        select(RunInstance.uid).where(
            RunInstance.project_id == project_id,
            RunInstance.workflow_uid == workflow_uid,
        )
    ).all()

    if not run_instance_uids:
        return

    task_instance_uids = session.scalars(
        select(TaskInstance.uid).where(TaskInstance.instance_uid.in_(run_instance_uids))
    ).all()

    if task_instance_uids:
        session.execute(delete(TaskMonitor).where(TaskMonitor.task_uid.in_(task_instance_uids)))

    session.execute(delete(TaskInstance).where(TaskInstance.instance_uid.in_(run_instance_uids)))
    session.execute(delete(RunInstance).where(RunInstance.uid.in_(run_instance_uids)))


async def package_workflow_runs(
    session: Session,
    project_id: int,
    workflow_uid: uuid.UUID,
    *,
    owner_id: int | None = None,
) -> StreamingResponse:
    """Build a zip response for all run outputs of a project workflow."""
    if not project_workflow_crud.exists(session, project_id, workflow_uid):
        raise ProjectWorkflowNotFoundError(_PROJECT_WORKFLOW_NOT_FOUND)

    workflow = wf_crud.get(session, workflow_uid)
    if workflow is None:
        raise ProjectWorkflowNotFoundError(_WORKFLOW_NOT_FOUND)
    runs = run_crud.get_runs_by_project_and_workflow(
        session,
        project_id,
        workflow_uid,
        owner_id=owner_id,
    )

    if workflow.execution_scope == ExecutionScope.PROJECT_LEVEL:
        if not runs:
            raise ProjectWorkflowNotFoundError(_RUN_NOT_FOUND)
        run_dirs = [Path(runs[0].base_dir)] if runs[0].base_dir and await AsyncPath(runs[0].base_dir).exists() else []
        zip_iter = _iter_project_level_zip(run_dirs)
    else:
        run_dirs = [Path(run.base_dir) for run in runs if run.base_dir and await AsyncPath(run.base_dir).exists()]
        zip_iter = _iter_sample_level_zip(run_dirs)

    return StreamingResponse(
        zip_iter,
        media_type='application/zip',
        headers={'Content-Disposition': f'attachment; filename="{workflow.name}_results.zip"'},
    )


def _iter_project_level_zip(run_dirs: list[Path]) -> Generator[bytes, None, None]:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        if run_dirs:
            base_dir = run_dirs[0]
            for disk_name, arc_name in _PACKAGE_SUBDIR_MAP.items():
                _write_subdir(zf, base_dir / disk_name, Path(arc_name))
            script = base_dir / 'run_downstream.sh'
            if script.exists():
                zf.write(script, arcname='run_downstream.sh')
        _write_agent_files(zf)
    buf.seek(0)
    yield from buf


def _iter_sample_level_zip(run_dirs: list[Path]) -> Generator[bytes, None, None]:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        root_script_written = False
        for sample_dir in run_dirs:
            sample_name = sample_dir.name
            for disk_name, arc_name in _PACKAGE_SUBDIR_MAP.items():
                _write_subdir(zf, sample_dir / disk_name, Path(arc_name) / sample_name)
            if not root_script_written:
                script = sample_dir / 'run_downstream.sh'
                if script.exists():
                    zf.write(script, arcname='run_downstream.sh')
                    root_script_written = True
        _write_agent_files(zf)
    buf.seek(0)
    yield from buf


def _write_subdir(zf: zipfile.ZipFile, sub_path: Path, arc_root: Path) -> None:
    if not sub_path.exists():
        return

    for file in sub_path.rglob('*'):
        if file.is_file():
            zf.write(file, arcname=arc_root / file.relative_to(sub_path))


def _write_agent_files(zf: zipfile.ZipFile) -> None:
    if not _AGENTS_MD.exists():
        return

    zf.writestr('AGENTS.md', _AGENTS_MD.read_text(encoding='utf-8'))
    zf.writestr('CLAUDE.md', '@AGENTS.md')
