"""Project workflow orchestration services."""

import io
import uuid
import zipfile
from collections.abc import Generator
from pathlib import Path

from anyio import Path as AsyncPath
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import delete
from sqlmodel import Session, col, select
from starlette.status import HTTP_404_NOT_FOUND

from app.crud import project_workflow as project_workflow_crud
from app.crud import run_instance as run_crud
from app.crud import workflow as wf_crud
from app.models.workflow import ExecutionScope, RunInstance, TaskInstance, TaskMonitor

_AGENTS_MD = Path(__file__).parents[2] / 'presets' / 'download_resources' / 'Agent.md'
_PACKAGE_SUBDIR_MAP = {'summary': 'summary', 'downstream': 'raw_data'}


def remove_workflow_from_project(session: Session, project_id: int, workflow_uid: uuid.UUID) -> None:
    """Remove a workflow from a project and clean up its run data."""
    if not project_workflow_crud.is_workflow_in_project(session, project_id, workflow_uid):
        raise HTTPException(HTTP_404_NOT_FOUND, 'Workflow not found in this project')

    _delete_workflow_runs(session, project_id, workflow_uid)
    project_workflow_crud.remove_workflow_from_project(session, project_id, workflow_uid)


def _delete_workflow_runs(session: Session, project_id: int, workflow_uid: uuid.UUID) -> None:
    """Delete run instances and dependent task data for a project workflow."""
    run_instance_uids = session.exec(
        select(RunInstance.uid).where(
            col(RunInstance.project_id) == project_id,
            col(RunInstance.workflow_uid) == workflow_uid,
        )
    ).all()

    if not run_instance_uids:
        return

    task_instance_uids = session.exec(
        select(TaskInstance.uid).where(col(TaskInstance.instance_uid).in_(run_instance_uids))
    ).all()

    if task_instance_uids:
        session.exec(delete(TaskMonitor).where(col(TaskMonitor.task_uid).in_(task_instance_uids)))

    session.exec(delete(TaskInstance).where(col(TaskInstance.instance_uid).in_(run_instance_uids)))
    session.exec(delete(RunInstance).where(col(RunInstance.uid).in_(run_instance_uids)))


async def package_workflow_runs(session: Session, project_id: int, workflow_uid: uuid.UUID) -> StreamingResponse:
    """Build a zip response for all run outputs of a project workflow."""
    if not project_workflow_crud.is_workflow_in_project(session, project_id, workflow_uid):
        raise HTTPException(HTTP_404_NOT_FOUND, 'Workflow not found in this project')

    workflow = wf_crud.get(session, workflow_uid)  # type: ignore
    runs = run_crud.get_runs_by_project_and_workflow(session, project_id, workflow_uid)

    if workflow.execution_scope == ExecutionScope.PROJECT_LEVEL:
        if not runs:
            raise HTTPException(HTTP_404_NOT_FOUND, 'No exist run instance for this workflow!')
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
