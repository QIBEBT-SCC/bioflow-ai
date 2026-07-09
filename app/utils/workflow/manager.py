"""Run instance manager.

Responsible for batch creating and managing workflow run instances.
"""

import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from loguru import logger
from sqlmodel import Session
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND, HTTP_409_CONFLICT

import app.crud.resource.sample as sample_crud
from app.core.config import get_settings
from app.crud import project as proj_crud
from app.crud import run_instance as run_crud
from app.crud import workflow as wf_crud
from app.models.user import User
from app.models.workflow import ExecutionScope, RunInstance, Status
from app.schemas.project_workflow import WorkflowRunResult
from app.utils.input_format import to_safe_path_segment
from app.utils.workflow.scheduler import build_task_list, send_task, update_node_id

STORAGE_PATH = get_settings().persistent.storage.STORAGE_PATH


def _get_project_run_base_dir(project_id: int, workflow_name: str, sample_name: str | None = None) -> Path:
    """Return the run output directory under the project's storage root."""
    base_dir = Path(STORAGE_PATH).absolute() / 'proj' / f'{project_id}' / 'runs' / to_safe_path_segment(workflow_name)
    if sample_name is not None:
        base_dir = base_dir / to_safe_path_segment(sample_name)
    return base_dir


def create_runs_from_samples(
    session: Session,
    project_id: int,
    workflow_uid: uuid.UUID,
    sample_uids: list[uuid.UUID],
    current_user: User,
    run_name_prefix: Optional[str] = None,
    *,
    auto_summary: bool = False,
    summary_prompt: Optional[str] = None,
) -> WorkflowRunResult:
    """Batch create run instances from a list of samples.

    Creates an independent RunInstance for each sample and submits tasks for execution.

    Args:
        session: Database session
        project_id: Project ID
        workflow_uid: Workflow template UID
        sample_uids: List of sample UIDs
        current_user: Current user
        run_name_prefix: Run name prefix (optional)

    Returns:
        WorkflowRunResult: Contains the list of created run instance UIDs

    Raises:
        HTTPException: Project, workflow, or sample not found
    """
    # Get project
    project = proj_crud.get(session, current_user.id, project_id)
    if not project:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Project not found')

    # Get workflow template
    workflow = wf_crud.get(session, workflow_uid)
    if not workflow:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Workflow not found')

    # Get all samples in this project
    samples = sample_crud.get_by_uids(session, sample_uids, project_id=project_id)
    if len(samples) != len(sample_uids):
        raise HTTPException(HTTP_404_NOT_FOUND, 'Some samples not found')

    # Create run instances for each sample
    run_uids = []

    for sample in samples:
        # Check if a run instance already exists for this sample+workflow combination
        existing_run = run_crud.get_latest_run_by_sample_and_workflow(session, sample.uid, workflow_uid)
        # Update workflow node IDs
        workflow_definition = update_node_id(workflow.workflow.model_copy(deep=True))
        base_dir = _get_project_run_base_dir(project_id, workflow.name, sample.sample_name)

        if existing_run:
            # If running, reject rerun
            if existing_run.status == Status.RUNNING:
                raise HTTPException(
                    HTTP_409_CONFLICT,
                    f'Sample {sample.sample_name} already has a running instance, please wait for it to finish',
                )

            # Clear old execution data and reuse the existing instance
            run_crud.clear_run_for_rerun(session, existing_run, workflow_definition)
            if existing_run.base_dir and Path(existing_run.base_dir).exists():
                shutil.rmtree(existing_run.base_dir)
            existing_run.base_dir = str(base_dir)
            session.add(existing_run)
            logger.debug(f'Reusing run instance {existing_run.uid} (Sample: {sample.sample_name})')

            run = existing_run
        else:
            # Build run name
            if run_name_prefix:
                run_name = f'{run_name_prefix}-{sample.sample_name}'
            else:
                run_name = f'{project.name}-{workflow.name}-{sample.sample_name}'

            # Create a new run instance
            run = RunInstance(
                project_id=project_id,
                workflow_uid=workflow_uid,
                sample_uid=sample.uid,
                owner_id=current_user.id,
                name=run_name,
                workflow=workflow_definition,
                base_dir=str(base_dir),
                status=Status.WAITING,
            )

            session.add(run)
            session.flush()  # 获取 run.uid

        # Build task list
        task_list = build_task_list(workflow_definition)

        # Commit the current run instance to the database
        session.commit()
        logger.debug(f'Run instance {run.uid} committed to database')

        # Submit tasks to Celery
        send_task(
            task_list,
            session,
            project_id,
            sample.uid,
            sample.sample_name,
            current_user,
            run.base_dir,
            run.uid,
            auto_summary=auto_summary,
            summary_prompt=summary_prompt,
        )

        run_uids.append(run.uid)

    return WorkflowRunResult(run_uids=run_uids, count=len(run_uids))


def create_project_level_run(
    session: Session,
    project_id: int,
    workflow_uid: uuid.UUID,
    current_user: User,
    run_name_prefix: Optional[str] = None,
    *,
    auto_summary: bool = False,
    summary_prompt: Optional[str] = None,
) -> WorkflowRunResult:
    """Create a project-level workflow run instance.

    Project-level workflows are not bound to samples, run only once, and are referenced using proj:keyword.

    Args:
        session: Database session
        project_id: Project ID
        workflow_uid: Workflow template UID
        current_user: Current user
        run_name_prefix: Run name prefix (optional)

    Returns:
        WorkflowRunResult: Contains the created run instance UID

    Raises:
        HTTPException: Project or workflow not found, or workflow is not project-level
    """
    # Get project
    project = proj_crud.get(session, current_user.id, project_id)
    if not project:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Project not found')

    # Get workflow template
    workflow = wf_crud.get(session, workflow_uid)
    if not workflow:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Workflow not found')

    # Verify the workflow is project-level
    if workflow.execution_scope != ExecutionScope.PROJECT_LEVEL:
        raise HTTPException(HTTP_400_BAD_REQUEST, 'Workflow is not project-level')

    # Update workflow node IDs
    workflow_definition = update_node_id(workflow.workflow.model_copy(deep=True))
    base_dir = _get_project_run_base_dir(project_id, workflow.name)

    # Check if a run instance already exists for this project+workflow combination
    existing_run = run_crud.get_latest_run_by_project_and_workflow(session, project_id, workflow_uid)

    if existing_run:
        # If running, reject rerun
        if existing_run.status == Status.RUNNING:
            raise HTTPException(
                HTTP_409_CONFLICT,
                'Project already has a running instance of this workflow',
            )

        # Clear old execution data and reuse the existing instance
        run_crud.clear_run_for_rerun(session, existing_run, workflow_definition)
        if existing_run.base_dir and Path(existing_run.base_dir).exists():
            shutil.rmtree(existing_run.base_dir)
        existing_run.base_dir = str(base_dir)
        session.add(existing_run)
        logger.debug(f'Reusing project-level run instance {existing_run.uid}')

        run = existing_run
    else:
        # Build run name
        run_name = f'{run_name_prefix}-{project.name}' if run_name_prefix else f'{project.name}-{workflow.name}'

        # Create a new run instance (sample_uid=None)
        run = RunInstance(
            project_id=project_id,
            workflow_uid=workflow_uid,
            sample_uid=None,  # Project-level workflows are not bound to samples
            owner_id=current_user.id,
            name=run_name,
            workflow=workflow_definition,
            base_dir=str(base_dir),
            status=Status.WAITING,
        )

        session.add(run)
        session.flush()

    # Build task list
    task_list = build_task_list(workflow_definition)

    # Commit the current run instance to the database
    session.commit()
    logger.debug(f'Project-level run instance {run.uid} committed to database')

    # Submit tasks to Celery (no sample context)
    send_task(
        task_list,
        session,
        project_id,
        None,
        None,
        current_user,
        run.base_dir,
        run.uid,
        auto_summary=auto_summary,
        summary_prompt=summary_prompt,
    )

    return WorkflowRunResult(run_uids=[run.uid], count=1)
