import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_403_FORBIDDEN, HTTP_404_NOT_FOUND, HTTP_409_CONFLICT

from app.core.config import get_settings
from app.core.session import SessionDep
from app.crud import project as project_crud
from app.crud import project_file_mapping as project_file_mapping_crud
from app.crud import project_workflow as project_workflow_crud
from app.crud import projetc_tag as tag_crud
from app.crud import run_instance as run_crud
from app.crud import workflow as wf_crud
from app.crud.resource import sample as sample_crud
from app.models.project import (
    Project,
    ProjectTag,
)
from app.models.resource import Sample, SampleFiles
from app.schemas.common import Paginated, Statistics
from app.schemas.project import (
    ProjectCreateProp,
    ProjectFileMappingCreate,
    ProjectFileMappingPublic,
    ProjectFileMappingUpdate,
    ProjectPublic,
    ProjectSort,
    ProjectUpdate,
    TagProp,
    TagWithCount,
)
from app.schemas.project_workflow import (
    ProjectWorkflowAdd,
    ProjectWorkflowPublic,
    WorkflowRunRequest,
    WorkflowRunResult,
)
from app.schemas.resource import (
    SampleBatchImportResult,
    SampleCreate,
    SampleFileCreate,
    SampleFilePublic,
    SampleListItem,
    SamplePublic,
    SampleUpdate,
)
from app.schemas.run_instance import SimpleRunPublic
from app.schemas.workflow_config import ExecutionScope
from app.services import project_workflow as project_workflow_service
from app.services import runtime_status
from app.utils.security import UserDep
from app.utils.workflow import create_project_level_run, create_runs_from_samples

if TYPE_CHECKING:
    from app.models.user import User

router = APIRouter()
tag_router = APIRouter()
workflow_router = APIRouter()

STORAGE_PATH = get_settings().persistent.storage.STORAGE_PATH


@tag_router.post('')
def new_tag(
    tag_prop: TagProp,
    session: SessionDep,
    _current_user: UserDep,
) -> ProjectTag:
    """Create a new project tag."""
    tag = ProjectTag(name=tag_prop.name, color=tag_prop.color)
    tag_crud.insert(session, tag)

    return tag


@tag_router.get('')
def get_tag_list(session: SessionDep, _current_user: UserDep) -> list[TagWithCount]:
    """Get all project tags with usage counts."""
    return tag_crud.get_all(session)  # type: ignore


@router.post('')
def new_project(
    project_prop: ProjectCreateProp,
    session: SessionDep,
    current_user: UserDep,
) -> Project:
    """Create a new project."""
    tags = []
    if project_prop.tag_ids:
        tags = tag_crud.get_all_by_id(session, project_prop.tag_ids)

    project = Project(
        owner_id=current_user.id,
        name=project_prop.name,
        description=project_prop.description,
        public=project_prop.public,
        tags=tags,
    )
    project = project_crud.insert(session, project)

    project_path = Path(STORAGE_PATH) / 'proj' / f'{project.id}'
    project_path.mkdir(parents=True, exist_ok=True)

    return project


@router.post('/{project_id}/star')
def star_project(
    project_id: int,
    session: SessionDep,
    current_user: UserDep,
) -> None:
    """Star (bookmark) a project for the current user."""
    project_crud.star_project(session, current_user.id, project_id)  # type: ignore


@router.post('/{project_id}/unstar')
def unstar_project(
    project_id: int,
    session: SessionDep,
    current_user: UserDep,
) -> None:
    """Remove a star (bookmark) from a project."""
    project_crud.unstar_project(session, current_user.id, project_id)  # type: ignore


@router.get('')
def get_project_list(
    session: SessionDep,
    current_user: UserDep,
    filter_type: Annotated[str | None, Query(description='Filter: starred, my, all', alias='filter')] = 'all',
    sort: Annotated[ProjectSort, Query(description='Sort: recent, nameAsc, nameDesc')] = 'recent',
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
) -> Paginated[ProjectPublic]:
    """Get project list.

    - filter=all: Get all visible projects (public or user-owned)
    - filter=starred: Get user's starred projects
    - filter=my: Get user's owned projects
    """
    match filter_type:
        case 'starred':
            projects, total = project_crud.get_page_star(session, limit, offset, current_user.id, sort)  # type: ignore
        case 'my':
            projects, total = project_crud.get_page_my(session, limit, offset, current_user.id, sort)  # type: ignore
        case 'all':
            projects, total = project_crud.get_page(session, limit, offset, current_user.id, sort)  # type: ignore
        case _:
            projects, total = [], 0

    return Paginated(
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + limit < total,
        data=projects,
    )


@router.get('/recent')
def get_recent_project(session: SessionDep, current_user: UserDep) -> ProjectPublic:
    """Get user's recently updated projects."""
    return project_crud.get_recent(session, current_user.id)  # type: ignore


@router.get('/{project_id}')
def get_project(
    project_id: int,
    session: SessionDep,
    current_user: UserDep,
) -> ProjectPublic:
    """Get project detail by ID."""
    return project_crud.get(session, current_user.id, project_id)  # type: ignore


@router.put('/{project_id}')
def update_project(
    project_id: int,
    project_update: ProjectUpdate,
    session: SessionDep,
    current_user: UserDep,
) -> ProjectPublic:
    """Update a project's name, description, or tags."""
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Project not found')
    if project.owner_id != current_user.id:
        raise HTTPException(HTTP_403_FORBIDDEN, 'Only the project owner can update this project')

    project_crud.update(session, project_id, project_update)
    return project_crud.get(session, current_user.id, project_id)


@router.delete('/{project_id}')
def delete_project(
    project_id: int,
    session: SessionDep,
    _current_user: UserDep,
) -> None:
    """Delete project."""
    project_crud.delete(session, project_id)

    project_path = Path(STORAGE_PATH) / 'proj' / f'{project_id}'
    shutil.rmtree(project_path)


# ==================== Sample Management Endpoints (Under Project) ====================


@router.post('/{project_id}/samples')
async def create_sample_in_project(
    project_id: int,
    sample_prop: SampleCreate,
    session: SessionDep,
    current_user: UserDep,
) -> SamplePublic:
    """Create a new sample in the specified project."""
    if sample_prop.project_id != project_id:
        raise HTTPException(400, 'Project ID mismatch')

    sample = Sample(
        # pyrefly: ignore [bad-argument-type]
        owner_id=current_user.id,
        project_id=project_id,
        sample_name=sample_prop.sample_name,
        meta_data=sample_prop.meta_data or {},
    )

    sample_crud.insert(session, sample)
    return sample_crud.get_with_files(session, sample.uid)  # type: ignore


@router.get('/{project_id}/samples')
async def get_project_samples(
    project_id: int,
    session: SessionDep,
    _current_user: UserDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 20,
) -> Paginated[SampleListItem]:
    """Get all samples under the project."""
    samples, total = sample_crud.get_page_by_project(session, project_id, limit, offset)
    return Paginated(
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + limit < total,
        data=samples,
    )


@router.get('/{project_id}/samples/count')
async def get_project_sample_count(
    project_id: int,
    session: SessionDep,
    _current_user: UserDep,
) -> int:
    """Get the number of samples under the project."""
    return sample_crud.get_count(session, project_id)


@router.post('/{project_id}/samples/batch-import')
async def batch_import_samples(
    project_id: int,
    session: SessionDep,
    current_user: UserDep,
    file: Annotated[UploadFile, File()],
) -> SampleBatchImportResult:
    """Batch import samples.

    Upload a CSV file to batch create samples. CSV file format requirements:
    - Encoding: UTF-8
    - Required column: sample_name
    - Optional file columns: r1, r2, single, spectrum, image (file path)
    - Other columns: Custom metadata, will be stored in meta_data

    Example CSV:
    ```
    sample_name,condition,batch,r1,r2
    Sample-001,control,2024-01,/data/s1_R1.fq.gz,/data/s1_R2.fq.gz
    Sample-002,treatment,2024-01,/data/s2_R1.fq.gz,/data/s2_R2.fq.gz
    ```
    """
    # Validate file type
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(400, 'Only CSV files are supported')

    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(400, f'Failed to read file: {e!s}') from e

    # Batch import
    return sample_crud.batch_import_from_csv(session, content, project_id, current_user.id)  # type: ignore


@router.get('/{project_id}/samples/{sample_uid}')
async def get_project_sample(
    project_id: int,
    sample_uid: uuid.UUID,
    session: SessionDep,
    _current_user: UserDep,
) -> SamplePublic:
    """Get the specified sample details under the project."""
    sample = sample_crud.get_with_files(session, sample_uid)
    if not sample:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Sample not found')
    if sample.project_id != project_id:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Sample not found in this project')
    return sample  # type: ignore


@router.put('/{project_id}/samples/{sample_uid}')
async def update_project_sample(
    project_id: int,
    sample_uid: uuid.UUID,
    sample_update: SampleUpdate,
    session: SessionDep,
    _current_user: UserDep,
) -> SamplePublic:
    """Update sample information under the project."""
    existing_sample = sample_crud.get(session, sample_uid)
    if not existing_sample or existing_sample.project_id != project_id:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Sample not found in this project')

    sample_crud.update(session, sample_uid, sample_update)
    return sample_crud.get_with_files(session, sample_uid)  # type: ignore


@router.delete('/{project_id}/samples/{sample_uid}')
async def delete_project_sample(
    project_id: int,
    sample_uid: uuid.UUID,
    session: SessionDep,
    _current_user: UserDep,
) -> dict:
    """Delete a sample under the project."""
    existing_sample = sample_crud.get(session, sample_uid)
    if not existing_sample or existing_sample.project_id != project_id:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Sample not found in this project')

    sample_crud.delete(session, sample_uid)
    return {'message': 'Sample deleted successfully'}


@router.post('/{project_id}/samples/{sample_uid}/files')
async def add_file_to_project_sample(
    project_id: int,
    sample_uid: uuid.UUID,
    file_prop: SampleFileCreate,
    session: SessionDep,
    _current_user: UserDep,
) -> SampleFilePublic:
    """Add a file to a sample under the project."""
    existing_sample = sample_crud.get(session, sample_uid)
    if not existing_sample or existing_sample.project_id != project_id:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Sample not found in this project')

    # File metadata is calculated automatically by the backend.
    sample_file = SampleFiles(
        sample_uid=sample_uid,
        file_path=file_prop.file_path,
        # pyrefly: ignore [bad-argument-type]
        tag=file_prop.tag,
        file_format='',
        file_size=0,
        md5_checksum='',
    )

    return sample_crud.add_file(session, sample_file)  # type: ignore


@router.delete('/{project_id}/samples/{sample_uid}/files/{file_uid}')
async def delete_file_from_project_sample(
    project_id: int,
    sample_uid: uuid.UUID,
    file_uid: uuid.UUID,
    session: SessionDep,
    _current_user: UserDep,
) -> dict:
    """Delete a file from a sample under the project."""
    existing_sample = sample_crud.get(session, sample_uid)
    if not existing_sample or existing_sample.project_id != project_id:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Sample not found in this project')

    sample_crud.delete_file(session, file_uid)
    return {'message': 'File deleted successfully'}


@router.get('/{project_id}/samples/{sample_uid}/files')
async def get_project_sample_files(
    project_id: int,
    sample_uid: uuid.UUID,
    session: SessionDep,
    _current_user: UserDep,
) -> list[SampleFilePublic]:
    """Get all files of a sample under the project."""
    existing_sample = sample_crud.get(session, sample_uid)
    if not existing_sample or existing_sample.project_id != project_id:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Sample not found in this project')

    return sample_crud.get_files(session, sample_uid)  # type: ignore


# ==================== Workflow Management Endpoints (Under Project) ====================


@router.post('/{project_id}/workflows')
async def add_workflow_to_project(
    project_id: int,
    workflow_add: ProjectWorkflowAdd,
    session: SessionDep,
    _current_user: UserDep,
) -> ProjectWorkflowPublic:
    """Add a workflow template to the project.

    Args:
        project_id: Project ID
        workflow_add: Request body containing workflow_uid

    Returns:
        ProjectWorkflowPublic: The added workflow information.
    """
    try:
        return project_workflow_service.add_workflow_to_project(
            session,
            project_id,
            workflow_add.workflow_uid,
        )
    except project_workflow_service.ProjectWorkflowNotFoundError as error:
        raise HTTPException(HTTP_404_NOT_FOUND, str(error)) from error
    except project_workflow_service.ProjectWorkflowConflictError as error:
        raise HTTPException(HTTP_409_CONFLICT, str(error)) from error


@router.get('/{project_id}/workflows')
async def get_project_workflows(
    project_id: int,
    session: SessionDep,
    _current_user: UserDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 20,
) -> list[ProjectWorkflowPublic]:
    """Get all workflow templates in the project.

    Args:
        project_id: Project ID
        offset: Offset
        limit: Limit count

    Returns:
        list[ProjectWorkflowPublic]: List of workflows.
    """
    return project_workflow_service.list_project_workflows(session, project_id, limit, offset)


@router.delete('/{project_id}/workflows/{workflow_uid}')
async def remove_workflow_from_project(
    project_id: int,
    workflow_uid: uuid.UUID,
    session: SessionDep,
    _current_user: UserDep,
) -> dict:
    """Remove a workflow template from the project.

    Args:
        project_id: Project ID
        workflow_uid: Workflow template UID

    Returns:
        Success message.
    """
    try:
        project_workflow_service.remove_workflow_from_project(session, project_id, workflow_uid)
    except project_workflow_service.ProjectWorkflowNotFoundError as error:
        raise HTTPException(HTTP_404_NOT_FOUND, str(error)) from error
    return {'message': 'Workflow removed successfully'}


@router.post('/{project_id}/workflows/{workflow_uid}/run')
async def run_workflow_on_samples(
    project_id: int,
    workflow_uid: uuid.UUID,
    run_request: WorkflowRunRequest,
    session: SessionDep,
    current_user: UserDep,
) -> WorkflowRunResult:
    """Run a workflow.

    - Sample-level workflow: Requires sample_uids, creates a run instance for each sample
    - Project-level workflow: Does not require sample_uids, creates only one run instance

    Args:
        project_id: Project ID
        workflow_uid: Workflow template UID
        run_request: Contains optional list of sample UIDs and optional run name prefix

    Returns:
        WorkflowRunResult: Contains the list of created run instance UIDs.
    """
    # Check if workflow is in the project
    if not project_workflow_crud.exists(session, project_id, workflow_uid):
        raise HTTPException(HTTP_404_NOT_FOUND, 'Workflow not found in this project')

    # Get workflow to determine execution scope
    workflow = wf_crud.get(session, workflow_uid)  # type: ignore
    if not workflow:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Workflow not found')

    # Dispatch based on execution scope
    if workflow.execution_scope == ExecutionScope.PROJECT_LEVEL:
        # Project-level execution
        if run_request.sample_uids:
            raise HTTPException(
                HTTP_400_BAD_REQUEST,
                'Project-level workflows do not accept sample_uids',
            )

        return create_project_level_run(
            session,
            project_id,
            workflow_uid,
            cast('User', current_user),
            run_request.run_name_prefix,
            auto_summary=run_request.auto_summary,
            summary_prompt=workflow.summary_prompt,
        )
    # Sample-level execution
    if not run_request.sample_uids:
        raise HTTPException(
            HTTP_400_BAD_REQUEST,
            'Sample-level workflows require sample_uids',
        )

    logger.debug(run_request)
    return create_runs_from_samples(
        session,
        project_id,
        workflow_uid,
        run_request.sample_uids,
        cast('User', current_user),
        run_request.run_name_prefix,
        auto_summary=run_request.auto_summary,
        summary_prompt=workflow.summary_prompt,
    )


@router.get('/{project_id}/workflows/{workflow_uid}/package')
async def package_workflow_runs(
    project_id: int,
    workflow_uid: uuid.UUID,
    session: SessionDep,
    _current_user: UserDep,
) -> StreamingResponse:
    """Download all run instance output directories for a workflow as a zip archive.

    Args:
        project_id: Project ID
        workflow_uid: Workflow template UID

    Returns:
        StreamingResponse: A zip file containing all run output directories.
    """
    try:
        return await project_workflow_service.package_workflow_runs(session, project_id, workflow_uid)
    except project_workflow_service.ProjectWorkflowNotFoundError as error:
        raise HTTPException(HTTP_404_NOT_FOUND, str(error)) from error


@router.get('/{project_id}/runs')
async def get_project_runs(
    project_id: int,
    session: SessionDep,
    _current_user: UserDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 20,
) -> Paginated[SimpleRunPublic]:
    """Get all run instances of the project.

    Args:
        project_id: Project ID
        offset: Offset
        limit: Limit count

    Returns:
        Paginated[SimpleRunPublic]: Page of run instances with refreshed runtime task statistics.
    """
    runs, total = run_crud.get_runs_by_project_page(session, project_id, limit, offset)

    result = []
    for run in runs:
        stats = runtime_status.refresh_run_status(run)
        run_data = SimpleRunPublic.model_validate(run)
        run_data.task_statistics = stats
        result.append(run_data)

    return Paginated(
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + limit < total,
        data=result,
    )


@router.get('/{project_id}/runs/stats')
async def get_project_run_stats(
    project_id: int,
    session: SessionDep,
    _current_user: UserDep,
) -> Statistics:
    """Get run instance statistics by status for a project.

    Args:
        project_id: Project ID

    Returns:
        Statistics: Counts of total, success, error, running, waiting run instances.
    """
    stats = run_crud.get_project_run_stats(session, project_id)
    return runtime_status.apply_effective_nonterminal_run_stats(
        stats,
        run_crud.get_nonterminal_runs_by_project(session, project_id),
    )


@router.get('/{project_id}/runs/{run_uid}')
async def get_project_run(
    project_id: int,
    run_uid: uuid.UUID,
    session: SessionDep,
    _current_user: UserDep,
) -> SimpleRunPublic:
    """Get the details of the specified run instance in the project.

    Args:
        project_id: Project ID
        run_uid: Run instance UID

    Returns:
        SimpleRunPublic: Details of the run instance with refreshed runtime task statistics.
    """
    run = run_crud.get(session, str(run_uid))
    if not run or run.project_id != project_id:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Run not found in this project')

    stats = runtime_status.refresh_run_status(run)
    run_data = SimpleRunPublic.model_validate(run)
    run_data.task_statistics = stats

    return run_data


# ==================== Project File Mapping Management Endpoints ====================


@router.post('/{project_id}/file-mappings')
async def create_project_file_mapping(
    project_id: int,
    mapping: ProjectFileMappingCreate,
    session: SessionDep,
    _current_user: UserDep,
) -> ProjectFileMappingPublic:
    """Create a new file mapping for the project."""
    result = project_file_mapping_crud.create(
        session, project_id, mapping.keyword, mapping.file_path, mapping.description
    )
    return ProjectFileMappingPublic.model_validate(result)


@router.get('/{project_id}/file-mappings')
async def get_project_file_mappings(
    project_id: int,
    session: SessionDep,
    _current_user: UserDep,
) -> list[ProjectFileMappingPublic]:
    """Get all file mappings of the project."""
    mappings = project_file_mapping_crud.get_all_by_project(session, project_id)
    return [ProjectFileMappingPublic.model_validate(m) for m in mappings]


@router.get('/{project_id}/file-mappings/{keyword}')
async def get_project_file_mapping(
    project_id: int,
    keyword: str,
    session: SessionDep,
    _current_user: UserDep,
) -> ProjectFileMappingPublic:
    """Get a specific file mapping by keyword."""
    mapping = project_file_mapping_crud.get_by_keyword(session, project_id, keyword)
    if not mapping:
        raise HTTPException(HTTP_404_NOT_FOUND, f'File mapping not found: {keyword}')
    return ProjectFileMappingPublic.model_validate(mapping)


@router.put('/{project_id}/file-mappings/{mapping_id}')
async def update_project_file_mapping(
    project_id: int,  # noqa: ARG001
    mapping_id: int,
    mapping_update: ProjectFileMappingUpdate,
    session: SessionDep,
    _current_user: UserDep,
) -> ProjectFileMappingPublic:
    """Update file mapping."""
    result = project_file_mapping_crud.update(session, mapping_id, mapping_update.file_path, mapping_update.description)
    return ProjectFileMappingPublic.model_validate(result)


@router.delete('/{project_id}/file-mappings/{mapping_id}')
async def delete_project_file_mapping(
    project_id: int,  # noqa: ARG001
    mapping_id: int,
    session: SessionDep,
    _current_user: UserDep,
) -> dict:
    """Delete file mapping."""
    project_file_mapping_crud.delete(session, mapping_id)
    return {'message': 'File mapping deleted successfully'}
