import uuid
from typing import Annotated, NoReturn

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_403_FORBIDDEN, HTTP_404_NOT_FOUND, HTTP_409_CONFLICT

from bioflow_ai.core.auth import AdminDep, MemberDep, UserDep
from bioflow_ai.crud import project as project_crud
from bioflow_ai.crud import project_file_mapping as project_file_mapping_crud
from bioflow_ai.crud import projetc_tag as tag_crud
from bioflow_ai.crud import workflow as wf_crud
from bioflow_ai.crud.resource import sample as sample_crud
from bioflow_ai.infrastructure.database import SessionDep
from bioflow_ai.models.resource import Sample, SampleFiles
from bioflow_ai.schemas.agent_file import AgentFilePublic, AgentFileUpdate
from bioflow_ai.schemas.common import Paginated
from bioflow_ai.schemas.project import (
    ProjectCreateProp,
    ProjectCreateResult,
    ProjectFileMappingCreate,
    ProjectFileMappingPublic,
    ProjectFileMappingUpdate,
    ProjectPublic,
    ProjectSort,
    ProjectTagPublic,
    ProjectUpdate,
    TagProp,
    TagWithCount,
)
from bioflow_ai.schemas.project_workflow import ProjectWorkflowAdd, ProjectWorkflowPublic
from bioflow_ai.schemas.resource import (
    SampleBatchImportResult,
    SampleCreate,
    SampleFileCreate,
    SampleFilePublic,
    SampleListItem,
    SamplePublic,
    SampleUpdate,
)
from bioflow_ai.schemas.user_config import UserRole
from bioflow_ai.services import agent_file as agent_file_service
from bioflow_ai.services import project as project_service
from bioflow_ai.services import project_file_mapping as project_file_mapping_service
from bioflow_ai.services import project_workflow as project_workflow_service
from bioflow_ai.services import sample as sample_service
from bioflow_ai.utils.file_browser import serve_file

router = APIRouter()
tag_router = APIRouter()
workflow_router = APIRouter()


@tag_router.post('')
def new_tag(
    tag_prop: TagProp,
    session: SessionDep,
    _current_user: AdminDep,
) -> ProjectTagPublic:
    """Create a new project tag."""
    return ProjectTagPublic.model_validate(project_service.create_tag(session, tag_prop))


@tag_router.get('')
def get_tag_list(session: SessionDep, _current_user: UserDep) -> list[TagWithCount]:
    """Get all project tags with usage counts."""
    return tag_crud.get_all(session)


@router.post('')
def new_project(
    project_prop: ProjectCreateProp,
    session: SessionDep,
    current_user: MemberDep,
) -> ProjectCreateResult:
    """Create a new project."""
    project = project_service.create_project(session, project_prop, current_user.id)
    return ProjectCreateResult.model_validate(project)


@router.post('/{project_id}/star')
def star_project(
    project_id: int,
    session: SessionDep,
    current_user: UserDep,
) -> None:
    """Star (bookmark) a project for the current user."""
    project_service.star_project(session, current_user.id, project_id)


@router.post('/{project_id}/unstar')
def unstar_project(
    project_id: int,
    session: SessionDep,
    current_user: UserDep,
) -> None:
    """Remove a star (bookmark) from a project."""
    project_service.unstar_project(session, current_user.id, project_id)


@router.get('')
def get_project_list(
    session: SessionDep,
    current_user: UserDep,
    filter_type: Annotated[str | None, Query(description='Filter: starred, my, all', alias='filter')] = 'all',
    sort: Annotated[ProjectSort, Query(description='Sort: recent, nameAsc, nameDesc')] = 'recent',
    search: Annotated[
        str | None,
        Query(max_length=200, description='Search project names by text fragment'),
    ] = None,
    tag_id: Annotated[int | None, Query(gt=0, description='Filter projects by tag ID')] = None,
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
            projects, total = project_crud.get_page_star(
                session,
                limit,
                offset,
                current_user.id,
                sort,
                search,
                tag_id,
                include_private=current_user.role == UserRole.ADMIN,
                include_owned=current_user.role >= UserRole.MEMBER,
            )
        case 'my':
            if current_user.role >= UserRole.MEMBER:
                projects, total = project_crud.get_page_my(
                    session, limit, offset, current_user.id, sort, search, tag_id
                )
            else:
                projects, total = [], 0
        case 'all':
            projects, total = project_crud.get_page(
                session,
                limit,
                offset,
                current_user.id,
                sort,
                search,
                tag_id,
                include_private=current_user.role == UserRole.ADMIN,
                include_owned=current_user.role >= UserRole.MEMBER,
            )
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
    project = project_crud.get_recent(
        session,
        current_user.id,
        include_owned=current_user.role >= UserRole.MEMBER,
    )
    if project is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Project not found')
    return project


@router.get('/{project_id}')
def get_project(
    project_id: int,
    session: SessionDep,
    current_user: UserDep,
) -> ProjectPublic:
    """Get project detail by ID."""
    project = project_crud.get(session, current_user.id, project_id)
    if project is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Project not found')
    return project


def _raise_agent_file_error(error: Exception) -> NoReturn:
    if isinstance(error, agent_file_service.AgentFileNotFoundError):
        raise HTTPException(HTTP_404_NOT_FOUND, 'Agent file not found') from error
    if isinstance(error, agent_file_service.AgentFileReadOnlyError):
        raise HTTPException(HTTP_403_FORBIDDEN, str(error)) from error
    if isinstance(error, agent_file_service.AgentFileConflictError):
        raise HTTPException(HTTP_409_CONFLICT, str(error)) from error
    if isinstance(error, agent_file_service.AgentFileValidationError):
        raise HTTPException(HTTP_400_BAD_REQUEST, str(error)) from error
    raise error


@router.get('/{project_id}/agent-files')
def get_project_agent_files(
    project_id: int,
    session: SessionDep,
    current_user: UserDep,
) -> list[AgentFilePublic]:
    """List current project documents and immutable reports produced by agents."""
    try:
        return agent_file_service.list_project_files(session, project_id, current_user.id)
    except agent_file_service.AgentFileError as error:
        _raise_agent_file_error(error)


@router.get('/{project_id}/agent-files/{file_id}')
async def get_project_agent_file(
    project_id: int,
    file_id: str,
    session: SessionDep,
    current_user: UserDep,
    *,
    download: Annotated[bool, Query()] = False,
) -> Response:
    """Preview or download one controlled agent file by logical ID."""
    try:
        path, metadata = agent_file_service.get_file(session, project_id, current_user.id, file_id)
    except agent_file_service.AgentFileError as error:
        _raise_agent_file_error(error)
    if download:
        return FileResponse(
            path,
            media_type='text/markdown',
            filename=metadata.name,
            headers={'Cache-Control': 'private, no-store'},
        )
    response = await serve_file(path)
    response.headers['Cache-Control'] = 'private, no-store'
    return response


@router.patch('/{project_id}/agent-files/{file_id}')
def update_project_agent_plan(
    project_id: int,
    file_id: str,
    data: AgentFileUpdate,
    session: SessionDep,
    current_user: UserDep,
) -> AgentFilePublic:
    """Update the current project plan when no agent run is active."""
    try:
        return agent_file_service.update_file(
            session,
            project_id,
            current_user.id,
            file_id,
            data.content,
            data.expected_revision,
        )
    except agent_file_service.AgentFileError as error:
        _raise_agent_file_error(error)


@router.put('/{project_id}')
def update_project(
    project_id: int,
    project_update: ProjectUpdate,
    session: SessionDep,
    current_user: UserDep,
) -> ProjectPublic:
    """Update a project's name, description, or tags."""
    try:
        project_service.update_project(
            session,
            project_id,
            project_update,
            current_user_id=current_user.id,
            is_admin=current_user.role == UserRole.ADMIN,
        )
    except project_service.ProjectNotFoundError as error:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Project not found') from error
    except project_service.ProjectPermissionError as error:
        raise HTTPException(HTTP_403_FORBIDDEN, 'Only the project owner can update this project') from error
    project = project_crud.get(session, current_user.id, project_id)
    if project is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Project not found')
    return project


@router.delete('/{project_id}')
def delete_project(
    project_id: int,
    session: SessionDep,
    _current_user: UserDep,
) -> None:
    """Delete project."""
    project_service.delete_project(session, project_id)


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
        owner_id=current_user.id,
        project_id=project_id,
        sample_name=sample_prop.sample_name,
        meta_data=sample_prop.meta_data or {},
    )

    sample_service.create_sample(session, sample)
    loaded = sample_crud.get_with_files(session, sample.uid)
    if loaded is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Sample not found')
    return SamplePublic.model_validate(loaded)


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
    try:
        return sample_service.import_samples_from_csv(session, content, project_id, current_user.id)
    except sample_service.SampleCsvError as error:
        raise HTTPException(400, str(error)) from error


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
    return SamplePublic.model_validate(sample)


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

    sample_service.update_sample(session, sample_uid, sample_update)
    sample = sample_crud.get_with_files(session, sample_uid)
    if sample is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Sample not found')
    return SamplePublic.model_validate(sample)


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

    sample_service.delete_sample(session, sample_uid)
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

    return SampleFilePublic.model_validate(sample_service.add_file(session, sample_file))


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

    try:
        sample_service.delete_file(session, file_uid)
    except sample_service.SampleFileNotFoundError as error:
        raise HTTPException(HTTP_404_NOT_FOUND, 'File not found') from error
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

    return [SampleFilePublic.model_validate(sample_file) for sample_file in sample_crud.get_files(session, sample_uid)]


# ==================== Workflow Management Endpoints (Under Project) ====================


@router.post('/{project_id}/workflows')
async def add_workflow_to_project(
    project_id: int,
    workflow_add: ProjectWorkflowAdd,
    session: SessionDep,
    current_user: UserDep,
) -> ProjectWorkflowPublic:
    """Add a workflow template to the project.

    Args:
        project_id: Project ID
        workflow_add: Request body containing workflow_uid

    Returns:
        ProjectWorkflowPublic: The added workflow information.
    """
    workflow = wf_crud.get(session, workflow_add.workflow_uid)
    if workflow is None or not (
        workflow.public or workflow.owner_id == current_user.id or current_user.role == UserRole.ADMIN
    ):
        raise HTTPException(HTTP_404_NOT_FOUND, 'Workflow not found')
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


# ==================== Project File Mapping Management Endpoints ====================


@router.post('/{project_id}/file-mappings')
async def create_project_file_mapping(
    project_id: int,
    mapping: ProjectFileMappingCreate,
    session: SessionDep,
    _current_user: UserDep,
) -> ProjectFileMappingPublic:
    """Create a new file mapping for the project."""
    result = project_file_mapping_service.create_mapping(
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
    try:
        result = project_file_mapping_service.update_mapping(
            session,
            mapping_id,
            file_path=mapping_update.file_path,
            description=mapping_update.description,
        )
    except project_file_mapping_service.ProjectFileMappingNotFoundError as error:
        raise HTTPException(HTTP_404_NOT_FOUND, 'File mapping not found') from error
    return ProjectFileMappingPublic.model_validate(result)


@router.delete('/{project_id}/file-mappings/{mapping_id}')
async def delete_project_file_mapping(
    project_id: int,  # noqa: ARG001
    mapping_id: int,
    session: SessionDep,
    _current_user: UserDep,
) -> dict:
    """Delete file mapping."""
    try:
        project_file_mapping_service.delete_mapping(session, mapping_id)
    except project_file_mapping_service.ProjectFileMappingNotFoundError as error:
        raise HTTPException(HTTP_404_NOT_FOUND, 'File mapping not found') from error
    return {'message': 'File mapping deleted successfully'}
