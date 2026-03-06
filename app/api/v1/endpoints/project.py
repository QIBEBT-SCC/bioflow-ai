import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from loguru import logger
from starlette.status import HTTP_404_NOT_FOUND

from app.core.redis_client import monitor_redis_client
from app.core.session import SessionDep
from app.crud import project as project_crud
from app.crud import project_workflow as project_workflow_crud
from app.crud import projetc_tag as tag_crud
from app.crud import run_instance as run_crud
from app.crud.resource import sample as sample_crud
from app.models.project import (
    Project,
    ProjectTag,
)
from app.models.resource import Sample, SampleFiles
from app.schemas.project import ProjectCreateProp, ProjectPublic, TagProp, TagWithCount
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
from app.utils.security import UserDep
from app.utils.workflow import create_runs_from_samples, get_run_stats_from_redis

router = APIRouter()
tag_router = APIRouter()
workflow_router = APIRouter()


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
    return tag_crud.get_all(session)


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
    project_crud.insert(session, project)
    return project


@router.post('/{project_id}/star')
def star_project(
    project_id: int,
    session: SessionDep,
    current_user: UserDep,
) -> None:
    """Star (bookmark) a project for the current user."""
    project_crud.star_project(session, current_user.id, project_id)


@router.post('/{project_id}/unstar')
def unstar_project(
    project_id: int,
    session: SessionDep,
    current_user: UserDep,
) -> None:
    """Remove a star (bookmark) from a project."""
    project_crud.unstar_project(session, current_user.id, project_id)


@router.get('')
def get_project_list(
    session: SessionDep,
    current_user: UserDep,
    filter_type: Annotated[Optional[str], Query(description='过滤条件: starred, my, all', alias='filter')] = 'all',
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
) -> list[ProjectPublic]:
    """获取项目列表.

    - filter=all: 获取所有可见项目(公开的或用户拥有的)
    - filter=starred: 获取用户收藏的项目
    - filter=my: 获取用户拥有的项目
    """
    match filter_type:
        case 'starred':
            return project_crud.get_all_star(session, limit, offset, current_user.id)
        case 'my':
            return project_crud.get_all_my(session, limit, offset, current_user.id)
        case 'all':
            return project_crud.get_all(session, limit, offset, current_user.id)
        case _:
            return []


@router.get('/recent')
def get_recent_project(session: SessionDep, current_user: UserDep) -> ProjectPublic:
    """获取用户最近更新的项目."""
    return project_crud.get_recent(session, current_user.id)


@router.get('/{project_id}')
def get_project(
    project_id: int,
    session: SessionDep,
    current_user: UserDep,
) -> ProjectPublic:
    """Get project detail by ID."""
    return project_crud.get(session, current_user.id, project_id)


# ==================== 样本管理端点(在项目下) ====================


@router.post('/{project_id}/samples')
async def create_sample_in_project(
    project_id: int,
    sample_prop: SampleCreate,
    session: SessionDep,
    current_user: UserDep,
) -> SamplePublic:
    """在指定项目中创建新样本."""
    # 确保样本属于指定的项目
    if sample_prop.project_id != project_id:
        raise HTTPException(400, 'Project ID mismatch')

    sample = Sample(
        owner_id=current_user.id,
        project_id=project_id,
        sample_name=sample_prop.sample_name,
        meta_data=sample_prop.meta_data or {},
    )

    # 如果提供了文件信息, 创建文件记录(文件信息由后端自动计算)
    files = []
    if sample_prop.files:
        for file_prop in sample_prop.files:
            # 只需要 data_type 和 file_path, 其他信息由 CRUD 层自动计算
            sample_file = SampleFiles(
                sample_uid=sample.uid,
                data_type=file_prop.data_type,
                file_path=file_prop.file_path,
                file_format='',  # 将由 CRUD 自动计算
                file_size=0,  # 将由 CRUD 自动计算
                md5_checksum='',  # 将由 CRUD 自动计算
            )
            files.append(sample_file)

    sample_crud.insert(session, sample, files)
    return sample_crud.get_with_files(session, sample.uid)


@router.get('/{project_id}/samples')
async def get_project_samples(
    project_id: int,
    session: SessionDep,
    _current_user: UserDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 20,
) -> list[SampleListItem]:
    """获取项目下的所有样本."""
    return sample_crud.get_all_by_project(session, project_id, limit, offset)


@router.get('/{project_id}/samples/count')
async def get_project_sample_count(
    project_id: int,
    session: SessionDep,
    _current_user: UserDep,
) -> int:
    """获取项目下的样本数量."""
    return sample_crud.get_count_by_project(session, project_id)


@router.post('/{project_id}/samples/batch-import')
async def batch_import_samples(
    project_id: int,
    session: SessionDep,
    current_user: UserDep,
    file: Annotated[UploadFile, File()],
) -> SampleBatchImportResult:
    """批量导入样本.

    上传 CSV 文件批量创建样本。CSV 文件格式要求:
    - 编码: UTF-8
    - 必需列: sample_name
    - 可选文件列: r1, r2, single, spectrum, image(文件路径)
    - 其他列: 自定义元数据, 将存储在 meta_data 中

    示例 CSV:
    ```
    sample_name,condition,batch,r1,r2
    Sample-001,control,2024-01,/data/s1_R1.fq.gz,/data/s1_R2.fq.gz
    Sample-002,treatment,2024-01,/data/s2_R1.fq.gz,/data/s2_R2.fq.gz
    ```
    """
    # 验证文件类型
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(400, '只支持 CSV 文件')

    # 读取文件内容
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(400, f'读取文件失败: {e!s}') from e

    # 批量导入
    return sample_crud.batch_import_from_csv(session, content, project_id, current_user.id)


@router.get('/{project_id}/samples/{sample_uid}')
async def get_project_sample(
    project_id: int,
    sample_uid: uuid.UUID,
    session: SessionDep,
    _current_user: UserDep,
) -> SamplePublic:
    """获取项目下的指定样本详情."""
    sample = sample_crud.get_with_files(session, sample_uid)
    if not sample:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Sample not found')
    if sample.project_id != project_id:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Sample not found in this project')
    return sample


@router.put('/{project_id}/samples/{sample_uid}')
async def update_project_sample(
    project_id: int,
    sample_uid: uuid.UUID,
    sample_update: SampleUpdate,
    session: SessionDep,
    _current_user: UserDep,
) -> SamplePublic:
    """更新项目下的样本信息."""
    existing_sample = sample_crud.get(session, sample_uid)
    if not existing_sample or existing_sample.project_id != project_id:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Sample not found in this project')

    sample_crud.update(session, sample_uid, sample_update)
    return sample_crud.get_with_files(session, sample_uid)


@router.delete('/{project_id}/samples/{sample_uid}')
async def delete_project_sample(
    project_id: int,
    sample_uid: uuid.UUID,
    session: SessionDep,
    _current_user: UserDep,
) -> dict:
    """删除项目下的样本."""
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
    """为项目下的样本添加文件."""
    existing_sample = sample_crud.get(session, sample_uid)
    if not existing_sample or existing_sample.project_id != project_id:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Sample not found in this project')

    # 只需要提供 data_type 和 file_path, 其他信息由后端自动计算
    sample_file = SampleFiles(
        sample_uid=sample_uid,
        data_type=file_prop.data_type,
        file_path=file_prop.file_path,
        file_format='',  # 将由 CRUD 自动计算
        file_size=0,  # 将由 CRUD 自动计算
        md5_checksum='',  # 将由 CRUD 自动计算
    )

    return sample_crud.add_file(session, sample_file)


@router.delete('/{project_id}/samples/{sample_uid}/files/{file_uid}')
async def delete_file_from_project_sample(
    project_id: int,
    sample_uid: uuid.UUID,
    file_uid: uuid.UUID,
    session: SessionDep,
    _current_user: UserDep,
) -> dict:
    """删除项目下样本的文件."""
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
    """获取项目下样本的所有文件."""
    existing_sample = sample_crud.get(session, sample_uid)
    if not existing_sample or existing_sample.project_id != project_id:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Sample not found in this project')

    return sample_crud.get_files(session, sample_uid)


# ==================== 工作流管理端点(在项目下) ====================


@router.post('/{project_id}/workflows')
async def add_workflow_to_project(
    project_id: int,
    workflow_add: ProjectWorkflowAdd,
    session: SessionDep,
    _current_user: UserDep,
) -> ProjectWorkflowPublic:
    """将工作流模板添加到项目中.

    Args:
        project_id: 项目 ID
        workflow_add: 包含 workflow_uid 的请求体

    Returns:
        ProjectWorkflowPublic: 添加的工作流信息.
    """
    project_workflow_crud.add_workflow_to_project(session, project_id, workflow_add.workflow_uid)

    # 返回工作流信息
    workflows = project_workflow_crud.get_project_workflows(session, project_id, limit=1000)
    for wf in workflows:
        if wf.workflow_uid == workflow_add.workflow_uid:
            return wf

    raise HTTPException(HTTP_404_NOT_FOUND, 'Workflow not found')


@router.get('/{project_id}/workflows')
async def get_project_workflows(
    project_id: int,
    session: SessionDep,
    _current_user: UserDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 20,
) -> list[ProjectWorkflowPublic]:
    """获取项目中的所有工作流模板.

    Args:
        project_id: 项目 ID
        offset: 偏移量
        limit: 限制数量

    Returns:
        list[ProjectWorkflowPublic]: 工作流列表.
    """
    return project_workflow_crud.get_project_workflows(session, project_id, limit, offset)


@router.delete('/{project_id}/workflows/{workflow_uid}')
async def remove_workflow_from_project(
    project_id: int,
    workflow_uid: uuid.UUID,
    session: SessionDep,
    _current_user: UserDep,
) -> dict:
    """从项目中移除工作流模板.

    Args:
        project_id: 项目 ID
        workflow_uid: 工作流模板 UID

    Returns:
        成功消息.
    """
    project_workflow_crud.remove_workflow_from_project(session, project_id, workflow_uid)
    return {'message': 'Workflow removed successfully'}


@router.post('/{project_id}/workflows/{workflow_uid}/run')
async def run_workflow_on_samples(
    project_id: int,
    workflow_uid: uuid.UUID,
    run_request: WorkflowRunRequest,
    session: SessionDep,
    current_user: UserDep,
) -> WorkflowRunResult:
    """在选定的样本上运行工作流.

    为每个样本创建一个独立的 RunInstance.

    Args:
        project_id: 项目 ID
        workflow_uid: 工作流模板 UID
        run_request: 包含样本 UID 列表和可选的运行名称前缀

    Returns:
        WorkflowRunResult: 包含创建的运行实例 UID 列表.
    """
    # 检查工作流是否在项目中
    if not project_workflow_crud.is_workflow_in_project(session, project_id, workflow_uid):
        raise HTTPException(HTTP_404_NOT_FOUND, 'Workflow not found in this project')

    # 批量创建运行实例
    logger.debug(run_request)
    return create_runs_from_samples(
        session, project_id, workflow_uid, run_request.sample_uids, current_user, run_request.run_name_prefix
    )


@router.get('/{project_id}/runs')
async def get_project_runs(
    project_id: int,
    session: SessionDep,
    _current_user: UserDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 20,
) -> list[SimpleRunPublic]:
    """获取项目的所有运行实例.

    Args:
        project_id: 项目 ID
        offset: 偏移量
        limit: 限制数量

    Returns:
        list[SimpleRunPublic]: 运行实例列表(包含从 Redis 读取的实时统计信息).
    """
    runs = run_crud.get_runs_by_project(session, project_id, limit, offset)

    # 从 Redis 读取每个运行实例的统计信息
    result = []
    for run in runs:
        run_data = SimpleRunPublic.model_validate(run)

        # 尝试从 Redis 读取统计信息
        stats = await get_run_stats_from_redis(monitor_redis_client, str(run.uid))
        if stats:
            run_data.task_statistics = stats
        else:
            # 如果 Redis 中没有, 从数据库计算
            run_data.task_statistics = run_crud.get_run_task_stats(session, str(run.uid))

        result.append(run_data)

    return result


@router.get('/{project_id}/runs/count')
async def get_project_run_count(
    project_id: int,
    session: SessionDep,
    _current_user: UserDep,
) -> int:
    """获取项目的运行实例数量.

    Args:
        project_id: 项目 ID

    Returns:
        int: 运行实例数量.
    """
    return run_crud.get_project_run_count(session, project_id)


@router.get('/{project_id}/runs/{run_uid}')
async def get_project_run(
    project_id: int,
    run_uid: uuid.UUID,
    session: SessionDep,
    _current_user: UserDep,
) -> SimpleRunPublic:
    """获取项目中的指定运行实例详情.

    Args:
        project_id: 项目 ID
        run_uid: 运行实例 UID

    Returns:
        SimpleRunPublic: 运行实例详情(包含从 Redis 读取的实时统计信息).
    """
    run = run_crud.get(session, str(run_uid))
    if not run or run.project_id != project_id:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Run not found in this project')

    run_data = SimpleRunPublic.model_validate(run)

    # 尝试从 Redis 读取统计信息
    stats = await get_run_stats_from_redis(monitor_redis_client, str(run_uid))
    if stats:
        run_data.task_statistics = stats
    else:
        # 如果 Redis 中没有, 从数据库计算
        run_data.task_statistics = run_crud.get_run_task_stats(session, str(run_uid))

    return run_data
